"""E-RQ3: recovery + hysteresis controller ablation (Kaggle T4).

Tests RQ3: at a matched fine budget, does recovery + hysteresis reduce mode
oscillation without worsening terminal / area-under-horizon reference regret,
relative to threshold-only switching?

Design (ablation isolates the controller while holding the signal fixed):
- The per-decision signal trace (warp, direction -1) is computed once from a
  development probe (F4-lite fixed_fine videos) and embedded here, so all
  controllers are compared on identical signals.
- Switch-producing threshold: tau at the pooled median (~50% fine decisions).
- Controllers (6): threshold_only, hysteresis_only, recovery_only, full,
  threshold_only_noisy, full_noisy. The *_noisy variants add deterministic
  Gaussian noise (sigma = 2*sigma_u) to the signal, directly stress-testing the
  hysteresis band (Proposition 1: Lambda > 2 sigma_u).
- All controllers share one coarse prefix per prompt (same seed) and the same
  canonical noise, and are scored against a regenerated c1-1 step-4 reference.

Outputs: results/rq3_controller/results.jsonl, summary JSON, videos.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import gc
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("/kaggle/working/results/rq3_controller")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR = OUT_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
RAW = OUT_DIR / "results.jsonl"
SUMMARY = OUT_DIR / "rq3_controller_summary.json"

MANIFEST_VERSION = "f4lite-v1"
SEED = 17
HORIZON = 57
FPS = 8
MID = [15 / 16, 5 / 6, 5 / 8]

TAU_PSI = -4.416192173957825     # fine if psi >= TAU_PSI (psi = -warp); pooled median
SIGMA_U = 3.915166421304675      # pooled std of warp
LAMBDA = 2.0 * SIGMA_U           # hysteresis band width
RECOVERY_R = 2
NOISE_SIGMA = 2.0 * SIGMA_U      # stress-test noise
NOISE_SEED = 20261026

PROMPTS = [
    {"stratum": "object_class", "prompt": "a boat"},
    {"stratum": "object_class", "prompt": "a book"},
    {"stratum": "scene", "prompt": "baseball stadium"},
    {"stratum": "scene", "prompt": "basement"},
    {"stratum": "motion", "prompt": "a bird flying over a snowy forest"},
    {"stratum": "motion", "prompt": "a bird soaring gracefully in the sky"},
    {"stratum": "human_action", "prompt": "A person is bungee jumping"},
    {"stratum": "human_action", "prompt": "A person is canoeing or kayaking"},
]

# per-prompt warp trace over the 4 decisions (development probe), psi = -warp
WARP_TRACE = {
    0: [6.4296, 6.1398, 6.5066, 6.0016],
    1: [0.7756, 2.4940, 2.4610, 1.4452],
    2: [1.3747, 1.0696, 2.0111, 1.0187],
    3: [7.3146, 3.5810, 1.2681, 1.2173],
    4: [20.3692, 5.2514, 5.7266, 6.6592],
    5: [7.8138, 2.4693, 3.4919, 5.7752],
    6: [7.9946, 3.5061, 2.6829, 2.2830],
    7: [10.9930, 8.5829, 7.3928, 9.0702],
}

CONTROLLERS = ("threshold_only", "hysteresis_only", "recovery_only", "full", "threshold_only_noisy", "full_noisy")


def log(message):
    print(message, flush=True)


def find_under(root, filename):
    root = Path(root)
    return next((p for p in root.rglob(filename) if p.is_file()), None) if root.exists() else None


def install_deps():
    packages = [
        "einops", "transformers", "sentencepiece", "ftfy", "regex", "imageio",
        "imageio-ffmpeg", "tqdm", "torchvision", "fvcore", "omegaconf",
        "attrs", "pyyaml", "iopath", "termcolor", "pynvml", "pandas",
        "loguru", "safetensors", "cloudpickle", "dill", "matplotlib",
        "packaging", "opencv-python-headless",
    ]
    for package in packages:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", package], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"pip install failed for {package}: {result.stderr[-500:]}")


def patch_file(path, replacements):
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in source:
            raise RuntimeError(f"missing upstream patch target in {path.name}: {old[:80]}")
        source = source.replace(old, new)
    path.write_text(source, encoding="utf-8")


def append_result(item):
    with RAW.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def completed_keys():
    keys = set()
    if RAW.exists():
        for line in RAW.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                if item.get("status") == "complete":
                    keys.add(item["sample_key"])
            except Exception:
                pass
    return keys


def to_frames(video):
    pixels = (1 + video.float().cpu().clamp(-1, 1)) / 2
    if pixels.ndim == 5:
        pixels = pixels[0]
    return (pixels.permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8)


def read_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(frames, dtype=np.uint8)


def save_mp4(path, frames):
    import imageio
    writer = imageio.get_writer(str(path), fps=FPS, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def segment_bounds(group):
    start = 1 + 4 * (3 * group - 1)
    end = HORIZON if group == 4 else 1 + 4 * (3 * (group + 1) - 1)
    return start, end


# ---- controller state machines (psi = -warp; fine when psi is high) ----

def threshold_only(psi):
    return ["F" if p >= TAU_PSI else "C" for p in psi]


def hysteresis_only(psi):
    mode = "C"
    out = []
    for p in psi:
        if mode == "C" and p >= TAU_PSI:
            mode = "F"
        elif mode == "F" and p <= TAU_PSI - LAMBDA:
            mode = "C"
        out.append(mode)
    return out


def recovery_only(psi):
    mode, r = "C", 0
    out = []
    for p in psi:
        if p >= TAU_PSI:
            mode, r = "F", RECOVERY_R
        elif r > 0:
            mode, r = "F", r - 1
        else:
            mode = "C"
        out.append(mode)
    return out


def full_controller(psi):
    mode, r = "C", 0
    out = []
    for p in psi:
        if p >= TAU_PSI:
            mode, r = "F", RECOVERY_R
        elif r > 0:
            mode, r = "F", r - 1
        elif p <= TAU_PSI - LAMBDA:
            mode = "C"
        out.append(mode)
    return out


def mode_sequence(controller, psi):
    if controller == "threshold_only" or controller == "threshold_only_noisy":
        return threshold_only(psi)
    if controller == "hysteresis_only":
        return hysteresis_only(psi)
    if controller == "recovery_only":
        return recovery_only(psi)
    return full_controller(psi)


def switch_count(seq):
    return sum(seq[i] != seq[i - 1] for i in range(1, len(seq)))


def main():
    report = {"run_id": RUN_ID, "phase": "rq3-controller-ablation", "status": "running",
              "start_time": datetime.now().isoformat(),
              "protocol": {"manifest_version": MANIFEST_VERSION, "seed": SEED, "horizon": HORIZON,
                           "tau_psi": TAU_PSI, "sigma_u": SIGMA_U, "lambda": LAMBDA, "recovery_r": RECOVERY_R,
                           "noise_sigma": NOISE_SIGMA, "noise_seed": NOISE_SEED, "controllers": CONTROLLERS,
                           "structure": "shared coarse prefix + 4 segment decisions of 3 latent frames",
                           "metric": "decoded-pixel L1 to matched internal c1-1 step-4 reference"}}
    try:
        install_deps()
        log(f"torch={torch.__version__} python={sys.version.split()[0]} cuda={torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/T4 unavailable")
        repo = Path("/kaggle/working/rcm")
        if repo.exists():
            shutil.rmtree(repo)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVlabs/rcm.git", str(repo)], check=True, capture_output=True)
        report["upstream_rcm_commit"] = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        inputs = Path("/kaggle/input")
        paths = {
            "fine": find_under(inputs, "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step2.pt"),
            "reference": find_under(inputs, "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt"),
            "coarse": find_under(inputs, "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step4.pt"),
            "vae": find_under(inputs, "Wan2.1_VAE.pth"),
            "t5": find_under(inputs, "models_t5_umt5-xxl-enc-bf16.pth"),
            "tokenizer": find_under(inputs, "tokenizer.json"),
        }
        if paths["tokenizer"]:
            paths["tokenizer"] = paths["tokenizer"].parent
        report["dataset_paths"] = {k: str(v) for k, v in paths.items()}
        missing = [k for k, v in paths.items() if v is None]
        if missing:
            raise FileNotFoundError(f"missing Kaggle inputs: {missing}")
        patch_file(repo / "rcm/inference/wan2pt1_t2v_causal_infer.py", [
            ('TENSOR_KWARGS = {"device": "cuda", "dtype": torch.bfloat16}', 'TENSOR_KWARGS = {"device": "cuda", "dtype": torch.float16}'),
            (".to(dtype=torch.bfloat16).cuda()", ".to(dtype=torch.float16).cuda()"),
        ])
        patch_file(repo / "rcm/utils/umt5.py", [
            ("dtype=torch.bfloat16,", "dtype=torch.float16,"),
            ('tokenizer_path="google/umt5-xxl",', f'tokenizer_path="{paths["tokenizer"]}",'),
        ])
        sys.path.insert(0, str(repo))
        from imaginaire.lazy_config import instantiate
        from rcm.inference.wan2pt1_t2v_causal_infer import (
            DIT_CONFIGS, TENSOR_KWARGS, build_few_step_t_steps,
            build_few_step_schedules_per_chunk, expand_steps_per_chunk,
            make_block_pattern, load_dit_weights, causal_rollout_sampling,
            causal_i2v_rollout_sampling,
        )
        from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
        from rcm.utils.model_utils import init_weights_on_device
        from rcm.utils.umt5 import clear_umt5_memory, umt5_xxl, HuggingfaceTokenizer
        from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
        from imaginaire.utils.easy_io import easy_io
        TENSOR_KWARGS["dtype"] = torch.float16

        tokenizer = Wan2pt1VAEInterface(vae_pth=str(paths["vae"]))
        width, height = VIDEO_RES_SIZE_INFO["480p"]["16:9"]
        latent_frames = tokenizer.get_latent_num_frames(HORIZON)
        if latent_frames != 15:
            raise RuntimeError(f"unexpected latent frames: {latent_frames}")
        latent_h = height // tokenizer.spatial_compression_factor
        latent_w = width // tokenizer.spatial_compression_factor
        latent_c = tokenizer.latent_ch

        def encode_prompt(prompt):
            model = umt5_xxl(encoder_only=True).eval().requires_grad_(False).to(dtype=torch.float16)
            ckpt = easy_io.load(str(paths["t5"]), backend_args=None, file_format="pt", map_location="cuda", weights_only=False)
            for key in list(ckpt.keys()):
                if torch.is_tensor(ckpt[key]):
                    ckpt[key] = ckpt[key].to(torch.float16)
            model.load_state_dict(ckpt, strict=False, assign=True)
            del ckpt
            tok = HuggingfaceTokenizer(name=str(paths["tokenizer"]), seq_len=512, clean="whitespace")
            ids, mask = tok(prompt, return_mask=True, add_special_tokens=True)
            ids, mask = ids.cuda(), mask.cuda()
            context = model(ids, mask)
            lengths = mask.gt(0).sum(dim=1).long()
            padded = []
            for item, length in zip(context, lengths):
                length = int(length)
                padded.append(item[:512] if length > 512 else torch.cat([item[:length], torch.zeros(512 - length, item.shape[1], device=item.device, dtype=item.dtype)], dim=0))
            emb = torch.stack(padded).to(torch.float16)
            del model, tok, ids, mask, context, padded
            clear_umt5_memory()
            gc.collect()
            torch.cuda.empty_cache()
            return emb

        def load_net(checkpoint):
            started = time.time()
            with init_weights_on_device():
                net = instantiate(DIT_CONFIGS["1.3B"]).eval()
            load_dit_weights(net, str(checkpoint))
            net.to(**TENSOR_KWARGS).cpu()
            torch.cuda.empty_cache()
            return net, time.time() - started

        def decode(latent):
            started = time.time()
            video = tokenizer.decode(latent.float().cuda())
            torch.cuda.synchronize()
            elapsed = time.time() - started
            frames = to_frames(video)
            peak = torch.cuda.max_memory_allocated() / 2**30
            del video
            torch.cuda.empty_cache()
            return frames, elapsed, peak

        @torch.no_grad()
        def run_reference(canonical_noise, condition):
            net, load_s = load_net(paths["reference"])
            steps = build_few_step_t_steps(4, 1600, MID, "cuda")
            _, blocks, _ = make_block_pattern(latent_frames, latent_h, latent_w, 1, 1, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(None, blocks, 4)
            schedules = build_few_step_schedules_per_chunk(counts, None, 1600, MID, [], "cuda")
            gen = torch.Generator(device="cuda").manual_seed(SEED + 1000)
            net.cuda()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            latent = causal_rollout_sampling(net, canonical_noise.cuda(), steps, schedules, None, condition, None, 1.0, 1, 1, False, generator=gen)
            torch.cuda.synchronize()
            dit_s = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            del net
            torch.cuda.empty_cache()
            frames, vae_s, dpeak = decode(latent)
            return frames, {"checkpoint_load_s": load_s, "dit_s": dit_s, "vae_s": vae_s, "nfe": int(sum(counts)), "peak_vram_gb": max(peak, dpeak)}

        @torch.no_grad()
        def run_initial(canonical_noise, condition, gen):
            net, load_s = load_net(paths["coarse"])
            steps = build_few_step_t_steps(4, 1600, MID, "cuda")
            _, blocks, _ = make_block_pattern(3, latent_h, latent_w, 3, 3, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(None, blocks, 4)
            schedules = build_few_step_schedules_per_chunk(counts, None, 1600, MID, [], "cuda")
            net.cuda()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            latent = causal_rollout_sampling(net, canonical_noise[:, :, :3].cuda(), steps, schedules, None, condition, None, 1.0, 3, 3, False, generator=gen)
            torch.cuda.synchronize()
            dit_s = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            del net
            torch.cuda.empty_cache()
            return latent.detach().cpu(), {"checkpoint_load_s": load_s, "dit_s": dit_s, "nfe": int(sum(counts)), "peak_vram_gb": peak}

        @torch.no_grad()
        def run_i2v(mode, boundary, group, canonical_noise, condition):
            net, load_s = load_net(paths["coarse"] if mode == "C" else paths["fine"])
            chunk = 3 if mode == "C" else 1
            steps_per_chunk = None if mode == "C" else [1]
            steps = build_few_step_t_steps(4, 1600, MID, "cuda")
            _, blocks, _ = make_block_pattern(4, latent_h, latent_w, 1, chunk, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(steps_per_chunk, blocks - 1, 4)
            schedules = build_few_step_schedules_per_chunk(counts, steps_per_chunk, 1600, MID, [], "cuda")
            noise = canonical_noise[:, :, 3 * group:3 * (group + 1)].cuda()
            gen = torch.Generator(device="cuda").manual_seed(SEED + 2000 + group)
            net.cuda()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            latent = causal_i2v_rollout_sampling(net, boundary.cuda(), noise, steps, schedules, steps_per_chunk, condition, None, 1.0, chunk, False, generator=gen)
            torch.cuda.synchronize()
            dit_s = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            fresh = latent[:, :, 1:].detach().cpu()
            del net, latent, noise
            torch.cuda.empty_cache()
            return fresh, {"checkpoint_load_s": load_s, "dit_s": dit_s, "nfe": int(sum(counts)), "peak_vram_gb": peak}

        done = completed_keys()
        for prompt_idx, spec in enumerate(PROMPTS):
            prompt = spec["prompt"]
            log(f"=== prompt {prompt_idx}: {prompt}")
            embedding = encode_prompt(prompt)
            condition = {"crossattn_emb": embedding.to(**TENSOR_KWARGS)}
            gen0 = torch.Generator(device="cuda").manual_seed(SEED)
            canonical_noise = torch.randn(1, latent_c, latent_frames, latent_h, latent_w, dtype=torch.float32, device="cuda", generator=gen0).cpu()
            noise_hash = hashlib.sha256(canonical_noise.numpy().tobytes()).hexdigest()

            ref_key = f"p{prompt_idx}_reference"
            if ref_key not in done:
                try:
                    ref_started = time.time()
                    ref_frames, ref_meta = run_reference(canonical_noise, condition)
                    save_mp4(VIDEO_DIR / f"{ref_key}.mp4", ref_frames)
                    append_result({"sample_key": ref_key, "controller": "reference", "status": "complete",
                                   "prompt_idx": prompt_idx, "prompt": prompt, "seed": SEED, "horizon": HORIZON,
                                   "initial_noise_sha256": noise_hash, "timing": {**ref_meta, "total_s": time.time() - ref_started},
                                   "decoded_shape": list(ref_frames.shape)})
                    done.add(ref_key)
                    del ref_frames
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as exc:
                    append_result({"sample_key": ref_key, "controller": "reference", "status": "failed", "prompt_idx": prompt_idx, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                    log(traceback.format_exc())
                    continue

            ref_frames = read_video(VIDEO_DIR / f"{ref_key}.mp4")

            # shared coarse prefix per prompt
            gen = torch.Generator(device="cuda").manual_seed(SEED + 2000)
            prefix, prefix_meta = run_initial(canonical_noise, condition, gen)
            prefix_hash = hashlib.sha256(prefix.numpy().tobytes()).hexdigest()

            warp = WARP_TRACE[prompt_idx]
            psi_real = [-w for w in warp]
            rng = random.Random(NOISE_SEED + prompt_idx)
            psi_noisy = [p + rng.gauss(0.0, NOISE_SIGMA) for p in psi_real]

            for controller in CONTROLLERS:
                key = f"p{prompt_idx}_{controller}"
                if key in done:
                    continue
                started = time.time()
                try:
                    psi = psi_noisy if controller.endswith("_noisy") else psi_real
                    seq = mode_sequence(controller, psi)
                    latents = [prefix]
                    timing = {"checkpoint_load_s": prefix_meta["checkpoint_load_s"], "dit_s": prefix_meta["dit_s"],
                              "vae_s": 0.0, "nfe": prefix_meta["nfe"], "peak_vram_gb": prefix_meta["peak_vram_gb"]}
                    decisions = []
                    for g in range(1, 5):
                        mode = seq[g - 1]
                        boundary = latents[-1][:, :, -1:]
                        fresh, meta = run_i2v(mode, boundary, g, canonical_noise, condition)
                        if fresh.shape[2] != 3:
                            raise RuntimeError(f"segment returned {fresh.shape[2]} frames")
                        latents.append(fresh)
                        timing["checkpoint_load_s"] += meta["checkpoint_load_s"]
                        timing["dit_s"] += meta["dit_s"]
                        timing["nfe"] += meta["nfe"]
                        timing["peak_vram_gb"] = max(timing["peak_vram_gb"], meta["peak_vram_gb"])
                        decisions.append({"group": g, "mode": mode, "psi": psi[g - 1], **meta})
                    latent = torch.cat(latents, dim=2)
                    if latent.shape[2] != latent_frames:
                        raise RuntimeError(f"latent frames mismatch: {latent.shape[2]}")
                    frames, elapsed, dpeak = decode(latent)
                    timing["vae_s"] += elapsed
                    timing["peak_vram_gb"] = max(timing["peak_vram_gb"], dpeak)
                    save_mp4(VIDEO_DIR / f"{key}.mp4", frames)
                    # per-segment regret and transition cost
                    segments = []
                    for g in range(1, 5):
                        s, e = segment_bounds(g)
                        actual = frames[s:e].astype(np.float32) / 255.0
                        target = ref_frames[s:e].astype(np.float32) / 255.0
                        segments.append({"group": g, "start_frame": s, "end_frame": e,
                                         "reference_l1": float(np.mean(np.abs(actual - target))),
                                         "adjacent_pixel_l1": float(np.mean(np.abs(actual[1:] - actual[:-1])))})
                    # transition cost: adjacent-frame L1 at switch boundaries vs interior
                    switch_bounds = []
                    for g in range(2, 5):
                        if seq[g - 1] != seq[g - 2]:
                            switch_bounds.append(segment_bounds(g)[0])
                    all_adj = np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32)).mean(axis=(1, 2, 3)) / 255.0
                    switch_adj = float(np.mean([all_adj[b - 1] for b in switch_bounds])) if switch_bounds else None
                    interior_mask = np.ones(len(all_adj), dtype=bool)
                    for b in switch_bounds:
                        interior_mask[b - 1] = False
                    interior_adj = float(all_adj[interior_mask].mean()) if interior_mask.any() else None
                    modes = [d["mode"] for d in decisions]
                    fine_count = sum(m == "F" for m in modes)
                    # mode run lengths
                    runs = []
                    cur, n = modes[0], 1
                    for m in modes[1:]:
                        if m == cur:
                            n += 1
                        else:
                            runs.append(n)
                            cur, n = m, 1
                    runs.append(n)
                    timing["total_s"] = time.time() - started
                    append_result({"sample_key": key, "controller": controller, "status": "complete", "prompt_idx": prompt_idx,
                                   "stratum": spec["stratum"], "prompt": prompt, "seed": SEED, "horizon": HORIZON,
                                   "initial_noise_sha256": noise_hash, "prefix_sha256": prefix_hash,
                                   "psi": psi, "mode_sequence": seq, "fine_count": fine_count,
                                   "fine_fraction": fine_count / 4, "switch_count": switch_count(seq),
                                   "mode_runs": runs, "switch_bound_frames": switch_bounds,
                                   "decisions": decisions, "segments": segments,
                                   "mean_reference_l1": float(np.mean([s["reference_l1"] for s in segments])),
                                   "terminal_reference_l1": float(segments[-1]["reference_l1"]),
                                   "auh_reference_l1": float(np.mean([s["reference_l1"] for s in segments])),
                                   "switch_adjacent_l1": switch_adj, "interior_adjacent_l1": interior_adj,
                                   "timing": timing, "video": f"videos/{key}.mp4",
                                   "metric": "decoded-pixel L1 to internal c1-1 step-4 reference; diagnostic"})
                    done.add(key)
                    log(f"  {controller}: seq={''.join(seq)} switches={switch_count(seq)} fine={fine_count}/4 meanref={float(np.mean([s['reference_l1'] for s in segments])):.4f} dit={timing['dit_s']:.1f}s total={timing['total_s']:.1f}s")
                    del latents, latent, frames
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as exc:
                    append_result({"sample_key": key, "controller": controller, "status": "failed", "prompt_idx": prompt_idx, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                    log(traceback.format_exc())
            del prefix, embedding, condition, canonical_noise
            gc.collect()
            torch.cuda.empty_cache()

        n_complete = len([k for k in completed_keys() if not k.endswith("_reference")])
        report["n_complete"] = n_complete
        report["n_expected"] = len(PROMPTS) * len(CONTROLLERS)
        report["verdict"] = "PASS" if n_complete == len(PROMPTS) * len(CONTROLLERS) else ("PARTIAL" if n_complete else "FAIL")
        report["status"] = "complete" if report["verdict"] == "PASS" else "partial"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        log(traceback.format_exc())
    finally:
        report["end_time"] = datetime.now().isoformat()
        SUMMARY.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        log(f"summary: {SUMMARY}")


if __name__ == "__main__":
    main()
