"""A100-01 / A2 online-policy demo (Google Colab A100, single-file kernel).

RQ2 on adequate hardware: with coarse now CHEAPER than fine on the A100 (per-forward
ratio 1.60 < break-even 2.33), can an online policy beat the best fixed policy at
matched measured cost? Same structure as F4-lite (T4): one shared coarse prefix
(3 latent frames) then 4 segment decisions of 3 latent frames each (57 frames).

Policies:
  fixed_coarse   : 4/4 coarse segments
  fixed_fine     : 4/4 fine segments
  fixed_hold     : 4/4 hold segments (copy boundary latent, no generation)
  random50       : 2 random fine groups (seeded)
  alternating50  : fine on groups 1 and 3
  adaptive_warp  : warp-threshold switching (T4 Phase 5B calibration; raw scores saved)

Uses the 12 fresh held-out prompts from config/a100_manifest.json (disjoint from
Phase 5B and F4-lite). Outputs /content/results/a100_online/ (results.jsonl,
summary, videos). T4 artifacts are never touched. Requires KAGGLE_USERNAME/KAGGLE_KEY.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
ROOT = Path("/content")
# Persistent cache on Google Drive so a reclaimed session does not lose inputs or
# partial results (the A100 runtime is frequently reclaimed). Falls back to local
# disk if Drive is not mounted.
DRIVE = Path("/content/drive/MyDrive")
CACHE = (DRIVE / "a100_cache") if DRIVE.exists() else (ROOT / "cache")
INPUTS_DIR = CACHE / "inputs"
OUT_DIR = CACHE / "results" / "a100_online_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR = OUT_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
RAW = OUT_DIR / "results.jsonl"
SUMMARY = OUT_DIR / "a100_online_v2_summary.json"
V1_VIDEO_DIR = CACHE / "results" / "a100_online" / "videos"

MANIFEST_VERSION = "a100-01"
SEED = 17
HORIZON = 57
FPS = 8
MID = [15 / 16, 5 / 6, 5 / 8]
WARP_DIRECTION = -1
WARP_THRESHOLD = -4.045816421508789  # A100-01 refit (measured on A2 v1 fixed_fine videos); refit_warp_threshold overrides if v1 videos exist
RANDOM_SEED = 20261025
PIN_COMMIT = "ed3cb14dd936f92cdc9f9381af7369991509b41f"

PROMPTS = [
    {"stratum": "object_class", "prompt": "a cake"},
    {"stratum": "object_class", "prompt": "a car"},
    {"stratum": "object_class", "prompt": "a carrot"},
    {"stratum": "scene", "prompt": "botanical garden"},
    {"stratum": "scene", "prompt": "bridge"},
    {"stratum": "scene", "prompt": "cafeteria"},
    {"stratum": "motion_smoothness", "prompt": "a bus accelerating to gain speed"},
    {"stratum": "motion_smoothness", "prompt": "a bus stuck in traffic during rush hour"},
    {"stratum": "motion_smoothness", "prompt": "a bus turning a corner"},
    {"stratum": "human_action", "prompt": "A person is clapping"},
    {"stratum": "human_action", "prompt": "A person is clay pottery making"},
    {"stratum": "human_action", "prompt": "A person is cleaning windows"},
]
POLICIES = ("fixed_coarse", "fixed_fine", "fixed_hold", "random50", "alternating50", "adaptive_warp")

CKPT_NAMES = {
    "fine": "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step2.pt",
    "reference": "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt",
    "coarse": "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step2.pt",
}


def log(m):
    print(m, flush=True)


def find_under(root, filename):
    root = Path(root)
    return next((p for p in root.rglob(filename) if p.is_file()), None) if root.exists() else None


def install_deps():
    packages = ["einops", "transformers", "sentencepiece", "ftfy", "regex", "imageio",
                "imageio-ffmpeg", "tqdm", "torchvision", "fvcore", "omegaconf", "attrs",
                "pyyaml", "iopath", "termcolor", "pynvml", "pandas", "loguru", "safetensors",
                "cloudpickle", "dill", "matplotlib", "packaging", "opencv-python-headless", "kaggle"]
    for pkg in packages:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pkg], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"pip install failed for {pkg}: {r.stderr[-400:]}")


def download_datasets():
    """rcm checkpoints from Kaggle; Wan2.1 T5+VAE+tokenizer from HuggingFace.

    The old Kaggle datasets ajjisan/wan21-t5 and ajjisan/wan21-dit-vae no longer
    exist (403, verified 2026-10-12), so those come from Wan-AI/Wan2.1-T2V-1.3B.
    """
    dst = INPUTS_DIR
    dst.mkdir(parents=True, exist_ok=True)
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not user or not key:
        raise RuntimeError("KAGGLE_USERNAME / KAGGLE_KEY required")
    kdir = Path.home() / ".kaggle"
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "kaggle.json").write_text(json.dumps({"username": user, "key": key}))
    try:
        os.chmod(kdir / "kaggle.json", 0o600)
    except Exception:
        pass

    ck = dst / "causal-rcm-ckpts"
    ck_done = ck.exists() and find_under(ck, CKPT_NAMES["reference"]) is not None
    if not ck_done:
        if ck.exists():
            shutil.rmtree(ck)
        ck.mkdir(parents=True, exist_ok=True)
        log("downloading ajjisan/causal-rcm-ckpts ...")
        subprocess.run(["kaggle", "datasets", "download", "-d", "ajjisan/causal-rcm-ckpts", "-p", str(ck), "--unzip"], check=True)
    else:
        log("causal-rcm-ckpts already present")

    hf = dst / "wan21"
    hf.mkdir(parents=True, exist_ok=True)
    if not (hf / "models_t5_umt5-xxl-enc-bf16.pth").exists():
        from huggingface_hub import snapshot_download
        log("downloading Wan-AI/Wan2.1-T2V-1.3B (T5 encoder + VAE + tokenizer) from HuggingFace ...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-T2V-1.3B",
            local_dir=str(hf),
            allow_patterns=["models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "config.json", "google/umt5-xxl/*"],
        )
    else:
        log("Wan2.1 T5/VAE already present")
    return dst


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
        fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


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
    px = (1 + video.float().cpu().clamp(-1, 1)) / 2
    if px.ndim == 5:
        px = px[0]
    return (px.permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8)


def save_mp4(path, frames):
    import imageio
    w = imageio.get_writer(str(path), fps=FPS, macro_block_size=1)
    try:
        for f in frames:
            w.append_data(f)
    finally:
        w.close()


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


def warp_from_context(frames):
    import cv2
    started = time.time()
    history = np.asarray(frames, dtype=np.float32)
    if len(history) < 3:
        raise ValueError("warp probe requires three decoded context frames")
    prevprev = history[-3].mean(axis=2)
    prev = history[-2].mean(axis=2)
    flow = cv2.calcOpticalFlowFarneback(prevprev, prev, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = prev.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = np.clip(xx + flow[..., 0], 0, w - 1)
    map_y = np.clip(yy + flow[..., 1], 0, h - 1)
    warped = cv2.remap(prevprev, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return float(np.mean(np.abs(prev - warped))), time.time() - started


def segment_bounds(group):
    start = 1 + 4 * (3 * group - 1)
    end = HORIZON if group == 4 else 1 + 4 * (3 * (group + 1) - 1)
    return start, end


def refit_warp_threshold(default):
    """Refit the warp threshold on A100 signals using A2-v1 fixed_fine videos.

    A2 v1 (the expensive-coarse run) already saved fixed_fine videos per prompt.
    We read their per-decision warp and take the pooled median of psi = -warp so
    the adaptive policy targets ~50% fine decisions on the A100 (not the T4 value).
    Falls back to `default` if the v1 videos are unavailable.
    """
    import glob
    files = sorted(glob.glob(str(V1_VIDEO_DIR / "p*_fixed_fine.mp4")))
    if not files:
        return default
    vals = []
    for p in files:
        vf = read_video(p)
        for g in range(1, 5):
            up = segment_bounds(g)[0]
            if len(vf) >= up and up >= 3:
                vals.append(warp_from_context(vf[:up])[0])
    if not vals:
        return default
    return float(np.median([-v for v in vals]))


def main():
    report = {"run_id": RUN_ID, "phase": "A100-A2-online-policies", "status": "running",
              "start_time": datetime.now().isoformat(),
              "protocol": {"manifest_version": MANIFEST_VERSION, "seed": SEED, "horizon": HORIZON,
                           "prompts": PROMPTS, "policies": POLICIES, "warp_direction": WARP_DIRECTION,
                           "warp_threshold": WARP_THRESHOLD, "random_seed": RANDOM_SEED,
                           "structure": "common coarse prefix (3 latent frames) + 4 segments of 3 latent frames",
                           "metric": "decoded-pixel L1 to matched internal c1-1 step-4 reference"}}
    try:
        install_deps()
        log(f"torch={torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        warp_threshold = refit_warp_threshold(WARP_THRESHOLD)
        log(f"warp threshold (A100-refit): {warp_threshold}")
        inputs = download_datasets()
        repo = ROOT / "rcm"
        if repo.exists():
            shutil.rmtree(repo)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVlabs/rcm.git", str(repo)], check=True, capture_output=True)
        report["upstream_rcm_commit"] = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        paths = {k: find_under(inputs, name) for k, name in CKPT_NAMES.items()}
        paths["vae"] = find_under(inputs, "Wan2.1_VAE.pth")
        paths["t5"] = find_under(inputs, "models_t5_umt5-xxl-enc-bf16.pth")
        tok = find_under(inputs, "tokenizer.json")
        paths["tokenizer"] = tok.parent if tok else None
        report["dataset_paths"] = {k: (str(v) if v else None) for k, v in paths.items()}
        missing = [k for k, v in paths.items() if v is None]
        if missing:
            raise FileNotFoundError(f"missing inputs: {missing}")
        patch_file(repo / "rcm/inference/wan2pt1_t2v_causal_infer.py", [
            ('TENSOR_KWARGS = {"device": "cuda", "dtype": torch.bfloat16}', 'TENSOR_KWARGS = {"device": "cuda", "dtype": torch.float16}'),
            (".to(dtype=torch.bfloat16).cuda()", ".to(dtype=torch.float16).cuda()"),
        ])
        patch_file(repo / "rcm/utils/umt5.py", [("dtype=torch.bfloat16,", "dtype=torch.float16,"),
                                                ('tokenizer_path="google/umt5-xxl",', f'tokenizer_path="{paths["tokenizer"]}",')])
        sys.path.insert(0, str(repo))
        from imaginaire.lazy_config import instantiate
        from rcm.inference.wan2pt1_t2v_causal_infer import (
            DIT_CONFIGS, TENSOR_KWARGS, build_few_step_t_steps, build_few_step_schedules_per_chunk,
            expand_steps_per_chunk, make_block_pattern, load_dit_weights,
            causal_rollout_sampling, causal_i2v_rollout_sampling,
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
        latent_h = height // tokenizer.spatial_compression_factor
        latent_w = width // tokenizer.spatial_compression_factor
        latent_c = tokenizer.latent_ch

        def encode_prompt(prompt):
            model = umt5_xxl(encoder_only=True).eval().requires_grad_(False).to(dtype=torch.float16)
            ckpt = easy_io.load(str(paths["t5"]), backend_args=None, file_format="pt", map_location="cuda", weights_only=False)
            for k in list(ckpt.keys()):
                if torch.is_tensor(ckpt[k]):
                    ckpt[k] = ckpt[k].to(torch.float16)
            model.load_state_dict(ckpt, strict=False, assign=True)
            del ckpt
            tokz = HuggingfaceTokenizer(name=str(paths["tokenizer"]), seq_len=512, clean="whitespace")
            ids, mask = tokz(prompt, return_mask=True, add_special_tokens=True)
            ids, mask = ids.cuda(), mask.cuda()
            context = model(ids, mask)
            lengths = mask.gt(0).sum(dim=1).long()
            padded = []
            for item, length in zip(context, lengths):
                length = int(length)
                padded.append(item[:512] if length > 512 else torch.cat([item[:length], torch.zeros(512 - length, item.shape[1], device=item.device, dtype=item.dtype)], dim=0))
            emb = torch.stack(padded).to(torch.float16)
            del model, tokz, ids, mask, context, padded
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
            if mode == "hold":
                # copy the boundary latent across the 3-frame segment; no generation
                return boundary.repeat(1, 1, 3, 1, 1).detach().cpu(), {"checkpoint_load_s": 0.0, "dit_s": 0.0, "nfe": 0, "peak_vram_gb": 0.0}
            net, load_s = load_net(paths[mode])
            chunk = 3 if mode == "coarse" else 1
            # A100 cheap regime: c3-3_step2 with a 1-step continuation (A1: DiT 2.69 s),
            # instead of the expensive c3-3_step4 4-step regime (DiT 5.58 s).
            steps_per_chunk = [1]
            steps = build_few_step_t_steps(4, 1600, MID, "cuda")
            _, blocks, _ = make_block_pattern(4, latent_h, latent_w, 1, chunk, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(steps_per_chunk, blocks - 1, 1)
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
            text_started = time.time()
            embedding = encode_prompt(prompt)
            text_encode_s = time.time() - text_started
            condition = {"crossattn_emb": embedding.to(**TENSOR_KWARGS)}
            gen0 = torch.Generator(device="cuda").manual_seed(SEED)
            canonical_noise = torch.randn(1, latent_c, latent_frames, latent_h, latent_w, dtype=torch.float32, device="cuda", generator=gen0).cpu()
            noise_hash = hashlib.sha256(canonical_noise.numpy().tobytes()).hexdigest()

            ref_key = f"p{prompt_idx}_reference"
            ref_path = VIDEO_DIR / f"{ref_key}.mp4"
            if ref_key not in done or not ref_path.exists():
                try:
                    ref_started = time.time()
                    ref_frames, ref_meta = run_reference(canonical_noise, condition)
                    save_mp4(VIDEO_DIR / f"{ref_key}.mp4", ref_frames)
                    append_result({"sample_key": ref_key, "policy": "reference", "status": "complete",
                                   "prompt_idx": prompt_idx, "prompt": prompt, "seed": SEED, "horizon": HORIZON,
                                   "initial_noise_sha256": noise_hash, "timing": {**ref_meta, "total_s": time.time() - ref_started},
                                   "decoded_shape": list(ref_frames.shape)})
                    done.add(ref_key)
                    del ref_frames
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as exc:
                    append_result({"sample_key": ref_key, "policy": "reference", "status": "failed", "prompt_idx": prompt_idx, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                    log(traceback.format_exc())
                    continue

            ref_frames = read_video(ref_path)

            for policy in POLICIES:
                key = f"p{prompt_idx}_{policy}"
                if key in done:
                    continue
                started = time.time()
                try:
                    gen = torch.Generator(device="cuda").manual_seed(SEED + 2000)
                    initial, initial_meta = run_initial(canonical_noise, condition, gen)
                    latents = [initial]
                    timing = {"checkpoint_load_s": initial_meta["checkpoint_load_s"], "dit_s": initial_meta["dit_s"],
                              "vae_s": 0.0, "signal_s": 0.0, "nfe": initial_meta["nfe"], "peak_vram_gb": initial_meta["peak_vram_gb"]}
                    context_frames = None
                    if policy == "adaptive_warp":
                        context_frames, elapsed, dpeak = decode(initial)
                        timing["vae_s"] += elapsed
                        timing["peak_vram_gb"] = max(timing["peak_vram_gb"], dpeak)
                    random_fine_groups = set()
                    if policy == "random50":
                        random_fine_groups = set(random.Random(RANDOM_SEED + prompt_idx).sample([1, 2, 3, 4], 2))
                    decisions = []
                    for group in range(1, 5):
                        raw_warp = None
                        oriented = None
                        signal_s = 0.0
                        if policy == "fixed_coarse":
                            mode = "coarse"
                        elif policy == "fixed_fine":
                            mode = "fine"
                        elif policy == "fixed_hold":
                            mode = "hold"
                        elif policy == "random50":
                            mode = "fine" if group in random_fine_groups else "coarse"
                        elif policy == "alternating50":
                            mode = "fine" if group in (1, 3) else "coarse"
                        else:
                            if len(context_frames) != segment_bounds(group)[0]:
                                raise RuntimeError(f"context alignment failed at group {group}")
                            raw_warp, signal_s = warp_from_context(context_frames)
                            oriented = WARP_DIRECTION * raw_warp
                            mode = "fine" if oriented >= warp_threshold else "coarse"
                        boundary = latents[-1][:, :, -1:]
                        fresh, meta = run_i2v(mode, boundary, group, canonical_noise, condition)
                        if fresh.shape[2] != 3:
                            raise RuntimeError(f"segment returned {fresh.shape[2]} frames")
                        latents.append(fresh)
                        timing["checkpoint_load_s"] += meta["checkpoint_load_s"]
                        timing["dit_s"] += meta["dit_s"]
                        timing["signal_s"] += signal_s
                        timing["nfe"] += meta["nfe"]
                        timing["peak_vram_gb"] = max(timing["peak_vram_gb"], meta["peak_vram_gb"])
                        decisions.append({"group": group, "mode": mode, "raw_warp": raw_warp, "oriented_score": oriented, "signal_s": signal_s, **meta})
                        if policy == "adaptive_warp" and group < 4:
                            context_frames, elapsed, dpeak = decode(torch.cat(latents, dim=2))
                            timing["vae_s"] += elapsed
                            timing["peak_vram_gb"] = max(timing["peak_vram_gb"], dpeak)
                    latent = torch.cat(latents, dim=2)
                    if latent.shape[2] != latent_frames:
                        raise RuntimeError(f"latent frames mismatch: {latent.shape[2]}")
                    frames, elapsed, dpeak = decode(latent)
                    timing["vae_s"] += elapsed
                    timing["peak_vram_gb"] = max(timing["peak_vram_gb"], dpeak)
                    save_mp4(VIDEO_DIR / f"{key}.mp4", frames)
                    segments = []
                    for group in range(1, 5):
                        start, end = segment_bounds(group)
                        actual = frames[start:end].astype(np.float32) / 255.0
                        target = ref_frames[start:end].astype(np.float32) / 255.0
                        segments.append({"group": group, "start_frame": start, "end_frame": end,
                                         "reference_l1": float(np.mean(np.abs(actual - target))),
                                         "adjacent_pixel_l1": float(np.mean(np.abs(actual[1:] - actual[:-1])))})
                    modes = [d["mode"] for d in decisions]
                    fine_count = sum(m == "fine" for m in modes)
                    hold_count = sum(m == "hold" for m in modes)
                    timing["total_s"] = time.time() - started
                    append_result({"sample_key": key, "policy": policy, "status": "complete", "prompt_idx": prompt_idx,
                                   "stratum": spec["stratum"], "prompt": prompt, "seed": SEED, "horizon": HORIZON,
                                   "hardware": "A100-SXM4-80GB", "initial_noise_sha256": noise_hash,
                                   "decoded_shape": list(frames.shape), "decisions": decisions, "segments": segments,
                                   "fine_count": fine_count, "hold_count": hold_count, "fine_fraction": fine_count / 4,
                                   "switch_count": sum(modes[i] != modes[i - 1] for i in range(1, 4)),
                                   "random_fine_groups": sorted(random_fine_groups),
                                   "mean_reference_l1": float(np.mean([s["reference_l1"] for s in segments])),
                                   "timing": timing, "text_encode_s": text_encode_s, "video": f"videos/{key}.mp4",
                                   "metric": "decoded-pixel L1 to internal c1-1 step-4 reference; diagnostic"})
                    done.add(key)
                    log(f"  {policy}: fine={fine_count}/4 hold={hold_count} switches={sum(modes[i] != modes[i-1] for i in range(1,4))} "
                        f"refL1={float(np.mean([s['reference_l1'] for s in segments])):.4f} dit={timing['dit_s']:.2f}s vae={timing['vae_s']:.2f}s total={timing['total_s']:.2f}s")
                    del initial, latents, latent, frames, context_frames
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as exc:
                    append_result({"sample_key": key, "policy": policy, "status": "failed", "prompt_idx": prompt_idx, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                    log(traceback.format_exc())
            del embedding, condition, canonical_noise
            gc.collect()
            torch.cuda.empty_cache()

        n_complete = len([k for k in completed_keys() if not k.endswith("_reference")])
        report["n_complete"] = n_complete
        report["n_expected"] = len(PROMPTS) * len(POLICIES)
        report["verdict"] = "PASS" if n_complete == len(PROMPTS) * len(POLICIES) else ("PARTIAL" if n_complete else "FAIL")
        report["status"] = "complete" if report["verdict"] == "PASS" else "partial"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        log(traceback.format_exc())
    finally:
        report["end_time"] = datetime.now().isoformat()
        SUMMARY.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        log("summary: " + str(SUMMARY))
        log("A100_ONLINE_DONE verdict=" + str(report.get("verdict")))


if __name__ == "__main__":
    main()
