"""One-case Phase 5C online reference and policy feasibility gate on Kaggle T4.

Uses real Causal-rCM inference. This calibration case is not an RQ2 estimate.
The locked choices are in config/phase5c_pilot_lock.json in the local project.
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
OUT_DIR = Path("/kaggle/working/results/phase5c_online_pilot")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW = OUT_DIR / "results.jsonl"
SUMMARY = OUT_DIR / "phase5c_pilot_summary.json"

PROMPT_IDX = 6
PROMPT = "a bear catching a salmon in its powerful jaws"
SEED = 17
HORIZON = 57
WARP_DIRECTION = -1
WARP_THRESHOLD = -8.452908515930176
RANDOM_SEED = 20261024
POLICIES = ("fixed_coarse", "fixed_fine_continuation", "warp50", "random_match", "alternating50")
MANIFEST_SHA256 = "cce4c8c35cea378a6f2bd040e5468692a04f7e66f324aaddaedba33e63188b01"
CALIBRATION_LOCK_SHA256 = "3795454f60770432de3aa290bc98492d5a8a3f84c5028e619ac72b08d39690e9"
MID = [15 / 16, 5 / 6, 5 / 8]


def log(message):
    print(message, flush=True)


def find_under(root, filename):
    root = Path(root)
    return next((path for path in root.rglob(filename) if path.is_file()), None) if root.exists() else None


def install_deps():
    packages = [
        "einops", "transformers", "sentencepiece", "ftfy", "regex", "imageio",
        "imageio-ffmpeg", "tqdm", "torchvision", "fvcore", "omegaconf",
        "attrs", "pyyaml", "iopath", "termcolor", "pynvml", "pandas",
        "loguru", "safetensors", "cloudpickle", "dill", "matplotlib",
        "packaging", "opencv-python-headless",
    ]
    for package in packages:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", package],
            capture_output=True, text=True,
        )
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
    with RAW.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")


def completed_records():
    records = {}
    for line in RAW.read_text(encoding="utf-8").splitlines() if RAW.exists() else []:
        try:
            item = json.loads(line)
            if item.get("status") == "complete" and item.get("policy") in POLICIES:
                records[item["policy"]] = item
        except Exception:
            pass
    return records


def to_frames(video):
    pixels = (1 + video.float().cpu().clamp(-1, 1)) / 2
    if pixels.ndim == 5:
        pixels = pixels[0]
    return (pixels.permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8)


def warp_from_context(frames):
    """Same preceding-frame warp calculation used in Phase 5B."""
    import cv2
    started = time.time()
    history = np.asarray(frames, dtype=np.float32)
    if len(history) < 3:
        raise ValueError("warp probe requires three decoded context frames")
    prevprev = history[-3].mean(axis=2)
    prev = history[-2].mean(axis=2)
    flow = cv2.calcOpticalFlowFarneback(prevprev, prev, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    height, width = prev.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    map_x = np.clip(xx + flow[..., 0], 0, width - 1)
    map_y = np.clip(yy + flow[..., 1], 0, height - 1)
    warped = cv2.remap(prevprev, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    value = float(np.mean(np.abs(prev - warped)))
    return value, time.time() - started


def segment_bounds(group):
    start = 1 + 4 * (3 * group - 1)
    end = HORIZON if group == 4 else 1 + 4 * (3 * (group + 1) - 1)
    return start, end


def main():
    report = {
        "run_id": RUN_ID, "phase": "phase5c-online-feasibility-pilot",
        "status": "running", "start_time": datetime.now().isoformat(),
        "protocol": {
            "split": "calibration", "prompt_idx": PROMPT_IDX, "prompt": PROMPT,
            "seed": SEED, "horizon": HORIZON, "policies": POLICIES,
            "initial_mode": "coarse", "initial_prefix_latent_frames": 3,
            "warp_direction": WARP_DIRECTION, "warp_threshold": WARP_THRESHOLD,
            "random_seed": RANDOM_SEED, "manifest_sha256": MANIFEST_SHA256,
            "calibration_lock_sha256": CALIBRATION_LOCK_SHA256,
            "metric": "decoded-pixel L1 to matched internal c1-1 step-4 reference",
            "scope": "one calibration prompt-seed; no held-out/RQ2 claim",
        },
    }
    try:
        if len(completed_records()) == len(POLICIES) and SUMMARY.exists():
            old = json.loads(SUMMARY.read_text(encoding="utf-8"))
            if old.get("verdict") == "PASS":
                log("all policies already complete; retaining previous output")
                return

        install_deps()
        log(f"torch={torch.__version__} python={sys.version.split()[0]} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'}")
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
        report["dataset_paths"] = {name: str(path) for name, path in paths.items()}
        missing = [name for name, path in paths.items() if path is None]
        if missing:
            raise FileNotFoundError(f"missing Kaggle inputs: {missing}")
        for name, path in paths.items():
            log(f"{name}: {path}")
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

        @torch.no_grad()
        def encode_prompt():
            model = umt5_xxl(encoder_only=True).eval().requires_grad_(False).to(dtype=torch.float16)
            checkpoint = easy_io.load(str(paths["t5"]), backend_args=None, file_format="pt", map_location="cuda", weights_only=False)
            for key in list(checkpoint):
                if torch.is_tensor(checkpoint[key]):
                    checkpoint[key] = checkpoint[key].to(torch.float16)
            model.load_state_dict(checkpoint, strict=False, assign=True)
            del checkpoint
            tokenizer_text = HuggingfaceTokenizer(name=str(paths["tokenizer"]), seq_len=512, clean="whitespace")
            ids, mask = tokenizer_text(PROMPT, return_mask=True, add_special_tokens=True)
            ids, mask = ids.cuda(), mask.cuda()
            context = model(ids, mask)
            lengths = mask.gt(0).sum(dim=1).long()
            padded = []
            for item, length in zip(context, lengths):
                length = int(length)
                padded.append(item[:512] if length > 512 else torch.cat([
                    item[:length], torch.zeros(512 - length, item.shape[1], device=item.device, dtype=item.dtype),
                ], dim=0))
            embedding = torch.stack(padded).to(torch.float16)
            del model, tokenizer_text, ids, mask, context, padded
            clear_umt5_memory(); gc.collect(); torch.cuda.empty_cache()
            return embedding

        tokenizer = Wan2pt1VAEInterface(vae_pth=str(paths["vae"]))
        width, height = VIDEO_RES_SIZE_INFO["480p"]["16:9"]
        latent_frames = tokenizer.get_latent_num_frames(HORIZON)
        if latent_frames != 15:
            raise RuntimeError(f"unexpected latent frame count: {latent_frames}")
        latent_h = height // tokenizer.spatial_compression_factor
        latent_w = width // tokenizer.spatial_compression_factor
        latent_c = tokenizer.latent_ch
        text_started = time.time()
        embedding = encode_prompt()
        report["text_encode_s"] = time.time() - text_started
        condition = {"crossattn_emb": embedding.to(**TENSOR_KWARGS)}
        canonical_generator = torch.Generator(device="cuda").manual_seed(SEED)
        canonical_noise = torch.randn(
            1, latent_c, latent_frames, latent_h, latent_w,
            dtype=torch.float32, device="cuda", generator=canonical_generator,
        ).cpu()
        report["initial_noise_sha256"] = hashlib.sha256(canonical_noise.numpy().tobytes()).hexdigest()

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
        def run_reference():
            net, load_s = load_net(paths["reference"])
            steps = build_few_step_t_steps(4, 1600, MID, "cuda")
            _, blocks, _ = make_block_pattern(latent_frames, latent_h, latent_w, 1, 1, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(None, blocks, 4)
            schedules = build_few_step_schedules_per_chunk(counts, None, 1600, MID, [], "cuda")
            generator = torch.Generator(device="cuda").manual_seed(SEED + 1000)
            net.cuda(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            started = time.time()
            latent = causal_rollout_sampling(
                net, canonical_noise.cuda(), steps, schedules, None, condition,
                None, 1.0, 1, 1, False, generator=generator,
            )
            torch.cuda.synchronize()
            dit_s = time.time() - started
            peak = torch.cuda.max_memory_allocated() / 2**30
            del net
            torch.cuda.empty_cache()
            frames, vae_s, decode_peak = decode(latent)
            peak = max(peak, decode_peak)
            del latent
            return frames, {"checkpoint_load_s": load_s, "dit_s": dit_s, "vae_s": vae_s,
                            "nfe": int(sum(counts)), "peak_vram_gb": peak}

        @torch.no_grad()
        def run_initial(generator):
            net, load_s = load_net(paths["coarse"])
            steps = build_few_step_t_steps(4, 1600, MID, "cuda")
            _, blocks, _ = make_block_pattern(3, latent_h, latent_w, 3, 3, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(None, blocks, 4)
            schedules = build_few_step_schedules_per_chunk(counts, None, 1600, MID, [], "cuda")
            net.cuda(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            started = time.time()
            latent = causal_rollout_sampling(
                net, canonical_noise[:, :, :3].cuda(), steps, schedules, None,
                condition, None, 1.0, 3, 3, False, generator=generator,
            )
            torch.cuda.synchronize()
            dit_s = time.time() - started
            peak = torch.cuda.max_memory_allocated() / 2**30
            del net
            torch.cuda.empty_cache()
            return latent.detach().cpu(), {"checkpoint_load_s": load_s, "dit_s": dit_s,
                                           "nfe": int(sum(counts)), "peak_vram_gb": peak}

        @torch.no_grad()
        def run_i2v(mode, boundary, group):
            net, load_s = load_net(paths[mode])
            chunk = 3 if mode == "coarse" else 1
            steps_per_chunk = None if mode == "coarse" else [1]
            steps = build_few_step_t_steps(4, 1600, MID, "cuda")
            _, blocks, _ = make_block_pattern(4, latent_h, latent_w, 1, chunk, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(steps_per_chunk, blocks - 1, 4)
            schedules = build_few_step_schedules_per_chunk(counts, steps_per_chunk, 1600, MID, [], "cuda")
            noise_cpu = canonical_noise[:, :, 3 * group:3 * (group + 1)]
            noise_hash = hashlib.sha256(noise_cpu.numpy().tobytes()).hexdigest()
            noise = noise_cpu.cuda()
            generator = torch.Generator(device="cuda").manual_seed(SEED + 2000 + group)
            net.cuda(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            started = time.time()
            latent = causal_i2v_rollout_sampling(
                net, boundary.cuda(), noise, steps, schedules, steps_per_chunk,
                condition, None, 1.0, chunk, False, generator=generator,
            )
            torch.cuda.synchronize()
            dit_s = time.time() - started
            peak = torch.cuda.max_memory_allocated() / 2**30
            fresh = latent[:, :, 1:].detach().cpu()
            del net, latent, noise
            torch.cuda.empty_cache()
            return fresh, {"checkpoint_load_s": load_s, "dit_s": dit_s,
                           "nfe": int(sum(counts)), "peak_vram_gb": peak,
                           "segment_noise_sha256": noise_hash}

        reference_started = time.time()
        reference_frames, reference_meta = run_reference()
        if list(reference_frames.shape) != [HORIZON, height, width, 3]:
            raise RuntimeError(f"reference decoded shape mismatch: {reference_frames.shape}")
        report["reference"] = {**reference_meta, "total_s": time.time() - reference_started,
                               "decoded_shape": list(reference_frames.shape)}
        log(f"reference complete: shape={reference_frames.shape} NFE={reference_meta['nfe']}")

        completed = completed_records()
        common_prefix_hash = next((item["initial_prefix_sha256"] for item in completed.values()), None)
        for policy in POLICIES:
            if policy in completed:
                log(f"skipping completed policy: {policy}")
                continue
            started = time.time()
            try:
                generator = torch.Generator(device="cuda").manual_seed(SEED + 2000)
                initial, initial_meta = run_initial(generator)
                initial_hash = hashlib.sha256(initial.numpy().tobytes()).hexdigest()
                if common_prefix_hash is None:
                    common_prefix_hash = initial_hash
                elif initial_hash != common_prefix_hash:
                    raise RuntimeError("policy initial coarse prefix differs")
                latents = [initial]
                timing = {"checkpoint_load_s": initial_meta["checkpoint_load_s"],
                          "dit_s": initial_meta["dit_s"], "vae_s": 0.0,
                          "signal_s": 0.0, "nfe": initial_meta["nfe"],
                          "peak_vram_gb": initial_meta["peak_vram_gb"]}
                context_frames = None
                if policy == "warp50":
                    context_frames, elapsed, decode_peak = decode(initial)
                    timing["vae_s"] += elapsed
                    timing["peak_vram_gb"] = max(timing["peak_vram_gb"], decode_peak)
                random_fine_groups = set()
                if policy == "random_match":
                    warp = completed.get("warp50")
                    if warp is None:
                        raise RuntimeError("warp policy must complete before matched random control")
                    count = sum(decision["mode"] == "fine" for decision in warp["decisions"])
                    random_fine_groups = set(random.Random(RANDOM_SEED).sample([1, 2, 3, 4], count))
                decisions = []
                for group in range(1, 5):
                    raw_warp = None
                    oriented_score = None
                    signal_s = 0.0
                    if policy == "fixed_coarse":
                        mode = "coarse"
                    elif policy == "fixed_fine_continuation":
                        mode = "fine"
                    elif policy == "warp50":
                        if len(context_frames) != segment_bounds(group)[0]:
                            raise RuntimeError(f"context frame alignment failed at group {group}")
                        raw_warp, signal_s = warp_from_context(context_frames)
                        oriented_score = WARP_DIRECTION * raw_warp
                        mode = "fine" if oriented_score >= WARP_THRESHOLD else "coarse"
                    elif policy == "random_match":
                        mode = "fine" if group in random_fine_groups else "coarse"
                    else:
                        mode = "fine" if group in (1, 3) else "coarse"
                    boundary = latents[-1][:, :, -1:]
                    fresh, meta = run_i2v(mode, boundary, group)
                    if fresh.shape[2] != 3:
                        raise RuntimeError(f"continuation returned {fresh.shape[2]} latent frames")
                    latents.append(fresh)
                    timing["checkpoint_load_s"] += meta["checkpoint_load_s"]
                    timing["dit_s"] += meta["dit_s"]
                    timing["signal_s"] += signal_s
                    timing["nfe"] += meta["nfe"]
                    timing["peak_vram_gb"] = max(timing["peak_vram_gb"], meta["peak_vram_gb"])
                    decisions.append({"group": group, "mode": mode, "context_frames": segment_bounds(group)[0],
                                      "raw_warp": raw_warp, "oriented_score": oriented_score,
                                      "signal_s": signal_s, **meta})
                    if policy == "warp50" and group < 4:
                        context_frames, elapsed, decode_peak = decode(torch.cat(latents, dim=2))
                        timing["vae_s"] += elapsed
                        timing["peak_vram_gb"] = max(timing["peak_vram_gb"], decode_peak)
                latent = torch.cat(latents, dim=2)
                if latent.shape[2] != latent_frames:
                    raise RuntimeError(f"latent frame count mismatch: {latent.shape[2]}")
                frames, elapsed, decode_peak = decode(latent)
                timing["vae_s"] += elapsed
                timing["peak_vram_gb"] = max(timing["peak_vram_gb"], decode_peak)
                if frames.shape != reference_frames.shape:
                    raise RuntimeError(f"policy decoded shape mismatch: {frames.shape}")
                segments = []
                for group in range(1, 5):
                    start, end = segment_bounds(group)
                    actual = frames[start:end].astype(np.float32) / 255.0
                    target = reference_frames[start:end].astype(np.float32) / 255.0
                    reference_l1 = float(np.mean(np.abs(actual - target)))
                    adjacent_l1 = float(np.mean(np.abs(actual[1:] - actual[:-1])))
                    segments.append({"group": group, "start_frame": start, "end_frame": end,
                                     "reference_l1": reference_l1, "adjacent_pixel_l1": adjacent_l1,
                                     "mode": decisions[group - 1]["mode"]})
                fine_count = sum(decision["mode"] == "fine" for decision in decisions)
                modes = [decision["mode"] for decision in decisions]
                timing["total_s"] = time.time() - started
                timing["standalone_total_s_including_text"] = timing["total_s"] + report["text_encode_s"]
                item = {
                    "policy": policy, "status": "complete", "prompt_idx": PROMPT_IDX,
                    "prompt": PROMPT, "seed": SEED, "horizon": HORIZON,
                    "initial_prefix_sha256": initial_hash,
                    "initial_noise_sha256": report["initial_noise_sha256"],
                    "decoded_shape": list(frames.shape), "decisions": decisions,
                    "segments": segments, "fine_count": fine_count,
                    "fine_fraction": fine_count / 4,
                    "switch_count_decisions": sum(modes[i] != modes[i - 1] for i in range(1, 4)),
                    "switch_count_including_initial": int(modes[0] != "coarse") + sum(modes[i] != modes[i - 1] for i in range(1, 4)),
                    "random_fine_groups": sorted(random_fine_groups),
                    "mean_reference_l1": float(np.mean([segment["reference_l1"] for segment in segments])),
                    "timing": timing,
                    "metric": "decoded-pixel L1 to matched internal c1-1 step-4 reference; not ground truth",
                }
                append_result(item)
                completed[policy] = item
                log(f"policy complete: {policy} fine={fine_count}/4 mean_reference_l1={item['mean_reference_l1']:.6f} total_s={timing['total_s']:.1f}")
                del initial, latents, latent, frames, context_frames
                gc.collect(); torch.cuda.empty_cache()
            except Exception as exc:
                append_result({"policy": policy, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                               "traceback": traceback.format_exc()})
                log(traceback.format_exc())
        completed = completed_records()
        report["n_complete"] = len(completed)
        report["n_failed_or_incomplete"] = len(POLICIES) - len(completed)
        report["status"] = "complete" if len(completed) == len(POLICIES) else "partial"
        report["verdict"] = "PASS" if len(completed) == len(POLICIES) else "PARTIAL"
        report["results_file"] = str(RAW)
        report["limitations"] = ["one calibration case only", "common coarse prefix for all policy continuations",
                                 "no residual controller, held-out inference, or RQ2 claim"]
    except Exception as exc:
        report["status"] = "failed"
        report["verdict"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        log(traceback.format_exc())
    finally:
        if not (SUMMARY.exists() and report.get("status") == "running"):
            report["end_time"] = datetime.now().isoformat()
            SUMMARY.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            log(f"summary: {SUMMARY}")


if __name__ == "__main__":
    main()
