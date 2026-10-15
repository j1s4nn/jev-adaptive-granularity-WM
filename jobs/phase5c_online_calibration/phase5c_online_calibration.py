"""Collect online coarse-context signals on the unused Phase 5C calibration prompts.

No held-out policies or quality comparison run here. Operating points are fit
locally only after every calibration trajectory passes its output audit.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import gc
import hashlib
import json
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
OUT_DIR = Path("/kaggle/working/results/phase5c_online_calibration")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW = OUT_DIR / "results.jsonl"
SUMMARY = OUT_DIR / "phase5c_calibration_summary.json"

PROMPTS = (
    (0, "a backpack"),
    (1, "a banana"),
    (2, "a baseball bat"),
    (3, "alley"),
    (4, "amusement park"),
    (5, "aquarium"),
    (7, "a bear climbing a tree"),
    (8, "a bear hunting for prey"),
    (9, "A person is air drumming"),
    (10, "A person is archery"),
    (11, "A person is arranging flowers"),
)
SEED = 17
HORIZON = 57
MANIFEST_SHA256 = "cce4c8c35cea378a6f2bd040e5468692a04f7e66f324aaddaedba33e63188b01"
PILOT_LOCK_SHA256 = "79a98b7685f633c23822ce7147fb9a42582ba0c00455057e22eb282de9fe8002"
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
            if item.get("status") == "complete" and item.get("prompt_idx") in dict(PROMPTS):
                records[item["prompt_idx"]] = item
        except Exception:
            pass
    return records


def to_frames(video):
    pixels = (1 + video.float().cpu().clamp(-1, 1)) / 2
    if pixels.ndim == 5:
        pixels = pixels[0]
    return (pixels.permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8)


def context_signals(frames):
    """Phase 5B-compatible preceding-context warp, motion and complexity."""
    import cv2
    started = time.time()
    history = np.asarray(frames, dtype=np.float32)
    if len(history) < 3:
        raise ValueError("context signal requires three decoded frames")
    prevprev = history[-3].mean(axis=2)
    prev = history[-2].mean(axis=2)
    current = history[-1].mean(axis=2)
    flow = cv2.calcOpticalFlowFarneback(prevprev, prev, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    height, width = prev.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    map_x = np.clip(xx + flow[..., 0], 0, width - 1)
    map_y = np.clip(yy + flow[..., 1], 0, height - 1)
    warped = cv2.remap(prevprev, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    gx = cv2.Sobel(current, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(current, cv2.CV_32F, 0, 1, ksize=3)
    return {
        "warp": float(np.mean(np.abs(prev - warped))),
        "motion": float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2))),
        "complexity": float(np.mean(np.sqrt(gx * gx + gy * gy)) + 0.5 * np.var(current)),
        "signal_s": time.time() - started,
    }


def segment_bounds(group):
    start = 1 + 4 * (3 * group - 1)
    end = HORIZON if group == 4 else 1 + 4 * (3 * (group + 1) - 1)
    return start, end


def main():
    report = {
        "run_id": RUN_ID, "phase": "phase5c-online-calibration",
        "status": "running", "start_time": datetime.now().isoformat(),
        "protocol": {
            "split": "calibration", "prompts": list(PROMPTS),
            "seed": SEED, "horizon": HORIZON,
            "mode": "coarse prefix plus four coarse continuations",
            "initial_prefix_latent_frames": 3,
            "signals": ["warp", "motion", "complexity"],
            "signal_context": "decoded prior coarse context before each segment",
            "manifest_sha256": MANIFEST_SHA256,
            "pilot_lock_sha256": PILOT_LOCK_SHA256,
            "scope": "online calibration only; no thresholds, held-out policies, quality or RQ2 claim",
        },
    }
    try:
        if len(completed_records()) == len(PROMPTS) and SUMMARY.exists():
            old = json.loads(SUMMARY.read_text(encoding="utf-8"))
            if old.get("verdict") == "PASS":
                log("all calibration cases already complete; retaining previous output")
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
        def encode_prompt(prompt):
            model = umt5_xxl(encoder_only=True).eval().requires_grad_(False).to(dtype=torch.float16)
            checkpoint = easy_io.load(str(paths["t5"]), backend_args=None, file_format="pt", map_location="cuda", weights_only=False)
            for key in list(checkpoint):
                if torch.is_tensor(checkpoint[key]):
                    checkpoint[key] = checkpoint[key].to(torch.float16)
            model.load_state_dict(checkpoint, strict=False, assign=True)
            del checkpoint
            tokenizer_text = HuggingfaceTokenizer(name=str(paths["tokenizer"]), seq_len=512, clean="whitespace")
            ids, mask = tokenizer_text(prompt, return_mask=True, add_special_tokens=True)
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
        def run_i2v(boundary, group):
            net, load_s = load_net(paths["coarse"])
            chunk = 3
            steps_per_chunk = None
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

        completed = completed_records()
        for prompt_idx, prompt in PROMPTS:
            if prompt_idx in completed:
                log(f"skipping completed calibration prompt {prompt_idx}")
                continue
            started = time.time()
            try:
                text_started = time.time()
                embedding = encode_prompt(prompt)
                text_encode_s = time.time() - text_started
                condition = {"crossattn_emb": embedding.to(**TENSOR_KWARGS)}
                canonical_generator = torch.Generator(device="cuda").manual_seed(SEED)
                canonical_noise = torch.randn(
                    1, latent_c, latent_frames, latent_h, latent_w,
                    dtype=torch.float32, device="cuda", generator=canonical_generator,
                ).cpu()
                noise_hash = hashlib.sha256(canonical_noise.numpy().tobytes()).hexdigest()
                generator = torch.Generator(device="cuda").manual_seed(SEED + 2000)
                initial, initial_meta = run_initial(generator)
                prefix_hash = hashlib.sha256(initial.numpy().tobytes()).hexdigest()
                latents = [initial]
                context_frames, vae_s, decode_peak = decode(initial)
                timing = {
                    "text_encode_s": text_encode_s,
                    "checkpoint_load_s": initial_meta["checkpoint_load_s"],
                    "dit_s": initial_meta["dit_s"],
                    "vae_s": vae_s,
                    "signal_s": 0.0,
                    "nfe": initial_meta["nfe"],
                    "peak_vram_gb": max(initial_meta["peak_vram_gb"], decode_peak),
                }
                decisions = []
                for group in range(1, 5):
                    expected_context = segment_bounds(group)[0]
                    if len(context_frames) != expected_context:
                        raise RuntimeError(f"context alignment failed at group {group}: {len(context_frames)} vs {expected_context}")
                    signals = context_signals(context_frames)
                    boundary = latents[-1][:, :, -1:]
                    fresh, meta = run_i2v(boundary, group)
                    if fresh.shape[2] != 3:
                        raise RuntimeError(f"continuation returned {fresh.shape[2]} latent frames")
                    latents.append(fresh)
                    timing["checkpoint_load_s"] += meta["checkpoint_load_s"]
                    timing["dit_s"] += meta["dit_s"]
                    timing["signal_s"] += signals["signal_s"]
                    timing["nfe"] += meta["nfe"]
                    timing["peak_vram_gb"] = max(timing["peak_vram_gb"], meta["peak_vram_gb"])
                    decisions.append({"group": group, "mode": "coarse", "context_frames": expected_context,
                                      **signals, **meta})
                    if group < 4:
                        context_frames, elapsed, decode_peak = decode(torch.cat(latents, dim=2))
                        timing["vae_s"] += elapsed
                        timing["peak_vram_gb"] = max(timing["peak_vram_gb"], decode_peak)
                if sum(latent.shape[2] for latent in latents) != latent_frames:
                    raise RuntimeError("final latent frame count mismatch")
                timing["total_s"] = time.time() - started
                item = {
                    "status": "complete", "prompt_idx": prompt_idx, "prompt": prompt,
                    "seed": SEED, "horizon": HORIZON, "latent_frames": latent_frames,
                    "initial_noise_sha256": noise_hash,
                    "initial_prefix_sha256": prefix_hash,
                    "decisions": decisions, "timing": timing,
                    "scope": "calibration-only online coarse-context signals; no reference or quality result",
                }
                append_result(item)
                completed[prompt_idx] = item
                log(f"calibration complete: prompt={prompt_idx} warp={[round(d['warp'], 3) for d in decisions]} total_s={timing['total_s']:.1f}")
                del embedding, condition, canonical_noise, initial, latents, context_frames
                gc.collect(); torch.cuda.empty_cache()
            except Exception as exc:
                append_result({"status": "failed", "prompt_idx": prompt_idx, "prompt": prompt,
                               "seed": SEED, "horizon": HORIZON,
                               "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                log(traceback.format_exc())
        completed = completed_records()
        report["n_complete"] = len(completed)
        report["n_failed_or_incomplete"] = len(PROMPTS) - len(completed)
        report["n_decisions"] = sum(len(row["decisions"]) for row in completed.values())
        report["status"] = "complete" if len(completed) == len(PROMPTS) else "partial"
        report["verdict"] = "PASS" if len(completed) == len(PROMPTS) else "PARTIAL"
        report["results_file"] = str(RAW)
        report["limitations"] = ["calibration prompts only", "coarse-context signal distribution only",
                                 "no fitted threshold, held-out result, reference metric, or RQ2 claim"]

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
