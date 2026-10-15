"""
Phase 5B: held-out reference-regret signal collection on Kaggle T4.

This job follows the verified Phase 0c recipe and measures only signals that
can be available before the next generated segment: denoising residual,
decoded-frame warp mismatch, motion, and frame complexity. EMA and action
signals are deliberately absent because the verified checkpoints expose
neither. Fixed c1-1 and c3-3 rollouts are generated with the same prompt and
seed and initial noise. Each scored segment uses decoded-pixel regret against
an internal c1-1 step-4 reference. Signals use only preceding coarse context.
Thresholds and grouped AUROCs are computed by the local Phase 5B audit.

The output is small JSON/CSV metadata only. No model files or videos are
written to the laptop.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import gc
import json
import math
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("/kaggle/working/results/phase5b_heldout_03")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSONL = OUT_DIR / "results.jsonl"
SUMMARY_JSON = OUT_DIR / "phase5b_summary.json"

# Locked held-out prompts from config/phase5_prompt_manifest.json.
# The list is embedded because Kaggle script kernels upload one code file.
PROMPTS = [
    "ballroom",
    "bar",
    "barn"
]
SEEDS = [17, 29, 41]
START_INDEX = 9
CASES = [(START_INDEX + i, prompt, seed) for i, prompt in enumerate(PROMPTS) for seed in SEEDS]
HORIZONS = [57]
BOOTSTRAP_REPS = 2000  # retained only for shared helper compatibility; this kernel does not score AUROC


def log(msg):
    print(msg, flush=True)


def find_under(root, filename):
    root = Path(root)
    return next((p for p in root.rglob(filename) if p.is_file()), None) if root.exists() else None


def install_deps():
    packages = [
        "einops", "transformers", "sentencepiece", "ftfy", "regex",
        "imageio", "imageio-ffmpeg", "tqdm", "torchvision", "fvcore",
        "omegaconf", "attrs", "pyyaml", "iopath", "termcolor", "pynvml",
        "pandas", "loguru", "safetensors", "cloudpickle", "dill",
        "matplotlib", "packaging", "opencv-python-headless", "scikit-learn",
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
            log(f"[warn] patch target not found in {path.name}: {old[:80]}")
        source = source.replace(old, new)
    path.write_text(source, encoding="utf-8")


def write_jsonl(item):
    with RESULTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def completed_keys():
    if not RESULTS_JSONL.exists():
        return set()
    keys = set()
    for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            if item.get("status") == "complete":
                keys.add(item["sample_key"])
        except Exception:
            pass
    return keys


def bootstrap_auc(scores, labels, groups=None, seed=20261005):
    from sklearn.metrics import roc_auc_score
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return {"auroc": None, "ci95": [None, None], "n": int(len(labels)), "positive": int(labels.sum()), "reason": "single_class"}
    observed = float(roc_auc_score(labels, scores))
    rng = np.random.default_rng(seed)
    boot = []
    groups = np.asarray(groups if groups is not None else np.arange(len(labels)))
    unique_groups = np.unique(groups)
    for _ in range(BOOTSTRAP_REPS):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == group) for group in sampled_groups])
        if len(np.unique(labels[idx])) < 2:
            continue
        boot.append(float(roc_auc_score(labels[idx], scores[idx])))
    ci = np.percentile(boot, [2.5, 97.5]).tolist() if boot else [None, None]
    return {"auroc": observed, "ci95": ci, "n": int(len(labels)), "positive": int(labels.sum())}


def youden_threshold(scores, labels):
    from sklearn.metrics import roc_curve
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return None
    fpr, tpr, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    if not np.any(finite):
        return None
    valid_idx = np.flatnonzero(finite)
    return float(thresholds[valid_idx[int(np.argmax((tpr - fpr)[valid_idx]))]])


def threshold_metrics(scores, labels, threshold):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if threshold is None or len(labels) == 0:
        return {"threshold": threshold, "accuracy": None, "precision": None, "recall": None, "specificity": None}
    pred = scores >= threshold
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    tn = int(np.sum((~pred) & (labels == 0)))
    fn = int(np.sum((~pred) & (labels == 1)))
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(labels)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def segment_bounds(num_frames, latent_frames, group_idx):
    # Common decision units are 3 latent frames: one c3-3 chunk. Wan's
    # temporal VAE maps latent index 0 to output frame 0, then advances four
    # output frames per latent index. The first c3 chunk therefore spans 9
    # decoded frames and each later 3-latent chunk spans 12 decoded frames.
    start_lat = 3 * group_idx
    end_lat = min(3 * (group_idx + 1), latent_frames)
    start_px = 0 if start_lat == 0 else 1 + 4 * (start_lat - 1)
    end_px = num_frames if end_lat == latent_frames else 1 + 4 * (end_lat - 1)
    return min(start_px, num_frames), min(end_px, num_frames)


def frame_signals(frames, start, min_start=0):
    """Compute real prior-context signals before a future segment."""
    import cv2
    t0 = time.time()
    arr = np.asarray(frames, dtype=np.float32)
    end = max(0, min(int(start), len(arr)))
    hist = arr[:end]
    if len(hist) < 3:
        return {"warp": 0.0, "motion": 0.0, "complexity": 0.0, "signal_s": time.time() - t0, "history_frames": int(len(hist))}
    prevprev = hist[-3].mean(axis=2)
    prev = hist[-2].mean(axis=2)
    current = hist[-1].mean(axis=2)
    flow = cv2.calcOpticalFlowFarneback(prevprev, prev, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = prev.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = np.clip(xx + flow[..., 0], 0, w - 1)
    map_y = np.clip(yy + flow[..., 1], 0, h - 1)
    warped = cv2.remap(prevprev, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warp = float(np.mean(np.abs(prev - warped)))
    motion = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
    gx = cv2.Sobel(current, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(current, cv2.CV_32F, 0, 1, ksize=3)
    complexity = float(np.mean(np.sqrt(gx * gx + gy * gy)) + 0.5 * np.var(current))
    return {"warp": warp, "motion": motion, "complexity": complexity, "signal_s": time.time() - t0, "history_frames": int(len(hist))}


def main():
    results = {"run_id": RUN_ID, "phase": "phase5b-heldout-03", "status": "running", "start_time": datetime.now().isoformat()}
    try:
        install_deps()
        log(f"torch={torch.__version__} python={sys.version.split()[0]} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'}")
        repo = Path("/kaggle/working/rcm")
        if repo.exists():
            shutil.rmtree(repo)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVlabs/rcm.git", str(repo)], check=True, capture_output=True)
        inp = Path("/kaggle/input")
        paths = {
            "c1_1_step2": find_under(inp, "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step2.pt"),
            "c1_1_step4": find_under(inp, "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt"),
            "c3_3_step4": find_under(inp, "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step4.pt"),
            "vae": find_under(inp, "Wan2.1_VAE.pth"),
            "t5": find_under(inp, "models_t5_umt5-xxl-enc-bf16.pth"),
            "tokenizer": find_under(inp, "tokenizer.json"),
        }
        if paths["tokenizer"]:
            paths["tokenizer"] = paths["tokenizer"].parent
        results["dataset_paths"] = {k: str(v) for k, v in paths.items()}
        missing = [k for k, v in paths.items() if v is None]
        if missing:
            raise FileNotFoundError(f"Missing nested Kaggle files: {missing}")
        patch_file(repo / "rcm/inference/wan2pt1_t2v_causal_infer.py", [
            ('TENSOR_KWARGS = {"device": "cuda", "dtype": torch.bfloat16}', 'TENSOR_KWARGS = {"device": "cuda", "dtype": torch.float16}'),
            (".to(dtype=torch.bfloat16).cuda()", ".to(dtype=torch.float16).cuda()"),
        ])
        patch_file(repo / "rcm/utils/umt5.py", [("dtype=torch.bfloat16,", "dtype=torch.float16,"), ('tokenizer_path="google/umt5-xxl",', f'tokenizer_path="{paths["tokenizer"]}",')])
        sys.path.insert(0, str(repo))
        from imaginaire.lazy_config import instantiate
        from rcm.inference.wan2pt1_t2v_causal_infer import (
            DIT_CONFIGS, TENSOR_KWARGS, build_few_step_t_steps,
            build_few_step_schedules_per_chunk, expand_steps_per_chunk,
            parse_rf_time_values, parse_mid_t_schedules, make_block_pattern,
            block_span, load_dit_weights, RECTIFIED_FLOW_T_SCALING,
        )
        from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
        from rcm.utils.model_utils import init_weights_on_device
        from rcm.utils.umt5 import clear_umt5_memory, umt5_xxl, HuggingfaceTokenizer
        from rcm.utils.blockmask import AttnMaskSpec
        from rcm.utils.kv_cache import CausalInferenceState, KVCacheMode
        from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
        from imaginaire.utils.easy_io import easy_io
        from einops import rearrange
        TENSOR_KWARGS["dtype"] = torch.float16

        def encode_prompt(prompt):
            model = umt5_xxl(encoder_only=True).eval().requires_grad_(False).to(dtype=torch.float16)
            ckpt = easy_io.load(str(paths["t5"]), backend_args=None, file_format="pt", map_location="cuda", weights_only=False)
            for key in list(ckpt.keys()):
                if torch.is_tensor(ckpt[key]): ckpt[key] = ckpt[key].to(torch.float16)
            model.load_state_dict(ckpt, strict=False, assign=True); del ckpt
            tok = HuggingfaceTokenizer(name=str(paths["tokenizer"]), seq_len=512, clean="whitespace")
            ids, mask = tok(prompt, return_mask=True, add_special_tokens=True); ids, mask = ids.cuda(), mask.cuda()
            context = model(ids, mask); lengths = mask.gt(0).sum(dim=1).long(); padded = []
            for item, length in zip(context, lengths):
                length = int(length)
                padded.append(item[:512] if length > 512 else torch.cat([item[:length], torch.zeros(512-length, item.shape[1], device=item.device, dtype=item.dtype)], dim=0))
            emb = torch.stack(padded).to(torch.float16)
            del model, tok, ids, mask, context, padded; clear_umt5_memory(); gc.collect(); torch.cuda.empty_cache()
            return emb

        @torch.no_grad()
        def instrumented_sampler(net, init_noise, t_steps, t_steps_per_chunk, steps_per_chunk, condition, first_chunk_t, chunk_t, generator):
            B, C, T, H, W = init_noise.shape
            frame_tokens, num_blocks, bp = make_block_pattern(T, H, W, first_chunk_t, chunk_t, net.get_spatial_patch_size())
            ones = torch.ones(B, 1, device=init_noise.device, dtype=torch.float64)
            step_counts = expand_steps_per_chunk(steps_per_chunk, num_blocks, len(t_steps)-1)
            max_steps = max(step_counts)
            noises = [torch.randn_like(init_noise, dtype=torch.float32, generator=generator) for _ in range(max_steps)]
            kv = net.allocate_kv_caches(max_len=T * frame_tokens); blocks = []; records = []
            for i in range(num_blocks):
                fs, fe, bs = block_span(bp, i)
                attn = AttnMaskSpec(mode="block_causal", pattern=bp, q_block_offset=i)
                ts = t_steps_per_chunk[i] if t_steps_per_chunk else t_steps
                x = init_noise[:, :, fs:fe].to(torch.float64) * ts[0]
                previous_v = None
                denoise_residual = None
                for step_idx, (tc, tn) in enumerate(zip(ts[:-1], ts[1:])):
                    tcur = (tc * ones * RECTIFIED_FLOW_T_SCALING).to(**TENSOR_KWARGS)
                    state = CausalInferenceState(mode=KVCacheMode.READONLY, kv_caches=kv, pattern=bp, block_cursor=i, fast_infer=True)
                    current_v = net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS), timesteps_B_T=tcur, **condition, inference_state=state, attn_meta=attn).float()
                    if previous_v is not None:
                        denoise_residual = float(torch.mean(torch.abs(current_v - previous_v)).item())
                    previous_v = current_v
                    noise = noises[step_idx][:, :, fs:fe]
                    x = (1 - tn) * (x - tc * current_v.to(torch.float64)) + tn * noise
                zero = torch.zeros(B, 1, **TENSOR_KWARGS)
                append = CausalInferenceState(mode=KVCacheMode.APPEND, kv_caches=kv, pattern=bp, block_cursor=i, fast_infer=True)
                net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS), timesteps_B_T=zero, **condition, inference_state=append, attn_meta=attn)
                blocks.append(x); records.append({"frame_start": fs, "frame_end": fe, "denoise_residual": denoise_residual})
            del kv; torch.cuda.empty_cache()
            return torch.cat(blocks, dim=2), records

        tokenizer = Wan2pt1VAEInterface(vae_pth=str(paths["vae"]))
        width, height = VIDEO_RES_SIZE_INFO["480p"]["16:9"]
        latent_c = tokenizer.latent_ch
        mid = [15/16, 5/6, 5/8]
        t_steps = build_few_step_t_steps(4, 1600, mid, "cuda")
        schedules = parse_mid_t_schedules("")
        done = completed_keys(); samples = []

        def run_mode(prompt, mode, seed, prompt_embedding):
            if mode == "reference":
                ckpt = paths["c1_1_step4"]
                first, chunk, steps, schedule = 1, 1, None, ""
            elif mode == "fine":
                ckpt = paths["c1_1_step2"]
                first, chunk, steps, schedule = 1, 1, [4, 1], "15/16,5/6,5/8;"
            else:
                ckpt = paths["c3_3_step4"]
                first, chunk, steps, schedule = 3, 3, None, ""
            started = time.time()
            text_encode_s = 0.0
            condition = {"crossattn_emb": prompt_embedding.to(**TENSOR_KWARGS)}
            load_started = time.time()
            with init_weights_on_device(): net = instantiate(DIT_CONFIGS["1.3B"]).eval()
            load_dit_weights(net, str(ckpt)); net.to(**TENSOR_KWARGS).cpu(); torch.cuda.empty_cache()
            checkpoint_load_s = time.time() - load_started
            _, blocks, _ = make_block_pattern(latent_frames, latent_h, latent_w, first, chunk, net.get_spatial_patch_size())
            spc = parse_mid_t_schedules(schedule)
            counts = expand_steps_per_chunk(steps, blocks, 4)
            per_chunk = build_few_step_schedules_per_chunk(counts, steps, 1600, mid, spc, "cuda")
            gen = torch.Generator(device="cuda").manual_seed(seed)
            noise = torch.randn(1, latent_c, latent_frames, latent_h, latent_w, dtype=torch.float32, device="cuda", generator=gen)
            torch.cuda.reset_peak_memory_stats()
            net.cuda(); torch.cuda.synchronize(); t0 = time.time()
            latent, records = instrumented_sampler(net, noise, t_steps, per_chunk, steps, condition, first, chunk, gen)
            torch.cuda.synchronize(); dit_s = time.time() - t0
            peak_vram_gb = torch.cuda.max_memory_allocated() / 2**30
            del net; torch.cuda.empty_cache(); t1 = time.time(); video = tokenizer.decode(latent.float().cuda()); torch.cuda.synchronize(); vae_s = time.time() - t1
            peak_vram_gb = max(peak_vram_gb, torch.cuda.max_memory_allocated() / 2**30)
            px = (1 + video.float().cpu().clamp(-1, 1)) / 2
            if px.ndim == 5: px = px[0]
            frames = (px.permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8)
            return {"mode": mode, "frames": frames, "denoise_residuals": [r["denoise_residual"] for r in records], "latent_frames": latent_frames, "text_encode_s": text_encode_s, "checkpoint_load_s": checkpoint_load_s, "dit_s": dit_s, "vae_s": vae_s, "total_s": time.time() - started, "peak_vram_gb": peak_vram_gb, "nfe": int(sum(counts)), "blocks": len(records)}

        complete = []
        prompt_cache = {}
        for prompt_idx, prompt, seed in CASES:
            for horizon in HORIZONS:
                key = f"prompt{prompt_idx}_seed{seed}_h{horizon}"
                if key in completed_keys():
                    continue
                try:
                    log(f"running {key}: {prompt}")
                    latent_frames = tokenizer.get_latent_num_frames(horizon)
                    latent_h = height // tokenizer.spatial_compression_factor
                    latent_w = width // tokenizer.spatial_compression_factor
                    if prompt not in prompt_cache:
                        text_started = time.time()
                        prompt_cache[prompt] = encode_prompt(prompt)
                        text_encode_s = time.time() - text_started
                    else:
                        text_encode_s = 0.0
                    prompt_embedding = prompt_cache[prompt]
                    reference = run_mode(prompt, "reference", seed, prompt_embedding)
                    fine = run_mode(prompt, "fine", seed, prompt_embedding)
                    coarse = run_mode(prompt, "coarse", seed, prompt_embedding)
                    if not (reference["frames"].shape == fine["frames"].shape == coarse["frames"].shape):
                        raise RuntimeError("decoded frame shapes are not aligned")
                    observations = []
                    for g in range(1, latent_frames // 3):
                        start, end = segment_bounds(horizon, latent_frames, g)
                        ref_seg = reference["frames"][start:end].astype(np.float32) / 255.0
                        fine_seg = fine["frames"][start:end].astype(np.float32) / 255.0
                        coarse_seg = coarse["frames"][start:end].astype(np.float32) / 255.0
                        fine_dist = float(np.mean(np.abs(fine_seg - ref_seg)))
                        coarse_dist = float(np.mean(np.abs(coarse_seg - ref_seg)))
                        sig = frame_signals(coarse["frames"], start)
                        residual = coarse["denoise_residuals"][g - 1] or 0.0
                        observations.append({
                            "group": g, "segment_start_frame": start, "segment_end_frame": end,
                            "fine_reference_l1": fine_dist, "coarse_reference_l1": coarse_dist,
                            "reference_regret_l1": coarse_dist - fine_dist,
                            "residual": float(residual), "warp": sig["warp"],
                            "motion": sig["motion"], "complexity": sig["complexity"],
                            "signal_s": sig["signal_s"], "history_frames": sig["history_frames"],
                        })
                    item = {
                        "sample_key": key, "prompt_idx": prompt_idx, "prompt": prompt,
                        "seed": seed, "horizon": horizon, "status": "complete",
                        "alignment": {"same_seed": True, "same_initial_noise": True,
                                      "decoded_shape": list(reference["frames"].shape),
                                      "reference_mode": "c1-1_step4"},
                        "reference": {k: reference[k] for k in ["text_encode_s", "checkpoint_load_s", "dit_s", "vae_s", "total_s", "peak_vram_gb", "nfe", "blocks"]},
                        "fine": {k: fine[k] for k in ["text_encode_s", "checkpoint_load_s", "dit_s", "vae_s", "total_s", "peak_vram_gb", "nfe", "blocks"]},
                        "coarse": {k: coarse[k] for k in ["text_encode_s", "checkpoint_load_s", "dit_s", "vae_s", "total_s", "peak_vram_gb", "nfe", "blocks"]},
                        "text_encode_s": text_encode_s,
                        "observations": observations,
                        "metric": "decoded pixel L1 to matched c1-1 step-4 reference; reference-defined, not ground truth",
                    }
                    write_jsonl(item)
                    complete.append(item)
                    del prompt_embedding, reference, fine, coarse
                    gc.collect(); torch.cuda.empty_cache()
                except Exception as exc:
                    write_jsonl({"sample_key": key, "prompt_idx": prompt_idx, "prompt": prompt, "seed": seed, "horizon": horizon, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                    log(traceback.format_exc())

        all_complete = []
        for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines() if RESULTS_JSONL.exists() else []:
            try:
                item = json.loads(line)
                if item.get("status") == "complete": all_complete.append(item)
            except Exception:
                pass
        regrets = [o["reference_regret_l1"] for x in all_complete for o in x["observations"]]
        results["protocol"] = {"split": "held_out", "prompts": PROMPTS, "seeds": SEEDS, "horizons": HORIZONS, "modes": ["c1-1_step4_reference", "c1-1_step2_fine", "c3-3_step4_coarse"], "signals": ["residual", "warp", "motion", "complexity"], "signal_context": "preceding fixed coarse context, before target segment", "prompt_embedding_cached_across_modes_and_seeds": True, "metric": "decoded pixel L1 to matched reference", "same_initial_noise": True, "limitations": ["held-out split; no threshold fitted in this kernel", "reference is an internal high-compute model, not ground truth", "decoded pixel L1 is a reference-defined proxy", "no adaptive-policy claim"]}
        results["n_complete"] = len(all_complete)
        results["n_failed_or_incomplete"] = len(CASES) * len(HORIZONS) - len(all_complete)
        results["reference_regret_summary"] = {"n_segments": len(regrets), "mean": float(np.mean(regrets)) if regrets else None, "min": float(np.min(regrets)) if regrets else None, "max": float(np.max(regrets)) if regrets else None}
        results["verdict"] = "PASS" if len(all_complete) == len(CASES) * len(HORIZONS) else ("PARTIAL" if all_complete else "FAIL")
        results["status"] = "complete" if results["verdict"] == "PASS" else "partial"
    except Exception as exc:
        results["status"] = "failed"; results["error"] = f"{type(exc).__name__}: {exc}"; results["traceback"] = traceback.format_exc(); log(traceback.format_exc())
    finally:
        results["end_time"] = datetime.now().isoformat(); SUMMARY_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8"); log(f"summary: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()

