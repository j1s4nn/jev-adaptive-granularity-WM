"""
Phase 0b: Checkpoint + VRAM Feasibility Probe (T4)
===================================================

GOAL: Confirm checkpoint loading, VRAM limits, and internals access (EMA)
      WITHOUT cloning the rcm repo (that is Phase 0c).

INPUT:
- Kaggle datasets mounted in /kaggle/input
  (ajjisan/causal-rcm-ckpts, ajjisan/wan21-t5, ajjisan/wan21-dit-vae)
- T4 GPU (16GB VRAM, Turing, no bf16)

OUTPUT (/kaggle/working/results/phase0/<run_id>_meta.json):
- environment: torch/CUDA versions, GPU name, FA2 availability, SDPA availability
- dataset files: checkpoint/VAE/T5 files actually found (recursive search)
- per-checkpoint: loaded?, size_mb, load_time_s, peak VRAM, top-level keys,
  EMA-related keys, sample tensor shapes/dtypes
- vae/t5: loaded?, size_mb, peak VRAM
- verdict: PASS/FAIL with criteria

PASS criteria:
- c1-1_step2 AND c3-3_step4 both load on T4
- peak VRAM during single-model load < 14.5 GB
- torch SDPA available (FA2 expected unavailable on Turing)
"""

import sys
import os
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

# The kernel runs as /kaggle/src/script.py with the bundled src/ folder
# uploaded next to it. Make sure both candidate roots are importable.
for p in ('/kaggle/src', os.path.dirname(os.path.abspath(__file__)), '/kaggle/working'):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.cuda

from src.causal_rcm import (
    load_causal_rcm_checkpoints,
    check_flash_attention,
    get_attention_backend
)
from src.causal_rcm.load_checkpoints import (
    find_kaggle_input_paths,
    list_available_checkpoints,
    safe_torch_load,
    find_file_recursive
)

TARGETS = [
    # (label, checkpoint_type, step_count, filename)
    ('c1_1_step2', 'c1-1', 2, 'Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step2.pt'),
    ('c1_1_step4', 'c1-1', 4, 'Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt'),
    ('c3_3_step4', 'c3-3', 4, 'Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step4.pt'),
]


def profile_vram():
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    return {
        "device": torch.cuda.get_device_name(),
        "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
    }


def inspect_ema(ckpt):
    """Look for EMA copies anywhere in the checkpoint structure."""
    info = {"has_ema": False, "ema_keys": []}
    if not isinstance(ckpt, dict):
        return info

    top_keys = [str(k) for k in ckpt.keys()]
    for k in top_keys:
        if 'ema' in k.lower():
            info["has_ema"] = True
            info["ema_keys"].append(f"top:{k}")
            v = ckpt[k]
            if isinstance(v, dict):
                info["ema_num_tensors"] = len(v)
                info["ema_sample_keys"] = list(v.keys())[:5]
            elif isinstance(v, torch.nn.Module):
                info["ema_type"] = type(v).__name__

    # Some checkpoints nest: {'model': {...}, 'model_ema': {...}}
    for k, v in ckpt.items():
        if isinstance(v, dict):
            for sub in v.keys():
                if 'ema' in str(sub).lower():
                    info["has_ema"] = True
                    info["ema_keys"].append(f"{k}.{sub}")
                    if isinstance(v[sub], dict):
                        info["ema_num_tensors"] = len(v[sub])
                        info["ema_sample_keys"] = list(v[sub].keys())[:5]
                    break

    return info


def inspect_state(ckpt):
    """Summary of checkpoint structure: types, tensor counts, shapes."""
    info = {"top_level_keys": [str(k) for k in ckpt.keys()][:30] if isinstance(ckpt, dict) else None}
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            if isinstance(v, dict) and len(v) > 0:
                first_val = next(iter(v.values()))
                info["state_dict_like"] = str(k)
                info["num_tensors"] = len(v)
                if torch.is_tensor(first_val):
                    info["sample_shape"] = list(first_val.shape)
                    info["sample_dtype"] = str(first_val.dtype)
                break
    return info


def probe_target(label, checkpoint_type, step_count, filename):
    out = {"label": label, "loaded": False}
    try:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        ckpts = load_causal_rcm_checkpoints(
            device='cuda',
            checkpoint_type=checkpoint_type,
            step_count=step_count,
            verify_hashes=False
        )
        load_s = time.time() - t0
        key = 'c1_1' if checkpoint_type == 'c1-1' else 'c3_3'
        out["load_time_s"] = round(load_s, 1)
        out["vram"] = profile_vram()

        if key in ckpts:
            ckpt = ckpts[key]
            out["loaded"] = True
            out["internals"] = inspect_state(ckpt)
            out["ema"] = inspect_ema(ckpt)
            out["size_mb"] = ckpts.get('metadata', {}).get(key, {}).get('size_mb')
        else:
            out["reason"] = f"{filename} not found in dataset"

        # Free memory before next target
        del ckpts
        torch.cuda.empty_cache()
    except Exception as e:
        out["error"] = str(e)
        out["traceback"] = traceback.format_exc()
    return out


def main():
    start_time = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "run_id": run_id,
        "phase": "phase0b",
        "start_time": datetime.now().isoformat(),
        "status": "running"
    }

    try:
        print("=" * 60)
        print("PHASE 0b: CHECKPOINT + VRAM FEASIBILITY PROBE")
        print("=" * 60)

        # 1. Environment
        print("\n[1] Environment")
        results["environment"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": profile_vram(),
            "flash_attention_2": check_flash_attention(),
            "attention_backend": get_attention_backend(),
            "sdpa_available": hasattr(torch.nn.functional, 'scaled_dot_product_attention'),
        }

        # 2. Dataset files
        print("\n[2] Dataset files")
        available = list_available_checkpoints()
        print(json.dumps(available, indent=2, default=str))
        results["dataset_files"] = available

        # 3. Per-checkpoint probe (one at a time)
        print("\n[3] Checkpoint probes")
        results["targets"] = []
        for label, ctype, step, fname in TARGETS:
            print(f"\n--- Probbing {label} ---")
            r = probe_target(label, ctype, step, fname)
            results["targets"].append(r)
            print(json.dumps({k: v for k, v in r.items() if k != 'traceback'}, indent=2, default=str))

        # 4. VAE + T5 sanity (T5 on CPU: 11 GB is too big to co-reside on T4)
        print("\n[4] VAE + T5")
        paths = find_kaggle_input_paths()
        results["aux"] = {}
        if paths['vae']:
            vae_path = find_file_recursive(paths['vae'], 'Wan2.1_VAE.pth')
            if vae_path:
                torch.cuda.reset_peak_memory_stats()
                vae = safe_torch_load(vae_path, map_location='cuda')
                results["aux"]["vae"] = {
                    "loaded": True, "size_mb": round(vae_path.stat().st_size / (1024**2), 1),
                    "vram": profile_vram(), "type": type(vae).__name__,
                    "top_keys": [str(k) for k in vae.keys()][:10] if isinstance(vae, dict) else None
                }
                del vae; torch.cuda.empty_cache()
            else:
                results["aux"]["vae"] = {"loaded": False, "reason": "Wan2.1_VAE.pth not found"}
        if paths['t5']:
            t5_path = find_file_recursive(paths['t5'], 'models_t5_umt5-xxl-enc-bf16.pth')
            if t5_path:
                t5 = safe_torch_load(t5_path, map_location='cpu')
                results["aux"]["t5"] = {
                    "loaded": True, "size_mb": round(t5_path.stat().st_size / (1024**2), 1),
                    "note": "loaded on CPU (11 GB; will be fp16 on GPU during inference)",
                    "top_keys": [str(k) for k in t5.keys()][:10] if isinstance(t5, dict) else None
                }
                del t5
            else:
                results["aux"]["t5"] = {"loaded": False, "reason": "models_t5_umt5-xxl-enc-bf16.pth not found"}

        # 5. Verdict
        print("\n[5] Verdict")
        c1 = next((t for t in results["targets"] if t["label"] == "c1_1_step2"), {})
        c3 = next((t for t in results["targets"] if t["label"] == "c3_3_step4"), {})
        worst_peak = max(
            (t.get("vram", {}).get("peak_allocated_gb", 0) or 0 for t in results["targets"]),
            default=0
        )
        criteria = [
            (c1.get("loaded", False), "c1-1_step2 checkpoint loads on T4"),
            (c3.get("loaded", False), "c3-3_step4 checkpoint loads on T4"),
            (worst_peak < 14.5, f"peak VRAM < 14.5 GB (measured {worst_peak:.2f} GB)"),
            (results["environment"]["sdpa_available"], "PyTorch SDPA available (T4 fallback)"),
        ]
        results["verdict"] = {
            "overall": "PASS" if all(c[0] for c in criteria) else "FAIL",
            "criteria": [{"pass": c[0], "description": c[1]} for c in criteria],
        }

        # EMA availability summary (needed for u_i^EMA decision)
        ema_found = any(t.get("ema", {}).get("has_ema", False) for t in results["targets"])
        results["ema_summary"] = {
            "found_in_any_checkpoint": ema_found,
            "note": "If False, u_i^EMA is dropped; residual+warp scores remain (per plan)"
        }

        results["status"] = "complete"

    except Exception as e:
        print(f"\nX PHASE 0b FAILED: {e}")
        traceback.print_exc()
        results["status"] = "failed"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        results["verdict"] = {"overall": "FAIL"}

    finally:
        results["end_time"] = datetime.now().isoformat()
        results["duration_seconds"] = round(time.time() - start_time, 1)
        out_dir = Path("/kaggle/working/results/phase0")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{run_id}_meta.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved: {out_dir / f'{run_id}_meta.json'}")

        verdict = results.get("verdict", {}).get("overall", "FAIL")
        print(f"VERDICT: {verdict}")
        for c in results.get("verdict", {}).get("criteria", []):
            print(f"  {'PASS' if c['pass'] else 'FAIL'}: {c['description']}")


if __name__ == "__main__":
    main()
