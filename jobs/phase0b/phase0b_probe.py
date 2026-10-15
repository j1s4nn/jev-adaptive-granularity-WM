"""
Phase 0b (minimal, SELF-CONTAINED): single-dataset checkpoint + VRAM probe
===========================================================================

No src imports: Kaggle script kernels upload ONLY the code file.
Only the causal-rcm-ckpts dataset is attached (fast mount).

OUTPUT: /kaggle/working/results/phase0/<run_id>_meta.json
"""

import sys
import os
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

import torch
import torch.cuda


def log(msg):
    print(msg, flush=True)


def safe_torch_load(path, map_location, verbose=True):
    try:
        return torch.load(str(path), map_location=map_location)
    except Exception as e:
        if verbose:
            log(f"    [warn] torch.load failed ({type(e).__name__}), retrying weights_only=False")
        return torch.load(str(path), map_location=map_location, weights_only=False)


def find_file_recursive(root, filename):
    if root is None or not root.exists():
        return None
    for f in root.rglob(filename):
        if f.is_file():
            return f
    return None


def find_rcm_mount():
    root = Path("/kaggle/input")
    for item in root.iterdir():
        if item.is_dir():
            n = item.name.lower()
            if 'rcm' in n or 'ckpt' in n:
                return item
    return None


def profile_vram():
    return {
        "device": torch.cuda.get_device_name(),
        "total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
    }


def inspect(ckpt):
    info = {}
    if isinstance(ckpt, dict):
        info["top_keys"] = [str(k) for k in ckpt.keys()]
        for k, v in ckpt.items():
            if isinstance(v, dict) and v:
                first_val = next(iter(v.values()))
                info["state_like_key"] = str(k)
                info["num_tensors"] = len(v)
                if torch.is_tensor(first_val):
                    info["sample_shape"] = list(first_val.shape)
                    info["sample_dtype"] = str(first_val.dtype)
                break
        ema = []
        for k in ckpt.keys():
            if 'ema' in str(k).lower():
                ema.append(str(k))
            elif isinstance(ckpt[k], dict):
                for sub in ckpt[k]:
                    if 'ema' in str(sub).lower():
                        ema.append(f"{k}.{sub}")
                        break
        info["ema_keys"] = ema
        info["has_ema"] = len(ema) > 0
    else:
        info["type"] = type(ckpt).__name__
    return info


def main():
    start = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {"run_id": run_id, "phase": "phase0b-minimal", "status": "running"}

    try:
        log("START phase0b-minimal")
        log(f"torch={torch.__version__} python={sys.version.split()[0]}")
        log(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"gpu={torch.cuda.get_device_name()}")

        # Environment
        fa2 = False
        try:
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                log(f"compute_capability={cap}")
                fa2 = (cap[0] + cap[1] / 10) >= 8.0
            try:
                import flash_attn
                fa2 = fa2 and True
            except ImportError:
                fa2 = False
        except Exception as e:
            log(f"fa2 check error: {e}")
        results["env"] = {
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "cuda_available": torch.cuda.is_available(),
            "fa2": fa2,
            "sdpa": hasattr(torch.nn.functional, 'scaled_dot_product_attention'),
        }

        # Dataset layout
        log("Listing /kaggle/input ...")
        layout = {}
        root = Path("/kaggle/input")
        for d in sorted(root.iterdir()):
            if d.is_dir():
                files = []
                for f in d.rglob("*"):
                    if f.is_file():
                        files.append((str(f.relative_to(d)), f.stat().st_size))
                        if len(files) >= 200:
                            break
                layout[d.name] = {"n_files_shown": len(files), "files": files[:50]}
        results["dataset_layout"] = layout
        log(f"mount dirs: {[d.name for d in root.iterdir() if d.is_dir()]}")

        # Probe targets — search the whole /kaggle/input tree (mount paths are
        # nested like /kaggle/input/datasets/ajjisan/causal-rcm-ckpts/rcm_ckpts/...)
        root = Path("/kaggle/input")
        targets = [
            ("c3_3_step4", 'Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step4.pt'),
            ("c1_1_step2", 'Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step2.pt'),
        ]
        results["targets"] = []
        for label, fname in targets:
            log(f"Probing {label} ...")
            t = {"label": label, "loaded": False}
            p = find_file_recursive(root, fname)
            if p is None:
                t["reason"] = "not found in mount"
                log(f"  NOT FOUND: {fname}")
                results["targets"].append(t)
                continue
            log(f"  found: {p}")
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                ckpt = safe_torch_load(p, map_location='cuda')
                t["load_time_s"] = round(time.time() - t0, 1)
                t["loaded"] = True
                t["size_mb"] = round(p.stat().st_size / (1024**2), 1)
                t["vram"] = profile_vram()
                t["internals"] = inspect(ckpt)
                log(f"  loaded in {t['load_time_s']}s, peak {t['vram']['peak_allocated_gb']} GB, ema={t['internals'].get('has_ema')}")
                del ckpt
            except Exception as e:
                t["error"] = f"{type(e).__name__}: {e}"
                t["traceback"] = traceback.format_exc()
                log(f"  FAILED: {t['error']}")
            torch.cuda.empty_cache()
            results["targets"].append(t)

        # Verdict
        loaded = [t for t in results["targets"] if t.get("loaded")]
        peaks = [t.get("vram", {}).get("peak_allocated_gb", 0) for t in loaded]
        peak_str = f"{max(peaks):.2f}" if peaks else "n/a"
        criteria = [
            (len(loaded) >= 1, f"{len(loaded)}/2 checkpoints loaded"),
            (len(loaded) == 2, "both c3-3_step4 and c1-1_step2 loaded"),
            (max(peaks) < 14.5 if peaks else False, f"peak VRAM < 14.5 GB (max {peak_str})"),
        ]
        results["verdict"] = {
            "overall": "PASS" if criteria[0][0] and criteria[2][0] else "FAIL",
            "criteria": [{"pass": c[0], "description": c[1]} for c in criteria],
        }
        ema_any = any(t.get("internals", {}).get("has_ema") for t in loaded)
        results["ema_summary"] = {"found_in_any_checkpoint": ema_any,
                                  "note": "False -> drop u_i^EMA, keep residual+warp"}
        results["status"] = "complete"

    except Exception as e:
        log(f"FATAL: {e}")
        traceback.print_exc()
        results["status"] = "failed"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()

    finally:
        results["duration_seconds"] = round(time.time() - start, 1)
        out_dir = Path("/kaggle/working/results/phase0")
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / f"{run_id}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        log(f"meta saved: {meta_path}")
        log(f"VERDICT: {results.get('verdict', {}).get('overall', 'FAIL')}")


if __name__ == "__main__":
    main()
