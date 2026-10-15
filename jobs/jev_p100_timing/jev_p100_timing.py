"""E-JEV-2 (Phase B): fine/coarse timing ratio on a non-T4 accelerator.

Purpose: test whether the fine-vs-coarse cost inversion is specific to the T4.
Runs the same operating modes as F1.3 but times the DiT only (VAE decode is
mode-independent, so it is skipped) and reports seconds-per-forward, which is
the quantity that decides the trade-off.

Runs on Kaggle with machine_shape set to a non-T4 accelerator (P100 if
available). Outputs results/jev_p100_timing/results.jsonl + summary.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import gc
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("/kaggle/working/results/jev_p100_timing")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW = OUT_DIR / "results.jsonl"
SUMMARY = OUT_DIR / "jev_p100_timing_summary.json"

SEED = 17
MID = [15 / 16, 5 / 6, 5 / 8]
PROMPTS = [{"stratum": "object_class", "prompt": "a boat"}, {"stratum": "motion", "prompt": "a bird flying over a snowy forest"}]
HORIZONS = [57, 81]

CKPT_NAMES = {
    "c1_1_step2": "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step2.pt",
    "c1_1_step4": "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt",
    "c3_3_step2": "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step2.pt",
    "c3_3_step4": "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step4.pt",
}
MODES = {
    "ref_c1_1_s4": {"ckpt": "c1_1_step4", "first": 1, "chunk": 1, "steps": None, "schedule": "", "default_steps": 4},
    "fine_c1_1_s1": {"ckpt": "c1_1_step2", "first": 1, "chunk": 1, "steps": [4, 1], "schedule": "15/16,5/6,5/8;", "default_steps": 4},
    "coarse_c3_3_s4": {"ckpt": "c3_3_step4", "first": 3, "chunk": 3, "steps": None, "schedule": "", "default_steps": 4},
    "coarse_c3_3_s1": {"ckpt": "c3_3_step2", "first": 3, "chunk": 3, "steps": [4, 1], "schedule": "15/16,5/6,5/8;", "default_steps": 1},
}


def log(m):
    print(m, flush=True)


def find_under(root, filename):
    root = Path(root)
    return next((p for p in root.rglob(filename) if p.is_file()), None) if root.exists() else None


def install_deps():
    for pkg in ["einops", "transformers", "sentencepiece", "ftfy", "regex", "tqdm", "torchvision", "fvcore", "omegaconf", "attrs", "pyyaml", "iopath", "termcolor", "pynvml", "pandas", "loguru", "safetensors", "cloudpickle", "dill", "matplotlib", "packaging"]:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pkg], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"pip install failed for {pkg}: {r.stderr[-400:]}")


def patch_file(path, repl):
    path = Path(path)
    s = path.read_text(encoding="utf-8")
    for o, n in repl:
        if o not in s:
            raise RuntimeError(f"missing patch target in {path.name}: {o[:70]}")
        s = s.replace(o, n)
    path.write_text(s, encoding="utf-8")


def append(item):
    with RAW.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    report = {"run_id": RUN_ID, "phase": "E-JEV-2-hardware-retime", "status": "running", "start_time": datetime.now().isoformat()}
    try:
        install_deps()
        gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "none"
        report["gpu"] = gpu
        log(f"torch={torch.__version__} gpu={gpu}")
        if not torch.cuda.is_available():
            raise RuntimeError("no GPU")
        repo = Path("/kaggle/working/rcm")
        if repo.exists():
            shutil.rmtree(repo)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVlabs/rcm.git", str(repo)], check=True, capture_output=True)
        report["upstream_rcm_commit"] = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        inp = Path("/kaggle/input")
        paths = {k: find_under(inp, n) for k, n in CKPT_NAMES.items()}
        paths["vae"] = find_under(inp, "Wan2.1_VAE.pth")
        paths["t5"] = find_under(inp, "models_t5_umt5-xxl-enc-bf16.pth")
        tk = find_under(inp, "tokenizer.json")
        paths["tokenizer"] = tk.parent if tk else None
        report["dataset_paths"] = {k: str(v) for k, v in paths.items()}
        missing = [k for k, v in paths.items() if v is None]
        if missing:
            raise FileNotFoundError(f"missing: {missing}")
        patch_file(repo / "rcm/inference/wan2pt1_t2v_causal_infer.py", [
            ('TENSOR_KWARGS = {"device": "cuda", "dtype": torch.bfloat16}', 'TENSOR_KWARGS = {"device": "cuda", "dtype": torch.float16}'),
            (".to(dtype=torch.bfloat16).cuda()", ".to(dtype=torch.float16).cuda()"),
        ])
        patch_file(repo / "rcm/utils/umt5.py", [("dtype=torch.bfloat16,", "dtype=torch.float16,"), ('tokenizer_path="google/umt5-xxl",', f'tokenizer_path="{paths["tokenizer"]}",')])
        sys.path.insert(0, str(repo))
        from imaginaire.lazy_config import instantiate
        from rcm.inference.wan2pt1_t2v_causal_infer import (
            DIT_CONFIGS, TENSOR_KWARGS, build_few_step_t_steps, build_few_step_schedules_per_chunk,
            expand_steps_per_chunk, parse_mid_t_schedules, make_block_pattern, block_span,
            load_dit_weights, RECTIFIED_FLOW_T_SCALING,
        )
        from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
        from rcm.utils.model_utils import init_weights_on_device
        from rcm.utils.umt5 import clear_umt5_memory, umt5_xxl, HuggingfaceTokenizer
        from rcm.utils.blockmask import AttnMaskSpec
        from rcm.utils.kv_cache import CausalInferenceState, KVCacheMode
        from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
        from imaginaire.utils.easy_io import easy_io
        TENSOR_KWARGS["dtype"] = torch.float16

        tokenizer = Wan2pt1VAEInterface(vae_pth=str(paths["vae"]))
        width, height = VIDEO_RES_SIZE_INFO["480p"]["16:9"]
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
            tok = HuggingfaceTokenizer(name=str(paths["tokenizer"]), seq_len=512, clean="whitespace")
            ids, mask = tok(prompt, return_mask=True, add_special_tokens=True)
            ids, mask = ids.cuda(), mask.cuda()
            ctx = model(ids, mask)
            lengths = mask.gt(0).sum(dim=1).long()
            padded = []
            for item, ln in zip(ctx, lengths):
                ln = int(ln)
                padded.append(item[:512] if ln > 512 else torch.cat([item[:ln], torch.zeros(512 - ln, item.shape[1], device=item.device, dtype=item.dtype)], dim=0))
            emb = torch.stack(padded).to(torch.float16)
            del model, tok, ids, mask, ctx, padded
            clear_umt5_memory()
            gc.collect()
            torch.cuda.empty_cache()
            return emb

        @torch.no_grad()
        def sampler(net, init_noise, t_steps, t_steps_per_chunk, step_counts, condition, first, chunk, gen):
            B, C, T, H, W = init_noise.shape
            frame_tokens, num_blocks, bp = make_block_pattern(T, H, W, first, chunk, net.get_spatial_patch_size())
            ones = torch.ones(B, 1, device=init_noise.device, dtype=torch.float64)
            max_steps = max(step_counts)
            noises = [torch.randn_like(init_noise, dtype=torch.float32, generator=gen) for _ in range(max_steps)]
            kv = net.allocate_kv_caches(max_len=T * frame_tokens)
            blocks, forwards = [], 0
            for i in range(num_blocks):
                fs, fe, _ = block_span(bp, i)
                attn = AttnMaskSpec(mode="block_causal", pattern=bp, q_block_offset=i)
                ts = t_steps_per_chunk[i] if t_steps_per_chunk else t_steps
                x = init_noise[:, :, fs:fe].to(torch.float64) * ts[0]
                for si, (tc, tn) in enumerate(zip(ts[:-1], ts[1:])):
                    tcur = (tc * ones * RECTIFIED_FLOW_T_SCALING).to(**TENSOR_KWARGS)
                    st = CausalInferenceState(mode=KVCacheMode.READONLY, kv_caches=kv, pattern=bp, block_cursor=i, fast_infer=True)
                    v = net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS), timesteps_B_T=tcur, **condition, inference_state=st, attn_meta=attn).float()
                    forwards += 1
                    nz = noises[si][:, :, fs:fe]
                    x = (1 - tn) * (x - tc * v.to(torch.float64)) + tn * nz
                zero = torch.zeros(B, 1, **TENSOR_KWARGS)
                ap = CausalInferenceState(mode=KVCacheMode.APPEND, kv_caches=kv, pattern=bp, block_cursor=i, fast_infer=True)
                net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS), timesteps_B_T=zero, **condition, inference_state=ap, attn_meta=attn)
                forwards += 1
                blocks.append(x)
            del kv
            torch.cuda.empty_cache()
            return torch.cat(blocks, dim=2), forwards

        def run_mode(emb, mode_key, latent_frames):
            cfg = MODES[mode_key]
            started = time.time()
            with init_weights_on_device():
                net = instantiate(DIT_CONFIGS["1.3B"]).eval()
            ls = time.time()
            load_dit_weights(net, str(paths[cfg["ckpt"]]))
            net.to(**TENSOR_KWARGS).cpu()
            torch.cuda.empty_cache()
            load_s = time.time() - ls
            _, blocks, _ = make_block_pattern(latent_frames, latent_h, latent_w, cfg["first"], cfg["chunk"], net.get_spatial_patch_size())
            spc = parse_mid_t_schedules(cfg["schedule"])
            counts = expand_steps_per_chunk(cfg["steps"], blocks, cfg["default_steps"])
            per_chunk = build_few_step_schedules_per_chunk(counts, cfg["steps"], 1600, MID, spc, "cuda")
            t_steps = build_few_step_t_steps(4, 1600, MID, "cuda")
            gen = torch.Generator(device="cuda").manual_seed(SEED)
            noise = torch.randn(1, latent_c, latent_frames, latent_h, latent_w, dtype=torch.float32, device="cuda", generator=gen)
            net.cuda()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            latent, forwards = sampler(net, noise, t_steps, per_chunk, counts, {"crossattn_emb": emb.to(**TENSOR_KWARGS)}, cfg["first"], cfg["chunk"], gen)
            torch.cuda.synchronize()
            dit_s = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            del net, latent
            torch.cuda.empty_cache()
            return {"mode": mode_key, "latent_frames": latent_frames, "dit_s": dit_s, "nfe": int(sum(counts)),
                    "forwards": forwards, "s_per_forward": dit_s / forwards, "checkpoint_load_s": load_s,
                    "peak_vram_gb": peak, "total_s": time.time() - started}

        for pi, spec in enumerate(PROMPTS):
            emb = encode_prompt(spec["prompt"])
            for horizon in HORIZONS:
                lf = tokenizer.get_latent_num_frames(horizon)
                for mode_key in MODES:
                    try:
                        r = run_mode(emb, mode_key, lf)
                        append({"sample_key": f"p{pi}_h{horizon}_{mode_key}", "status": "complete", "prompt": spec["prompt"], "horizon": horizon, "gpu": gpu, **r})
                        log(f"  {spec['prompt'][:16]} h{horizon} {mode_key}: nfe={r['nfe']} fwd={r['forwards']} dit={r['dit_s']:.1f}s s/fwd={r['s_per_forward']:.3f}")
                    except Exception as exc:
                        append({"sample_key": f"p{pi}_h{horizon}_{mode_key}", "status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                        log(traceback.format_exc())
            del emb
            gc.collect()
            torch.cuda.empty_cache()

        report["n_complete"] = len([json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if json.loads(l).get("status") == "complete"]) if RAW.exists() else 0
        report["status"] = "complete"
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
