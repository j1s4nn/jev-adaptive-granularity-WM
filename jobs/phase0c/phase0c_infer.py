"""
Phase 0c: REAL causal-rCM inference test on T4 (self-contained)
================================================================

Clones NVlabs/rcm (code only, no HF model downloads), patches bf16 -> fp16
and the umT5 tokenizer path to the local dataset, then runs two real
generations in-process:

  Test A: c1-1 (frame-wise)  1-step from c1-1_step2 ckpt, num_frames=13 (latent T=4)
  Test B: c3-3 (chunk-wise)  4-step from c3-3_step4 ckpt, num_frames=21 (latent T=6)

Measures: peak VRAM, DiT sampling ms, VAE decode ms, saves keyframes.
OUTPUT: /kaggle/working/results/phase0c/<run_id>_meta.json + frames/*.png
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import sys
import json
import time
import shutil
import subprocess
import traceback
from pathlib import Path
from datetime import datetime

import torch

PROMPT = "A cinematic shot of a snowy mountain at sunrise"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("/kaggle/working/results/phase0c") / RUN_ID
FRAME_DIR = OUT_DIR / "frames"
FRAME_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


def log(msg):
    print(msg, flush=True)


def pip_install(pkgs):
    for pkg in pkgs:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pkg],
                           capture_output=True, text=True)
        log(f"pip install {pkg}: exit={r.returncode}")


def find_under(root, filename):
    root = Path(root)
    if not root.exists():
        return None
    for f in root.rglob(filename):
        if f.is_file():
            return f
    return None


def patch_file(path, replacements):
    """Apply (old, new) string replacements to a file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in text:
            log(f"  [warn] patch target not found in {p.name}: {old[:60]}...")
        text = text.replace(old, new)
    p.write_text(text, encoding="utf-8")


def main():
    results = {"run_id": RUN_ID, "phase": "phase0c", "status": "running",
               "start_time": datetime.now().isoformat()}
    try:
        log("=== Phase 0c: real inference test ===")
        log(f"torch={torch.__version__} python={sys.version.split()[0]}")
        log(f"cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'}")
        results["env"] = {
            "torch": torch.__version__, "python": sys.version.split()[0],
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "none",
        }

        # 1. Dependencies
        log("\n[1] pip deps")
        pip_install(["einops", "transformers", "sentencepiece", "ftfy", "regex",
                     "imageio", "imageio-ffmpeg", "tqdm", "torchvision",
                     "fvcore", "omegaconf", "attrs", "pyyaml", "iopath",
                     "termcolor", "pynvml", "pandas", "loguru", "safetensors",
                     "cloudpickle", "dill", "matplotlib", "packaging"])

        # 2. Clone repo (code only)
        log("\n[2] clone NVlabs/rcm")
        repo = Path("/kaggle/working/rcm")
        if repo.exists():
            shutil.rmtree(repo)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVlabs/rcm.git", str(repo)],
                       check=True, capture_output=True)
        log("  cloned")

        # 3. Locate dataset files (recursive - mount paths are nested)
        log("\n[3] locate dataset files")
        inp = Path("/kaggle/input")
        paths = {
            "c1_1_step2": find_under(inp, "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step2.pt"),
            "c3_3_step4": find_under(inp, "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step4.pt"),
            "vae": find_under(inp, "Wan2.1_VAE.pth"),
            "t5": find_under(inp, "models_t5_umt5-xxl-enc-bf16.pth"),
            "t5_tokenizer": find_under(inp, "tokenizer.json"),
        }
        if paths["t5_tokenizer"] is not None:
            paths["t5_tokenizer"] = paths["t5_tokenizer"].parent
        for k, v in paths.items():
            log(f"  {k}: {v}")
        results["dataset_paths"] = {k: str(v) for k, v in paths.items()}
        missing = [k for k, v in paths.items() if v is None]
        if missing:
            raise FileNotFoundError(f"Missing dataset files: {missing}")

        # 4. Patch repo for T4 (bf16 -> fp16, local tokenizer)
        log("\n[4] patch repo for T4")
        infer_py = repo / "rcm/inference/wan2pt1_t2v_causal_infer.py"
        patch_file(infer_py, [
            ('TENSOR_KWARGS = {"device": "cuda", "dtype": torch.bfloat16}',
             'TENSOR_KWARGS = {"device": "cuda", "dtype": torch.float16}'),
            (".to(dtype=torch.bfloat16).cuda()", ".to(dtype=torch.float16).cuda()"),
        ])
        umt5_py = repo / "rcm/utils/umt5.py"
        patch_file(umt5_py, [
            ("dtype=torch.bfloat16,", "dtype=torch.float16,"),
            ('tokenizer_path="google/umt5-xxl",', f'tokenizer_path="{paths["t5_tokenizer"]}",'),
        ])

        # 5. In-process inference
        log("\n[5] import repo and run inference")
        sys.path.insert(0, str(repo))
        from imaginaire.lazy_config import instantiate
        from rcm.inference.wan2pt1_t2v_causal_infer import (
            DIT_CONFIGS, TENSOR_KWARGS, build_few_step_t_steps,
            build_few_step_schedules_per_chunk, expand_steps_per_chunk,
            parse_rf_time_values, parse_mid_t_schedules, make_block_pattern,
            causal_rollout_sampling, load_dit_weights, RECTIFIED_FLOW_T_SCALING,
        )
        from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
        from rcm.utils.model_utils import init_weights_on_device
        from rcm.utils.umt5 import get_umt5_embedding, clear_umt5_memory
        from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
        from einops import rearrange
        from PIL import Image

        TENSOR_KWARGS["dtype"] = torch.float16  # belt and suspenders

        def encode_prompt_fp16(prompt):
            """Build umT5 on CPU (fp16), load bf16 ckpt straight to GPU, assign."""
            from rcm.utils.umt5 import umt5_xxl, HuggingfaceTokenizer
            from imaginaire.utils.easy_io import easy_io
            log("  building umT5 encoder (CPU)...")
            model = umt5_xxl(encoder_only=True).eval().requires_grad_(False)
            model = model.to(dtype=torch.float16)  # CPU ~11 GB (v4 proved RAM OK)
            log("  loading T5 weights to GPU (bf16)...")
            ckpt = easy_io.load(str(paths["t5"]), backend_args=None, file_format="pt",
                                map_location="cuda", weights_only=False)
            log("  casting T5 weights to fp16 in place on GPU...")
            for k in list(ckpt.keys()):
                if torch.is_tensor(ckpt[k]):
                    ckpt[k] = ckpt[k].to(torch.float16)
            torch.cuda.empty_cache()
            log("  assigning weights (no copy)...")
            missing, unexpected = model.load_state_dict(ckpt, strict=False, assign=True)
            log(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
            del ckpt
            gc.collect()
            torch.cuda.empty_cache()
            tok = HuggingfaceTokenizer(name=str(paths["t5_tokenizer"]), seq_len=512, clean="whitespace")
            ids, mask = tok(prompt, return_mask=True, add_special_tokens=True)
            ids = ids.cuda()
            mask = mask.cuda()
            seq_lens = mask.gt(0).sum(dim=1).long()
            context = model(ids, mask)
            emb = []
            for u, length in zip(context, seq_lens):
                if length > 512:
                    emb.append(u[:512])
                else:
                    zeros = torch.zeros(512 - length, u.shape[1], device=u.device, dtype=u.dtype)
                    emb.append(torch.cat([u[:length], zeros], dim=0))
            text_emb = torch.stack(emb).to(dtype=torch.float16)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return text_emb

        def run_test(label, dit_path, num_frames, first_chunk_t, chunk_t,
                     steps_per_chunk, mid_t_schedules, num_steps, mid_t):
            log(f"\n=== Test {label} ===")
            t = {"label": label, "ok": False}
            try:
                with init_weights_on_device():
                    net = instantiate(DIT_CONFIGS["1.3B"]).eval()
                load_dit_weights(net, str(dit_path))
                net.to(**TENSOR_KWARGS).cpu()
                torch.cuda.empty_cache()

                tokenizer = Wan2pt1VAEInterface(vae_pth=str(paths["vae"]))
                w, h = VIDEO_RES_SIZE_INFO["480p"]["16:9"]
                text_emb = encode_prompt_fp16(PROMPT)
                condition = {"crossattn_emb": text_emb.to(**TENSOR_KWARGS)}

                latent_frames = tokenizer.get_latent_num_frames(num_frames)
                latent_h = h // tokenizer.spatial_compression_factor
                latent_w = w // tokenizer.spatial_compression_factor
                state_shape = [tokenizer.latent_ch, latent_frames, latent_h, latent_w]
                t["latent_shape"] = state_shape

                generator = torch.Generator(device=TENSOR_KWARGS["device"])
                generator.manual_seed(0)

                mid_t_vals = parse_rf_time_values(mid_t)
                t_steps = build_few_step_t_steps(num_steps, 1600, mid_t_vals, TENSOR_KWARGS["device"])
                schedules = parse_mid_t_schedules(mid_t_schedules)
                _, num_blocks, _ = make_block_pattern(
                    latent_frames, latent_h, latent_w, first_chunk_t, chunk_t,
                    net.get_spatial_patch_size())
                step_counts = expand_steps_per_chunk(steps_per_chunk, num_blocks, num_steps)
                t_steps_per_chunk = build_few_step_schedules_per_chunk(
                    step_counts, steps_per_chunk, 1600, mid_t_vals, schedules,
                    TENSOR_KWARGS["device"])
                t["chunks"] = num_blocks
                t["steps_per_chunk"] = step_counts

                noise = torch.randn(1, *state_shape, dtype=torch.float32,
                                    device=TENSOR_KWARGS["device"], generator=generator)

                net.cuda()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t0 = time.time()
                samples = causal_rollout_sampling(
                    net, noise, t_steps, t_steps_per_chunk, steps_per_chunk,
                    condition, None, 1.0, first_chunk_t, chunk_t,
                    ode=False, generator=generator)
                torch.cuda.synchronize()
                dit_s = time.time() - t0
                t["dit_s"] = round(dit_s, 2)
                t["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 3)

                del net
                torch.cuda.empty_cache()
                t1 = time.time()
                video = tokenizer.decode(samples.float())
                torch.cuda.synchronize()
                t["vae_s"] = round(time.time() - t1, 2)

                # Save keyframes
                px = (1.0 + video.float().cpu().clamp(-1, 1)) / 2.0  # [B?,]C,T,H,W
                if px.ndim == 5:
                    px = px[0]
                for fidx in [0, px.shape[1] - 1]:
                    frame = (px[:, fidx].permute(1, 2, 0).numpy() * 255).astype("uint8")
                    Image.fromarray(frame).save(FRAME_DIR / f"{label}_frame{fidx}.png")
                t["frames_saved"] = True
                t["ok"] = True
                log(f"  DONE: dit={dit_s:.1f}s vae={t['vae_s']}s peak_vram={t['peak_vram_gb']}GB chunks={num_blocks}")

                del net, tokenizer, samples, video
                clear_umt5_memory()
                torch.cuda.empty_cache()
            except Exception as e:
                t["error"] = f"{type(e).__name__}: {e}"
                t["traceback"] = traceback.format_exc()
                log(f"  FAILED: {t['error']}")
            return t

        tests = []
        tests.append(run_test(
            "c1-1_1step", paths["c1_1_step2"], num_frames=13,
            first_chunk_t=1, chunk_t=1, steps_per_chunk=[4, 1],
            mid_t_schedules="15/16,5/6,5/8;", num_steps=4, mid_t=[15/16, 5/6, 5/8]))
        tests.append(run_test(
            "c3-3_4step", paths["c3_3_step4"], num_frames=21,
            first_chunk_t=3, chunk_t=3, steps_per_chunk=None,
            mid_t_schedules="", num_steps=4, mid_t=[15/16, 5/6, 5/8]))

        results["tests"] = tests
        ok = [t for t in tests if t.get("ok")]
        results["verdict"] = {
            "overall": "PASS" if len(ok) == 2 else ("PARTIAL" if len(ok) == 1 else "FAIL"),
            "n_passed": len(ok),
        }
        results["status"] = "complete"

    except Exception as e:
        log(f"FATAL: {e}")
        traceback.print_exc()
        results["status"] = "failed"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()

    finally:
        results["end_time"] = datetime.now().isoformat()
        meta = OUT_DIR / f"{RUN_ID}_meta.json"
        with open(meta, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log(f"\nmeta saved: {meta}")
        log(f"VERDICT: {results.get('verdict', {}).get('overall', 'FAIL')}")


if __name__ == "__main__":
    main()
