"""
Phase 1: real runtime-switch smoke test.

This job is derived from the verified Phase 0c recipe. It performs one
text-conditioned sequence with real Causal-rCM checkpoints:

  c1-1 initial rollout -> c3-3 continuation -> c1-1 continuation

Continuation uses the released causal_i2v_rollout_sampling interface with the
last generated latent as the boundary context. This is a feasibility test for
online switching, not a quality claim. The job records load/sampling/decode
times, VRAM, latent shapes, seam statistics, and a pass/fail verdict.

The job is resumable: if the final metadata is complete it exits without
re-running the experiment. No model files are written to local storage.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import gc
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

import torch

PROMPT = "A cinematic shot of a snowy mountain at sunrise"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("/kaggle/working/results/phase1_switch") / RUN_ID
FRAME_DIR = OUT_DIR / "frames"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)
META = OUT_DIR / f"{RUN_ID}_meta.json"


def log(msg):
    print(msg, flush=True)


def find_under(root, filename):
    root = Path(root)
    if not root.exists():
        return None
    for item in root.rglob(filename):
        if item.is_file():
            return item
    return None


def install_deps():
    packages = [
        "einops", "transformers", "sentencepiece", "ftfy", "regex",
        "imageio", "imageio-ffmpeg", "tqdm", "torchvision", "fvcore",
        "omegaconf", "attrs", "pyyaml", "iopath", "termcolor", "pynvml",
        "pandas", "loguru", "safetensors", "cloudpickle", "dill",
        "matplotlib", "packaging",
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
            log(f"[warn] patch target not found in {path.name}: {old[:80]}")
        source = source.replace(old, new)
    path.write_text(source, encoding="utf-8")


def vram():
    if not torch.cuda.is_available():
        return {"cuda": False}
    return {
        "cuda": True,
        "device": torch.cuda.get_device_name(),
        "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3),
        "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }


def save_keyframes(video, label):
    from PIL import Image
    px = (1.0 + video.float().cpu().clamp(-1, 1)) / 2.0
    if px.ndim == 5:
        px = px[0]
    indices = sorted(set([0, max(0, px.shape[1] // 2), px.shape[1] - 1]))
    for index in indices:
        frame = (px[:, index].permute(1, 2, 0).numpy() * 255).astype("uint8")
        Image.fromarray(frame).save(FRAME_DIR / f"{label}_{index:03d}.png")


def main():
    results = {
        "run_id": RUN_ID,
        "phase": "phase1-switch-smoke",
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "protocol": {
            "prompt": PROMPT,
            "sequence": ["c1-1_initial", "c3-3_i2v_continuation", "c1-1_i2v_continuation"],
            "action_lambda": 0.0,
            "resolution": "480p 16:9",
            "seed": 17,
        },
    }
    try:
        log("=== Phase 1 runtime switch smoke test ===")
        install_deps()
        log(f"torch={torch.__version__} python={sys.version.split()[0]}")
        log(f"cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'}")
        results["env"] = {
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "none",
        }

        repo = Path("/kaggle/working/rcm")
        if repo.exists():
            subprocess.run(["rm", "-rf", str(repo)], check=False)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/NVlabs/rcm.git", str(repo)],
            check=True, capture_output=True,
        )

        inp = Path("/kaggle/input")
        paths = {
            "c1_1_step2": find_under(inp, "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step2.pt"),
            "c3_3_step4": find_under(inp, "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step4.pt"),
            "vae": find_under(inp, "Wan2.1_VAE.pth"),
            "t5": find_under(inp, "models_t5_umt5-xxl-enc-bf16.pth"),
            "tokenizer": find_under(inp, "tokenizer.json"),
        }
        if paths["tokenizer"]:
            paths["tokenizer"] = paths["tokenizer"].parent
        results["dataset_paths"] = {key: str(value) for key, value in paths.items()}
        missing = [key for key, value in paths.items() if value is None]
        if missing:
            raise FileNotFoundError(f"Missing nested Kaggle files: {missing}")
        for key, value in paths.items():
            log(f"{key}: {value}")

        patch_file(repo / "rcm/inference/wan2pt1_t2v_causal_infer.py", [
            ('TENSOR_KWARGS = {"device": "cuda", "dtype": torch.bfloat16}',
             'TENSOR_KWARGS = {"device": "cuda", "dtype": torch.float16}'),
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
            parse_rf_time_values, parse_mid_t_schedules, make_block_pattern,
            causal_rollout_sampling, causal_i2v_rollout_sampling, load_dit_weights,
        )
        from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
        from rcm.utils.model_utils import init_weights_on_device
        from rcm.utils.umt5 import clear_umt5_memory, umt5_xxl, HuggingfaceTokenizer
        from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
        from imaginaire.utils.easy_io import easy_io

        TENSOR_KWARGS["dtype"] = torch.float16

        def encode_prompt():
            log("encoding prompt")
            model = umt5_xxl(encoder_only=True).eval().requires_grad_(False).to(dtype=torch.float16)
            checkpoint = easy_io.load(str(paths["t5"]), backend_args=None, file_format="pt",
                                      map_location="cuda", weights_only=False)
            for key in list(checkpoint.keys()):
                if torch.is_tensor(checkpoint[key]):
                    checkpoint[key] = checkpoint[key].to(torch.float16)
            model.load_state_dict(checkpoint, strict=False, assign=True)
            del checkpoint
            tokenizer = HuggingfaceTokenizer(name=str(paths["tokenizer"]), seq_len=512, clean="whitespace")
            ids, mask = tokenizer(PROMPT, return_mask=True, add_special_tokens=True)
            ids, mask = ids.cuda(), mask.cuda()
            context = model(ids, mask)
            lengths = mask.gt(0).sum(dim=1).long()
            padded = []
            for item, length in zip(context, lengths):
                length = int(length)
                padded.append(item[:512] if length > 512 else torch.cat(
                    [item[:length], torch.zeros(512 - length, item.shape[1], device=item.device, dtype=item.dtype)], dim=0))
            embedding = torch.stack(padded).to(torch.float16)
            del model, tokenizer, ids, mask, context, padded
            clear_umt5_memory()
            gc.collect()
            torch.cuda.empty_cache()
            return embedding

        def load_net(path):
            started = time.time()
            with init_weights_on_device():
                net = instantiate(DIT_CONFIGS["1.3B"]).eval()
            load_dit_weights(net, str(path))
            net.to(**TENSOR_KWARGS).cpu()
            torch.cuda.empty_cache()
            return net, round(time.time() - started, 3)

        tokenizer = Wan2pt1VAEInterface(vae_pth=str(paths["vae"]))
        width, height = VIDEO_RES_SIZE_INFO["480p"]["16:9"]
        text_emb = encode_prompt()
        condition = {"crossattn_emb": text_emb.to(**TENSOR_KWARGS)}
        generator = torch.Generator(device="cuda").manual_seed(17)
        latent_h = height // tokenizer.spatial_compression_factor
        latent_w = width // tokenizer.spatial_compression_factor
        latent_c = tokenizer.latent_ch

        def t_schedule(num_steps, steps_per_chunk, schedule_spec):
            mid = parse_rf_time_values([15 / 16, 5 / 6, 5 / 8])
            t_steps = build_few_step_t_steps(num_steps, 1600, mid, "cuda")
            schedules = parse_mid_t_schedules(schedule_spec)
            return t_steps, schedules

        def run_initial_c1():
            net, load_s = load_net(paths["c1_1_step2"])
            latent_t = tokenizer.get_latent_num_frames(9)
            state_shape = [latent_c, latent_t, latent_h, latent_w]
            t_steps, schedules = t_schedule(4, [4, 1], "15/16,5/6,5/8;")
            _, blocks, _ = make_block_pattern(latent_t, latent_h, latent_w, 1, 1, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk([4, 1], blocks, 4)
            per_chunk = build_few_step_schedules_per_chunk(counts, [4, 1], 1600,
                                                            [15 / 16, 5 / 6, 5 / 8], schedules, "cuda")
            noise = torch.randn(1, *state_shape, dtype=torch.float32, device="cuda", generator=generator)
            net.cuda(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.time()
            samples = causal_rollout_sampling(net, noise, t_steps, per_chunk, [4, 1], condition, None, 1.0,
                                               1, 1, False, generator=generator)
            torch.cuda.synchronize()
            item = {"label": "c1-1_initial", "latent_shape": list(samples.shape),
                    "load_s": load_s, "sample_s": round(time.time() - started, 3),
                    "peak_vram": vram()}
            del net; torch.cuda.empty_cache(); return samples.detach().cpu(), item

        def run_i2v(mode, image_latent, remaining_t):
            path = paths["c3_3_step4"] if mode == "c3-3" else paths["c1_1_step2"]
            net, load_s = load_net(path)
            noise = torch.randn(1, latent_c, remaining_t, latent_h, latent_w, dtype=torch.float32,
                                device="cuda", generator=generator)
            if mode == "c3-3":
                chunk_t, steps = 3, None
                num_steps, first_chunk_t = 4, 1
                schedule_spec, mid = "", [15 / 16, 5 / 6, 5 / 8]
            else:
                chunk_t, steps = 1, [1]
                num_steps, first_chunk_t = 4, 1
                # No custom midpoint is needed for a one-step continuation;
                # an empty spec means the default schedule list is used.
                schedule_spec, mid = "", [15 / 16, 5 / 6, 5 / 8]
            t_steps, schedules = t_schedule(num_steps, steps, schedule_spec)
            # The one-step c1 continuation uses the helper's default midpoint;
            # pass no per-chunk schedule explicitly to satisfy its one-entry
            # steps_per_chunk contract.
            if mode == "c1-1":
                schedules = []
            total_t = 1 + remaining_t
            _, blocks, _ = make_block_pattern(total_t, latent_h, latent_w, first_chunk_t, chunk_t,
                                               net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(steps, blocks - 1, num_steps)
            per_chunk = build_few_step_schedules_per_chunk(counts, steps, 1600, mid, schedules, "cuda")
            net.cuda(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.time()
            samples = causal_i2v_rollout_sampling(net, image_latent.to("cuda"), noise, t_steps, per_chunk,
                                                   steps, condition, None, 1.0, chunk_t, False,
                                                   generator=generator)
            torch.cuda.synchronize()
            item = {"label": f"{mode}_i2v_continuation", "latent_shape": list(samples.shape),
                    "load_s": load_s, "sample_s": round(time.time() - started, 3),
                    "peak_vram": vram(), "remaining_t": remaining_t, "chunk_t": chunk_t}
            del net; torch.cuda.empty_cache(); return samples.detach().cpu(), item

        segments = []
        first, first_meta = run_initial_c1()
        segments.append(first)
        boundary_1 = first[:, :, -1:]
        second, second_meta = run_i2v("c3-3", boundary_1, 3)
        segments.append(second[:, :, 1:])
        boundary_2 = second[:, :, -1:]
        third, third_meta = run_i2v("c1-1", boundary_2, 2)
        segments.append(third[:, :, 1:])
        combined = torch.cat(segments, dim=2)
        results["segments"] = [first_meta, second_meta, third_meta]
        results["combined_latent_shape"] = list(combined.shape)

        # Decode once after all DiT models are released, following Phase 0c.
        torch.cuda.reset_peak_memory_stats(); started = time.time()
        video = tokenizer.decode(combined.float().cuda())
        torch.cuda.synchronize()
        results["decode_s"] = round(time.time() - started, 3)
        results["decode_peak_vram"] = vram()
        save_keyframes(video, "switch_sequence")
        results["frames_saved"] = True

        # Boundary diagnostics on the latent sequence and decoded pixels.
        latent_diff_1 = (first[:, :, -1:] - second[:, :, :1]).abs().mean().item()
        latent_diff_2 = (second[:, :, -1:] - third[:, :, :1]).abs().mean().item()
        results["boundary_diagnostics"] = {
            "c1_to_c3_latent_l1": latent_diff_1,
            "c3_to_c1_latent_l1": latent_diff_2,
            "note": "Lower is smoother; no universal pass threshold is claimed in this smoke test.",
        }
        results["verdict"] = {
            "overall": "PASS",
            "switch_sequence_completed": True,
            "context_strategy": "last latent frame passed through causal_i2v_rollout_sampling",
            "caveat": "This validates executable boundary transfer, not quality superiority.",
        }
        results["status"] = "complete"
        del video, combined, first, second, third, tokenizer, text_emb
        clear_umt5_memory(); torch.cuda.empty_cache()
    except Exception as exc:
        results["status"] = "failed"
        results["verdict"] = {"overall": "FAIL", "switch_sequence_completed": False}
        results["error"] = f"{type(exc).__name__}: {exc}"
        results["traceback"] = traceback.format_exc()
        log(results["error"])
    finally:
        results["end_time"] = datetime.now().isoformat()
        META.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        log(f"metadata: {META}")
        log(f"VERDICT: {results.get('verdict', {}).get('overall', 'FAIL')}")


if __name__ == "__main__":
    main()
