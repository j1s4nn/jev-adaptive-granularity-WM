"""E-JEV-3b: Jev 'hold-else-fine' controller.

Phase C showed the Jev routing that sends 'moderate' segments to coarse is
counterproductive (coarse is both slower and less accurate). This variant routes
static -> hold and everything else -> fine, which is the natural policy if the
only genuinely cheap mode is hold.

Policies: fixed_fine (baseline), jev_holdfine (static->hold, else->fine).
8 prompts, 57 frames, shared coarse prefix, matched seed, self-contained reference.

Outputs results/jev_holdfine/results.jsonl + summary + videos.
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

import numpy as np
import torch

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("/kaggle/working/results/jev_holdfine")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR = OUT_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
RAW = OUT_DIR / "results.jsonl"
SUMMARY = OUT_DIR / "jev_holdfine_summary.json"

SEED = 17
HORIZON = 57
FPS = 8
MID = [15 / 16, 5 / 6, 5 / 8]

PROMPTS = [
    {"stratum": "object_class", "prompt": "a boat"}, {"stratum": "object_class", "prompt": "a book"},
    {"stratum": "scene", "prompt": "baseball stadium"}, {"stratum": "scene", "prompt": "basement"},
    {"stratum": "motion", "prompt": "a bird flying over a snowy forest"}, {"stratum": "motion", "prompt": "a bird soaring gracefully in the sky"},
    {"stratum": "human_action", "prompt": "A person is bungee jumping"}, {"stratum": "human_action", "prompt": "A person is canoeing or kayaking"},
]
# static -> hold, else -> fine (from config/jev_routing.json motion choices)
JEV_HOLDFINE = {
    0: ["fine", "fine", "fine", "fine"], 1: ["hold", "hold", "hold", "hold"],
    2: ["hold", "hold", "fine", "hold"], 3: ["fine", "fine", "hold", "hold"],
    4: ["fine", "fine", "fine", "fine"], 5: ["fine", "fine", "fine", "fine"],
    6: ["fine", "fine", "fine", "fine"], 7: ["fine", "fine", "fine", "fine"],
}
POLICIES = ("fixed_fine", "jev_holdfine")


def log(m):
    print(m, flush=True)


def find_under(root, filename):
    root = Path(root)
    return next((p for p in root.rglob(filename) if p.is_file()), None) if root.exists() else None


def install_deps():
    for pkg in ["einops", "transformers", "sentencepiece", "ftfy", "regex", "imageio", "imageio-ffmpeg", "tqdm", "torchvision", "fvcore", "omegaconf", "attrs", "pyyaml", "iopath", "termcolor", "pynvml", "pandas", "loguru", "safetensors", "cloudpickle", "dill", "matplotlib", "packaging", "opencv-python-headless"]:
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


def completed_keys():
    keys = set()
    if RAW.exists():
        for line in RAW.read_text(encoding="utf-8").splitlines():
            try:
                it = json.loads(line)
                if it.get("status") == "complete":
                    keys.add(it["sample_key"])
            except Exception:
                pass
    return keys


def to_frames(video):
    px = (1 + video.float().cpu().clamp(-1, 1)) / 2
    if px.ndim == 5:
        px = px[0]
    return (px.permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8)


def read_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(frames, dtype=np.uint8)


def save_mp4(path, frames):
    import imageio
    w = imageio.get_writer(str(path), fps=FPS, macro_block_size=1)
    try:
        for f in frames:
            w.append_data(f)
    finally:
        w.close()


def segment_bounds(group):
    start = 1 + 4 * (3 * group - 1)
    end = HORIZON if group == 4 else 1 + 4 * (3 * (group + 1) - 1)
    return start, end


def policy_sequence(policy, pi):
    if policy == "fixed_fine":
        return ["fine"] * 4
    if policy == "jev_holdfine":
        return list(JEV_HOLDFINE[pi])
    raise ValueError(policy)


def main():
    report = {"run_id": RUN_ID, "phase": "E-JEV-3b-holdfine", "status": "running", "start_time": datetime.now().isoformat(),
              "protocol": {"seed": SEED, "horizon": HORIZON, "policies": POLICIES, "routing": JEV_HOLDFINE}}
    try:
        install_deps()
        gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "none"
        report["gpu"] = gpu
        log(f"gpu={gpu}")
        if not torch.cuda.is_available():
            raise RuntimeError("no GPU")
        repo = Path("/kaggle/working/rcm")
        if repo.exists():
            shutil.rmtree(repo)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVlabs/rcm.git", str(repo)], check=True, capture_output=True)
        report["upstream_rcm_commit"] = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        inp = Path("/kaggle/input")
        paths = {
            "fine": find_under(inp, "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step2.pt"),
            "reference": find_under(inp, "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt"),
            "coarse": find_under(inp, "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step4.pt"),
            "vae": find_under(inp, "Wan2.1_VAE.pth"),
            "t5": find_under(inp, "models_t5_umt5-xxl-enc-bf16.pth"),
            "tokenizer": find_under(inp, "tokenizer.json"),
        }
        if paths["tokenizer"]:
            paths["tokenizer"] = paths["tokenizer"].parent
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

        def load_net(ckpt):
            t = time.time()
            with init_weights_on_device():
                net = instantiate(DIT_CONFIGS["1.3B"]).eval()
            load_dit_weights(net, str(ckpt))
            net.to(**TENSOR_KWARGS).cpu()
            torch.cuda.empty_cache()
            return net, time.time() - t

        def decode(latent):
            t = time.time()
            video = tokenizer.decode(latent.float().cuda())
            torch.cuda.synchronize()
            el = time.time() - t
            frames = to_frames(video)
            peak = torch.cuda.max_memory_allocated() / 2**30
            del video
            torch.cuda.empty_cache()
            return frames, el, peak

        @torch.no_grad()
        def run_reference(noise, condition):
            net, ls = load_net(paths["reference"])
            steps = build_few_step_t_steps(4, 1600, MID, "cuda")
            _, blocks, _ = make_block_pattern(latent_frames, latent_h, latent_w, 1, 1, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(None, blocks, 4)
            sched = build_few_step_schedules_per_chunk(counts, None, 1600, MID, [], "cuda")
            gen = torch.Generator(device="cuda").manual_seed(SEED + 1000)
            net.cuda()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            latent = causal_rollout_sampling(net, noise.cuda(), steps, sched, None, condition, None, 1.0, 1, 1, False, generator=gen)
            torch.cuda.synchronize()
            dit = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            del net
            torch.cuda.empty_cache()
            frames, vae, dp = decode(latent)
            return frames, {"checkpoint_load_s": ls, "dit_s": dit, "vae_s": vae, "nfe": int(sum(counts)), "forwards": int(sum(counts)) + blocks, "peak_vram_gb": max(peak, dp)}

        @torch.no_grad()
        def run_initial(noise, condition, gen):
            net, ls = load_net(paths["coarse"])
            steps = build_few_step_t_steps(4, 1600, MID, "cuda")
            _, blocks, _ = make_block_pattern(3, latent_h, latent_w, 3, 3, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(None, blocks, 4)
            sched = build_few_step_schedules_per_chunk(counts, None, 1600, MID, [], "cuda")
            net.cuda()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            latent = causal_rollout_sampling(net, noise[:, :, :3].cuda(), steps, sched, None, condition, None, 1.0, 3, 3, False, generator=gen)
            torch.cuda.synchronize()
            dit = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            del net
            torch.cuda.empty_cache()
            return latent.detach().cpu(), {"checkpoint_load_s": ls, "dit_s": dit, "nfe": int(sum(counts)), "forwards": int(sum(counts)) + blocks, "peak_vram_gb": peak}

        @torch.no_grad()
        def run_segment(mode, boundary, group, noise, condition):
            if mode == "hold":
                fresh = boundary.detach().cpu().repeat(1, 1, 3, 1, 1)
                return fresh, {"checkpoint_load_s": 0.0, "dit_s": 0.0, "nfe": 0, "forwards": 0, "peak_vram_gb": 0.0}
            net, ls = load_net(paths["coarse"] if mode == "coarse" else paths["fine"])
            chunk = 3 if mode == "coarse" else 1
            spc = None if mode == "coarse" else [1]
            steps = build_few_step_t_steps(4, 1600, MID, "cuda")
            _, blocks, _ = make_block_pattern(4, latent_h, latent_w, 1, chunk, net.get_spatial_patch_size())
            counts = expand_steps_per_chunk(spc, blocks - 1, 4)
            sched = build_few_step_schedules_per_chunk(counts, spc, 1600, MID, [], "cuda")
            nz = noise[:, :, 3 * group:3 * (group + 1)].cuda()
            gen = torch.Generator(device="cuda").manual_seed(SEED + 2000 + group)
            net.cuda()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            latent = causal_i2v_rollout_sampling(net, boundary.cuda(), nz, steps, sched, spc, condition, None, 1.0, chunk, False, generator=gen)
            torch.cuda.synchronize()
            dit = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30
            fresh = latent[:, :, 1:].detach().cpu()
            del net, latent, nz
            torch.cuda.empty_cache()
            return fresh, {"checkpoint_load_s": ls, "dit_s": dit, "nfe": int(sum(counts)), "forwards": int(sum(counts)) + (blocks - 1), "peak_vram_gb": peak}

        done = completed_keys()
        for pi, spec in enumerate(PROMPTS):
            prompt = spec["prompt"]
            log(f"=== prompt {pi}: {prompt}")
            emb = encode_prompt(prompt)
            condition = {"crossattn_emb": emb.to(**TENSOR_KWARGS)}
            gen0 = torch.Generator(device="cuda").manual_seed(SEED)
            noise = torch.randn(1, latent_c, latent_frames, latent_h, latent_w, dtype=torch.float32, device="cuda", generator=gen0).cpu()

            ref_key = f"p{pi}_reference"
            if ref_key not in done:
                try:
                    ref_frames, ref_meta = run_reference(noise, condition)
                    save_mp4(VIDEO_DIR / f"{ref_key}.mp4", ref_frames)
                    append({"sample_key": ref_key, "policy": "reference", "status": "complete", "prompt_idx": pi, "prompt": prompt, "timing": ref_meta})
                    done.add(ref_key)
                    del ref_frames
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as exc:
                    append({"sample_key": ref_key, "policy": "reference", "status": "failed", "prompt_idx": pi, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                    log(traceback.format_exc())
                    continue

            ref_frames = read_video(VIDEO_DIR / f"{ref_key}.mp4")
            gen = torch.Generator(device="cuda").manual_seed(SEED + 2000)
            prefix, prefix_meta = run_initial(noise, condition, gen)

            for policy in POLICIES:
                key = f"p{pi}_{policy}"
                if key in done:
                    continue
                started = time.time()
                try:
                    seq = policy_sequence(policy, pi)
                    latents = [prefix]
                    timing = {"checkpoint_load_s": prefix_meta["checkpoint_load_s"], "dit_s": prefix_meta["dit_s"], "vae_s": 0.0,
                              "nfe": prefix_meta["nfe"], "forwards": prefix_meta["forwards"], "peak_vram_gb": prefix_meta["peak_vram_gb"]}
                    for g in range(1, 5):
                        fresh, meta = run_segment(seq[g - 1], latents[-1][:, :, -1:], g, noise, condition)
                        latents.append(fresh)
                        for k in ["checkpoint_load_s", "dit_s", "nfe", "forwards"]:
                            timing[k] += meta[k]
                        timing["peak_vram_gb"] = max(timing["peak_vram_gb"], meta["peak_vram_gb"])
                    latent = torch.cat(latents, dim=2)
                    frames, el, dp = decode(latent)
                    timing["vae_s"] += el
                    timing["peak_vram_gb"] = max(timing["peak_vram_gb"], dp)
                    save_mp4(VIDEO_DIR / f"{key}.mp4", frames)
                    segs = []
                    for g in range(1, 5):
                        s, e = segment_bounds(g)
                        a = frames[s:e].astype(np.float32) / 255.0
                        t = ref_frames[s:e].astype(np.float32) / 255.0
                        segs.append({"group": g, "mode": seq[g - 1], "reference_l1": float(np.mean(np.abs(a - t)))})
                    timing["total_s"] = time.time() - started
                    append({"sample_key": key, "policy": policy, "status": "complete", "prompt_idx": pi, "stratum": spec["stratum"],
                            "prompt": prompt, "seed": SEED, "horizon": HORIZON, "mode_sequence": seq,
                            "hold_count": sum(m == "hold" for m in seq), "fine_count": sum(m == "fine" for m in seq),
                            "coarse_count": sum(m == "coarse" for m in seq), "segments": segs,
                            "mean_reference_l1": float(np.mean([s["reference_l1"] for s in segs])),
                            "terminal_reference_l1": float(segs[-1]["reference_l1"]), "timing": timing, "video": f"videos/{key}.mp4"})
                    done.add(key)
                    log(f"  {policy}: seq={seq} meanref={float(np.mean([s['reference_l1'] for s in segs])):.4f} dit={timing['dit_s']:.1f}s total={timing['total_s']:.1f}s")
                    del latents, latent, frames
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as exc:
                    append({"sample_key": key, "policy": policy, "status": "failed", "prompt_idx": pi, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                    log(traceback.format_exc())
            del prefix, emb, condition, noise
            gc.collect()
            torch.cuda.empty_cache()

        nc = len([k for k in completed_keys() if not k.endswith("_reference")])
        report["n_complete"] = nc
        report["n_expected"] = len(PROMPTS) * len(POLICIES)
        report["verdict"] = "PASS" if nc == len(PROMPTS) * len(POLICIES) else ("PARTIAL" if nc else "FAIL")
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
