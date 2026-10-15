"""A100-JEV / online Jev-routed policies on the A100 (Google Colab, single-file kernel).

Phase 1 of the Jev-maximized plan. Same structure as colab/a100_online.py (shared coarse
prefix + 4 segments of 3 latent frames, 57 frames, cheap coarse c3-3 step-1) but the online
signal is **Jev** (TypeSafe System One) instead of the warp threshold:

  - BLIP caption of the last decoded context frame + motion/warp stats -> Jev state
  - Jev questions: coarse_drift (noul), difficulty (score), motion (choice)
  - policies:
      fixed_fine / fixed_coarse / fixed_hold / random50
      warp                 : prior A100 baseline (threshold -4.0458)
      jev_difficulty       : fine if difficulty >= 1.72 else coarse
      jev_routed           : static->hold, moderate->coarse, fast/scene_change->fine
      jev_holdfine         : static->hold, else->fine

Signal is computed off the DiT path. Independent CLIP + classical endpoints are computed
in-kernel over the saved videos. Outputs /content/cache/results/a100_online_jev/.
Requires KAGGLE_USERNAME/KAGGLE_KEY and TYPESAFE_API_KEY. Never touches prior artifacts.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import glob
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT = Path("/content")
DRIVE = Path("/content/drive/MyDrive")
CACHE = (DRIVE / "a100_cache") if DRIVE.exists() else (ROOT / "cache")
INPUTS_DIR = CACHE / "inputs"
OUT_NAME = os.environ.get("JEV_OUT_NAME", "a100_online_jev")
OUT_DIR = CACHE / "results" / OUT_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR = OUT_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
RAW = OUT_DIR / "results.jsonl"
SUMMARY = OUT_DIR / "a100_online_jev_summary.json"
CLIP_JSON = OUT_DIR / "a100_online_jev_quality_clip.json"
CLASSICAL_JSON = OUT_DIR / "a100_online_jev_quality_classical.json"

MANIFEST_VERSION = "a100-01"
SEED = 17
HORIZON = 57
FPS = 8
MID = [15 / 16, 5 / 6, 5 / 8]
WARP_DIRECTION = -1
WARP_THRESHOLD = -4.045816421508789  # A100 refit (A2 v2)
JEV_DIFFICULTY_THRESHOLD = 1.72      # Phase 0.2 Youden optimum
RANDOM_SEED = 20261025
JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
PIN_COMMIT = "ed3cb14dd936f92cdc9f9381af7369991509b41f"

ALL_PROMPTS = [
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
# Optional external prompt manifest (Phase 3): {"prompts":[{"stratum":..,"prompt":..},..]}
_pf = os.environ.get("JEV_PROMPTS_FILE")
if _pf and Path(_pf).exists():
    ALL_PROMPTS = [{"stratum": x.get("stratum", ""), "prompt": x["prompt"]}
                   for x in json.loads(Path(_pf).read_text(encoding="utf-8"))["prompts"]]

# Chunked execution: run a prompt range per session so a reclaimed runtime only
# loses one chunk. Global prompt index is preserved (used for filenames + CLIP).
PROMPT_START = int(os.environ.get("JEV_PROMPT_START", "0"))
PROMPT_END = int(os.environ.get("JEV_PROMPT_END", str(len(ALL_PROMPTS))))
PROMPTS = ALL_PROMPTS  # full list; iteration range below
SELECTED = list(range(PROMPT_START, min(PROMPT_END, len(ALL_PROMPTS))))
POLICIES = ("fixed_coarse", "fixed_fine", "fixed_hold", "random50", "warp",
            "jev_difficulty", "jev_routed", "jev_holdfine", "jev_single", "latent")
# Optional subset (comma-separated) for focused runs, e.g. Phase 2 signal variants.
_policy_env = os.environ.get("JEV_POLICIES", "").strip()
if _policy_env:
    POLICIES = tuple(p for p in _policy_env.split(",") if p)

CKPT_NAMES = {
    "fine": "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step2.pt",
    "reference": "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt",
    "coarse": "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step2.pt",
}

JEV_QUESTIONS = {
    "coarse_drift": {
        "type": "noul",
        "instructions": "Given this video context, would generating the next few frames with a coarse 3-frame-chunk method likely drift or become temporally inconsistent compared with a frame-wise method?",
        "criteria": {
            "true": "Fast or complex motion, occlusion, or a scene change that a coarse chunk would blur or smear",
            "false": "Simple, slow, or static motion that a coarse chunk would reproduce faithfully",
        },
    },
    "difficulty": {
        "type": "score",
        "instructions": "How difficult is it to predict the next few frames of this video accurately?",
        "criteria": ["very easy", "easy", "moderate", "hard", "very hard"],
    },
    "motion": {
        "type": "choice",
        "instructions": "What kind of motion best describes this context?",
        "criteria": {"static": "Almost no motion", "moderate": "Some smooth motion",
                     "fast": "Fast or complex motion", "scene_change": "A cut or large scene change"},
    },
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
                "cloudpickle", "dill", "matplotlib", "packaging", "opencv-python-headless",
                "kaggle", "requests", "huggingface_hub"]
    for pkg in packages:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pkg], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"pip install failed for {pkg}: {r.stderr[-400:]}")


def download_datasets():
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
    if not (ck.exists() and find_under(ck, CKPT_NAMES["reference"]) is not None):
        if ck.exists():
            shutil.rmtree(ck)
        ck.mkdir(parents=True, exist_ok=True)
        log("downloading ajjisan/causal-rcm-ckpts ...")
        subprocess.run(["kaggle", "datasets", "download", "-d", "ajjisan/causal-rcm-ckpts", "-p", str(ck), "--unzip"], check=True)
    hf = dst / "wan21"
    hf.mkdir(parents=True, exist_ok=True)
    if not (hf / "models_t5_umt5-xxl-enc-bf16.pth").exists():
        from huggingface_hub import snapshot_download
        log("downloading Wan-AI/Wan2.1-T2V-1.3B ...")
        snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-1.3B", local_dir=str(hf),
                          allow_patterns=["models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "config.json", "google/umt5-xxl/*"])
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


def segment_bounds(group):
    start = 1 + 4 * (3 * group - 1)
    end = HORIZON if group == 4 else 1 + 4 * (3 * (group + 1) - 1)
    return start, end


def warp_and_motion(frames):
    import cv2
    history = np.asarray(frames, dtype=np.float32)
    prevprev = history[-3].mean(axis=2)
    prev = history[-2].mean(axis=2)
    flow = cv2.calcOpticalFlowFarneback(prevprev, prev, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = prev.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx = np.clip(xx + flow[..., 0], 0, w - 1)
    my = np.clip(yy + flow[..., 1], 0, h - 1)
    warped = cv2.remap(prevprev, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warp = float(np.mean(np.abs(prev - warped)))
    motion = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
    return warp, motion


def main():
    report = {"run_id": RUN_ID, "phase": "A100-JEV-online-policies", "status": "running",
              "start_time": datetime.now().isoformat(),
              "protocol": {"manifest_version": MANIFEST_VERSION, "seed": SEED, "horizon": HORIZON,
                           "prompts": PROMPTS, "policies": POLICIES,
                           "warp_threshold": WARP_THRESHOLD, "jev_difficulty_threshold": JEV_DIFFICULTY_THRESHOLD,
                           "jev_rule": "static->hold, moderate->coarse, fast/scene_change->fine",
                           "structure": "common coarse prefix (3 latent frames) + 4 segments of 3 latent frames"}}
    try:
        install_deps()
        log(f"torch={torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        jev_key = os.environ.get("TYPESAFE_API_KEY")
        if not jev_key:
            raise RuntimeError("TYPESAFE_API_KEY required")
        import requests
        from PIL import Image
        from transformers import BlipForConditionalGeneration, BlipProcessor

        blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval().cuda()
        cap_cache = {}

        def caption(frame):
            key = int(frame[::8, ::8].sum())
            if key in cap_cache:
                return cap_cache[key]
            image = Image.fromarray(frame)
            with torch.no_grad():
                out = blip_model.generate(**blip_proc(image, return_tensors="pt").to("cuda"), max_new_tokens=24)
            text = blip_proc.decode(out[0], skip_special_tokens=True)
            cap_cache[key] = text
            return text

        def query_jev(prompt, cap, seg_idx, motion, warp):
            state = {"video_prompt": prompt, "context_caption": cap, "segment_index": seg_idx,
                     "horizon_frames": HORIZON, "recent_motion_magnitude": round(motion, 3),
                     "recent_temporal_warp": round(warp, 3)}
            payload = {"model": "jev-latest", "state": state, "questions": JEV_QUESTIONS}
            started = time.time()
            answer, error = None, None
            for attempt in range(4):
                try:
                    resp = requests.post(JEV_ENDPOINT, headers={"Authorization": "Bearer " + jev_key, "Content-Type": "application/json"}, json=payload, timeout=40)
                    if resp.status_code in (429, 529):
                        time.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    answer = resp.json()
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    time.sleep(1 + attempt)
            return answer, time.time() - started, error

        inputs = download_datasets()
        repo = ROOT / "rcm"
        if repo.exists():
            shutil.rmtree(repo)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVlabs/rcm.git", str(repo)], check=True, capture_output=True)
        report["upstream_rcm_commit"] = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        paths = {k: find_under(inputs, name) for k, name in CKPT_NAMES.items()}
        paths["vae"] = find_under(inputs, "Wan2.1_VAE.pth")
        paths["t5"] = find_under(inputs, "models_t5_umt5-xxl-enc-bf16.pth")
        tk = find_under(inputs, "tokenizer.json")
        paths["tokenizer"] = tk.parent if tk else None
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
                return boundary.repeat(1, 1, 3, 1, 1).detach().cpu(), {"checkpoint_load_s": 0.0, "dit_s": 0.0, "nfe": 0, "peak_vram_gb": 0.0}
            net, load_s = load_net(paths["coarse"] if mode == "coarse" else paths["fine"])
            chunk = 3 if mode == "coarse" else 1
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

        def jev_mode(policy, answer):
            a = answer["answers"]
            motion = a["motion"]["choice"]
            diff = a["difficulty"]["score"]
            if policy == "jev_difficulty":
                return "fine" if diff >= JEV_DIFFICULTY_THRESHOLD else "coarse"
            if policy == "jev_routed":
                if motion == "static":
                    return "hold"
                return "coarse" if motion == "moderate" else "fine"
            if policy == "jev_holdfine":
                return "hold" if motion == "static" else "fine"
            raise ValueError(policy)

        done = completed_keys()
        resumed = len(done)
        log(f"resume: {resumed} completed runs already present in {RAW}")
        for pi in SELECTED:
            spec = PROMPTS[pi]
            prompt = spec["prompt"]
            if all(f"p{pi}_{pol}" in done for pol in POLICIES):
                log(f"=== prompt {pi}: all {len(POLICIES)} policies already done, skipping")
                continue
            log(f"=== prompt {pi}: {prompt}")
            embedding = encode_prompt(prompt)
            condition = {"crossattn_emb": embedding.to(**TENSOR_KWARGS)}
            gen0 = torch.Generator(device="cuda").manual_seed(SEED)
            canonical_noise = torch.randn(1, latent_c, latent_frames, latent_h, latent_w, dtype=torch.float32, device="cuda", generator=gen0).cpu()
            noise_hash = hashlib.sha256(canonical_noise.numpy().tobytes()).hexdigest()

            ref_key = f"p{pi}_reference"
            ref_path = VIDEO_DIR / f"{ref_key}.mp4"
            if ref_key not in done or not ref_path.exists():
                try:
                    ref_started = time.time()
                    ref_frames, ref_meta = run_reference(canonical_noise, condition)
                    save_mp4(ref_path, ref_frames)
                    append_result({"sample_key": ref_key, "policy": "reference", "status": "complete", "prompt_idx": pi,
                                   "prompt": prompt, "seed": SEED, "horizon": HORIZON, "initial_noise_sha256": noise_hash,
                                   "timing": {**ref_meta, "total_s": time.time() - ref_started}, "decoded_shape": list(ref_frames.shape)})
                    done.add(ref_key)
                    del ref_frames
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as exc:
                    append_result({"sample_key": ref_key, "policy": "reference", "status": "failed", "prompt_idx": pi, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                    log(traceback.format_exc())
                    continue
            ref_frames = read_video(ref_path)

            for policy in POLICIES:
                key = f"p{pi}_{policy}"
                if key in done:
                    continue
                started = time.time()
                try:
                    gen = torch.Generator(device="cuda").manual_seed(SEED + 2000)
                    initial, initial_meta = run_initial(canonical_noise, condition, gen)
                    latents = [initial]
                    timing = {"checkpoint_load_s": initial_meta["checkpoint_load_s"], "dit_s": initial_meta["dit_s"],
                              "vae_s": 0.0, "signal_s": 0.0, "jev_s": 0.0, "blip_s": 0.0,
                              "nfe": initial_meta["nfe"], "peak_vram_gb": initial_meta["peak_vram_gb"]}
                    context_frames = None
                    needs_signal = policy in ("warp", "jev_difficulty", "jev_routed", "jev_holdfine", "jev_single")
                    if needs_signal:
                        context_frames, elapsed, dpeak = decode(initial)
                        timing["vae_s"] += elapsed
                        timing["peak_vram_gb"] = max(timing["peak_vram_gb"], dpeak)
                    random_fine_groups = set()
                    if policy == "random50":
                        import random
                        random_fine_groups = set(random.Random(RANDOM_SEED + pi).sample([1, 2, 3, 4], 2))
                    decisions = []
                    single_answer = None
                    latent_ref = None
                    for group in range(1, 5):
                        raw_warp = motion = oriented = None
                        answer = None
                        signal_s = 0.0
                        latent_sig = None
                        if policy == "fixed_coarse":
                            mode = "coarse"
                        elif policy == "fixed_fine":
                            mode = "fine"
                        elif policy == "fixed_hold":
                            mode = "hold"
                        elif policy == "random50":
                            mode = "fine" if group in random_fine_groups else "coarse"
                        elif policy == "warp":
                            raw_warp, motion = warp_and_motion(context_frames)
                            oriented = WARP_DIRECTION * raw_warp
                            mode = "fine" if oriented >= WARP_THRESHOLD else "coarse"
                        elif policy == "latent":
                            # FREE latent-space proxy: std of the boundary latent frame, NO decode
                            t0 = time.time()
                            latent_sig = float(latents[-1][:, :, -1:].float().std())
                            if latent_ref is None:
                                latent_ref = latent_sig
                            mode = "fine" if latent_sig > latent_ref else "coarse"
                            latent_ref = 0.5 * latent_ref + 0.5 * latent_sig
                            signal_s = time.time() - t0
                        elif policy == "jev_single":
                            if group == 1:
                                raw_warp, motion = warp_and_motion(context_frames)
                                t0 = time.time()
                                cap = caption(context_frames[-1])
                                timing["blip_s"] += time.time() - t0
                                t0 = time.time()
                                single_answer, jev_lat, err = query_jev(prompt, cap, 1, motion, raw_warp)
                                timing["jev_s"] += time.time() - t0
                                if single_answer is None:
                                    raise RuntimeError(f"jev failed: {err}")
                            answer = single_answer
                            mode = jev_mode("jev_routed", answer)
                        else:  # jev_difficulty / jev_routed / jev_holdfine (per-segment)
                            raw_warp, motion = warp_and_motion(context_frames)
                            t0 = time.time()
                            cap = caption(context_frames[-1])
                            timing["blip_s"] += time.time() - t0
                            t0 = time.time()
                            answer, jev_lat, err = query_jev(prompt, cap, group, motion, raw_warp)
                            timing["jev_s"] += time.time() - t0
                            if answer is None:
                                raise RuntimeError(f"jev failed: {err}")
                            mode = jev_mode(policy, answer)
                        boundary = latents[-1][:, :, -1:]
                        fresh, meta = run_i2v(mode, boundary, group, canonical_noise, condition)
                        if fresh.shape[2] != 3:
                            raise RuntimeError(f"segment returned {fresh.shape[2]} frames")
                        latents.append(fresh)
                        timing["checkpoint_load_s"] += meta["checkpoint_load_s"]
                        timing["dit_s"] += meta["dit_s"]
                        timing["nfe"] += meta["nfe"]
                        timing["peak_vram_gb"] = max(timing["peak_vram_gb"], meta["peak_vram_gb"])
                        decisions.append({"group": group, "mode": mode, "raw_warp": raw_warp, "motion": motion,
                                          "oriented_score": oriented, "latent_sig": latent_sig, "jev_answer": answer, **meta})
                        if needs_signal and group < 4 and policy != "jev_single":
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
                        s, e = segment_bounds(group)
                        actual = frames[s:e].astype(np.float32) / 255.0
                        target = ref_frames[s:e].astype(np.float32) / 255.0
                        segments.append({"group": group, "start_frame": s, "end_frame": e,
                                         "reference_l1": float(np.mean(np.abs(actual - target))),
                                         "adjacent_pixel_l1": float(np.mean(np.abs(actual[1:] - actual[:-1])))})
                    modes = [d["mode"] for d in decisions]
                    timing["total_s"] = time.time() - started
                    append_result({"sample_key": key, "policy": policy, "status": "complete", "prompt_idx": pi,
                                   "stratum": spec["stratum"], "prompt": prompt, "seed": SEED, "horizon": HORIZON,
                                   "hardware": "A100-SXM4-80GB", "initial_noise_sha256": noise_hash,
                                   "decoded_shape": list(frames.shape), "decisions": decisions, "segments": segments,
                                   "fine_count": sum(m == "fine" for m in modes), "coarse_count": sum(m == "coarse" for m in modes),
                                   "hold_count": sum(m == "hold" for m in modes), "fine_fraction": sum(m == "fine" for m in modes) / 4,
                                   "switch_count": sum(modes[i] != modes[i - 1] for i in range(1, 4)),
                                   "mean_reference_l1": float(np.mean([s["reference_l1"] for s in segments])),
                                   "timing": timing, "video": f"videos/{key}.mp4",
                                   "metric": "decoded-pixel L1 to internal c1-1 step-4 reference; diagnostic"})
                    done.add(key)
                    log(f"  {policy}: modes={''.join(m[0].upper() for m in modes)} fine={sum(m=='fine' for m in modes)} hold={sum(m=='hold' for m in modes)} "
                        f"refL1={float(np.mean([s['reference_l1'] for s in segments])):.4f} dit={timing['dit_s']:.2f}s vae={timing['vae_s']:.2f}s jev={timing['jev_s']:.2f}s total={timing['total_s']:.2f}s")
                    del initial, latents, latent, frames, context_frames
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as exc:
                    append_result({"sample_key": key, "policy": policy, "status": "failed", "prompt_idx": pi, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                    log(traceback.format_exc())
            del embedding, condition, canonical_noise
            gc.collect()
            torch.cuda.empty_cache()

        # independent quality endpoints
        def clip_quality():
            import cv2
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor
            name_re = re.compile(r"^p(\d+)_(" + "|".join(POLICIES) + r")$")
            device = "cuda"
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
            proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            per = {}
            with torch.no_grad():
                for path in sorted(glob.glob(str(VIDEO_DIR / "*.mp4"))):
                    stem = os.path.splitext(os.path.basename(path))[0]
                    m = name_re.match(stem)
                    if not m:
                        continue
                    pi, pol = int(m.group(1)), m.group(2)
                    frames = read_video(path)[::2]
                    if not frames:
                        continue
                    imgs = [Image.fromarray(f) for f in frames]
                    inp = proc(images=imgs, return_tensors="pt", padding=True).to(device)
                    out = model.get_image_features(**inp)
                    emb = (out if torch.is_tensor(out) else out.pooler_output)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    arr = emb.cpu().numpy()
                    cons = float(np.mean(np.sum(arr[:-1] * arr[1:], axis=1))) if len(arr) > 1 else None
                    txt = proc(text=[PROMPTS[pi]["prompt"]], return_tensors="pt", padding=True).to(device)
                    to = model.get_text_features(**txt)
                    te = (to if torch.is_tensor(to) else to.pooler_output)
                    te = te / te.norm(dim=-1, keepdim=True)
                    sims = (emb @ te.t()).squeeze(-1).cpu().numpy()
                    qtxt = proc(text=["a high quality, sharp, detailed, natural photo",
                                      "a blurry, low quality, distorted, artifact-ridden image"],
                                return_tensors="pt", padding=True).to(device)
                    qo = model.get_text_features(**qtxt)
                    qe = (qo if torch.is_tensor(qo) else qo.pooler_output)
                    qe = qe / qe.norm(dim=-1, keepdim=True)
                    qsims = (emb @ qe.t()).cpu().numpy()
                    per[stem] = {"prompt_idx": pi, "policy": pol, "subject_consistency": cons,
                                 "text_alignment": float(np.mean(sims)),
                                 "clip_quality": float(np.mean(qsims[:, 0] - qsims[:, 1]))}
            agg = {}
            for pol in POLICIES:
                rs = [v for v in per.values() if v["policy"] == pol]
                agg[pol] = {"n": len(rs),
                            "subject_consistency": statistics.fmean([r["subject_consistency"] for r in rs if r["subject_consistency"] is not None]) if rs else None,
                            "text_alignment": statistics.fmean([r["text_alignment"] for r in rs if r["text_alignment"] is not None]) if rs else None,
                            "clip_quality": statistics.fmean([r.get("clip_quality") for r in rs if r.get("clip_quality") is not None]) if rs else None}
            CLIP_JSON.write_text(json.dumps({"phase": "A100-JEV-clip", "by_policy": agg, "per_video": per}, indent=2), encoding="utf-8")
            log("CLIP quality written: " + str(CLIP_JSON))

        def classical_quality():
            import cv2
            name_re = re.compile(r"^p(\d+)_(" + "|".join(POLICIES) + r")$")
            per = {}
            for path in sorted(glob.glob(str(VIDEO_DIR / "*.mp4"))):
                stem = os.path.splitext(os.path.basename(path))[0]
                m = name_re.match(stem)
                if not m:
                    continue
                frames = read_video(path)
                if len(frames) < 4:
                    continue
                grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
                small = [cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA) for g in grays]
                h, w = small[0].shape
                yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
                warps = []
                for a, b in zip(small[:-1], small[1:]):
                    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mx = np.clip(xx + flow[..., 0], 0, w - 1)
                    my = np.clip(yy + flow[..., 1], 0, h - 1)
                    wp = cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    warps.append(float(np.mean(np.abs(b.astype(np.float32) - wp.astype(np.float32))) / 255.0))
                per[stem] = {"prompt_idx": int(m.group(1)), "policy": m.group(2), "temporal_warp_mean": float(np.mean(warps))}
            agg = {}
            for pol in POLICIES:
                rs = [v for v in per.values() if v["policy"] == pol]
                agg[pol] = {"n": len(rs), "temporal_warp_mean": statistics.fmean([r["temporal_warp_mean"] for r in rs]) if rs else None}
            CLASSICAL_JSON.write_text(json.dumps({"phase": "A100-JEV-classical", "by_policy": agg, "per_video": per}, indent=2), encoding="utf-8")
            log("classical quality written: " + str(CLASSICAL_JSON))

        try:
            clip_quality()
        except Exception as exc:
            log("clip quality failed: " + repr(exc))
        try:
            classical_quality()
        except Exception as exc:
            log("classical quality failed: " + repr(exc))

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
        log("A100_JEV_DONE verdict=" + str(report.get("verdict")))


if __name__ == "__main__":
    main()
