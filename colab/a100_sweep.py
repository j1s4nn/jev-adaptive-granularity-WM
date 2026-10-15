"""A100-01 / A1 mode-cost + quality sweep (Google Colab A100, single-file kernel).

Hardware-conditional arm of the adaptive temporal-granularity study. Repeats the
F1.3 six-mode sweep on an A100-SXM4-80GB WITH real umT5 text encoding and real VAE
decode (the earlier results/a100_timing run was timing-only: dummy text embedding,
no decode). Same prompts, modes, seed and horizons as the T4 F1 job so T4-vs-A100 is
a clean hardware comparison.

Modes (verified recipe, master plan PII.1):
  ref_c1_1_s4     c1-1_step4, first=1 chunk=1, 4-step      (internal reference)
  fine_c1_1_s1    c1-1_step2, first=1 chunk=1, [4,1]
  fine_c1_1_s2    c1-1_step2, first=1 chunk=1, [4,2]
  coarse_c3_3_s4  c3-3_step4, first=3 chunk=3, 4-step
  coarse_c3_3_s2  c3-3_step2, first=3 chunk=3, [4,2]
  coarse_c3_3_s1  c3-3_step2, first=3 chunk=3, [4,1]

Outputs under /content/results/a100_mode_cost/: results.jsonl, summary JSON,
a100_quality_classical.json, a100_quality_clip.json, videos/*.mp4.
T4 artifacts are never touched. Requires env KAGGLE_USERNAME / KAGGLE_KEY.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import glob
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
OUT_DIR = ROOT / "results" / "a100_mode_cost"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR = OUT_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSONL = OUT_DIR / "results.jsonl"
SUMMARY_JSON = OUT_DIR / "a100_mode_cost_summary.json"
CLASSICAL_JSON = OUT_DIR / "a100_quality_classical.json"
CLIP_JSON = OUT_DIR / "a100_quality_clip.json"

FPS = 8
SEED = 17
HORIZONS = [45, 57, 81]
MID_T = [15 / 16, 5 / 6, 5 / 8]

# Same development prompts as F1.3 for direct T4-vs-A100 comparability.
PROMPTS = [
    {"id": "dev_object", "stratum": "object_class", "prompt": "a banana"},
    {"id": "dev_motion", "stratum": "motion_deformation", "prompt": "a bear catching a salmon in its powerful jaws"},
    {"id": "dev_human", "stratum": "human_action", "prompt": "A person is air drumming"},
]

CKPT_NAMES = {
    "c1_1_step2": "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step2.pt",
    "c1_1_step4": "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt",
    "c3_3_step2": "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step2.pt",
    "c3_3_step4": "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step4.pt",
}

MODE_ORDER = ["ref_c1_1_s4", "fine_c1_1_s1", "fine_c1_1_s2", "coarse_c3_3_s4", "coarse_c3_3_s2", "coarse_c3_3_s1"]
MODES = {
    "ref_c1_1_s4": {"ckpt": "c1_1_step4", "first": 1, "chunk": 1, "steps": None, "schedule": "", "default_steps": 4},
    "fine_c1_1_s1": {"ckpt": "c1_1_step2", "first": 1, "chunk": 1, "steps": [4, 1], "schedule": "15/16,5/6,5/8;", "default_steps": 4},
    "fine_c1_1_s2": {"ckpt": "c1_1_step2", "first": 1, "chunk": 1, "steps": [4, 2], "schedule": "15/16,5/6,5/8;5/6", "default_steps": 4},
    "coarse_c3_3_s4": {"ckpt": "c3_3_step4", "first": 3, "chunk": 3, "steps": None, "schedule": "", "default_steps": 4},
    "coarse_c3_3_s2": {"ckpt": "c3_3_step2", "first": 3, "chunk": 3, "steps": [4, 2], "schedule": "15/16,5/6,5/8;5/6", "default_steps": 2},
    "coarse_c3_3_s1": {"ckpt": "c3_3_step2", "first": 3, "chunk": 3, "steps": [4, 1], "schedule": "15/16,5/6,5/8;", "default_steps": 1},
}

PIN_COMMIT = "ed3cb14dd936f92cdc9f9381af7369991509b41f"


def log(msg):
    print(msg, flush=True)


def find_under(root, filename):
    root = Path(root)
    return next((p for p in root.rglob(filename) if p.is_file()), None) if root.exists() else None


def install_deps():
    packages = [
        "einops", "transformers", "sentencepiece", "ftfy", "regex", "imageio",
        "imageio-ffmpeg", "tqdm", "torchvision", "fvcore", "omegaconf", "attrs",
        "pyyaml", "iopath", "termcolor", "pynvml", "pandas", "loguru",
        "safetensors", "cloudpickle", "dill", "matplotlib", "packaging",
        "opencv-python-headless", "scikit-learn", "kaggle",
    ]
    for pkg in packages:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pkg], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"pip install failed for {pkg}: {r.stderr[-400:]}")


def download_datasets():
    """Fetch inputs into /content/inputs.

    The old Kaggle datasets ajjisan/wan21-t5 and ajjisan/wan21-dit-vae no longer
    exist (verified 2026-10-12: 403 / not in `kaggle datasets list --user ajjisan`).
    So the rcm checkpoints come from the private Kaggle dataset ajjisan/causal-rcm-ckpts,
    and the Wan2.1 T5 encoder + VAE + umt5 tokenizer come from the public
    HuggingFace repo Wan-AI/Wan2.1-T2V-1.3B.
    """
    dst = ROOT / "inputs"
    dst.mkdir(parents=True, exist_ok=True)
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not user or not key:
        raise RuntimeError("KAGGLE_USERNAME / KAGGLE_KEY env vars are required")
    kdir = Path.home() / ".kaggle"
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "kaggle.json").write_text(json.dumps({"username": user, "key": key}))
    try:
        os.chmod(kdir / "kaggle.json", 0o600)
    except Exception:
        pass

    ck = dst / "causal-rcm-ckpts"
    if not (ck.exists() and any(ck.iterdir())):
        ck.mkdir(parents=True, exist_ok=True)
        log("downloading ajjisan/causal-rcm-ckpts ...")
        subprocess.run(["kaggle", "datasets", "download", "-d", "ajjisan/causal-rcm-ckpts", "-p", str(ck), "--unzip"], check=True)
    else:
        log("causal-rcm-ckpts already present")

    hf = dst / "wan21"
    hf.mkdir(parents=True, exist_ok=True)
    if not (hf / "models_t5_umt5-xxl-enc-bf16.pth").exists():
        from huggingface_hub import snapshot_download
        log("downloading Wan-AI/Wan2.1-T2V-1.3B (T5 encoder + VAE + tokenizer) from HuggingFace ...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-T2V-1.3B",
            local_dir=str(hf),
            allow_patterns=["models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "config.json", "google/umt5-xxl/*"],
        )
    else:
        log("Wan2.1 T5/VAE already present")
    return dst


def patch_file(path, replacements):
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in source:
            log(f"[warn] patch target not found in {path.name}: {old[:70]}")
        source = source.replace(old, new)
    path.write_text(source, encoding="utf-8")


def write_jsonl(item):
    with RESULTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


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


def segment_bounds(num_frames, latent_frames, group_idx):
    start_lat = 3 * group_idx
    end_lat = min(3 * (group_idx + 1), latent_frames)
    start_px = 0 if start_lat == 0 else 1 + 4 * (start_lat - 1)
    end_px = num_frames if end_lat == latent_frames else 1 + 4 * (end_lat - 1)
    return min(start_px, num_frames), min(end_px, num_frames)


def save_mp4(path, frames):
    import imageio
    writer = imageio.get_writer(str(path), fps=FPS, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


# ---------------- independent quality endpoints (identical to T4 scripts) ----------------

def _read_video_cv2(path):
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def classical_metrics(frames):
    import cv2
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    small = [cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA) for g in grays]
    h, w = small[0].shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    warps, diffs, mags, hists = [], [], [], []
    for a, b, fa, fb in zip(small[:-1], small[1:], frames[:-1], frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        map_x = np.clip(xx + flow[..., 0], 0, w - 1)
        map_y = np.clip(yy + flow[..., 1], 0, h - 1)
        warped = cv2.remap(a, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        warps.append(float(np.mean(np.abs(b.astype(np.float32) - warped.astype(np.float32))) / 255.0))
        diffs.append(float(np.mean(np.abs(b.astype(np.float32) - a.astype(np.float32))) / 255.0))
        mags.append(float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2))))
        h1 = cv2.calcHist([cv2.cvtColor(fa, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
        h2 = cv2.calcHist([cv2.cvtColor(fb, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        hists.append(float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)))
    first_last = float(np.mean(np.abs(frames[-1].astype(np.float32) - frames[0].astype(np.float32))) / 255.0)
    return {
        "n_frames": len(frames),
        "temporal_warp_mean": statistics.fmean(warps) if warps else None,
        "frame_diff_l1_mean": statistics.fmean(diffs) if diffs else None,
        "flow_mag_mean": statistics.fmean(mags) if mags else None,
        "hist_corr_mean": statistics.fmean(hists) if hists else None,
        "first_last_l1": first_last,
    }


def run_classical(video_dir, out_json):
    name_re = re.compile(r"^(?P<case>.+)_(?P<mode>" + "|".join(MODES) + r")$")
    per_video = {}
    for path in sorted(glob.glob(str(Path(video_dir) / "*.mp4"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        m = name_re.match(stem)
        if not m:
            continue
        frames = _read_video_cv2(path)
        if not frames:
            continue
        per_video[stem] = {"case": m.group("case"), "mode": m.group("mode"), **classical_metrics(frames)}
    agg = {}
    for mode in MODES:
        rows = [v for v in per_video.values() if v["mode"] == mode]
        agg[mode] = {"n": len(rows)}
        for key in ["temporal_warp_mean", "frame_diff_l1_mean", "flow_mag_mean", "hist_corr_mean", "first_last_l1"]:
            vals = [r[key] for r in rows if r[key] is not None]
            agg[mode][key] = statistics.fmean(vals) if vals else None
    Path(out_json).write_text(json.dumps({
        "phase": "A100-A1-classical-no-reference-quality",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Independent of the internal c1-1 reference.",
        "n_videos": len(per_video), "by_mode": agg, "per_video": per_video,
    }, indent=2), encoding="utf-8")
    log("classical quality written: " + str(out_json))


def run_clip(video_dir, out_json, prompt_by_case):
    import cv2
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    name_re = re.compile(r"^(?P<case>.+)_(?P<mode>" + "|".join(MODES) + r")$")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def feat(out):
        return out if torch.is_tensor(out) else out.pooler_output

    def read_frames(path, stride=2):
        cap = cv2.VideoCapture(path)
        frames, idx = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()
        return frames

    per_video = {}
    with torch.no_grad():
        for path in sorted(glob.glob(str(Path(video_dir) / "*.mp4"))):
            stem = os.path.splitext(os.path.basename(path))[0]
            m = name_re.match(stem)
            if not m:
                continue
            case, mode = m.group("case"), m.group("mode")
            prompt = prompt_by_case.get(re.sub(r"_h\d+$", "", case))
            frames = read_frames(path)
            if not frames:
                continue
            images = [Image.fromarray(f) for f in frames]
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            img_emb = feat(model.get_image_features(**inputs))
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            emb = img_emb.cpu().numpy()
            cons = float(np.mean(np.sum(emb[:-1] * emb[1:], axis=1))) if len(emb) > 1 else None
            ta_mean, ta_std = None, None
            if prompt:
                txt = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
                txt_emb = feat(model.get_text_features(**txt))
                txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
                sims = (img_emb @ txt_emb.t()).squeeze(-1).cpu().numpy()
                ta_mean, ta_std = float(np.mean(sims)), float(np.std(sims))
            per_video[stem] = {"case": case, "mode": mode, "prompt": prompt,
                               "n_frames_used": len(emb), "subject_consistency": cons,
                               "text_alignment": ta_mean, "text_alignment_std": ta_std}
    agg = {}
    for mode in MODES:
        rows = [v for v in per_video.values() if v["mode"] == mode]
        agg[mode] = {"n": len(rows)}
        for key in ["subject_consistency", "text_alignment", "text_alignment_std"]:
            vals = [r[key] for r in rows if r[key] is not None]
            agg[mode][key] = statistics.fmean(vals) if vals else None
    Path(out_json).write_text(json.dumps({
        "phase": "A100-A1-clip-independent-quality",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": "openai/clip-vit-base-patch32",
        "note": "Independent of the internal c1-1 reference.",
        "n_videos": len(per_video), "by_mode": agg, "per_video": per_video,
    }, indent=2), encoding="utf-8")
    log("CLIP quality written: " + str(out_json))


def main():
    results = {"run_id": RUN_ID, "phase": "A100-A1-mode-cost", "status": "running",
               "start_time": datetime.now().isoformat()}
    try:
        install_deps()
        log(f"torch={torch.__version__} python={sys.version.split()[0]} "
            f"cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'}")

        inputs = download_datasets()

        repo = ROOT / "rcm"
        if repo.exists():
            shutil.rmtree(repo)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVlabs/rcm.git", str(repo)], check=True, capture_output=True)
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        log(f"rcm HEAD = {head} (pinned {PIN_COMMIT})")
        results["rcm_commit"] = head

        paths = {key: find_under(inputs, name) for key, name in CKPT_NAMES.items()}
        paths["vae"] = find_under(inputs, "Wan2.1_VAE.pth")
        paths["t5"] = find_under(inputs, "models_t5_umt5-xxl-enc-bf16.pth")
        tok = find_under(inputs, "tokenizer.json")
        paths["tokenizer"] = tok.parent if tok else None
        results["dataset_paths"] = {k: (str(v) if v else None) for k, v in paths.items()}
        missing = [k for k, v in paths.items() if v is None]
        if missing:
            raise FileNotFoundError(f"missing dataset files: {missing}")

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

        @torch.no_grad()
        def instrumented_sampler(net, init_noise, t_steps, t_steps_per_chunk, step_counts, condition, first_chunk_t, chunk_t, generator):
            B, C, T, H, W = init_noise.shape
            frame_tokens, num_blocks, bp = make_block_pattern(T, H, W, first_chunk_t, chunk_t, net.get_spatial_patch_size())
            ones = torch.ones(B, 1, device=init_noise.device, dtype=torch.float64)
            max_steps = max(step_counts)
            noises = [torch.randn_like(init_noise, dtype=torch.float32, generator=generator) for _ in range(max_steps)]
            kv = net.allocate_kv_caches(max_len=T * frame_tokens)
            blocks, records = [], []
            for i in range(num_blocks):
                fs, fe, bs = block_span(bp, i)
                attn = AttnMaskSpec(mode="block_causal", pattern=bp, q_block_offset=i)
                ts = t_steps_per_chunk[i] if t_steps_per_chunk else t_steps
                x = init_noise[:, :, fs:fe].to(torch.float64) * ts[0]
                for step_idx, (tc, tn) in enumerate(zip(ts[:-1], ts[1:])):
                    tcur = (tc * ones * RECTIFIED_FLOW_T_SCALING).to(**TENSOR_KWARGS)
                    state = CausalInferenceState(mode=KVCacheMode.READONLY, kv_caches=kv, pattern=bp, block_cursor=i, fast_infer=True)
                    current_v = net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS), timesteps_B_T=tcur, **condition, inference_state=state, attn_meta=attn).float()
                    noise = noises[step_idx][:, :, fs:fe]
                    x = (1 - tn) * (x - tc * current_v.to(torch.float64)) + tn * noise
                zero = torch.zeros(B, 1, **TENSOR_KWARGS)
                append = CausalInferenceState(mode=KVCacheMode.APPEND, kv_caches=kv, pattern=bp, block_cursor=i, fast_infer=True)
                net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS), timesteps_B_T=zero, **condition, inference_state=append, attn_meta=attn)
                blocks.append(x)
                records.append({"frame_start": fs, "frame_end": fe})
            del kv
            torch.cuda.empty_cache()
            return torch.cat(blocks, dim=2), records

        tokenizer = Wan2pt1VAEInterface(vae_pth=str(paths["vae"]))
        width, height = VIDEO_RES_SIZE_INFO["480p"]["16:9"]
        latent_c = tokenizer.latent_ch
        t_steps = build_few_step_t_steps(4, 1600, MID_T, "cuda")

        def run_mode(prompt_embedding, mode_key, latent_frames, latent_h, latent_w, seed):
            cfg = MODES[mode_key]
            first, chunk = cfg["first"], cfg["chunk"]
            steps, schedule, default_steps = cfg["steps"], cfg["schedule"], cfg["default_steps"]
            started = time.time()
            condition = {"crossattn_emb": prompt_embedding.to(**TENSOR_KWARGS)}
            load_started = time.time()
            with init_weights_on_device():
                net = instantiate(DIT_CONFIGS["1.3B"]).eval()
            load_dit_weights(net, str(paths[cfg["ckpt"]]))
            net.to(**TENSOR_KWARGS).cpu()
            torch.cuda.empty_cache()
            checkpoint_load_s = time.time() - load_started
            _, blocks, _ = make_block_pattern(latent_frames, latent_h, latent_w, first, chunk, net.get_spatial_patch_size())
            spc = parse_mid_t_schedules(schedule)
            counts = expand_steps_per_chunk(steps, blocks, default_steps)
            per_chunk = build_few_step_schedules_per_chunk(counts, steps, 1600, MID_T, spc, "cuda")
            gen = torch.Generator(device="cuda").manual_seed(seed)
            noise = torch.randn(1, latent_c, latent_frames, latent_h, latent_w, dtype=torch.float32, device="cuda", generator=gen)
            torch.cuda.reset_peak_memory_stats()
            net.cuda()
            torch.cuda.synchronize()
            t0 = time.time()
            latent, records = instrumented_sampler(net, noise, t_steps, per_chunk, counts, condition, first, chunk, gen)
            torch.cuda.synchronize()
            dit_s = time.time() - t0
            peak_vram_gb = torch.cuda.max_memory_allocated() / 2**30
            del net
            torch.cuda.empty_cache()
            t1 = time.time()
            video = tokenizer.decode(latent.float().cuda())
            torch.cuda.synchronize()
            vae_s = time.time() - t1
            peak_vram_gb = max(peak_vram_gb, torch.cuda.max_memory_allocated() / 2**30)
            px = (1 + video.float().cpu().clamp(-1, 1)) / 2
            if px.ndim == 5:
                px = px[0]
            frames = (px.permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8)
            return {"mode": mode_key, "frames": frames, "latent_frames": latent_frames,
                    "checkpoint_load_s": checkpoint_load_s, "dit_s": dit_s, "vae_s": vae_s,
                    "total_s": time.time() - started, "peak_vram_gb": peak_vram_gb,
                    "nfe": int(sum(counts)), "blocks": len(records), "step_counts": [int(c) for c in counts]}

        done = completed_keys()
        for prompt_idx, prompt_spec in enumerate(PROMPTS):
            prompt = prompt_spec["prompt"]
            log(f"encoding prompt {prompt_idx}: {prompt}")
            text_started = time.time()
            emb = encode_prompt(prompt)
            text_encode_s = time.time() - text_started
            order = MODE_ORDER if prompt_idx % 2 == 0 else list(reversed(MODE_ORDER))
            for horizon in HORIZONS:
                key = f"{prompt_spec['id']}_seed{SEED}_h{horizon}"
                if key in done:
                    continue
                try:
                    log(f"running {key}: {prompt}")
                    latent_frames = tokenizer.get_latent_num_frames(horizon)
                    latent_h = height // tokenizer.spatial_compression_factor
                    latent_w = width // tokenizer.spatial_compression_factor
                    mode_runs = {}
                    for mode_key in order:
                        mode_runs[mode_key] = run_mode(emb, mode_key, latent_frames, latent_h, latent_w, SEED)
                        log(f"  {mode_key}: nfe={mode_runs[mode_key]['nfe']} dit={mode_runs[mode_key]['dit_s']:.2f}s "
                            f"vae={mode_runs[mode_key]['vae_s']:.2f}s total={mode_runs[mode_key]['total_s']:.2f}s "
                            f"vram={mode_runs[mode_key]['peak_vram_gb']:.2f}GB")
                    ref_frames = mode_runs["ref_c1_1_s4"]["frames"]
                    shapes = {m: tuple(r["frames"].shape) for m, r in mode_runs.items()}
                    if len(set(shapes.values())) != 1:
                        raise RuntimeError(f"decoded frame shapes not aligned: {shapes}")
                    observations = []
                    for g in range(1, latent_frames // 3):
                        start, end = segment_bounds(horizon, latent_frames, g)
                        ref_seg = ref_frames[start:end].astype(np.float32) / 255.0
                        seg_row = {"group": g, "segment_start_frame": start, "segment_end_frame": end}
                        for mode_key, r in mode_runs.items():
                            mode_seg = r["frames"][start:end].astype(np.float32) / 255.0
                            seg_row[f"{mode_key}_ref_l1"] = float(np.mean(np.abs(mode_seg - ref_seg)))
                        observations.append(seg_row)
                    for mode_key, r in mode_runs.items():
                        save_mp4(VIDEO_DIR / f"{key}_{mode_key}.mp4", r["frames"])
                    item = {
                        "sample_key": key, "prompt_id": prompt_spec["id"], "stratum": prompt_spec["stratum"],
                        "prompt": prompt, "seed": SEED, "horizon": horizon, "status": "complete",
                        "run_order": order, "text_encode_s": text_encode_s,
                        "hardware": "A100-SXM4-80GB",
                        "alignment": {"same_seed": True, "same_initial_noise": True, "decoded_shape": list(ref_frames.shape), "reference_mode": "c1-1_step4"},
                        "modes": {mode_key: {k: r[k] for k in ["checkpoint_load_s", "dit_s", "vae_s", "total_s", "peak_vram_gb", "nfe", "blocks", "step_counts"]} for mode_key, r in mode_runs.items()},
                        "videos": {mode_key: f"videos/{key}_{mode_key}.mp4" for mode_key in mode_runs},
                        "observations": observations,
                        "metric": "decoded pixel L1 to matched internal c1-1 step-4 reference; reference-defined diagnostic, not ground truth",
                    }
                    write_jsonl(item)
                    del mode_runs
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as exc:
                    write_jsonl({"sample_key": key, "prompt_id": prompt_spec["id"], "prompt": prompt, "seed": SEED,
                                 "horizon": horizon, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                 "traceback": traceback.format_exc()})
                    log(traceback.format_exc())

        # independent quality endpoints over the saved videos
        prompt_by_case = {f"{p['id']}_seed{SEED}": p["prompt"] for p in PROMPTS}
        try:
            run_classical(VIDEO_DIR, CLASSICAL_JSON)
        except Exception as exc:
            log("classical quality failed: " + repr(exc))
        try:
            run_clip(VIDEO_DIR, CLIP_JSON, prompt_by_case)
        except Exception as exc:
            log("CLIP quality failed: " + repr(exc))

        complete = []
        if RESULTS_JSONL.exists():
            for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
                try:
                    it = json.loads(line)
                    if it.get("status") == "complete":
                        complete.append(it)
                except Exception:
                    pass
        results["n_complete"] = len(complete)
        results["n_expected"] = len(PROMPTS) * len(HORIZONS)
        results["verdict"] = "PASS" if len(complete) == len(PROMPTS) * len(HORIZONS) else ("PARTIAL" if complete else "FAIL")
        results["status"] = "complete" if results["verdict"] == "PASS" else "partial"
    except Exception as exc:
        results["status"] = "failed"
        results["error"] = f"{type(exc).__name__}: {exc}"
        results["traceback"] = traceback.format_exc()
        log(traceback.format_exc())
    finally:
        results["end_time"] = datetime.now().isoformat()
        SUMMARY_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        log("summary: " + str(SUMMARY_JSON))
        log("A100_SWEEP_DONE verdict=" + str(results.get("verdict")))


if __name__ == "__main__":
    main()
