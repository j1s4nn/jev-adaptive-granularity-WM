"""F4-lite independent quality: CLIP + classical no-reference metrics on the policy videos.

Reads  results/f4lite/videos/p{idx}_{policy}.mp4
Writes results/f4lite/f4lite_quality.json
"""

import glob
import json
import os
import re
import statistics
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(ROOT, "results", "f4lite", "videos")
OUT_JSON = os.path.join(ROOT, "results", "f4lite", "f4lite_quality.json")

POLICIES = ["reference", "fixed_coarse", "fixed_fine", "random50", "alternating50", "adaptive_warp"]
NAME_RE = re.compile(r"^p(?P<idx>\d+)_(?P<policy>reference|fixed_coarse|fixed_fine|random50|alternating50|adaptive_warp)$")
PROMPTS = {
    0: "a boat", 1: "a book", 2: "baseball stadium", 3: "basement",
    4: "a bird flying over a snowy forest", 5: "a bird soaring gracefully in the sky",
    6: "A person is bungee jumping", 7: "A person is canoeing or kayaking",
}


def read_frames(path, stride=2):
    cap = cv2.VideoCapture(str(path))
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


def classical(frames):
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    small = [cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA) for g in grays]
    h, w = small[0].shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    warps, mags = [], []
    for a, b in zip(small[:-1], small[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mx = np.clip(xx + flow[..., 0], 0, w - 1)
        my = np.clip(yy + flow[..., 1], 0, h - 1)
        warped = cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        warps.append(float(np.mean(np.abs(b.astype(np.float32) - warped.astype(np.float32))) / 255.0))
        mags.append(float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2))))
    first_last = float(np.mean(np.abs(frames[-1].astype(np.float32) - frames[0].astype(np.float32))) / 255.0)
    return {"temporal_warp_mean": statistics.fmean(warps) if warps else None,
            "flow_mag_mean": statistics.fmean(mags) if mags else None, "first_last_l1": first_last}


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.fmean(v) if v else None


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    per_video = {}
    with torch.no_grad():
        for path in files:
            stem = os.path.splitext(os.path.basename(path))[0]
            m = NAME_RE.match(stem)
            if not m:
                continue
            idx = int(m.group("idx"))
            policy = m.group("policy")
            prompt = PROMPTS[idx]
            frames = read_frames(path)
            if not frames:
                continue
            inputs = processor(images=[Image.fromarray(f) for f in frames], return_tensors="pt", padding=True).to(device)
            emb = model.get_image_features(**inputs)
            emb = (emb.pooler_output if hasattr(emb, "pooler_output") else emb)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            e = emb.cpu().numpy()
            cons = float(np.mean(np.sum(e[:-1] * e[1:], axis=1))) if len(e) > 1 else None
            txt = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
            t = model.get_text_features(**txt)
            t = (t.pooler_output if hasattr(t, "pooler_output") else t)
            t = t / t.norm(dim=-1, keepdim=True)
            ta = float(np.mean((emb @ t.t()).squeeze(-1).cpu().numpy()))
            per_video[stem] = {"idx": idx, "policy": policy, "prompt": prompt,
                               "subject_consistency": cons, "text_alignment": ta, **classical(frames)}
            print(f"{stem}: cons={cons:.4f} ta={ta:.4f}")

    agg = {}
    for policy in POLICIES:
        rows = [v for v in per_video.values() if v["policy"] == policy]
        agg[policy] = {"n": len(rows)}
        for key in ["subject_consistency", "text_alignment", "temporal_warp_mean", "flow_mag_mean", "first_last_l1"]:
            agg[policy][key] = mean([r[key] for r in rows])

    report = {"phase": "F4-lite-quality", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "model": "openai/clip-vit-base-patch32", "n_videos": len(per_video), "by_policy": agg, "per_video": per_video}
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\nF4-lite quality:", os.path.relpath(OUT_JSON, ROOT))
    print(f"{'policy':<16}{'subj_cons':>11}{'text_align':>12}{'warp':>9}{'flow_mag':>10}{'drift':>9}")
    for p in POLICIES:
        a = agg[p]
        print(f"{p:<16}{a['subject_consistency']:>11.4f}{a['text_alignment']:>12.4f}{a['temporal_warp_mean']:>9.4f}{a['flow_mag_mean']:>10.3f}{a['first_last_l1']:>9.4f}")


if __name__ == "__main__":
    main()
