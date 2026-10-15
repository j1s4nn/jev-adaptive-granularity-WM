"""E-JEV-1 step 1: build the per-segment states for the Jev signal test.

For every paired fine/coarse segment in the existing data (F4-lite @57f and
RQ4 @81f) this builds:
  - a text state (prompt + BLIP caption of the pre-decision context frame + segment index)
  - the label coarse_unsafe = (regret_coarse > regret_fine)
  - the hand-crafted signals (warp, motion, complexity) computed at the same point

Reads  results/f4lite/results.jsonl, results/rq4_monotonicity/results.jsonl,
       results/f4lite/videos/*.mp4, results/rq4_monotonicity/videos/*.mp4
Writes results/jev_signal/states.jsonl
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F4 = os.path.join(ROOT, "results", "f4lite")
RQ4 = os.path.join(ROOT, "results", "rq4_monotonicity")
OUT_DIR = os.path.join(ROOT, "results", "jev_signal")
OUT = os.path.join(OUT_DIR, "states.jsonl")

PROMPTS = {
    0: "a boat", 1: "a book", 2: "baseball stadium", 3: "basement",
    4: "a bird flying over a snowy forest", 5: "a bird soaring gracefully in the sky",
    6: "A person is bungee jumping", 7: "A person is canoeing or kayaking",
}
STRATA = {0: "object_class", 1: "object_class", 2: "scene", 3: "scene",
          4: "motion", 5: "motion", 6: "human_action", 7: "human_action"}


def seg_bounds(group, horizon, n_segments):
    start = 1 + 4 * (3 * group - 1)
    end = horizon if group == n_segments else 1 + 4 * (3 * (group + 1) - 1)
    return start, end


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(frames, dtype=np.uint8)


def handcrafted(frames, end):
    g1 = frames[end - 2].mean(axis=2).astype(np.float32)
    g2 = frames[end - 1].mean(axis=2).astype(np.float32)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    motion = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
    prevprev = frames[end - 3].mean(axis=2).astype(np.float32)
    h, w = g1.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx = np.clip(xx + flow[..., 0], 0, w - 1)
    my = np.clip(yy + flow[..., 1], 0, h - 1)
    warped = cv2.remap(prevprev, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warp = float(np.mean(np.abs(g1 - warped)))
    gray = cv2.cvtColor(frames[end - 1], cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    complexity = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
    return {"warp": warp, "motion": motion, "complexity": complexity}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").eval()

    cache = {}

    def caption(frame):
        key = int(frame[::8, ::8].sum())
        if key in cache:
            return cache[key]
        image = Image.fromarray(frame)
        with torch.no_grad():
            out = model.generate(**proc(image, return_tensors="pt"), max_new_tokens=24)
        text = proc.decode(out[0], skip_special_tokens=True)
        cache[key] = text
        return text

    rows = []

    # 57-frame F4-lite paired segments
    ff, fc = {}, {}
    with open(os.path.join(F4, "results.jsonl"), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") != "complete":
                continue
            if r.get("policy") == "fixed_fine":
                ff[r["prompt_idx"]] = r["segments"]
            elif r.get("policy") == "fixed_coarse":
                fc[r["prompt_idx"]] = r["segments"]
    for idx in range(8):
        if idx not in ff or idx not in fc:
            continue
        frames = read_video(os.path.join(F4, "videos", f"p{idx}_fixed_fine.mp4"))
        for g in range(1, 5):
            end = seg_bounds(g, 57, 4)[0]
            rf = ff[idx][g - 1]["reference_l1"]
            rc = fc[idx][g - 1]["reference_l1"]
            rows.append({
                "segment_id": f"p{idx}_57f_g{g}", "prompt_idx": idx, "stratum": STRATA[idx],
                "prompt": PROMPTS[idx], "horizon": 57, "group": g, "segment_index": g,
                "context_frame_index": end - 1, "caption": caption(frames[end - 1]),
                "regret_fine": rf, "regret_coarse": rc, "fine_gap": rf - rc,
                "coarse_unsafe": int(rc > rf), **handcrafted(frames, end),
            })

    # 81-frame RQ4 paired segments
    paired = {}
    with open(os.path.join(RQ4, "results.jsonl"), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") == "complete" and r.get("kind") == "paired":
                paired[r["prompt_idx"]] = r["paired"]
    for idx in range(8):
        if idx not in paired:
            continue
        frames = read_video(os.path.join(RQ4, "videos", f"p{idx}_fixed_fine.mp4"))
        for p in paired[idx]:
            g = p["group"]
            end = seg_bounds(g, 81, 6)[0]
            rf, rc = p["regret_fine"], p["regret_coarse"]
            rows.append({
                "segment_id": f"p{idx}_81f_g{g}", "prompt_idx": idx, "stratum": STRATA[idx],
                "prompt": PROMPTS[idx], "horizon": 81, "group": g, "segment_index": g,
                "context_frame_index": end - 1, "caption": caption(frames[end - 1]),
                "regret_fine": rf, "regret_coarse": rc, "fine_gap": rf - rc,
                "coarse_unsafe": int(rc > rf), **handcrafted(frames, end),
            })

    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    pos = sum(r["coarse_unsafe"] for r in rows)
    print(f"wrote {len(rows)} states -> {os.path.relpath(OUT, ROOT)}")
    print(f"coarse_unsafe positives: {pos}/{len(rows)} ({pos/len(rows):.2f})")
    print("sample captions:")
    for r in rows[:5]:
        print(f"  [{r['segment_id']}] prompt='{r['prompt']}' cap='{r['caption']}' unsafe={r['coarse_unsafe']} gap={r['fine_gap']:+.4f}")


if __name__ == "__main__":
    main()
