"""F1.4a classical no-reference quality audit.

Computes a lightweight, documented no-reference suite on the saved F1.3 videos.
These metrics are INDEPENDENT of the internal c1-1 step-4 reference (which may
favor the fine family). They are a first independent read; the paper-grade
endpoint (VBench dimensions / CLIP subject consistency) is a later step.

Metrics per video:
  temporal_warp_mean : mean |frame_t - warp(frame_{t-1})| under Farneback flow
                       (temporal smoothness proxy; lower = smoother)
  frame_diff_l1_mean : mean |frame_t - frame_{t-1}| (temporal variation)
  flow_mag_mean      : mean optical-flow magnitude (dynamic degree)
  hist_corr_mean     : mean HSV-histogram correlation between consecutive frames
                       (scene/color consistency; higher = more consistent)
  first_last_l1      : |last frame - first frame| (long-horizon drift proxy)

Reads:  results/f1_mode_cost/videos/*.mp4
Writes: results/f1_mode_cost/f1_quality_classical.json
"""

import glob
import json
import os
import re
import statistics
from datetime import datetime, timezone

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(ROOT, "results", "f1_mode_cost", "videos")
OUT_JSON = os.path.join(ROOT, "results", "f1_mode_cost", "f1_quality_classical.json")

MODES = ["ref_c1_1_s4", "fine_c1_1_s1", "fine_c1_1_s2", "coarse_c3_3_s4", "coarse_c3_3_s2", "coarse_c3_3_s1"]
NAME_RE = re.compile(r"^(?P<case>.+)_(?P<mode>ref_c1_1_s4|fine_c1_1_s1|fine_c1_1_s2|coarse_c3_3_s4|coarse_c3_3_s2|coarse_c3_3_s1)$")


def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def video_metrics(frames):
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


def main():
    files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    per_video = {}
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        m = NAME_RE.match(stem)
        if not m:
            print("skip unmatched:", stem)
            continue
        frames = read_video(path)
        if not frames:
            print("no frames:", stem)
            continue
        metrics = video_metrics(frames)
        per_video[stem] = {"case": m.group("case"), "mode": m.group("mode"), **metrics}

    agg = {}
    for mode in MODES:
        rows = [v for v in per_video.values() if v["mode"] == mode]
        agg[mode] = {}
        for key in ["temporal_warp_mean", "frame_diff_l1_mean", "flow_mag_mean", "hist_corr_mean", "first_last_l1"]:
            vals = [r[key] for r in rows if r[key] is not None]
            agg[mode][key] = statistics.fmean(vals) if vals else None
        agg[mode]["n"] = len(rows)

    report = {
        "phase": "F1.4a-classical-no-reference-quality",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Independent of the internal c1-1 reference. Lightweight proxies, not VBench.",
        "n_videos": len(per_video),
        "by_mode": agg,
        "per_video": per_video,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("F1.4a classical quality written:", os.path.relpath(OUT_JSON, ROOT))
    print(f"{'mode':<18}{'warp':>8}{'framediff':>11}{'flowmag':>9}{'histcorr':>10}{'drift':>8}")
    for mode in MODES:
        a = agg[mode]
        print(
            f"{mode:<18}{a['temporal_warp_mean']:>8.4f}{a['frame_diff_l1_mean']:>11.4f}"
            f"{a['flow_mag_mean']:>9.3f}{a['hist_corr_mean']:>10.4f}{a['first_last_l1']:>8.4f}"
        )


if __name__ == "__main__":
    main()
