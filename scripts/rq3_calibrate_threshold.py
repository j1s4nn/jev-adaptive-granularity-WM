"""Calibrate RQ3 threshold: compare candidate signals for switch-producing behavior.

Computes warp, motion, and complexity per decision from the F4-lite fixed_fine
development videos and reports, for each signal, how often a pooled-median
threshold is crossed within a rollout (i.e. switch-producing potential).
Writes config/rq3_threshold.json with the chosen signal and threshold.
"""

import json
import os
import statistics

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(ROOT, "results", "f4lite", "videos")
OUT = os.path.join(ROOT, "config", "rq3_threshold.json")
HORIZON = 57


def segment_bounds(group):
    start = 1 + 4 * (3 * group - 1)
    end = HORIZON if group == 4 else 1 + 4 * (3 * (group + 1) - 1)
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


def signals_at(frames, end):
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


def crossings(trace, tau):
    modes = [1 if v <= tau else 0 for v in trace]
    return sum(modes[i] != modes[i - 1] for i in range(1, len(modes))), modes


def main():
    traces = {k: {} for k in ["warp", "motion", "complexity"]}
    for idx in range(8):
        frames = read_video(os.path.join(VIDEO_DIR, f"p{idx}_fixed_fine.mp4"))
        for g in range(1, 5):
            s = signals_at(frames, segment_bounds(g)[0])
            for k, v in s.items():
                traces[k].setdefault(f"p{idx}", []).append(v)

    summary = {}
    for key, per_prompt in traces.items():
        pooled = [v for t in per_prompt.values() for v in t]
        tau = statistics.median(pooled)
        total_sw, per_prompt_sw = 0, {}
        for p, t in per_prompt.items():
            sw, _ = crossings(t, tau)
            per_prompt_sw[p] = sw
            total_sw += sw
        summary[key] = {
            "tau_median": tau, "mean": statistics.fmean(pooled), "std": statistics.pstdev(pooled),
            "min": min(pooled), "max": max(pooled),
            "within_prompt_range_mean": statistics.fmean([max(t) - min(t) for t in per_prompt.values()]),
            "total_switches_at_median": total_sw, "per_prompt_switches": per_prompt_sw,
            "per_prompt_trace": per_prompt,
        }
        print(f"{key:<11} median={tau:.3f} std={statistics.pstdev(pooled):.3f} within_range={summary[key]['within_prompt_range_mean']:.3f} total_switches={total_sw}")

    # choose the signal with the most switches at the pooled median
    best = max(summary, key=lambda k: summary[k]["total_switches_at_median"])
    report = {
        "source": "results/f4lite/videos/p*_fixed_fine.mp4 (development)",
        "candidates": summary,
        "chosen_signal": best,
        "tau_median": summary[best]["tau_median"],
        "sigma_u": summary[best]["std"],
        "note": "tau at pooled median to force switch-producing behavior; low-warp/low-signal maps to fine (direction -1).",
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nchosen signal:", best, "tau:", summary[best]["tau_median"])
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
