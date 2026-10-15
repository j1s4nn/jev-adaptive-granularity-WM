"""E-JEV-1 step 3: AUROC of the Jev signal vs hand-crafted signals.

Label: coarse_unsafe = (regret_coarse > regret_fine).
Signals: Jev drift (noul), Jev difficulty (score), Jev motion (choice->ordinal),
         and the hand-crafted warp / motion / complexity on the same segments.
Reports prompt-grouped bootstrap 95% CIs (resampling the 8 prompts).

Reads  results/jev_signal/jev_answers.jsonl
Writes results/jev_signal/jev_audit.json
"""

import json
import os
import random
import statistics
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "results", "jev_signal", "jev_answers.jsonl")
OUT = os.path.join(ROOT, "results", "jev_signal", "jev_audit.json")

MOTION_SCORE = {"static": 0.0, "moderate": 1.0, "fast": 2.0, "scene_change": 3.0}
BOOTSTRAP = 2000
SEED = 20261026


def auroc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def extract(rows):
    out = {"jev_drift": [], "jev_difficulty": [], "jev_motion": [], "warp": [], "motion": [], "complexity": [], "label": [], "prompt": []}
    for r in rows:
        if not r.get("answer"):
            continue
        a = r["answer"]["answers"]
        out["jev_drift"].append(a["coarse_drift"]["noul"])
        out["jev_difficulty"].append(a["difficulty"]["score"])
        out["jev_motion"].append(MOTION_SCORE.get(a["motion"]["choice"], 1.0))
        out["warp"].append(r["warp"])
        out["motion"].append(r["motion"])
        out["complexity"].append(r["complexity"])
        out["label"].append(r["coarse_unsafe"])
        out["prompt"].append(r["prompt_idx"])
    return out


def grouped_bootstrap(scores, labels, groups, rng):
    by_prompt = {}
    for s, y, g in zip(scores, labels, groups):
        by_prompt.setdefault(g, []).append((s, y))
    prompts = list(by_prompt)
    vals = []
    for _ in range(BOOTSTRAP):
        ss, yy = [], []
        for p in rng.choices(prompts, k=len(prompts)):
            for s, y in by_prompt[p]:
                ss.append(s)
                yy.append(y)
        v = auroc(ss, yy)
        if v is not None:
            vals.append(v)
    vals.sort()
    if not vals:
        return None, None
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def main():
    rows = []
    with open(IN, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    data = extract(rows)
    rng = random.Random(SEED)
    labels = data["label"]
    report = {"phase": "E-JEV-1-auroc", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "n_segments": len(labels), "positives": sum(labels), "signals": {},
              "signal_cost": {"jev_latency_mean_s": statistics.fmean([r["latency_s"] for r in rows if r.get("answer")]),
                              "note": "Jev ~0.7 s per decision vs the warp signal which required a VAE decode (~156 s @57f) per decision"}}

    for name in ["jev_drift", "jev_difficulty", "jev_motion", "warp", "motion", "complexity"]:
        a = auroc(data[name], labels)
        lo, hi = grouped_bootstrap(data[name], labels, data["prompt"], rng)
        report["signals"][name] = {"auroc": a, "ci_low": lo, "ci_high": hi}

    best_jev = max(["jev_drift", "jev_difficulty", "jev_motion"], key=lambda k: report["signals"][k]["auroc"] or 0)
    best_hc = max(["warp", "motion", "complexity"], key=lambda k: report["signals"][k]["auroc"] or 0)
    report["best_jev_signal"] = best_jev
    report["best_handcrafted_signal"] = best_hc
    report["auroc_gain_jev_over_handcrafted"] = (report["signals"][best_jev]["auroc"] or 0) - (report["signals"][best_hc]["auroc"] or 0)
    report["reference_phase5b_residual_auroc"] = 0.491

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("E-JEV-1 audit:", os.path.relpath(OUT, ROOT))
    print(f"n={len(labels)} positives={sum(labels)} ({sum(labels)/len(labels):.2f})")
    print(f"{'signal':<16}{'AUROC':>8}{'95% CI':>18}")
    for name in ["jev_drift", "jev_difficulty", "jev_motion", "warp", "motion", "complexity"]:
        s = report["signals"][name]
        ci = f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]" if s["ci_low"] is not None else "n/a"
        print(f"{name:<16}{s['auroc']:>8.3f}{ci:>18}")
    print(f"\nbest Jev: {best_jev} | best hand-crafted: {best_hc} | gain {report['auroc_gain_jev_over_handcrafted']:+.3f}")
    print(f"Phase 5B residual reference AUROC: 0.491")


if __name__ == "__main__":
    main()
