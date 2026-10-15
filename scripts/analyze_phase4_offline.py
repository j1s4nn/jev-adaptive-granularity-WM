"""Phase 4 fallback: held-out offline threshold-controller reanalysis.

The Phase 3 paired trajectories are real c1-1/c3-3 generations. This script
applies a residual threshold policy fitted only on calibration observations to
the held-out observations, then compares selected proxy/cost with fixed modes
and an offline oracle. It does not claim online frame generation.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/phase3_calibration_rerun/results/phase3_calibration"
OUT = ROOT / "results/phase4_adaptive_offline"
OUT.mkdir(parents=True, exist_ok=True)
scores = [json.loads(x) for x in (SRC / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
summary3 = json.loads((SRC / "phase3_summary_audited.json").read_text(encoding="utf-8"))
held_prompts = summary3["protocol"]["held_out_prompts"]
cal = [x for x in scores if x["sample_split"] == "calibration"]
test = [x for x in scores if x["sample_split"] == "held_out"]
res = np.asarray([x["residual"] for x in cal], dtype=float)
# High threshold is the calibration-only Youden operating point reported by
# Phase 3. The lower threshold is the calibration first quartile, creating a
# documented hysteresis band without touching held-out outcomes.
tau_low = float(np.quantile(res, 0.25))
tau_high = float(summary3["metrics"]["residual"]["threshold_calibration_youden"])

def decisions(obs):
    mode = "coarse"
    rows = []
    for x in obs:
        u = float(x["residual"])
        if u >= tau_high:
            mode = "fine"
        elif u <= tau_low:
            mode = "coarse"
        rows.append({"group": int(x["group"]), "residual": u, "mode": mode,
                     "fine_l1": float(x["fine_l1"]), "coarse_l1": float(x["coarse_l1"]),
                     "selected_l1": float(x["fine_l1"] if mode == "fine" else x["coarse_l1"]),
                     "oracle_l1": float(min(x["fine_l1"], x["coarse_l1"]))})
    return rows

by_prompt = []
for idx in sorted({x["prompt_idx"] for x in test}):
    obs = sorted([x for x in test if x["prompt_idx"] == idx], key=lambda x: x["group"])
    rows = decisions(obs)
    n_fine = sum(r["mode"] == "fine" for r in rows)
    n_coarse = len(rows) - n_fine
    switches = sum(rows[i]["mode"] != rows[i-1]["mode"] for i in range(1, len(rows)))
    # Phase 3 end-to-end totals include three scored segments. Use equal
    # per-segment shares only as an operational cost proxy.
    fixed = next(x for x in summary3.get("timing", {}) if False) if False else None
    by_prompt.append({
        "prompt_idx": int(idx),
        "prompt": held_prompts[int(idx)],
        "segments": rows,
        "fine_fraction": n_fine / len(rows),
        "switch_count": switches,
        "adaptive_mean_l1": float(np.mean([r["selected_l1"] for r in rows])),
        "fixed_fine_mean_l1": float(np.mean([r["fine_l1"] for r in rows])),
        "fixed_coarse_mean_l1": float(np.mean([r["coarse_l1"] for r in rows])),
        "oracle_mean_l1": float(np.mean([r["oracle_l1"] for r in rows])),
    })

def mean(key):
    return float(np.mean([x[key] for x in by_prompt]))

out = {
    "phase": "phase4-adaptive-controller",
    "status": "complete",
    "verdict": "PARTIAL_OFFLINE_SELECTION",
    "source_results": str(SRC / "results.jsonl"),
    "source_scores": str(SRC / "scores.jsonl"),
    "protocol": {
        "split": "held_out only for evaluation; calibration only for threshold fitting",
        "signal": "Phase 3 residual denoising discrepancy",
        "tau_low": tau_low, "tau_high": tau_high,
        "initial_mode": "coarse", "recovery_length": 0, "lambda_action": 0.0,
        "quality_proxy": "adjacent decoded-pixel temporal L1",
        "cost_proxy": "mode fraction; Phase 3 timing means are reported separately",
        "online_generation": False,
        "oracle": "per-segment min(fine_l1, coarse_l1), offline upper bound",
    },
    "limitations": [
        "selection is offline over paired fixed trajectories, not a new online rollout",
        "pixel temporal L1 is motion/content dependent and not perceptual quality",
        "four held-out prompts, one seed; three segments per prompt are correlated",
        "thresholds are residual quartiles, not optimized for held-out outcomes",
    ],
    "n_prompts": len(by_prompt),
    "prompt_results": by_prompt,
    "aggregate": {
        "adaptive_mean_l1": mean("adaptive_mean_l1"),
        "fixed_fine_mean_l1": mean("fixed_fine_mean_l1"),
        "fixed_coarse_mean_l1": mean("fixed_coarse_mean_l1"),
        "oracle_mean_l1": mean("oracle_mean_l1"),
        "adaptive_fine_fraction": mean("fine_fraction"),
        "adaptive_switches_mean": mean("switch_count"),
        "adaptive_selected_l1_minus_oracle": mean("adaptive_mean_l1") - mean("oracle_mean_l1"),
    },
    "phase3_timing_means": summary3.get("timing", {}),
}
(OUT / "phase4_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
(OUT / "phase4_decisions.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in by_prompt), encoding="utf-8")
print(json.dumps(out["aggregate"], indent=2))
