"""Audit the completed online Phase 4 run against paired Phase 3 baselines."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ONLINE = ROOT / "results/phase4_online_v2_download/results/phase4_adaptive_online"
P3 = ROOT / "results/phase3_calibration_rerun/results/phase3_calibration"
OUT = ROOT / "results/phase4_online_audit"
OUT.mkdir(parents=True, exist_ok=True)
summary = json.loads((ONLINE / "phase4_summary.json").read_text(encoding="utf-8"))
adaptive = [json.loads(x) for x in (ONLINE / "results.jsonl").read_text(encoding="utf-8").splitlines()]
fixed = [json.loads(x) for x in (P3 / "scores.jsonl").read_text(encoding="utf-8").splitlines() if json.loads(x)["sample_split"] == "held_out"]
assert summary["status"] == "complete" and summary["n_complete"] == 4 and summary["n_failed"] == 0
assert len(adaptive) == 4 and all(x["status"] == "complete" for x in adaptive)

fixed_by_prompt = {}
for x in fixed:
    fixed_by_prompt.setdefault(int(x["prompt_idx"]), []).append(x)
adaptive_by_prompt = {int(x["prompt_idx"]): x for x in adaptive}
prompt_rows = []
for idx in sorted(adaptive_by_prompt):
    a = adaptive_by_prompt[idx]
    fobs = sorted(fixed_by_prompt[idx], key=lambda x: int(x["group"]))
    aseg = sorted(a["segments"], key=lambda x: int(x["group"]))
    assert [int(x["group"]) for x in aseg] == [int(x["group"]) for x in fobs]
    fine_frac = sum(d["mode"] == "fine" for d in a["decisions"]) / len(a["decisions"])
    prompt_rows.append({
        "prompt_idx": idx,
        "prompt": a["prompt"],
        "adaptive_mean_adjacent_pixel_l1": float(np.mean([x["adjacent_pixel_l1"] for x in aseg])),
        "fixed_fine_mean_adjacent_pixel_l1": float(np.mean([x["fine_l1"] for x in fobs])),
        "fixed_coarse_mean_adjacent_pixel_l1": float(np.mean([x["coarse_l1"] for x in fobs])),
        "oracle_mean_adjacent_pixel_l1": float(np.mean([min(x["fine_l1"], x["coarse_l1"]) for x in fobs])),
        "fine_fraction": fine_frac,
        "switch_count": int(a["controller"]["switch_count"]),
        "adaptive_modes": [x["mode"] for x in a["decisions"]],
        "segments": [{
            "group": int(s["group"]),
            "complexity": float(d["complexity"]),
            "mode": s["mode"],
            "adaptive_l1": float(s["adjacent_pixel_l1"]),
            "fixed_fine_l1": float(f["fine_l1"]),
            "fixed_coarse_l1": float(f["coarse_l1"]),
            "oracle_l1": float(min(f["fine_l1"], f["coarse_l1"])),
        } for s, d, f in zip(aseg, a["decisions"], fobs)],
        "timing": a["timing"],
    })

def prompt_mean(field):
    return float(np.mean([r[field] for r in prompt_rows]))

def paired_bootstrap_diff(field_a, field_b, reps=20000, seed=20261006):
    diffs = np.asarray([r[field_a] - r[field_b] for r in prompt_rows], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(diffs, size=(reps, len(diffs)), replace=True), axis=1)
    return {"mean_difference_adaptive_minus_baseline": float(diffs.mean()),
            "prompt_paired_bootstrap_ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "n_prompts": len(diffs), "bootstrap_reps": reps,
            "limitation": "prompt-level resampling with four prompts; exploratory only"}

fields = {
    "adaptive": "adaptive_mean_adjacent_pixel_l1",
    "fixed_fine": "fixed_fine_mean_adjacent_pixel_l1",
    "fixed_coarse": "fixed_coarse_mean_adjacent_pixel_l1",
    "oracle": "oracle_mean_adjacent_pixel_l1",
}
metrics = {k: prompt_mean(v) for k, v in fields.items()}
timing_keys = ["checkpoint_load_s", "dit_s", "vae_s", "signal_s", "total_s", "nfe", "peak_vram_gb"]
timing = {k: float(np.mean([r["timing"][k] for r in prompt_rows])) for k in timing_keys}
online_vae_caveat = (
    "The recorded VAE component omits the initial coarse-prefix decode used to compute the first online signal; "
    "that decode is included in total_s. Therefore vae_s is a lower bound, while total_s includes observed setup, "
    "all decodes, controller/signal work, model loads, and generation."
)
out = {
    "phase": "phase4-adaptive-controller",
    "status": "complete",
    "verdict": "PASS_EXPLORATORY_ONLINE",
    "online_artifact": str(ONLINE),
    "protocol": summary["protocol"],
    "n_prompts": len(prompt_rows),
    "n_segments": sum(len(x["segments"]) for x in prompt_rows),
    "prompt_results": prompt_rows,
    "aggregate_proxy": metrics,
    "adaptive_minus_baseline_paired_bootstrap": {
        "fixed_fine": paired_bootstrap_diff(fields["adaptive"], fields["fixed_fine"]),
        "fixed_coarse": paired_bootstrap_diff(fields["adaptive"], fields["fixed_coarse"]),
        "offline_oracle": paired_bootstrap_diff(fields["adaptive"], fields["oracle"]),
    },
    "aggregate_online_usage": {
        "fine_fraction": float(np.mean([r["fine_fraction"] for r in prompt_rows])),
        "coarse_fraction": float(1.0 - np.mean([r["fine_fraction"] for r in prompt_rows])),
        "mean_switch_count_per_prompt": float(np.mean([r["switch_count"] for r in prompt_rows])),
        "timing_means": timing,
        "vae_timing_caveat": online_vae_caveat,
    },
    "comparison_caveat": "Fixed-mode and adaptive runs are prompt/seed paired but follow different trajectories; oracle uses both fixed outcomes and is offline only.",
    "limitations": [
        "four held-out prompts and one seed; bootstrap intervals are exploratory",
        "adjacent-pixel temporal L1 depends on motion/content and is not perceptual quality",
        "oracle is offline and uses the paired fixed-mode outcomes",
        "online run uses the calibration-fitted complexity selector, while the earlier residual analysis is an offline secondary analysis",
        "the per-component VAE timer omits the initial prefix decode; full total_s includes that decode",
    ],
    "proposal_coverage": {
        "RQ2": "real online adaptive generation completed; exploratory comparison against fixed modes and offline oracle",
        "RQ3": "not evaluated in this phase; recovery is Phase 5",
        "deviations": ["text-to-video, not action/camera-conditioned", "lambda_action=0", "no EMA", "no reference-video quality metrics"]
    }
}
(OUT / "phase4_online_audit.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
(OUT / "phase4_online_prompt_table.csv").write_text(
    "prompt_idx,prompt,adaptive_l1,fixed_fine_l1,fixed_coarse_l1,oracle_l1,fine_fraction,switch_count\n" +
    "".join(f'{r["prompt_idx"]},"{r["prompt"].replace(chr(34), chr(34)*2)}",{r["adaptive_mean_adjacent_pixel_l1"]},{r["fixed_fine_mean_adjacent_pixel_l1"]},{r["fixed_coarse_mean_adjacent_pixel_l1"]},{r["oracle_mean_adjacent_pixel_l1"]},{r["fine_fraction"]},{r["switch_count"]}\n' for r in prompt_rows),
    encoding="utf-8")
print(json.dumps({"aggregate_proxy": metrics, "usage": out["aggregate_online_usage"], "bootstrap": out["adaptive_minus_baseline_paired_bootstrap"]}, indent=2))
