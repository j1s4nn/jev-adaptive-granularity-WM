"""Audit the online residual-threshold controller against paired Phase 3 runs.

This script requires the filtered Kaggle output (phase4_summary.json and
results.jsonl) from the residual-controller kernel. It reports exploratory
prompt-paired proxy, latency, and NFE comparisons; it does not claim a
perceptual-quality result or a fully compute-matched oracle.
"""
import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT / "results/phase3_calibration_rerun/results/phase3_calibration"
COMPLEXITY = ROOT / "results/phase4_online_v2_download/results/phase4_adaptive_online"
OUT = ROOT / "results/phase4_residual_audit"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def locate(root, name):
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    if len(matches) > 1:
        exact = [p for p in matches if p.parent.name == "phase4_adaptive_online"]
        if len(exact) == 1:
            return exact[0]
        raise RuntimeError(f"Multiple {name} files found under {root}: {matches}")
    return matches[0]


def prompt_bootstrap(diffs, reps=20000, seed=20261006):
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(diffs, size=(reps, len(diffs)), replace=True), axis=1)
    return {
        "mean_difference_adaptive_minus_baseline": float(np.mean(diffs)),
        "prompt_paired_bootstrap_ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "n_prompts": int(len(diffs)),
        "bootstrap_reps": reps,
        "interpretation": "exploratory only; resampling unit is the prompt",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--online-dir", type=Path,
        default=ROOT / "results/phase4_residual_output",
        help="directory containing the filtered residual kernel output",
    )
    args = parser.parse_args()
    summary_path = locate(args.online_dir, "phase4_summary.json")
    raw_path = locate(args.online_dir, "results.jsonl")
    summary = read_json(summary_path)
    raw = read_jsonl(raw_path)
    p3_summary = read_json(P3 / "phase3_summary_audited.json")
    p3_raw = read_jsonl(P3 / "results.jsonl")
    p3_scores = read_jsonl(P3 / "scores.jsonl")

    if summary.get("status") != "complete" or summary.get("n_complete") != 4 or summary.get("n_failed") != 0:
        raise RuntimeError(f"Residual kernel did not complete all prompts: {summary.get('status')}, "
                           f"complete={summary.get('n_complete')}, failed={summary.get('n_failed')}")
    complete = [x for x in raw if x.get("status") == "complete"]
    failures = [x for x in raw if x.get("status") != "complete"]
    if failures or len(complete) != 4 or len({int(x["prompt_idx"]) for x in complete}) != 4:
        raise RuntimeError(f"Expected four unique successful prompt records; got {len(complete)} "
                           f"complete and {len(failures)} failed/other records")

    protocol = summary["protocol"]
    if protocol.get("signal") != "denoising residual between adjacent predictions in preceding coarse segment":
        raise RuntimeError(f"Unexpected online signal definition: {protocol.get('signal')}")
    if int(protocol.get("seed")) != 17 or int(protocol.get("horizon")) != 45:
        raise RuntimeError("The residual run does not match the declared seed-17, 45-frame protocol")
    expected_tau_low = float(np.quantile(
        [x["residual"] for x in p3_scores if x["sample_split"] == "calibration"], 0.25
    ))
    expected_tau_high = float(p3_summary["metrics"]["residual"]["threshold_calibration_youden"])
    if not np.isclose(float(protocol["tau_low"]), expected_tau_low, rtol=0, atol=1e-7):
        raise RuntimeError("Online low threshold does not match the calibration-only Phase 3 Q25")
    if not np.isclose(float(protocol["tau_high"]), expected_tau_high, rtol=0, atol=1e-7):
        raise RuntimeError("Online high threshold does not match the calibration-only Phase 3 Youden point")

    p3_held = {int(x["prompt_idx"]): x for x in p3_raw if x.get("split") == "held_out"}
    p3_scored = {}
    for x in p3_scores:
        if x.get("sample_split") == "held_out":
            p3_scored.setdefault(int(x["prompt_idx"]), []).append(x)
    adaptive = {int(x["prompt_idx"]): x for x in complete}
    if set(adaptive) != set(p3_held) or set(adaptive) != set(p3_scored):
        raise RuntimeError("Online and Phase 3 held-out prompt indices do not align")

    prompt_rows = []
    for idx in sorted(adaptive):
        a = adaptive[idx]
        base = p3_held[idx]
        fixed_obs = sorted(p3_scored[idx], key=lambda x: int(x["group"]))
        aseg = sorted(a["segments"], key=lambda x: int(x["group"]))
        decisions = sorted(a["decisions"], key=lambda x: int(x["group"]))
        groups = [1, 2, 3]
        if [int(x["group"]) for x in aseg] != groups or [int(x["group"]) for x in decisions] != groups:
            raise RuntimeError(f"Prompt {idx} has an unexpected segment/decision trace")
        if len(fixed_obs) != 3 or [int(x["group"]) for x in fixed_obs] != groups:
            raise RuntimeError(f"Prompt {idx} has an unexpected Phase 3 segment baseline")

        adaptive_l1 = float(np.mean([x["adjacent_pixel_l1"] for x in aseg]))
        fine_l1 = float(np.mean([x["fine_l1"] for x in fixed_obs]))
        coarse_l1 = float(np.mean([x["coarse_l1"] for x in fixed_obs]))
        oracle_l1 = float(np.mean([min(x["fine_l1"], x["coarse_l1"]) for x in fixed_obs]))
        modes = [x["mode"] for x in decisions]
        n_fine = sum(m == "fine" for m in modes)
        prompt_rows.append({
            "prompt_idx": idx,
            "prompt": a["prompt"],
            "segments": [{
                "group": g,
                "residual": float(d["residual"]),
                "mode": d["mode"],
                "adaptive_l1": float(s["adjacent_pixel_l1"]),
                "fixed_fine_l1": float(f["fine_l1"]),
                "fixed_coarse_l1": float(f["coarse_l1"]),
                "oracle_l1": float(min(f["fine_l1"], f["coarse_l1"])),
                "signal_s": float(d["signal_s"]),
                "checkpoint_load_s": float(d["checkpoint_load_s"]),
                "dit_s": float(d["dit_s"]),
                "nfe": int(d["nfe"]),
            } for g, s, d, f in zip(groups, aseg, decisions, fixed_obs)],
            "adaptive_mean_adjacent_pixel_l1": adaptive_l1,
            "fixed_fine_mean_adjacent_pixel_l1": fine_l1,
            "fixed_coarse_mean_adjacent_pixel_l1": coarse_l1,
            "offline_oracle_mean_adjacent_pixel_l1": oracle_l1,
            "fine_fraction": n_fine / len(modes),
            "switch_count": int(a["controller"]["switch_count"]),
            "adaptive_modes": modes,
            "timing": a["timing"],
            "fixed_fine_timing": base["fine"],
            "fixed_coarse_timing": base["coarse"],
        })

    def mean(field):
        return float(np.mean([r[field] for r in prompt_rows]))

    quality_fields = {
        "residual_threshold_online_R0": "adaptive_mean_adjacent_pixel_l1",
        "fixed_fine_c1_1": "fixed_fine_mean_adjacent_pixel_l1",
        "fixed_coarse_c3_3": "fixed_coarse_mean_adjacent_pixel_l1",
        "offline_segment_oracle": "offline_oracle_mean_adjacent_pixel_l1",
    }
    aggregate_proxy = {name: mean(field) for name, field in quality_fields.items()}
    comparisons = {
        baseline: prompt_bootstrap([
            row[quality_fields["residual_threshold_online_R0"]] - row[field]
            for row in prompt_rows
        ], seed=20261006 + offset)
        for offset, (baseline, field) in enumerate(list(quality_fields.items())[1:], start=1)
    }

    curve = []
    curve.append({
        "method": "residual_threshold_online_R0",
        "quality_proxy_mean_adjacent_pixel_l1": aggregate_proxy["residual_threshold_online_R0"],
        "mean_total_s": float(np.mean([r["timing"]["total_s"] for r in prompt_rows])),
        "mean_nfe": float(np.mean([r["timing"]["nfe"] for r in prompt_rows])),
        "mean_fine_fraction": mean("fine_fraction"),
        "mean_switches_per_prompt": float(np.mean([r["switch_count"] for r in prompt_rows])),
        "latency_status": "measured end-to-end; total includes prompt encoding, loads, controller path, sampling, and final decode",
    })
    for label, quality_key, mode in [
        ("fixed_fine_c1_1", "fixed_fine_mean_adjacent_pixel_l1", "fine"),
        ("fixed_coarse_c3_3", "fixed_coarse_mean_adjacent_pixel_l1", "coarse"),
    ]:
        curve.append({
            "method": label,
            "quality_proxy_mean_adjacent_pixel_l1": mean(quality_key),
            "mean_total_s": float(np.mean([r[f"fixed_{mode}_timing"]["total_s"] for r in prompt_rows])),
            "mean_nfe": float(np.mean([r[f"fixed_{mode}_timing"]["nfe"] for r in prompt_rows])),
            "mean_fine_fraction": 1.0 if mode == "fine" else 0.0,
            "mean_switches_per_prompt": 0.0,
            "latency_status": "measured fixed-mode Phase 3 end-to-end run; paired prompt/seed",
        })
    curve.append({
        "method": "offline_segment_oracle",
        "quality_proxy_mean_adjacent_pixel_l1": aggregate_proxy["offline_segment_oracle"],
        "mean_total_s": None,
        "mean_nfe": None,
        "mean_fine_fraction": None,
        "mean_switches_per_prompt": None,
        "latency_status": "not measured; per-segment min(fine, coarse) uses both future fixed outcomes",
    })

    complexity_rows = []
    complexity_summary_path = COMPLEXITY / "phase4_summary.json"
    complexity_raw_path = COMPLEXITY / "results.jsonl"
    if complexity_summary_path.exists() and complexity_raw_path.exists():
        csummary = read_json(complexity_summary_path)
        craw = [x for x in read_jsonl(complexity_raw_path) if x.get("status") == "complete"]
        if csummary.get("status") == "complete" and len(craw) == 4:
            for item in craw:
                segs = item["segments"]
                complexity_rows.append({
                    "prompt_idx": int(item["prompt_idx"]),
                    "adaptive_mean_adjacent_pixel_l1": float(np.mean([x["adjacent_pixel_l1"] for x in segs])),
                    "fine_fraction": float(item["controller"]["fine_fraction"]),
                    "switch_count": int(item["controller"]["switch_count"]),
                    "timing": item["timing"],
                })
            if {r["prompt_idx"] for r in complexity_rows} != set(adaptive):
                raise RuntimeError("Complexity and residual online prompt indices do not align")
            c_by_idx = {r["prompt_idx"]: r for r in complexity_rows}
            c_quality = float(np.mean([r["adaptive_mean_adjacent_pixel_l1"] for r in complexity_rows]))
            curve.append({
                "method": "complexity_threshold_online_R0",
                "quality_proxy_mean_adjacent_pixel_l1": c_quality,
                "mean_total_s": float(np.mean([r["timing"]["total_s"] for r in complexity_rows])),
                "mean_nfe": float(np.mean([r["timing"]["nfe"] for r in complexity_rows])),
                "mean_fine_fraction": float(np.mean([r["fine_fraction"] for r in complexity_rows])),
                "mean_switches_per_prompt": float(np.mean([r["switch_count"] for r in complexity_rows])),
                "latency_status": "measured online; separate pilot kernel using decoded-frame complexity signal",
            })
            comparisons["complexity_threshold_online_R0"] = prompt_bootstrap([
                prompt_rows[i]["adaptive_mean_adjacent_pixel_l1"] - c_by_idx[i]["adaptive_mean_adjacent_pixel_l1"]
                for i in sorted(adaptive)
            ], seed=20261010)

    OUT.mkdir(parents=True, exist_ok=True)
    timing_means = {
        key: float(np.mean([r["timing"][key] for r in prompt_rows]))
        for key in ["text_encode_s", "checkpoint_load_s", "dit_s", "vae_s", "signal_s", "total_s", "nfe", "peak_vram_gb"]
    }
    out = {
        "phase": "phase4-adaptive-controller",
        "status": "complete",
        "verdict": "PASS_EXPLORATORY_ONLINE_RESIDUAL_R0",
        "online_artifacts": {"summary": str(summary_path), "raw": str(raw_path)},
        "protocol": protocol,
        "calibration_threshold_audit": {
            "tau_low_q25_recomputed": expected_tau_low,
            "tau_high_youden_from_phase3_calibration": expected_tau_high,
            "split": "Phase 3 calibration prompts only",
        },
        "n_prompts": len(prompt_rows),
        "n_segments": sum(len(r["segments"]) for r in prompt_rows),
        "prompt_results": prompt_rows,
        "aggregate_proxy": aggregate_proxy,
        "residual_adaptive_minus_baseline_prompt_bootstrap": comparisons,
        "quality_latency_compute_points": curve,
        "aggregate_online_usage": {
            "fine_fraction": float(np.mean([r["fine_fraction"] for r in prompt_rows])),
            "coarse_fraction": float(1.0 - np.mean([r["fine_fraction"] for r in prompt_rows])),
            "mean_switch_count_per_prompt": float(np.mean([r["switch_count"] for r in prompt_rows])),
            "timing_means": timing_means,
            "signal_timing_note": "signal_s measures the decision/read path; denoising residual calculation occurs within the timed DiT sampling path",
        },
        "practical_target": {
            "target": "0.1% improvement",
            "assessment": "not resolvable in this pilot: four held-out prompts, one seed, and an adjacent-pixel temporal proxy rather than a perceptual/reference quality metric",
        },
        "limitations": [
            "four held-out prompts and one seed; prompt-level bootstrap intervals are exploratory",
            "adjacent-pixel temporal L1 depends on motion/content and is not perceptual quality",
            "online adaptive and fixed trajectories are prompt/seed paired but are not identical trajectories",
            "offline oracle uses both future fixed-mode outcomes and has no measured end-to-end cost",
            "no recovery spell was evaluated; recovery and hysteresis remain Phase 5",
            "proposal deviations: text-to-video on Kaggle T4, no EMA, lambda_action=0, and no reference-video metrics",
        ],
        "proposal_coverage": {
            "RQ2": "online residual threshold controller completed with exploratory fixed/oracle comparisons and measured runtime/NFE",
            "RQ3": "not evaluated; recovery/hysteresis remain Phase 5",
        },
    }
    (OUT / "phase4_residual_audit.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "phase4_residual_prompt_table.csv").write_text(
        "prompt_idx,prompt,adaptive_l1,fixed_fine_l1,fixed_coarse_l1,oracle_l1,fine_fraction,switch_count,total_s,nfe\n" +
        "".join(
            f'{r["prompt_idx"]},"{r["prompt"].replace(chr(34), chr(34) * 2)}",'
            f'{r["adaptive_mean_adjacent_pixel_l1"]},{r["fixed_fine_mean_adjacent_pixel_l1"]},'
            f'{r["fixed_coarse_mean_adjacent_pixel_l1"]},{r["offline_oracle_mean_adjacent_pixel_l1"]},'
            f'{r["fine_fraction"]},{r["switch_count"]},{r["timing"]["total_s"]},{r["timing"]["nfe"]}\n'
            for r in prompt_rows
        ),
        encoding="utf-8",
    )
    headers = ["method", "quality_proxy_mean_adjacent_pixel_l1", "mean_total_s", "mean_nfe",
               "mean_fine_fraction", "mean_switches_per_prompt", "latency_status"]
    with (OUT / "phase4_quality_latency_compute.csv").open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(headers) + "\n")
        for row in curve:
            values = [row.get(h) for h in headers]
            f.write(",".join("" if v is None else str(v).replace(",", ";") for v in values) + "\n")
    print(json.dumps({
        "aggregate_proxy": aggregate_proxy,
        "usage": out["aggregate_online_usage"],
        "paired_bootstrap": comparisons,
        "curve": curve,
        "outputs": [str(OUT / "phase4_residual_audit.json"), str(OUT / "phase4_residual_prompt_table.csv"), str(OUT / "phase4_quality_latency_compute.csv")],
    }, indent=2))


if __name__ == "__main__":
    main()

