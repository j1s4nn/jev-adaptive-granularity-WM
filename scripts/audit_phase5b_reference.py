"""Audit the locked Phase 5B reference-regret calibration and held-out split."""

from __future__ import annotations

import json
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config/phase5_prompt_manifest.json"
OUT = ROOT / "results/phase5b_audit/phase5b_audit.json"
SIGNALS = ("residual", "warp", "motion", "complexity")
REPS = 2000


def read_rows(name: str) -> tuple[list[dict], dict]:
    base = ROOT / "results" / name / "results" / name
    raw = base / "results.jsonl"
    summary = base / "phase5b_summary.json"
    return (
        [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()],
        json.loads(summary.read_text(encoding="utf-8")),
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc_or_none(labels: np.ndarray, scores: np.ndarray) -> float | None:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None


def clustered_bootstrap(
    labels: np.ndarray,
    score_columns: dict[str, np.ndarray],
    groups: np.ndarray,
    seed: int,
) -> dict[str, dict]:
    unique = np.unique(groups)
    indices = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in score_columns}
    skipped = 0
    for _ in range(REPS):
        selected = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([indices[i] for i in selected])
        if len(np.unique(labels[idx])) < 2:
            skipped += 1
            continue
        for name, values in score_columns.items():
            draws[name].append(float(roc_auc_score(labels[idx], values[idx])))
    result = {}
    for name, values in score_columns.items():
        observed = auc_or_none(labels, values)
        sample = np.asarray(draws[name])
        result[name] = {
            "auroc": observed,
            "ci95": np.quantile(sample, [0.025, 0.975]).tolist() if len(sample) else [None, None],
            "bootstrap_valid": len(sample),
            "bootstrap_skipped_single_class": skipped,
        }
    return result


def choose_youden(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    fpr, tpr, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    candidates = np.flatnonzero(finite)
    return float(thresholds[candidates[np.argmax((tpr - fpr)[candidates])]])


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    batch_names = ["phase5b_reference_calibration_a", "phase5b_reference_calibration_b"]
    held_names = [f"phase5b_heldout_{i:02d}" for i in range(8)]
    all_rows: dict[str, list[dict]] = {"calibration": [], "held_out": []}
    batches = {}
    for split, names in (("calibration", batch_names), ("held_out", held_names)):
        for name in names:
            rows, summary = read_rows(name)
            index_offset = 6 if name == "phase5b_reference_calibration_b" else 0
            # Batch B stores local 0-5 indices. Keep raw files unchanged and
            # map those indices to the locked manifest's global 6-11 indices.
            rows = [{**row, "prompt_idx": row["prompt_idx"] + index_offset} for row in rows]
            batches[name] = {
                "manifest_index_offset": index_offset,
                "raw_rows": len(rows),
                "complete": sum(r.get("status") == "complete" for r in rows),
                "summary_verdict": summary.get("verdict"),
                "summary_complete": summary.get("n_complete"),
                "raw_sha256": sha256(ROOT / "results" / name / "results" / name / "results.jsonl"),
                "summary_sha256": sha256(ROOT / "results" / name / "results" / name / "phase5b_summary.json"),
                "job_source_sha256": sha256(ROOT / "jobs" / name / (
                    "phase5b_heldout.py" if name.startswith("phase5b_heldout_") else "phase5b_reference_calibration.py"
                )),
            }
            all_rows[split].extend(rows)

    expected = {
        "calibration": {
            (i, seed, 57, item["prompt"])
            for i, item in enumerate(manifest["calibration"])
            for seed in manifest["calibration_seeds"]
        },
        "held_out": {
            (i, seed, 57, item["prompt"])
            for i, item in enumerate(manifest["held_out"])
            for seed in manifest["held_out_seeds"]
        },
    }
    checks = {}
    observations = {}
    for split, rows in all_rows.items():
        keys = [(r.get("prompt_idx"), r.get("seed"), r.get("horizon"), r.get("prompt")) for r in rows]
        complete = [r for r in rows if r.get("status") == "complete"]
        checks[split] = {
            "expected_groups": len(expected[split]),
            "observed_groups": len(rows),
            "duplicate_groups": len(keys) - len(set(keys)),
            "missing_groups": len(expected[split] - set(keys)),
            "unexpected_groups": len(set(keys) - expected[split]),
            "failures": [r.get("sample_key") for r in rows if r.get("status") != "complete"],
            "shape_ok": all(r["alignment"]["decoded_shape"] == [57, 480, 832, 3] for r in complete),
            "seed_noise_metadata_ok": all(
                r["alignment"]["same_seed"] and r["alignment"]["same_initial_noise"]
                and r["alignment"]["reference_mode"] == "c1-1_step4"
                for r in complete
            ),
            "segment_index_ok": all(
                [(o["group"], o["segment_start_frame"], o["segment_end_frame"], o["history_frames"])
                 for o in r["observations"]]
                == [(1, 9, 21, 9), (2, 21, 33, 21), (3, 33, 45, 33), (4, 45, 57, 45)]
                for r in complete
            ),
            "regret_arithmetic_ok": all(
                abs(o["reference_regret_l1"] - (o["coarse_reference_l1"] - o["fine_reference_l1"])) < 1e-8
                for r in complete for o in r["observations"]
            ),
            "finite_signals_ok": all(
                np.isfinite(o[name]) for r in complete for o in r["observations"] for name in SIGNALS
            ),
            "positive_nfe_and_vram_ok": all(
                r[mode]["nfe"] > 0 and r[mode]["peak_vram_gb"] > 0
                for r in complete for mode in ("reference", "fine", "coarse")
            ),
        }
        observations[split] = [
            {"prompt_idx": r["prompt_idx"], "seed": r["seed"], **o}
            for r in complete for o in r["observations"]
        ]

    calibration = observations["calibration"]
    held_out = observations["held_out"]
    by_cal_group = {}
    for item in calibration:
        by_cal_group.setdefault((item["prompt_idx"], item["seed"]), []).append(item["reference_regret_l1"])
    margin = float(np.median([np.median(values) for values in by_cal_group.values()]))
    labels = {
        split: np.asarray([row["reference_regret_l1"] > margin for row in values], dtype=int)
        for split, values in observations.items()
    }
    direction = {}
    for name in SIGNALS:
        score = np.asarray([row[name] for row in calibration], dtype=float)
        raw_auc = auc_or_none(labels["calibration"], score)
        direction[name] = 1 if raw_auc is None or raw_auc >= 0.5 else -1
    scores = {
        split: {
            name: direction[name] * np.asarray([row[name] for row in values], dtype=float)
            for name in SIGNALS
        }
        for split, values in observations.items()
    }
    grouped_auc = {}
    for split, values in observations.items():
        prompt_seed = np.asarray([f"{r['prompt_idx']}_{r['seed']}" for r in values])
        prompt = np.asarray([str(r["prompt_idx"]) for r in values])
        grouped_auc[split] = {
            "prompt_seed": clustered_bootstrap(labels[split], scores[split], prompt_seed, 20261006),
            "prompt": clustered_bootstrap(labels[split], scores[split], prompt, 20261007),
        }
    baseline = max(
        ("warp", "motion", "complexity"),
        key=lambda name: (
            grouped_auc["calibration"]["prompt_seed"][name]["auroc"]
            if grouped_auc["calibration"]["prompt_seed"][name]["auroc"] is not None else -1
        ),
    )
    differences = {}
    for split, values in observations.items():
        paired = {"residual_minus_calibration_selected_baseline": scores[split]["residual"]}
        paired["baseline"] = scores[split][baseline]
        # Resample the same prompt-seed groups for both signals.
        groups = np.asarray([f"{r['prompt_idx']}_{r['seed']}" for r in values])
        unique = np.unique(groups)
        indices = [np.flatnonzero(groups == group) for group in unique]
        rng = np.random.default_rng(20261008)
        boot = []
        for _ in range(REPS):
            idx = np.concatenate([indices[i] for i in rng.integers(0, len(indices), len(indices))])
            if len(np.unique(labels[split][idx])) == 2:
                boot.append(roc_auc_score(labels[split][idx], paired["residual_minus_calibration_selected_baseline"][idx])
                            - roc_auc_score(labels[split][idx], paired["baseline"][idx]))
        residual_auc = grouped_auc[split]["prompt_seed"]["residual"]["auroc"]
        baseline_auc = grouped_auc[split]["prompt_seed"][baseline]["auroc"]
        observed = residual_auc - baseline_auc if residual_auc is not None and baseline_auc is not None else None
        differences[split] = {"difference": observed, "ci95": np.quantile(boot, [0.025, 0.975]).tolist() if boot else [None, None]}
    thresholds = {
        name: choose_youden(scores["calibration"][name], labels["calibration"])
        for name in SIGNALS
    }
    lock_path = ROOT / "results/phase5b_calibration_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_ok = (
        lock["manifest_sha256"] == sha256(MANIFEST)
        and abs(lock["unsafe_margin"] - margin) < 1e-12
        and lock["strongest_calibration_baseline"] == baseline
        and all(
            lock["sources"][name]["raw_sha256"] == batches[name]["raw_sha256"]
            for name in batch_names
        )
        and all(
            lock["signals"][name]["direction"] == direction[name]
            and abs(lock["signals"][name]["youden_threshold_on_oriented_score"] - thresholds[name]) < 1e-12
            for name in SIGNALS
        )
    )
    threshold_results = {}
    for split in ("calibration", "held_out"):
        threshold_results[split] = {}
        for name in SIGNALS:
            predictions = scores[split][name] >= thresholds[name]
            truth = labels[split].astype(bool)
            threshold_results[split][name] = {
                "tp": int(np.sum(predictions & truth)),
                "fp": int(np.sum(predictions & ~truth)),
                "tn": int(np.sum(~predictions & ~truth)),
                "fn": int(np.sum(~predictions & truth)),
            }
    data_pass = lock_ok and all(
        batch["summary_verdict"] == "PASS" and batch["complete"] == batch["summary_complete"]
        for batch in batches.values()
    ) and all(
        check["observed_groups"] == check["expected_groups"]
        and check["duplicate_groups"] == check["missing_groups"] == check["unexpected_groups"] == 0
        and not check["failures"]
        and all(check[field] for field in (
            "shape_ok", "seed_noise_metadata_ok", "segment_index_ok", "regret_arithmetic_ok",
            "finite_signals_ok", "positive_nfe_and_vram_ok"
        ))
        for check in checks.values()
    )
    auc_defined = all(grouped_auc["held_out"]["prompt_seed"][name]["auroc"] is not None for name in SIGNALS)
    claim = bool(
        data_pass and auc_defined and differences["held_out"]["ci95"][0] is not None
        and differences["held_out"]["ci95"][0] > 0
        and grouped_auc["held_out"]["prompt_seed"]["residual"]["auroc"]
        > max(grouped_auc["held_out"]["prompt_seed"][name]["auroc"] for name in ("warp", "motion", "complexity"))
    )
    audit = {
        "phase": "phase5b-reference-regret-calibration",
        "verdict": "PASS" if data_pass and auc_defined else "FAIL",
        "calibration_lock": {"path": str(lock_path), "sha256": sha256(lock_path), "matches_recomputed_choices": lock_ok},
        "batches": batches,
        "integrity": checks,
        "metric": "decoded-pixel L1 to matched internal c1-1 step-4 reference",
        "unsafe_margin_rule": "median of 12 calibration prompt-seed median regrets; strictly greater is unsafe",
        "unsafe_margin": margin,
        "label_counts": {split: dict(Counter(labels[split].tolist())) for split in labels},
        "regret_summary": {
            split: {
                "n_segments": len(values),
                "mean": float(np.mean([r["reference_regret_l1"] for r in values])),
                "min": float(np.min([r["reference_regret_l1"] for r in values])),
                "max": float(np.max([r["reference_regret_l1"] for r in values])),
            }
            for split, values in observations.items()
        },
        "signal_direction_fitted_on_calibration": direction,
        "calibration_youden_thresholds_on_oriented_scores": thresholds,
        "fixed_threshold_confusion": threshold_results,
        "auroc": grouped_auc,
        "strongest_calibration_baseline": baseline,
        "residual_minus_baseline": differences,
        "rq1_supported": claim,
        "limitations": [
            "reference-defined decoded-pixel metric, not ground-truth perceptual quality",
            "signals use a fixed coarse trajectory; online controllers have different contexts",
            "prompt-seed intervals may understate shared-prompt dependence; prompt-cluster intervals are also reported",
            "calibration margin is a relative high-regret cutoff, not an absolute safety threshold",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": audit["verdict"], "margin": margin, "label_counts": audit["label_counts"],
        "held_out_auc": grouped_auc["held_out"]["prompt_seed"], "baseline": baseline,
        "held_out_difference": differences["held_out"], "rq1_supported": claim,
    }, indent=2))
    if audit["verdict"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
