"""Freeze Phase 5B choices from calibration raw data before held-out scoring."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase5b_calibration_lock.json"
SIGNALS = ("residual", "warp", "motion", "complexity")


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"calibration lock already exists: {OUT}")
    manifest_path = ROOT / "config/phase5_prompt_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    sources = {}
    for name, offset in (("phase5b_reference_calibration_a", 0), ("phase5b_reference_calibration_b", 6)):
        path = ROOT / "results" / name / "results" / name / "results.jsonl"
        source_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(source_rows) != 6 or any(row.get("status") != "complete" for row in source_rows):
            raise ValueError(f"incomplete calibration batch: {name}")
        rows.extend({**row, "prompt_idx": row["prompt_idx"] + offset} for row in source_rows)
        sources[name] = {"raw_sha256": hash_file(path), "manifest_index_offset": offset}
    if [row["prompt"] for row in rows] != [item["prompt"] for item in manifest["calibration"]]:
        raise ValueError("calibration rows do not match the locked manifest")
    margin = float(np.median([
        np.median([o["reference_regret_l1"] for o in row["observations"]])
        for row in rows
    ]))
    observations = [o for row in rows for o in row["observations"]]
    labels = np.asarray([o["reference_regret_l1"] > margin for o in observations], dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError("calibration labels have only one class")
    signals = {}
    for name in SIGNALS:
        raw = np.asarray([o[name] for o in observations], dtype=float)
        raw_auc = float(roc_auc_score(labels, raw))
        direction = 1 if raw_auc >= 0.5 else -1
        oriented = direction * raw
        fpr, tpr, thresholds = roc_curve(labels, oriented)
        finite = np.flatnonzero(np.isfinite(thresholds))
        chosen = float(thresholds[finite[np.argmax((tpr - fpr)[finite])]])
        signals[name] = {
            "direction": direction,
            "calibration_auroc": float(roc_auc_score(labels, oriented)),
            "youden_threshold_on_oriented_score": chosen,
        }
    strongest = max(("warp", "motion", "complexity"), key=lambda name: signals[name]["calibration_auroc"])
    lock = {
        "phase": "phase5b-reference-regret-calibration",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_version": manifest["manifest_version"],
        "manifest_sha256": hash_file(manifest_path),
        "sources": sources,
        "margin_rule": "median of 12 calibration prompt-seed median regrets; strictly greater is high regret",
        "unsafe_margin": margin,
        "calibration_groups": len(rows),
        "calibration_segments": len(observations),
        "positive_segments": int(labels.sum()),
        "signals": signals,
        "strongest_calibration_baseline": strongest,
    }
    OUT.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
