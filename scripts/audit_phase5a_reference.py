"""Audit Phase 5A reference/capacity output without changing raw artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/phase5a_reference_capacity/results/phase5a_reference_capacity"
RAW = BASE / "results.jsonl"
SUMMARY = BASE / "phase5a_summary.json"
OUT = ROOT / "results/phase5a_reference_capacity/phase5a_audit.json"


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = []
    for line in RAW.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("status") == "complete":
            rows.append(row)

    expected = {(0, 57), (0, 81), (1, 57), (1, 81)}
    observed = {(r["prompt_idx"], r["horizon"]) for r in rows}
    shape_ok = all(r["alignment"]["decoded_shape"][0] == r["horizon"] for r in rows)
    alignment_ok = all(
        r["alignment"]["same_seed"]
        and r["alignment"]["same_initial_noise"]
        and r["alignment"]["reference_mode"] == "c1-1_step4"
        for r in rows
    )
    mode_ok = all(set(r) >= {"reference", "fine", "coarse", "observations"} for r in rows)
    regrets = [o["reference_regret_l1"] for r in rows for o in r["observations"]]
    audit = {
        "phase": "phase5a-reference-capacity",
        "source_summary": str(SUMMARY),
        "source_raw": str(RAW),
        "summary_verdict": summary.get("verdict"),
        "expected_samples": len(expected),
        "observed_samples": len(rows),
        "sample_keys_ok": observed == expected,
        "shape_ok": shape_ok,
        "alignment_metadata_ok": alignment_ok,
        "mode_records_ok": mode_ok,
        "all_nfe_positive": all(r[m]["nfe"] > 0 for r in rows for m in ("reference", "fine", "coarse")),
        "all_peak_vram_recorded": all(r[m]["peak_vram_gb"] > 0 for r in rows for m in ("reference", "fine", "coarse")),
        "reference_regret": {
            "n": len(regrets),
            "mean": sum(regrets) / len(regrets) if regrets else None,
            "min": min(regrets) if regrets else None,
            "max": max(regrets) if regrets else None,
        },
        "limitations": [
            "two prompts and one seed; capacity/reference gate only",
            "decoded pixel L1 against an internal c1-1 step-4 reference, not ground truth",
            "no signal AUROC or adaptive-policy claim is made",
        ],
    }
    audit["verdict"] = "PASS" if all(
        [
            audit["summary_verdict"] == "PASS",
            audit["sample_keys_ok"],
            audit["shape_ok"],
            audit["alignment_metadata_ok"],
            audit["mode_records_ok"],
            audit["all_nfe_positive"],
            audit["all_peak_vram_recorded"],
        ]
    ) else "FAIL"
    OUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

