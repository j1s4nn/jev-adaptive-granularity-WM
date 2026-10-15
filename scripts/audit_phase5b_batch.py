"""Audit one locked Phase 5B held-out batch before submitting the next."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIGNALS = ("residual", "warp", "motion", "complexity")
SEGMENTS = [(1, 9, 21, 9), (2, 21, 33, 21), (3, 33, 45, 33), (4, 45, 57, 45)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_nonnegative(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def audit(batch_id: int) -> dict:
    name = f"phase5b_heldout_{batch_id:02d}"
    base = ROOT / "results" / name / "results" / name
    raw_path = base / "results.jsonl"
    summary_path = base / "phase5b_summary.json"
    manifest_path = ROOT / "config" / "phase5_prompt_manifest.json"
    lock_path = ROOT / "results" / "phase5b_calibration_lock.json"
    source_path = ROOT / "jobs" / name / "phase5b_heldout.py"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seeds = manifest["held_out_seeds"]
    prompts = manifest["held_out"][batch_id * 3:(batch_id + 1) * 3]
    expected = {(i, seed, 57, item["prompt"])
                for i, item in enumerate(prompts, batch_id * 3) for seed in seeds}
    keys = [(row.get("prompt_idx"), row.get("seed"), row.get("horizon"), row.get("prompt")) for row in rows]
    complete = [row for row in rows if row.get("status") == "complete"]
    observations = [obs for row in complete for obs in row.get("observations", [])]
    errors = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(sha256(manifest_path) == lock.get("manifest_sha256"), "manifest vs calibration lock")
    require(summary.get("verdict") == "PASS" and summary.get("status") == "complete", "summary verdict/status")
    require(summary.get("phase") == f"phase5b-heldout-{batch_id:02d}", "summary phase")
    require(summary.get("n_complete") == 9 and summary.get("n_failed_or_incomplete") == 0, "summary counts")
    protocol = summary.get("protocol", {})
    require(protocol.get("split") == "held_out" and protocol.get("prompts") == [p["prompt"] for p in prompts]
            and protocol.get("seeds") == seeds and protocol.get("horizons") == [57], "summary protocol/manifest")
    require(len(rows) == 9 and len(complete) == 9, "raw/complete row counts")
    require(len(keys) == len(set(keys)) and set(keys) == expected, "raw keys vs locked manifest")
    require(len(observations) == 36, "observation count")
    for row in complete:
        key = f"prompt{row['prompt_idx']}_seed{row['seed']}_h57"
        require(row.get("sample_key") == key, f"{key}: sample key")
        alignment = row.get("alignment", {})
        require(alignment.get("decoded_shape") == [57, 480, 832, 3]
                and alignment.get("same_seed") is True
                and alignment.get("same_initial_noise") is True
                and alignment.get("reference_mode") == "c1-1_step4", f"{key}: alignment")
        require(finite_nonnegative(row.get("text_encode_s")), f"{key}: text timing")
        for mode in ("reference", "fine", "coarse"):
            values = row.get(mode, {})
            require(all(finite_nonnegative(values.get(field)) for field in
                        ("text_encode_s", "checkpoint_load_s", "dit_s", "vae_s", "total_s")),
                    f"{key}: {mode} timings")
            require(all(isinstance(values.get(field), (int, float)) and math.isfinite(values[field]) and values[field] > 0
                        for field in ("nfe", "peak_vram_gb", "blocks")), f"{key}: {mode} compute/VRAM")
        segments = row.get("observations", [])
        require([(o.get("group"), o.get("segment_start_frame"), o.get("segment_end_frame"),
                  o.get("history_frames")) for o in segments] == SEGMENTS, f"{key}: segment indices")
        for obs in segments:
            group = obs.get("group")
            require(all(finite_nonnegative(obs.get(field)) for field in
                        ("fine_reference_l1", "coarse_reference_l1", "signal_s")),
                    f"{key}/segment{group}: metric/signal timing")
            require(all(isinstance(obs.get(field), (int, float)) and math.isfinite(obs[field])
                        for field in (*SIGNALS, "reference_regret_l1")),
                    f"{key}/segment{group}: finite signals/regret")
            if all(isinstance(obs.get(field), (int, float)) for field in
                   ("coarse_reference_l1", "fine_reference_l1", "reference_regret_l1")):
                require(abs(obs["reference_regret_l1"] -
                            (obs["coarse_reference_l1"] - obs["fine_reference_l1"])) < 1e-8,
                        f"{key}/segment{group}: regret arithmetic")
    regrets = [obs["reference_regret_l1"] for obs in observations]
    stats = summary.get("reference_regret_summary", {})
    if regrets:
        require(stats.get("n_segments") == len(regrets) and
                all(math.isclose(stats.get(field, float("nan")), value, rel_tol=0, abs_tol=1e-12)
                    for field, value in (("mean", sum(regrets) / len(regrets)),
                                         ("min", min(regrets)), ("max", max(regrets)))),
                "summary regret statistics")
    return {
        "batch": name, "verdict": "PASS" if not errors else "FAIL", "errors": errors,
        "expected_cases": len(expected), "observed_cases": len(rows), "complete_cases": len(complete),
        "segments": len(observations), "prompt_indices": sorted({row.get("prompt_idx") for row in rows}),
        "seeds": sorted({row.get("seed") for row in rows}),
        "regret_summary": stats, "source_hashes": {
            "manifest_sha256": sha256(manifest_path), "calibration_lock_sha256": sha256(lock_path),
            "raw_sha256": sha256(raw_path),
            "summary_sha256": sha256(summary_path), "job_source_sha256": sha256(source_path),
        },
        "label_counts_at_locked_calibration_cutoff": dict(Counter(
            str(obs["reference_regret_l1"] > lock["unsafe_margin"]) for obs in observations)),
        "scope": "single-batch data integrity only; final AUROC and RQ1 gate require all eight batches",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_id", type=int, choices=range(8), metavar="{0..7}")
    args = parser.parse_args()
    result = audit(args.batch_id)
    out = ROOT / "results" / result["batch"] / "heldout_batch_audit.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
