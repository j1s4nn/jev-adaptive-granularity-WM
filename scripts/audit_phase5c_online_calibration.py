"""Audit online calibration trajectories, then fit the locked warp quantiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_FRAMES = [9, 21, 33, 45]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: object, positive: bool = False) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def audit(base: Path) -> dict:
    lock = json.loads((ROOT / "config/phase5c_online_calibration_lock.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "config/phase5_prompt_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((base / "phase5c_calibration_summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (base / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    complete = [row for row in rows if row.get("status") == "complete"]
    expected = {i: manifest["calibration"][i]["prompt"] for i in lock["calibration_prompt_indices"]}
    errors: list[str] = []

    def require(ok: bool, message: str) -> None:
        if not ok:
            errors.append(message)

    require(sha256(ROOT / "config/phase5_prompt_manifest.json") == lock["manifest_sha256"], "manifest hash")
    require(sha256(ROOT / "config/phase5c_pilot_lock.json") == lock["pilot_lock_sha256"], "pilot lock hash")
    require(lock["calibration_prompt_indices"] == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
            and lock["excluded_pilot_prompt_index"] == 6, "locked prompt set")
    require(summary.get("phase") == "phase5c-online-calibration"
            and summary.get("status") == "complete" and summary.get("verdict") == "PASS", "remote verdict")
    require(summary.get("n_complete") == 11 and summary.get("n_failed_or_incomplete") == 0
            and summary.get("n_decisions") == 44, "remote counts")
    protocol = summary.get("protocol", {})
    require(protocol.get("split") == "calibration"
            and protocol.get("prompts") == [[i, prompt] for i, prompt in expected.items()]
            and protocol.get("seed") == lock["seed"] and protocol.get("horizon") == lock["horizon"]
            and protocol.get("mode") == "coarse prefix plus four coarse continuations"
            and protocol.get("manifest_sha256") == lock["manifest_sha256"]
            and protocol.get("pilot_lock_sha256") == lock["pilot_lock_sha256"], "remote protocol")
    require(len(rows) == len(complete) == 11, "11 complete rows and no failures")
    require(len({row.get("prompt_idx") for row in complete}) == 11
            and {row.get("prompt_idx") for row in complete} == set(expected), "unique expected prompt indices")

    noise_hashes: set[str] = set()
    group_hashes: dict[int, set[str]] = {i: set() for i in range(1, 5)}
    oriented_warp: list[float] = []
    for row in complete:
        index = row.get("prompt_idx")
        tag = f"prompt {index}: "
        require(row.get("prompt") == expected.get(index) and row.get("seed") == lock["seed"]
                and row.get("horizon") == lock["horizon"] and row.get("latent_frames") == 15, tag + "identity/shape")
        noise_hashes.add(row.get("initial_noise_sha256"))
        prefix_hash = row.get("initial_prefix_sha256")
        require(isinstance(prefix_hash, str) and len(prefix_hash) == 64, tag + "prefix hash")
        decisions = row.get("decisions", [])
        require(len(decisions) == 4, tag + "decision count")
        if len(decisions) != 4:
            continue
        for group, decision in enumerate(decisions, 1):
            require(decision.get("group") == group and decision.get("mode") == "coarse"
                    and decision.get("context_frames") == CONTEXT_FRAMES[group - 1], tag + f"decision {group} mode/context")
            for signal in ("warp", "motion", "complexity"):
                require(finite(decision.get(signal)), tag + f"decision {group} {signal}")
            if finite(decision.get("warp")):
                oriented_warp.append(-decision["warp"])
            for field in ("checkpoint_load_s", "dit_s", "signal_s", "peak_vram_gb"):
                require(finite(decision.get(field)), tag + f"decision {group} {field}")
            require(finite(decision.get("nfe"), positive=True), tag + f"decision {group} NFE")
            group_hashes[group].add(decision.get("segment_noise_sha256"))
        timing = row.get("timing", {})
        for field in ("text_encode_s", "checkpoint_load_s", "dit_s", "vae_s", "signal_s", "total_s", "peak_vram_gb"):
            require(finite(timing.get(field)), tag + "timing " + field)
        require(timing.get("nfe") == 20, tag + "total NFE")
        require(math.isclose(timing.get("signal_s", -1), sum(d["signal_s"] for d in decisions), rel_tol=1e-7), tag + "signal timing sum")
        require(timing.get("total_s", 0) >= sum(timing.get(k, 0) for k in
                ("text_encode_s", "checkpoint_load_s", "dit_s", "vae_s", "signal_s")) - 2, tag + "total timing")
    require(len(noise_hashes) == 1 and all(isinstance(h, str) and len(h) == 64 for h in noise_hashes), "common full-noise hash")
    require(all(len(hashes) == 1 and all(isinstance(h, str) and len(h) == 64 for h in hashes)
                for hashes in group_hashes.values()), "common per-group noise hashes")
    require(len(oriented_warp) == 44, "44 finite warp scores")

    thresholds = None
    if not errors:
        thresholds = {str(fraction): float(np.quantile(oriented_warp, 1 - fraction, method="linear"))
                      for fraction in lock["threshold_fit_after_audit"]["target_fine_fractions"]}
    return {
        "verdict": "PASS" if not errors else "FAIL", "errors": errors,
        "n_cases": len(complete), "n_decisions": sum(len(row.get("decisions", [])) for row in complete),
        "raw_sha256": sha256(base / "results.jsonl"),
        "thresholds_on_oriented_warp": thresholds,
        "threshold_rule": lock["threshold_fit_after_audit"],
        "limitations": lock["limitations"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "results/phase5c_online_calibration/v1_selected/results/phase5c_online_calibration")
    args = parser.parse_args()
    result = audit(args.base)
    output = ROOT / "results/phase5c_online_calibration/phase5c_online_calibration_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["verdict"] == "PASS" else 1)
