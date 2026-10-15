"""Audit the single calibration-case Phase 5C online feasibility pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOUNDS = [(9, 21), (21, 33), (33, 45), (45, 57)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: object, nonnegative: bool = True) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value >= 0 if nonnegative else True))


def audit(base: Path) -> dict:
    lock = json.loads((ROOT / "config/phase5c_pilot_lock.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "config/phase5_prompt_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((base / "phase5c_pilot_summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (base / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    policies = lock["policies"]
    complete = [row for row in rows if row.get("status") == "complete"]
    by_policy = {row["policy"]: row for row in complete}
    errors = []

    def require(ok: bool, message: str) -> None:
        if not ok:
            errors.append(message)

    require(sha256(ROOT / "config/phase5_prompt_manifest.json") == lock["manifest_sha256"], "manifest hash")
    require(sha256(ROOT / "results/phase5b_calibration_lock.json") == lock["phase5b_calibration_lock_sha256"], "calibration lock hash")
    selected = manifest["calibration"][lock["calibration_prompt_index"]]
    require(selected["prompt"] == lock["prompt"], "calibration prompt/position")
    scores = []
    for name in ("phase5b_reference_calibration_a", "phase5b_reference_calibration_b"):
        source = ROOT / "results" / name / "results" / name / "results.jsonl"
        for line in source.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("status") == "complete":
                scores.extend(lock["warp_direction"] * obs["warp"] for obs in row["observations"])
    scores.sort()
    require(len(scores) == 48, "48 calibration warp scores")
    if len(scores) == 48:
        require(math.isclose((scores[23] + scores[24]) / 2, lock["warp_threshold_on_oriented_score"], abs_tol=1e-9), "warp median threshold")

    require(summary.get("status") == "complete" and summary.get("verdict") == "PASS", "remote status/verdict")
    require(summary.get("n_complete") == 5 and summary.get("n_failed_or_incomplete") == 0, "remote counts")
    protocol = summary.get("protocol", {})
    require(protocol.get("split") == "calibration" and protocol.get("prompt_idx") == lock["calibration_prompt_index"]
            and protocol.get("prompt") == lock["prompt"] and protocol.get("seed") == lock["seed"]
            and protocol.get("horizon") == lock["horizon"], "remote case identity")
    require(protocol.get("manifest_sha256") == lock["manifest_sha256"]
            and protocol.get("calibration_lock_sha256") == lock["phase5b_calibration_lock_sha256"]
            and protocol.get("warp_direction") == lock["warp_direction"]
            and protocol.get("warp_threshold") == lock["warp_threshold_on_oriented_score"], "remote locked choices")
    require(list(protocol.get("policies", [])) == policies, "remote policy list")
    require(summary.get("reference", {}).get("decoded_shape") == [57, 480, 832, 3], "reference decoded shape")
    for field in ("checkpoint_load_s", "dit_s", "vae_s", "total_s", "peak_vram_gb"):
        require(finite(summary.get("reference", {}).get(field)), f"reference {field}")
    require(finite(summary.get("reference", {}).get("nfe")) and summary.get("reference", {}).get("nfe", 0) > 0, "reference NFE")
    require(len(rows) == 5 and len(complete) == 5 and set(by_policy) == set(policies), "five unique completed policies, no failed rows")

    prefix_hashes, noise_hashes = set(), set()
    group_noise_hashes = {group: set() for group in range(1, 5)}
    for policy in policies:
        if policy not in by_policy:
            continue
        row = by_policy[policy]
        tag = policy + ": "
        require((row.get("prompt_idx"), row.get("prompt"), row.get("seed"), row.get("horizon"))
                == (lock["calibration_prompt_index"], lock["prompt"], lock["seed"], lock["horizon"]), tag + "case identity")
        require(row.get("decoded_shape") == [57, 480, 832, 3], tag + "decoded shape")
        prefix_hashes.add(row.get("initial_prefix_sha256"))
        noise_hashes.add(row.get("initial_noise_sha256"))
        decisions, segments = row.get("decisions", []), row.get("segments", [])
        require(len(decisions) == len(segments) == 4, tag + "four decisions/segments")
        if len(decisions) != 4 or len(segments) != 4:
            continue
        modes = [d.get("mode") for d in decisions]
        fine_count = modes.count("fine")
        require(set(modes) <= {"coarse", "fine"}, tag + "valid modes")
        require(row.get("fine_count") == fine_count and row.get("fine_fraction") == fine_count / 4, tag + "fine count/fraction")
        switches = sum(modes[i] != modes[i - 1] for i in range(1, 4))
        require(row.get("switch_count_decisions") == switches, tag + "decision switches")
        require(row.get("switch_count_including_initial") == switches + int(modes[0] != "coarse"), tag + "prefix switches")
        for i, (decision, segment) in enumerate(zip(decisions, segments), 1):
            start, end = BOUNDS[i - 1]
            require(decision.get("group") == segment.get("group") == i
                    and decision.get("context_frames") == start
                    and (segment.get("start_frame"), segment.get("end_frame")) == (start, end)
                    and segment.get("mode") == modes[i - 1], tag + f"segment {i} alignment")
            for field in ("reference_l1", "adjacent_pixel_l1"):
                require(finite(segment.get(field)) and segment.get(field, 2) <= 1, tag + f"segment {i} {field}")
            for field in ("checkpoint_load_s", "dit_s", "signal_s", "peak_vram_gb"):
                require(finite(decision.get(field)), tag + f"decision {i} {field}")
            require(finite(decision.get("nfe")) and decision.get("nfe", 0) > 0, tag + f"decision {i} NFE")
            group_noise_hashes[i].add(decision.get("segment_noise_sha256"))
            if policy == "warp50":
                raw, oriented = decision.get("raw_warp"), decision.get("oriented_score")
                require(finite(raw) and finite(oriented, False), tag + f"decision {i} warp score")
                if finite(raw) and finite(oriented, False):
                    require(math.isclose(oriented, lock["warp_direction"] * raw, abs_tol=1e-9)
                            and modes[i - 1] == ("fine" if oriented >= lock["warp_threshold_on_oriented_score"] else "coarse"), tag + f"decision {i} threshold")
            else:
                require(decision.get("raw_warp") is None and decision.get("oriented_score") is None, tag + f"decision {i} unscored")
        mean_l1 = sum(s["reference_l1"] for s in segments) / 4
        require(math.isclose(row.get("mean_reference_l1", -1), mean_l1, abs_tol=1e-9), tag + "mean regret arithmetic")
        timing = row.get("timing", {})
        for field in ("checkpoint_load_s", "dit_s", "vae_s", "signal_s", "total_s", "standalone_total_s_including_text", "peak_vram_gb"):
            require(finite(timing.get(field)), tag + field)
        require(timing.get("nfe") == sum(d["nfe"] for d in decisions) + 4, tag + "NFE sum")
        for field in ("checkpoint_load_s", "dit_s", "signal_s"):
            require(math.isclose(timing.get(field, -1), sum(d[field] for d in decisions) + (0 if field == "signal_s" else 0), rel_tol=1e-7) if field == "signal_s" else timing.get(field, -1) >= sum(d[field] for d in decisions), tag + field + " sum")

    require(len(prefix_hashes) == len(noise_hashes) == 1 and all(isinstance(h, str) and len(h) == 64 for h in prefix_hashes | noise_hashes), "shared prefix/noise hashes")
    require(noise_hashes == {summary.get("initial_noise_sha256")}, "summary noise hash")
    require(all(len(v) == 1 and all(isinstance(h, str) and len(h) == 64 for h in v) for v in group_noise_hashes.values()), "shared per-segment noise hashes")
    if all(p in by_policy for p in policies):
        coarse, fine, warp, matched, alternating = (by_policy[p] for p in policies)
        require(all(d["mode"] == "coarse" for d in coarse["decisions"]), "fixed coarse modes")
        require(all(d["mode"] == "fine" for d in fine["decisions"]), "fixed fine modes")
        expected_groups = sorted(random.Random(lock["random_seed"]).sample([1, 2, 3, 4], warp["fine_count"]))
        require(matched["fine_count"] == warp["fine_count"] and matched["random_fine_groups"] == expected_groups
                and [d["group"] for d in matched["decisions"] if d["mode"] == "fine"] == expected_groups, "matched random schedule")
        require([d["mode"] for d in alternating["decisions"]] == ["fine", "coarse", "fine", "coarse"]
                and alternating["switch_count_decisions"] == 3, "alternating switches")
        fixed_regrets = [coarse["segments"][i]["reference_l1"] - fine["segments"][i]["reference_l1"] for i in range(4)]
    else:
        fixed_regrets = []
    return {"verdict": "PASS" if not errors else "FAIL", "errors": errors,
            "n_policies": len(complete), "n_segments": sum(len(r.get("segments", [])) for r in complete),
            "fixed_coarse_minus_fine_reference_l1_by_segment": fixed_regrets,
            "mean_fixed_coarse_minus_fine_reference_l1": sum(fixed_regrets) / 4 if fixed_regrets else None,
            "policy_summary": {p: {"fine_count": r["fine_count"], "mean_reference_l1": r["mean_reference_l1"],
                                   "total_s": r["timing"]["total_s"], "nfe": r["timing"]["nfe"]}
                               for p, r in by_policy.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "results/phase5c_online_pilot/results/phase5c_online_pilot")
    args = parser.parse_args()
    result = audit(args.base)
    output = ROOT / "results/phase5c_online_pilot/phase5c_pilot_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["verdict"] == "PASS" else 1)
