"""RQ3 audit: controller ablation, regret, transition cost, and Lambda/R sweeps.

Reads  results/rq3_controller/results.jsonl
Writes results/rq3_controller/rq3_audit.json
"""

import hashlib
import json
import os
import statistics
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_JSONL = os.path.join(ROOT, "results", "rq3_controller", "results.jsonl")
OUT_JSON = os.path.join(ROOT, "results", "rq3_controller", "rq3_audit.json")

CONTROLLERS = ["threshold_only", "hysteresis_only", "recovery_only", "full", "threshold_only_noisy", "full_noisy"]
TAU_PSI = -4.416192173957825
SIGMA_U = 3.915166421304675
RECOVERY_R = 2

WARP_TRACE = {
    0: [6.4296, 6.1398, 6.5066, 6.0016], 1: [0.7756, 2.4940, 2.4610, 1.4452],
    2: [1.3747, 1.0696, 2.0111, 1.0187], 3: [7.3146, 3.5810, 1.2681, 1.2173],
    4: [20.3692, 5.2514, 5.7266, 6.6592], 5: [7.8138, 2.4693, 3.4919, 5.7752],
    6: [7.9946, 3.5061, 2.6829, 2.2830], 7: [10.9930, 8.5829, 7.3928, 9.0702],
}


def mean(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.fmean(v) if v else None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def sw(seq):
    return sum(seq[i] != seq[i - 1] for i in range(1, len(seq)))


def threshold_only(psi):
    return ["F" if p >= TAU_PSI else "C" for p in psi]


def hysteresis_only(psi, lam):
    m, o = "C", []
    for p in psi:
        if m == "C" and p >= TAU_PSI:
            m = "F"
        elif m == "F" and p <= TAU_PSI - lam:
            m = "C"
        o.append(m)
    return o


def recovery_only(psi, r):
    m, rr, o = "C", 0, []
    for p in psi:
        if p >= TAU_PSI:
            m, rr = "F", r
        elif rr > 0:
            m, rr = "F", rr - 1
        else:
            m = "C"
        o.append(m)
    return o


def full_controller(psi, lam, r):
    m, rr, o = "C", 0, []
    for p in psi:
        if p >= TAU_PSI:
            m, rr = "F", r
        elif rr > 0:
            m, rr = "F", rr - 1
        elif p <= TAU_PSI - lam:
            m = "C"
        o.append(m)
    return o


def main():
    rows = []
    with open(IN_JSONL, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("status") == "complete" and r.get("controller") in CONTROLLERS:
                    rows.append(r)

    report = {"phase": "RQ3-controller-ablation-audit", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "input_sha256": sha256_file(IN_JSONL), "n_rollouts": len(rows), "by_controller": {}, "paired": {}}

    for c in CONTROLLERS:
        sub = [r for r in rows if r["controller"] == c]
        entry = {"n": len(sub),
                 "switch_count_mean": mean([r["switch_count"] for r in sub]),
                 "fine_fraction_mean": mean([r["fine_fraction"] for r in sub]),
                 "mean_reference_l1": mean([r["mean_reference_l1"] for r in sub]),
                 "terminal_reference_l1": mean([r["terminal_reference_l1"] for r in sub]),
                 "auh_reference_l1": mean([r["auh_reference_l1"] for r in sub]),
                 "dit_s": mean([r["timing"]["dit_s"] for r in sub]),
                 "total_s": mean([r["timing"]["total_s"] for r in sub]),
                 "switch_adjacent_l1": mean([r["switch_adjacent_l1"] for r in sub]),
                 "interior_adjacent_l1": mean([r["interior_adjacent_l1"] for r in sub]),
                 "mode_run_length_mean": mean([statistics.fmean(r["mode_runs"]) for r in sub])}
        report["by_controller"][c] = entry

    # paired: full vs threshold_only (real) and full_noisy vs threshold_only_noisy
    for base, treat, tag in [("threshold_only", "full", "real"), ("threshold_only_noisy", "full_noisy", "noisy")]:
        ds, dr, dt = [], [], []
        for idx in range(8):
            a = next((r for r in rows if r["prompt_idx"] == idx and r["controller"] == base), None)
            b = next((r for r in rows if r["prompt_idx"] == idx and r["controller"] == treat), None)
            if a and b:
                ds.append(b["switch_count"] - a["switch_count"])
                dr.append(b["auh_reference_l1"] - a["auh_reference_l1"])
                dt.append(b["terminal_reference_l1"] - a["terminal_reference_l1"])
        report["paired"][tag] = {"switch_diff_mean": mean(ds), "auh_regret_diff_mean": mean(dr),
                                 "terminal_regret_diff_mean": mean(dt), "n": len(ds)}

    # offline Lambda and R sweeps on the real traces
    lam_sweep, r_sweep = {}, {}
    for mult in [0.5, 1.0, 2.0, 4.0]:
        lam = mult * SIGMA_U
        tot = 0
        for idx, warp in WARP_TRACE.items():
            psi = [-w for w in warp]
            tot += sw(hysteresis_only(psi, lam))
        lam_sweep[str(mult)] = {"lambda": lam, "total_switches_hysteresis_only": tot,
                                "total_switches_full": sum(sw(full_controller([-w for w in w2], lam, RECOVERY_R)) for w2 in WARP_TRACE.values())}
    for r in [1, 2, 3]:
        tot = 0
        for idx, warp in WARP_TRACE.items():
            psi = [-w for w in warp]
            tot += sw(recovery_only(psi, r))
        r_sweep[str(r)] = {"R": r, "total_switches_recovery_only": tot,
                           "total_switches_full": sum(sw(full_controller([-w for w in w2], 2.0 * SIGMA_U, r)) for w2 in WARP_TRACE.values())}
    report["lambda_sweep"] = lam_sweep
    report["recovery_sweep"] = r_sweep

    # verdict
    real = report["paired"]["real"]
    noisy = report["paired"]["noisy"]
    rq3_real = (real["switch_diff_mean"] is not None and real["switch_diff_mean"] <= 0
                and real["auh_regret_diff_mean"] is not None and real["auh_regret_diff_mean"] <= 1e-4)
    rq3_noisy = (noisy["switch_diff_mean"] is not None and noisy["switch_diff_mean"] < 0
                 and noisy["auh_regret_diff_mean"] is not None and noisy["auh_regret_diff_mean"] <= 1e-3)
    report["verdict"] = {
        "rq3_supported_real_signal": rq3_real,
        "rq3_supported_noisy_stress": rq3_noisy,
        "note": "RQ3 supported iff recovery+hysteresis reduces switches without worsening terminal/AUH regret at matched fine budget.",
    }

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("RQ3 audit:", os.path.relpath(OUT_JSON, ROOT))
    print(f"{'controller':<22}{'switches':>9}{'fine_frac':>10}{'meanRef':>9}{'termRef':>9}{'AUHRef':>9}{'swAdj':>8}{'intAdj':>8}")
    for c in CONTROLLERS:
        e = report["by_controller"][c]
        sa = e["switch_adjacent_l1"]
        ia = e["interior_adjacent_l1"]
        print(f"{c:<22}{e['switch_count_mean']:>9.2f}{e['fine_fraction_mean']:>10.2f}{e['mean_reference_l1']:>9.4f}"
              f"{e['terminal_reference_l1']:>9.4f}{e['auh_reference_l1']:>9.4f}"
              f"{(sa if sa is not None else float('nan')):>8.4f}{(ia if ia is not None else float('nan')):>8.4f}")
    print()
    for tag in ["real", "noisy"]:
        p = report["paired"][tag]
        print(f"paired {tag}: switch_diff={p['switch_diff_mean']:+.2f} AUH_regret_diff={p['auh_regret_diff_mean']:+.5f} terminal_diff={p['terminal_regret_diff_mean']:+.5f}")
    print("Lambda sweep (hysteresis_only switches):", {k: v["total_switches_hysteresis_only"] for k, v in lam_sweep.items()})
    print("R sweep (recovery_only switches):", {k: v["total_switches_recovery_only"] for k, v in r_sweep.items()})
    print("verdict:", report["verdict"])


if __name__ == "__main__":
    main()
