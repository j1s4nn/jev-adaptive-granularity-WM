"""RQ4 audit: is the fine-vs-coarse benefit monotone in the uncertainty signal?

Combines the 81-frame paired segments (results/rq4_monotonicity) with the
57-frame F4-lite paired segments (results/f4lite) and computes the
uncertainty-benefit relationship.

Reads  results/rq4_monotonicity/results.jsonl
       results/f4lite/results.jsonl
       config/rq3_threshold.json  (57-frame warp traces)
Writes results/rq4_monotonicity/rq4_audit.json
"""

import hashlib
import json
import os
import statistics
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RQ4_JSONL = os.path.join(ROOT, "results", "rq4_monotonicity", "results.jsonl")
F4_JSONL = os.path.join(ROOT, "results", "f4lite", "results.jsonl")
RQ3_CFG = os.path.join(ROOT, "config", "rq3_threshold.json")
OUT_JSON = os.path.join(ROOT, "results", "rq4_monotonicity", "rq4_audit.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    mx, my = statistics.fmean(x), statistics.fmean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def main():
    points = []  # (signal, fine_gap, source)

    with open(RQ4_JSONL, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") == "complete" and r.get("kind") == "paired":
                for p in r["paired"]:
                    points.append({"signal": p["signal"], "fine_gap": p["fine_gap"], "source": "81f"})

    # 57-frame F4-lite paired segments
    with open(RQ3_CFG, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    traces = cfg["candidates"]["warp"]["per_prompt_trace"]
    ff, fc = {}, {}
    with open(F4_JSONL, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") != "complete":
                continue
            if r.get("policy") == "fixed_fine":
                ff[r["prompt_idx"]] = r["segments"]
            elif r.get("policy") == "fixed_coarse":
                fc[r["prompt_idx"]] = r["segments"]
    for idx in range(8):
        if idx in ff and idx in fc and f"p{idx}" in traces:
            tr = traces[f"p{idx}"]
            for g in range(4):
                points.append({"signal": tr[g],
                               "fine_gap": ff[idx][g]["reference_l1"] - fc[idx][g]["reference_l1"],
                               "source": "57f"})

    sig = [p["signal"] for p in points]
    gap = [p["fine_gap"] for p in points]
    rho = spearman(sig, gap) if len(points) > 2 else None

    # tercile bins
    order = sorted(range(len(points)), key=lambda i: points[i]["signal"])
    bins = []
    n = len(order)
    for b in range(3):
        lo, hi = b * n // 3, (b + 1) * n // 3
        idxs = order[lo:hi]
        if idxs:
            bins.append({"bin": b, "n": len(idxs),
                         "signal_mean": statistics.fmean(points[i]["signal"] for i in idxs),
                         "fine_gap_mean": statistics.fmean(points[i]["fine_gap"] for i in idxs)})

    if rho is None:
        verdict = "insufficient data"
    elif rho >= 0.3:
        verdict = "monotone (fine benefit grows with uncertainty)"
    elif rho <= -0.3:
        verdict = "anti-monotone (fine benefit shrinks with uncertainty)"
    else:
        verdict = "no monotone relationship"

    report = {"phase": "RQ4-monotonicity-audit", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": {"rq4_jsonl_sha256": sha256_file(RQ4_JSONL), "f4lite_jsonl_sha256": sha256_file(F4_JSONL)},
              "n_points": len(points), "n_81f": sum(1 for p in points if p["source"] == "81f"),
              "n_57f": sum(1 for p in points if p["source"] == "57f"),
              "spearman_rho": rho, "fine_gap_mean": statistics.fmean(gap), "signal_mean": statistics.fmean(sig),
              "bins": bins, "verdict": verdict, "points": points}
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("RQ4 audit:", os.path.relpath(OUT_JSON, ROOT))
    print(f"n_points={len(points)} (81f={report['n_81f']}, 57f={report['n_57f']})")
    print(f"Spearman rho(signal, fine_gap) = {rho:+.4f}" if rho is not None else "rho: n/a")
    print(f"mean fine_gap = {report['fine_gap_mean']:+.5f}")
    for b in bins:
        print(f"  bin {b['bin']}: n={b['n']} signal={b['signal_mean']:.3f} fine_gap={b['fine_gap_mean']:+.5f}")
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
