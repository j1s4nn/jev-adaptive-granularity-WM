"""Phase 0.1: RQ4 monotonicity re-analyzed with the Jev signal (zero GPU).

Uses the EXISTING results/jev_signal/jev_answers.jsonl (80 paired segments, 57f + 81f)
which already contains Jev difficulty/drift/motion plus fine_gap
(fine_gap = regret_fine - regret_coarse; negative => fine is closer).

Reports, for each candidate signal (Jev difficulty, Jev coarse_drift, warp, motion):
  - Spearman rho vs fine_gap (all 80 and per horizon),
  - mean fine_gap per tercile of the signal,
  - AUROC for the label coarse_unsafe.

Writes results/rq4_jev/jev_rq4_audit.json. Never modifies prior artifacts.
"""

import json
import os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "results", "jev_signal", "jev_answers.jsonl")
OUT_DIR = os.path.join(ROOT, "results", "rq4_jev")
OUT = os.path.join(OUT_DIR, "jev_rq4_audit.json")


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else None


def auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def terciles(values, gaps):
    idx = sorted(range(len(values)), key=lambda i: values[i])
    n = len(idx)
    out = []
    for t in range(3):
        lo = t * n // 3
        hi = (t + 1) * n // 3
        sl = idx[lo:hi]
        if not sl:
            out.append(None)
            continue
        out.append({"n": len(sl), "signal_range": [values[sl[0]], values[sl[-1]]],
                    "mean_fine_gap": sum(gaps[i] for i in sl) / len(sl)})
    return out


def main():
    rows = [json.loads(l) for l in open(IN, encoding="utf-8") if l.strip()]
    for r in rows:
        a = r["answer"]["answers"]
        r["_diff"] = a["difficulty"]["score"]
        r["_drift"] = a["coarse_drift"]["noul"]
        r["_motion"] = {"static": 0, "moderate": 1, "fast": 2, "scene_change": 3}[a["motion"]["choice"]]

    def analyze(subset, name):
        gaps = [r["fine_gap"] for r in subset]
        labels = [r["coarse_unsafe"] for r in subset]
        res = {"n": len(subset)}
        for key in ("_diff", "_drift", "warp", "_motion"):
            vals = [r[key] for r in subset]
            res[key] = {
                "spearman_vs_fine_gap": spearman(vals, gaps),
                "auroc_coarse_unsafe": auroc(vals, labels),
                "terciles": terciles(vals, gaps),
            }
        return res

    audit = {
        "phase": "RQ4-monotonicity-with-Jev",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "results/jev_signal/jev_answers.jsonl",
        "note": "fine_gap = regret_fine - regret_coarse; negative => fine closer. Warp is the prior (T4) signal; Jev is the new signal.",
        "all": analyze(rows, "all"),
        "h57": analyze([r for r in rows if r["horizon"] == 57], "57f"),
        "h81": analyze([r for r in rows if r["horizon"] == 81], "81f"),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, default=str)

    print("RQ4-with-Jev audit written:", os.path.relpath(OUT, ROOT))
    for scope in ("all", "h57", "h81"):
        s = audit[scope]
        print(f"\n== {scope} (n={s['n']}) ==")
        for key in ("_diff", "_drift", "warp", "_motion"):
            v = s[key]
            rho = v["spearman_vs_fine_gap"]
            auc = v["auroc_coarse_unsafe"]
            rho_s = f"{rho:+.3f}" if rho is not None else "  -  "
            auc_s = f"{auc:.3f}" if auc is not None else "  -  "
            print(f"  {key:8s} rho={rho_s}  AUROC={auc_s}")
        print("  terciles by Jev difficulty:", [(round(t['mean_fine_gap'], 4) if t else None) for t in s["_diff"]["terciles"]])


if __name__ == "__main__":
    main()
