"""Phase 4.2: RQ4 monotonicity on the A100 with the Jev signal (zero GPU).

Uses the A100 Phase 3 run (results/a100_online_jev_phase3/results.jsonl), which contains
per-group decoded-pixel L1 for fixed_fine and fixed_coarse, and per-group Jev answers for
jev_routed. For each (prompt, group): fine_gap = refL1_fine - refL1_coarse (negative => fine
closer), label coarse_unsafe = fine_gap < 0. Correlate fine_gap with Jev difficulty / drift /
motion and report Spearman, AUROC, and terciles. Writes results/a100_online_jev_phase3/rq4_jev_a100.json.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "a100_online_jev_phase3", "results.jsonl")
OUT = os.path.join(ROOT, "results", "a100_online_jev_phase3", "rq4_jev_a100.json")


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
    rs = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rs += avg
        i = j + 1
    return (rs - pos * (pos + 1) / 2.0) / (pos * neg)


def terciles(vals, gaps):
    idx = sorted(range(len(vals)), key=lambda i: vals[i])
    n = len(idx)
    out = []
    for t in range(3):
        lo, hi = t * n // 3, (t + 1) * n // 3
        sl = idx[lo:hi]
        out.append(round(sum(gaps[i] for i in sl) / len(sl), 5) if sl else None)
    return out


def main():
    rows = [json.loads(l) for l in open(RAW, encoding="utf-8") if l.strip()]
    by = {}
    for r in rows:
        if r.get("status") != "complete":
            continue
        by.setdefault(r["prompt_idx"], {})[r["policy"]] = r

    recs = []
    for p, pols in by.items():
        if "fixed_fine" not in pols or "fixed_coarse" not in pols or "jev_routed" not in pols:
            continue
        fine_seg = {s["group"]: s["reference_l1"] for s in pols["fixed_fine"].get("segments", [])}
        coarse_seg = {s["group"]: s["reference_l1"] for s in pols["fixed_coarse"].get("segments", [])}
        jev = {}
        for d in pols["jev_routed"].get("decisions", []):
            a = d.get("jev_answer")
            if a and "answers" in a:
                jev[d["group"]] = a["answers"]
        for g in sorted(set(fine_seg) & set(coarse_seg) & set(jev)):
            gap = fine_seg[g] - coarse_seg[g]
            ans = jev[g]
            recs.append({"prompt": p, "group": g, "fine_gap": gap,
                         "coarse_unsafe": 1 if gap < 0 else 0,
                         "difficulty": ans["difficulty"]["score"],
                         "drift": ans["coarse_drift"]["noul"],
                         "motion": {"static": 0, "moderate": 1, "fast": 2, "scene_change": 3}[ans["motion"]["choice"]]})

    out = {"phase": "RQ4-with-Jev-A100", "n_segments": len(recs),
           "note": "fine_gap = refL1_fine - refL1_coarse (negative => fine closer). A100 Phase 3 data."}
    for key in ("difficulty", "drift", "motion"):
        vals = [r[key] for r in recs]
        gaps = [r["fine_gap"] for r in recs]
        labels = [r["coarse_unsafe"] for r in recs]
        out[key] = {"spearman_vs_fine_gap": spearman(vals, gaps),
                    "auroc_coarse_unsafe": auroc(vals, labels),
                    "terciles_mean_fine_gap": terciles(vals, gaps)}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Phase 4.2 RQ4-with-Jev (A100) written:", os.path.relpath(OUT, ROOT))
    print("n_segments:", len(recs))
    for key in ("difficulty", "drift", "motion"):
        v = out[key]
        rho = v["spearman_vs_fine_gap"]
        auc = v["auroc_coarse_unsafe"]
        print(f"  {key:10s} rho={rho:+.3f}  AUROC={auc:.3f}  terciles={v['terciles_mean_fine_gap']}")


if __name__ == "__main__":
    main()
