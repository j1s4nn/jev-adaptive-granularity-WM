"""Phase 0.2: pick the Jev gate threshold for coarse-safety (zero GPU).

Sweeps thresholds on Jev difficulty and Jev coarse_drift against the label
coarse_unsafe and reports the Youden-optimal operating point (TPR, FPR, threshold),
plus the implied per-segment routing split. Writes results/rq4_jev/jev_signal_select.json.
"""

import json
import os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "results", "jev_signal", "jev_answers.jsonl")
OUT = os.path.join(ROOT, "results", "rq4_jev", "jev_signal_select.json")


def roc_curve(scores, labels):
    pts = sorted(range(len(scores)), key=lambda i: -scores[i])
    P = sum(labels)
    N = len(labels) - P
    out = []
    tp = fp = 0
    for i in pts:
        if labels[i] == 1:
            tp += 1
        else:
            fp += 1
        out.append({"threshold": scores[i], "tpr": tp / P if P else 0.0, "fpr": fp / N if N else 0.0})
    return out


def best_youden(curve):
    best = None
    for pt in curve:
        j = pt["tpr"] - pt["fpr"]
        if best is None or j > best["youden"]:
            best = {**pt, "youden": j}
    return best


def main():
    rows = [json.loads(l) for l in open(IN, encoding="utf-8") if l.strip()]
    labels = [r["coarse_unsafe"] for r in rows]
    out = {"phase": "Jev-gate-selection", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "n": len(rows), "positives": sum(labels)}

    for key, getter in (("difficulty", lambda r: r["answer"]["answers"]["difficulty"]["score"]),
                        ("coarse_drift", lambda r: r["answer"]["answers"]["coarse_drift"]["noul"])):
        scores = [getter(r) for r in rows]
        curve = roc_curve(scores, labels)
        best = best_youden(curve)
        # routing: "unsafe -> fine" if score >= threshold (higher score = harder/unsafe)
        n_fine = sum(1 for s in scores if s >= best["threshold"])
        out[key] = {"youden": best["youden"], "threshold": best["threshold"],
                    "tpr": best["tpr"], "fpr": best["fpr"],
                    "n_flagged_fine": n_fine, "n_total": len(scores),
                    "fine_fraction": n_fine / len(scores)}
        print(f"{key}: Youden={best['youden']:.3f} thr={best['threshold']:.3f} "
              f"TPR={best['tpr']:.3f} FPR={best['fpr']:.3f} fine_frac={n_fine/len(scores):.2f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("written:", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
