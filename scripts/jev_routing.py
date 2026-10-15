"""Compute the Jev routing traces (57f) for the Phase C hold-mode controller.

Routing rule: motion choice static -> hold, moderate -> coarse, fast/scene_change -> fine.
Writes config/jev_routing.json
"""

import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "results", "jev_signal", "jev_answers.jsonl")
OUT = os.path.join(ROOT, "config", "jev_routing.json")


def main():
    rows = [json.loads(l) for l in open(IN, encoding="utf-8") if l.strip()]
    routing = {}
    detail = {}
    for r in rows:
        if r["horizon"] != 57:
            continue
        a = r["answer"]["answers"]
        mc = a["motion"]["choice"]
        drift = a["coarse_drift"]["noul"]
        diff = a["difficulty"]["score"]
        if mc == "static":
            mode = "hold"
        elif mc == "moderate":
            mode = "coarse"
        else:
            mode = "fine"
        routing.setdefault(str(r["prompt_idx"]), {})[str(r["group"])] = mode
        detail.setdefault(str(r["prompt_idx"]), {})[str(r["group"])] = {"motion": mc, "drift": drift, "difficulty": diff}

    traces = {p: [routing[p][str(g)] for g in range(1, 5)] for p in routing}
    counts = collections.Counter(m for t in traces.values() for m in t)

    report = {"source": "results/jev_signal/jev_answers.jsonl (57f)", "rule": "static->hold, moderate->coarse, fast/scene_change->fine",
              "traces": traces, "mode_counts": dict(counts), "detail": detail}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("wrote", os.path.relpath(OUT, ROOT))
    print("mode counts:", dict(counts))
    for p in sorted(traces, key=int):
        print(f"  p{p}: {traces[p]}  ({detail[p]})")


if __name__ == "__main__":
    main()
