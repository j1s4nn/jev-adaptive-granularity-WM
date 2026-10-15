"""Phase 3.2: paired bootstrap CIs on the independent endpoint + cost deltas.

Reads results/a100_online_jev_phase3/results.jsonl and
      results/a100_online_jev_phase3/a100_online_jev_quality_clip.json
Writes results/a100_online_jev_phase3/phase3_ci.json

For each Jev policy vs fixed_fine: paired per-prompt deltas of
  CLIP subject consistency, CLIP text alignment, clip_quality,
  DiT seconds, total seconds,
with 10k-resample bootstrap 95% CIs. Purely offline (no GPU).
"""

import json
import os
import random
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "results", "a100_online_jev_phase3")
RAW = os.path.join(DIR, "results.jsonl")
CLIP = os.path.join(DIR, "a100_online_jev_quality_clip.json")
OUT = os.path.join(DIR, "phase3_ci.json")

BASE = "fixed_fine"
JEV = ["jev_single", "jev_routed", "latent"]
NBOOT = 10000
random.seed(20261013)


def load_cost():
    rows = {}
    for line in open(RAW, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("status") == "complete" and r.get("policy") in JEV + [BASE]:
            rows[(r["prompt_idx"], r["policy"])] = {
                "dit_s": r["timing"]["dit_s"], "total_s": r["timing"]["total_s"],
                "vae_s": r["timing"]["vae_s"], "jev_s": r["timing"].get("jev_s", 0.0)}
    return rows


def load_clip():
    d = json.load(open(CLIP, encoding="utf-8"))
    out = {}
    for stem, v in d["per_video"].items():
        out[(v["prompt_idx"], v["policy"])] = {
            "subject_consistency": v.get("subject_consistency"),
            "text_alignment": v.get("text_alignment"),
            "clip_quality": v.get("clip_quality")}
    return out


def boot_ci(deltas):
    if len(deltas) < 2:
        return {"mean": (st.fmean(deltas) if deltas else None), "ci95": None, "n": len(deltas)}
    n = len(deltas)
    means = []
    for _ in range(NBOOT):
        s = 0.0
        for _ in range(n):
            s += deltas[random.randrange(n)]
        means.append(s / n)
    means.sort()
    return {"mean": st.fmean(deltas),
            "ci95": [means[int(0.025 * NBOOT)], means[int(0.975 * NBOOT)]], "n": n}


def main():
    cost = load_cost()
    clip = load_clip()
    prompts = sorted(set(k[0] for k in cost))
    result = {"phase": "phase3-bootstrap-ci", "baseline": BASE, "n_prompts": len(prompts),
              "n_boot": NBOOT, "comparisons": {}}

    for pol in JEV:
        comp = {}
        for metric, src in (("subject_consistency", clip), ("text_alignment", clip),
                            ("clip_quality", clip), ("dit_s", cost), ("total_s", cost), ("vae_s", cost)):
            deltas = []
            for p in prompts:
                a = src.get((p, pol))
                b = src.get((p, BASE))
                if a is None or b is None:
                    continue
                va = a.get(metric) if isinstance(a, dict) else None
                vb = b.get(metric) if isinstance(b, dict) else None
                if va is None or vb is None:
                    continue
                deltas.append(va - vb)
            comp[metric] = boot_ci(deltas)
        result["comparisons"][pol] = comp

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Phase 3 CI written:", os.path.relpath(OUT, ROOT))
    print(f"n_prompts={len(prompts)}  baseline={BASE}")
    for pol in JEV:
        print(f"\n== {pol} vs {BASE} (paired delta, 95% CI) ==")
        for metric in ("subject_consistency", "text_alignment", "clip_quality", "dit_s", "total_s"):
            c = result["comparisons"][pol][metric]
            ci = c["ci95"]
            ci_s = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "n/a"
            m = c["mean"]
            m_s = f"{m:+.4f}" if m is not None else "  n/a "
            print(f"  {metric:<20} {m_s}  {ci_s}  (n={c['n']})")


if __name__ == "__main__":
    main()
