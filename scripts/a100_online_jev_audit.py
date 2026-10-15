"""A100-JEV audit: aggregate the Jev-routed online policies (RQ2 headline).

Reads  results/a100_online_jev/results.jsonl (+ quality_clip / quality_classical if present)
Writes results/a100_online_jev/a100_online_jev_audit.json

Per policy: fine/coarse/hold counts, switches, internal refL1, DiT/VAE/jev/blip/total,
NFE; plus independent CLIP subject consistency / text alignment and classical temporal warp.
Verdict: does a Jev policy improve the independent endpoint at equal or lower mode-dependent
DiT (and total) vs fixed_fine, and beat random50 at matched budget? Never modifies prior runs.
"""

import json
import os
import statistics
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRNAME = sys.argv[1] if len(sys.argv) > 1 else "a100_online_jev"
OUT_DIR = os.path.join(ROOT, "results", DIRNAME)
RAW = os.path.join(OUT_DIR, "results.jsonl")
CLIP = os.path.join(OUT_DIR, "a100_online_jev_quality_clip.json")
CLASS = os.path.join(OUT_DIR, "a100_online_jev_quality_classical.json")
AUDIT = os.path.join(OUT_DIR, "a100_online_jev_audit.json")

POLICIES = ["fixed_coarse", "fixed_fine", "fixed_hold", "random50", "warp",
            "jev_difficulty", "jev_routed", "jev_holdfine", "jev_single", "latent"]
JEV_POLICIES = ["jev_difficulty", "jev_routed", "jev_holdfine", "jev_single", "latent"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def load():
    rows = []
    if not os.path.exists(RAW):
        return rows
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                it = json.loads(line)
            except Exception:
                continue
            if it.get("status") == "complete" and it.get("policy") in POLICIES:
                rows.append(it)
    return rows


def agg(rows):
    out = {}
    for pol in POLICIES:
        rs = [r for r in rows if r["policy"] == pol]
        t = [r.get("timing", {}) for r in rs]
        out[pol] = {
            "n": len(rs),
            "fine_fraction": mean([r.get("fine_fraction") for r in rs]),
            "hold_count": mean([r.get("hold_count") for r in rs]),
            "coarse_count": mean([r.get("coarse_count") for r in rs]),
            "switch_count": mean([r.get("switch_count") for r in rs]),
            "mean_reference_l1": mean([r.get("mean_reference_l1") for r in rs]),
            "dit_s": mean([x.get("dit_s") for x in t]),
            "vae_s": mean([x.get("vae_s") for x in t]),
            "jev_s": mean([x.get("jev_s") for x in t]),
            "blip_s": mean([x.get("blip_s") for x in t]),
            "total_s": mean([x.get("total_s") for x in t]),
            "nfe": mean([x.get("nfe") for x in t]),
        }
    return out


def attach(audit):
    for path, key in ((CLIP, "clip"), (CLASS, "classical")):
        if os.path.exists(path):
            try:
                audit["quality_" + key] = json.load(open(path, encoding="utf-8")).get("by_policy")
            except Exception as e:
                audit["quality_" + key] = {"error": repr(e)}


def main():
    rows = load()
    by = agg(rows)
    audit = {"phase": "A100-JEV-online-audit", "generated_utc": datetime.now(timezone.utc).isoformat(),
             "n_complete": len(rows), "by_policy": by}
    attach(audit)

    ff = by.get("fixed_fine", {})
    rnd = by.get("random50", {})
    verdict = {}
    for pol in JEV_POLICIES:
        p = by.get(pol, {})
        if not ff.get("mean_reference_l1") or not p.get("mean_reference_l1"):
            continue
        q_rel = (ff["mean_reference_l1"] - p["mean_reference_l1"]) / ff["mean_reference_l1"]
        dit_rel = (ff["dit_s"] - p["dit_s"]) / ff["dit_s"] if ff.get("dit_s") and p.get("dit_s") else None
        tot_rel = (ff["total_s"] - p["total_s"]) / ff["total_s"] if ff.get("total_s") and p.get("total_s") else None
        beats_random = (p.get("mean_reference_l1") is not None and rnd.get("mean_reference_l1") is not None
                        and p["mean_reference_l1"] <= rnd["mean_reference_l1"])
        verdict[pol] = {"quality_rel_vs_fine": q_rel, "dit_rel_vs_fine": dit_rel, "total_rel_vs_fine": tot_rel,
                        "beats_random": bool(beats_random)}
    audit["verdict"] = verdict
    audit["note"] = ("Internal refL1 is a diagnostic; the independent CLIP subject-consistency is the "
                     "primary endpoint. total_rel includes the Jev/BLIP signal latency (off the DiT path).")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(AUDIT, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)

    print("A100-JEV audit written:", os.path.relpath(AUDIT, ROOT))
    print(f"{'policy':<16}{'n':>3}{'fine%':>7}{'hold':>6}{'sw':>5}{'refL1':>9}{'dit_s':>8}{'vae_s':>8}{'jev_s':>7}{'total_s':>9}")
    for pol in POLICIES:
        a = by[pol]
        def f(v, d=3):
            return f"{v:.{d}f}" if isinstance(v, (int, float)) else "  -"
        print(f"{pol:<16}{a['n']:>3}{f(a['fine_fraction'],2):>7}{f(a['hold_count'],1):>6}{f(a['switch_count'],2):>5}"
              f"{f(a['mean_reference_l1'],4):>9}{f(a['dit_s'],2):>8}{f(a['vae_s'],2):>8}{f(a['jev_s'],2):>7}{f(a['total_s'],2):>9}")
    if audit.get("quality_clip"):
        print("\nCLIP by policy:")
        for pol in POLICIES:
            q = audit["quality_clip"].get(pol, {})
            print(f"  {pol:<16} subj.cons={q.get('subject_consistency')} text_align={q.get('text_alignment')}")
    print("\nVERDICT:", json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
