"""Build the F4-lite fresh-prompt manifest, disjoint from the Phase 5B manifest.

Reads the VBench prompt pool and the existing Phase 5B manifest, excludes every
already-used prompt, and selects a deterministic stratified sample of fresh prompts.

Usage:
  python scripts/make_f4lite_manifest.py --pool <prompts.json> --n-per-stratum 2
Writes config/f4lite_manifest.json
"""

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE5 = os.path.join(ROOT, "config", "phase5_prompt_manifest.json")
OUT = os.path.join(ROOT, "config", "f4lite_manifest.json")

STRATA = {
    "object_class": ["object_class"],
    "scene": ["scene", "background_consistency"],
    "motion": ["motion_smoothness", "dynamic_degree", "subject_consistency"],
    "human_action": ["human_action"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--n-per-stratum", type=int, default=2)
    args = ap.parse_args()

    with open(args.pool, "r", encoding="utf-8") as fh:
        pool = json.load(fh)
    with open(PHASE5, "r", encoding="utf-8") as fh:
        phase5 = json.load(fh)

    used = set(phase5.get("excluded_prior_prompts", []))
    for entry in phase5.get("calibration", []) + phase5.get("held_out", []):
        used.add(entry["prompt"])

    selected = []
    for stratum, dims in STRATA.items():
        picked = 0
        for item in sorted(pool, key=lambda x: x["prompt"]):
            if picked >= args.n_per_stratum:
                break
            prompt = item["prompt"]
            if prompt in used:
                continue
            if any(d in item.get("dimensions", []) for d in dims):
                selected.append({"stratum": stratum, "prompt": prompt, "dimensions": item.get("dimensions", [])})
                used.add(prompt)
                picked += 1
        if picked < args.n_per_stratum:
            print(f"[warn] stratum {stratum}: only {picked} fresh prompts found")

    manifest = {
        "manifest_version": "f4lite-v1",
        "source": os.path.basename(args.pool),
        "selection_rule": "first sorted fresh candidates per stratum, disjoint from phase5-v1",
        "n_per_stratum": args.n_per_stratum,
        "seed": 17,
        "horizon": 57,
        "prompts": selected,
        "status": "locked_protocol_only; no results",
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print("wrote", os.path.relpath(OUT, ROOT))
    for s in selected:
        print(f"  {s['stratum']:<13} {s['prompt']}")


if __name__ == "__main__":
    main()
