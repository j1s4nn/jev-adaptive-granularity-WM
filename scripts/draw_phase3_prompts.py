"""Phase 3: draw 24 fresh VBench prompts (6 per stratum), disjoint from all prior sets.

Excludes: A100-01 (12 cal + 12 held), A1 dev (3), Phase 5B (12 cal + 24 held + 8 prior),
F4-lite (8). Writes config/a100_phase3_manifest.json. Never modifies prior manifests.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = os.path.join(ROOT, "results", "phase3_calibration_rerun", "rcm", "evaluation",
                    "vbench_text2video", "prompts.json")
OUT = os.path.join(ROOT, "config", "a100_phase3_manifest.json")

STRATA = {
    "object_class": "object_class",
    "scene": "scene",
    "motion_smoothness": "motion_smoothness",
    "human_action": "human_action",
}
PER_STRATUM = 6


def load_used():
    used = set()

    def add(path, key_paths):
        p = os.path.join(ROOT, "config", path)
        if not os.path.exists(p):
            return
        d = json.load(open(p, encoding="utf-8"))
        for kp in key_paths:
            cur = d
            for k in kp:
                cur = cur.get(k, []) if isinstance(cur, dict) else []
            for it in cur:
                used.add(it["prompt"] if isinstance(it, dict) else it)

    add("phase5_prompt_manifest.json",
        [["excluded_prior_prompts"], ["calibration"], ["held_out"]])
    add("f4lite_manifest.json", [["prompts"]])
    add("a100_manifest.json",
        [["a1_dev_prompts"], ["calibration"], ["held_out"]])
    return used


def main():
    pool = json.load(open(POOL, encoding="utf-8"))
    used = load_used()
    print("excluded prompts:", len(used))

    manifest = {"manifest_version": "a100-phase3", "source": os.path.relpath(POOL, ROOT),
                "selection_rule": "fresh, disjoint from A100-01 / Phase 5B / F4-lite",
                "n_excluded": len(used), "horizons": [57], "seeds": [17, 29, 41],
                "strata": list(STRATA), "prompts": []}
    idx = 0
    for stratum, dim in STRATA.items():
        cands = [e["prompt"] for e in pool
                 if dim in e.get("dimensions", []) and e["prompt"] not in used]
        # stable, deterministic order
        cands = sorted(set(cands))[:PER_STRATUM]
        for p in cands:
            manifest["prompts"].append({"manifest_id": f"p3_{idx:02d}", "stratum": stratum, "prompt": p})
            idx += 1
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("wrote", os.path.relpath(OUT, ROOT), "with", len(manifest["prompts"]), "prompts")
    for it in manifest["prompts"]:
        print(f"  {it['manifest_id']} [{it['stratum']}] {it['prompt']}")


if __name__ == "__main__":
    main()
