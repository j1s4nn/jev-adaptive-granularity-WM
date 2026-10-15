"""Create the locked Phase 5 prompt manifest from the bundled VBench list.

The selection is deterministic and excludes every prompt used by Phases 2--4.
The generated JSON is a protocol artifact, not an experiment result.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/phase3_calibration_rerun/rcm/evaluation/vbench_text2video/prompts.json"
OUT = ROOT / "config/phase5_prompt_manifest.json"

USED = {
    "In a still frame, a stop sign",
    "A tranquil tableau of alley",
    "A person is taking a shower",
    "a cat drinking water",
    "Splash of turquoise water in extreme slow motion, alpha channel included.",
    "The Bund Shanghai, with an intense shaking effect",
    "A beautiful coastal beach in spring, waves lapping on sand, watercolor painting",
    "A cyclist rides quickly along a forest trail in bright sunlight",
}

# These are VBench dimensions, not labels inferred from the text.
CATEGORIES = ("object_class", "scene", "dynamic_degree", "human_action")


def main() -> None:
    records = json.loads(SOURCE.read_text(encoding="utf-8"))
    candidates = [r for r in records if r.get("prompt") not in USED]
    buckets = {category: [] for category in CATEGORIES}
    for record in candidates:
        dims = set(record.get("dimensions", []))
        category = next((c for c in CATEGORIES if c in dims), None)
        if category is not None:
            buckets[category].append(record)
    for values in buckets.values():
        values.sort(key=lambda r: r["prompt"])

    calibration = []
    held_out = []
    for category in CATEGORIES:
        calibration.extend(buckets[category][:3])
        held_out.extend(buckets[category][3:9])

    selected = calibration + held_out
    if len(calibration) != 12 or len(held_out) != 24:
        raise RuntimeError(f"unexpected selection sizes: {len(calibration)} / {len(held_out)}")
    if len({r["prompt"] for r in selected}) != len(selected):
        raise RuntimeError("duplicate prompt in manifest")

    def serialise(records, split):
        return [
            {
                "manifest_id": f"{split}_{i:02d}",
                "split": split,
                "prompt": r["prompt"],
                "vbench_dimensions": r.get("dimensions", []),
                "source_index": records_source_index[r["prompt"]],
            }
            for i, r in enumerate(records)
        ]

    records_source_index = {r["prompt"]: i for i, r in enumerate(records)}
    manifest = {
        "manifest_version": "phase5-v1",
        "source": str(SOURCE.relative_to(ROOT)),
        "selection_rule": "first sorted candidates per VBench dimension after excluding prior pilot prompts",
        "excluded_prior_prompts": sorted(USED),
        "calibration_seeds": [17],
        "held_out_seeds": [17, 29, 41],
        "horizons_requested": [45, 57, 81],
        "categories": list(CATEGORIES),
        "calibration": serialise(calibration, "calibration"),
        "held_out": serialise(held_out, "held_out"),
        "status": "locked_protocol_only; no results",
    }
    OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(calibration)} calibration, {len(held_out)} held-out)")


if __name__ == "__main__":
    main()

