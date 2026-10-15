"""Build self-contained Kaggle scripts for the locked Phase 5B held-out split."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config/phase5_prompt_manifest.json"
TEMPLATE = ROOT / "jobs/phase5b_reference_calibration_b/phase5b_reference_calibration.py"
METADATA = ROOT / "jobs/phase5b_reference_calibration_b/kernel-metadata.json"
BATCH_SIZE = 3


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"expected one occurrence of {old!r}, found {count}")
    return source.replace(old, new, 1)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    held_out = manifest["held_out"]
    seeds = manifest["held_out_seeds"]
    assert len(held_out) == 24 and seeds == [17, 29, 41]
    template = TEMPLATE.read_text(encoding="utf-8")
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))

    for start in range(0, len(held_out), BATCH_SIZE):
        batch_id = start // BATCH_SIZE
        name = f"phase5b_heldout_{batch_id:02d}"
        slug = f"phase-5b-heldout-{batch_id:02d}"
        batch = held_out[start : start + BATCH_SIZE]
        prompts = [item["prompt"] for item in batch]
        source = template
        source = replace_once(
            source,
            'OUT_DIR = Path("/kaggle/working/results/phase5b_reference_calibration_b")',
            f'OUT_DIR = Path("/kaggle/working/results/{name}")',
        )
        prompt_start = source.index("PROMPTS = [")
        prompt_end = source.index("BOOTSTRAP_REPS =", prompt_start)
        prompt_block = (
            f"PROMPTS = {json.dumps(prompts, ensure_ascii=False, indent=4)}\n"
            f"SEEDS = {json.dumps(seeds)}\n"
            f"START_INDEX = {start}\n"
            "CASES = [(START_INDEX + i, prompt, seed) "
            "for i, prompt in enumerate(PROMPTS) for seed in SEEDS]\n"
            "HORIZONS = [57]\n"
        )
        source = source[:prompt_start] + prompt_block + source[prompt_end:]
        source = replace_once(
            source,
            '"phase": "phase5b-reference-calibration-b"',
            f'"phase": "{name.replace("_", "-")}"',
        )
        source = replace_once(
            source,
            "for prompt_idx, prompt in enumerate(PROMPTS):",
            "for prompt_idx, prompt, seed in CASES:",
        )
        source = replace_once(source, 'f"prompt{prompt_idx}_seed{SEED}_h{horizon}"', 'f"prompt{prompt_idx}_seed{seed}_h{horizon}"')
        source = source.replace(', SEED, prompt_embedding)', ', seed, prompt_embedding)')
        source = source.replace('"seed": SEED,', '"seed": seed,')
        source = replace_once(source, '"split": "calibration"', '"split": "held_out"')
        source = replace_once(source, '"seed": seed, "horizons": HORIZONS', '"seeds": SEEDS, "horizons": HORIZONS')
        source = source.replace('len(PROMPTS) * len(HORIZONS)', 'len(CASES) * len(HORIZONS)')
        source = source.replace('"calibration split only"', '"held-out split; no threshold fitted in this kernel"')
        source = replace_once(
            source,
            '        complete = []\n        for prompt_idx, prompt, seed in CASES:',
            '        complete = []\n        prompt_cache = {}\n        for prompt_idx, prompt, seed in CASES:',
        )
        source = replace_once(
            source,
            '                    text_started = time.time()\n'
            '                    prompt_embedding = encode_prompt(prompt)\n'
            '                    text_encode_s = time.time() - text_started',
            '                    if prompt not in prompt_cache:\n'
            '                        text_started = time.time()\n'
            '                        prompt_cache[prompt] = encode_prompt(prompt)\n'
            '                        text_encode_s = time.time() - text_started\n'
            '                    else:\n'
            '                        text_encode_s = 0.0\n'
            '                    prompt_embedding = prompt_cache[prompt]',
        )
        source = source.replace(
            '"prompt_embedding_cached_across_modes": True,',
            '"prompt_embedding_cached_across_modes_and_seeds": True,',
        )
        source = source.replace('Phase 3 / E2: real signal calibration on Kaggle T4.', 'Phase 5B: held-out reference-regret signal collection on Kaggle T4.')
        source = source.replace(
            'seed. A temporal-stability proxy gap labels coarse-unsafe segments; it is not\n'
            'a ground-truth quality label. Calibration prompts fit the label margin and\n'
            'Youden thresholds, while held-out prompts report AUROC and bootstrap CIs.',
            'seed and initial noise. Each scored segment uses decoded-pixel regret against\n'
            'an internal c1-1 step-4 reference. Signals use only preceding coarse context.\n'
            'Thresholds and grouped AUROCs are computed by the local Phase 5B audit.',
        )
        source = source.replace(
            '# Calibration batch B: six locked prompts from config/phase5_prompt_manifest.json.',
            '# Locked held-out prompts from config/phase5_prompt_manifest.json.',
        )
        source = source.replace('Phase 5A does not score AUROC', 'this kernel does not score AUROC')
        if "SEED" in source.replace("SEEDS", ""):
            raise ValueError(f"unconverted seed reference in {name}")

        job_dir = ROOT / "jobs" / name
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "phase5b_heldout.py").write_text(source, encoding="utf-8")
        job_metadata = dict(metadata)
        job_metadata.update(
            id=f"ajjisan/{slug}",
            title=f"Phase 5B Heldout {batch_id:02d}",
            code_file="phase5b_heldout.py",
        )
        (job_dir / "kernel-metadata.json").write_text(
            json.dumps(job_metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{name}: prompts {start}-{start + len(batch) - 1}, {len(batch) * len(seeds)} cases")


if __name__ == "__main__":
    main()
