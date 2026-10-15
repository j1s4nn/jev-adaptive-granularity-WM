"""
E1: Fixed-Mode Baseline Experiment
===================================

INPUT:
- 100 VBench-T2V prompts
- c1-1 checkpoint (frame-wise, K=1, S={1,2,4})
- c3-3 checkpoint (chunk-wise, K=3, S=4)
- 3 seeds per prompt

OUTPUT:
- raw/fixed_fine_step{1,2,4}.jsonl
- raw/fixed_coarse_step4.jsonl
- raw/compute_costs.json
- raw/unsafe_labels.jsonl
- figures/fig_error_curves.pdf
- tables/tab_baseline.tex

Run configuration: This script runs ONE mode (pass via env var MODE=c1-1-step2)
Submit 4 jobs in parallel for full E1 sweep.
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/kaggle/working')

from src.causal_rcm import load_causal_rcm_checkpoints, CausalRCMInference


# VBench-T2V prompts (100 selected)
VBENCH_PROMPTS = [
    "A cat sitting on a windowsill watching birds",
    "Ocean waves crashing on a beach at sunset",
    "A person walking through a autumn forest",
    "Time-lapse of clouds moving across blue sky",
    "A coffee cup on a table, steam rising",
    # ... (placeholder - full 100 prompts would come from VBench dataset)
    # For Phase 0/E1 testing, we'll use 20 prompts
]

# Extend to 100 with variations
VBENCH_PROMPTS = VBENCH_PROMPTS[:5] * 20  # Repeat 5 prompts 20 times for testing


def load_prompts(subset_size: int = 100) -> List[str]:
    """Load VBench prompts."""
    return VBENCH_PROMPTS[:subset_size]


def compute_fvd(generated: torch.Tensor, reference_dist: torch.Tensor) -> float:
    """
    Placeholder FVD computation.
    Real implementation requires pretrained I3D model.
    """
    # Dummy FVD for now
    return float(torch.randn(1).abs() * 50)


def compute_lpips(frame1: torch.Tensor, frame2: torch.Tensor) -> float:
    """
    Placeholder LPIPS (perceptual similarity).
    Real implementation requires lpips library.
    """
    # Dummy LPIPS
    return float(torch.randn(1).abs() * 0.3)


def compute_psnr(frame1: torch.Tensor, frame2: torch.Tensor) -> float:
    """Placeholder PSNR."""
    mse = torch.nn.functional.mse_loss(frame1, frame2)
    if mse == 0:
        return 100.0
    return float(20 * torch.log10(1.0 / torch.sqrt(mse)))


def measure_compute_cost(mode: str, num_frames: int) -> Dict:
    """
    Measure GPU time and FLOPs for a rollout.

    Returns:
        Dict with gpu_ms, flops_estimate
    """
    # Placeholder - real measurement requires torch.cuda.Event timing
    K = 1 if 'c1-1' in mode else 3
    S = int(mode.split('step')[-1]) if 'step' in mode else 2

    # Rough estimates based on Causal-rCM paper
    base_ms_per_frame = 230 if K == 1 else 400

    return {
        "mode": mode,
        "K": K,
        "S": S,
        "gpu_ms_per_step": base_ms_per_frame,
        "total_gpu_ms": base_ms_per_frame * (num_frames // K),
        "flops_estimate": 1e12  # Placeholder
    }


def run_baseline_experiment(
    mode: str,
    prompts: List[str],
    seeds: List[int],
    rollout_lengths: List[int],
    output_dir: Path
):
    """
    Run baseline experiment for one fixed mode.

    Args:
        mode: 'c1-1-step1', 'c1-1-step2', 'c1-1-step4', or 'c3-3-step4'
        prompts: List of text prompts
        seeds: Random seeds
        rollout_lengths: [10, 25, 50]
        output_dir: Output directory
    """
    print(f"\n{'='*60}")
    print(f"E1 BASELINE: {mode}")
    print(f"{'='*60}\n")

    # Parse mode
    if 'c1-1' in mode:
        checkpoint_mode = 'c1-1'
        step_count = int(mode.split('step')[-1])
    else:
        checkpoint_mode = 'c3-3'
        step_count = 4

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_mode}, S={step_count}...")
    checkpoints = load_causal_rcm_checkpoints(
        device='cuda',
        checkpoint_type=checkpoint_mode,
        step_count=step_count
    )

    # Initialize inference
    inference = CausalRCMInference(checkpoints, device='cuda', dtype=torch.float16)

    # Results storage
    results = []

    # Generate for each prompt × seed × length
    total_runs = len(prompts) * len(seeds) * len(rollout_lengths)
    pbar = tqdm(total=total_runs, desc=f"Generating {mode}")

    for prompt_idx, prompt in enumerate(prompts):
        for seed in seeds:
            # Set seed
            torch.manual_seed(seed)
            np.random.seed(seed)

            for num_frames in rollout_lengths:
                try:
                    # Generate
                    config = GenerationConfig(
                        num_frames=num_frames,
                        seed=seed,
                        mode=checkpoint_mode,
                        step_count=step_count
                    )

                    start_time = time.time()
                    output = inference.generate([prompt], config)
                    gen_time = time.time() - start_time

                    frames = output['frames']
                    latents = output['latents']

                    # Compute metrics (placeholder implementations)
                    fvd = compute_fvd(frames, None)

                    # Frame-to-frame LPIPS
                    lpips_scores = []
                    for i in range(1, frames.shape[1]):
                        lpips_scores.append(
                            compute_lpips(frames[0, i-1], frames[0, i])
                        )
                    lpips_mean = float(np.mean(lpips_scores))

                    # Measure compute cost
                    compute_cost = measure_compute_cost(mode, num_frames)

                    # Record result
                    result = {
                        "mode": mode,
                        "prompt_idx": prompt_idx,
                        "prompt": prompt,
                        "seed": seed,
                        "num_frames": num_frames,
                        "fvd": fvd,
                        "lpips_mean": lpips_mean,
                        "psnr": 0.0,  # No reference for text-to-video
                        "gen_time_sec": gen_time,
                        "gpu_ms": compute_cost["total_gpu_ms"],
                        "K": compute_cost["K"],
                        "S": compute_cost["S"]
                    }

                    results.append(result)

                except Exception as e:
                    print(f"\n✗ Failed: prompt={prompt_idx}, seed={seed}, frames={num_frames}")
                    print(f"  Error: {e}")

                    # Record failure
                    results.append({
                        "mode": mode,
                        "prompt_idx": prompt_idx,
                        "seed": seed,
                        "num_frames": num_frames,
                        "status": "failed",
                        "error": str(e)
                    })

                pbar.update(1)

    pbar.close()

    # Save results
    output_file = output_dir / f"{mode}.jsonl"
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"\n✓ Saved {len(results)} results to {output_file}")

    # Compute summary statistics
    successful = [r for r in results if 'fvd' in r]
    if successful:
        fvd_mean = np.mean([r['fvd'] for r in successful])
        lpips_mean = np.mean([r['lpips_mean'] for r in successful])

        print(f"\nSummary ({len(successful)}/{len(results)} successful):")
        print(f"  FVD (mean): {fvd_mean:.2f}")
        print(f"  LPIPS (mean): {lpips_mean:.4f}")

    return results


def compute_unsafe_labels(results_dir: Path, delta_threshold: float = 10.0):
    """
    Label segments as coarse-unsafe if fixed-coarse quality drops >Δ below fixed-fine.

    Args:
        results_dir: Directory with all E1 results
        delta_threshold: FVD threshold for unsafe label
    """
    print("\nComputing coarse-unsafe labels...")

    # Load all results
    fine_results = []
    coarse_results = []

    for result_file in results_dir.glob("*.jsonl"):
        with open(result_file) as f:
            results = [json.loads(line) for line in f]

            if 'c1-1' in result_file.name:
                fine_results.extend(results)
            else:
                coarse_results.extend(results)

    # Match prompts and compute labels
    labels = []

    for coarse_r in coarse_results:
        if 'fvd' not in coarse_r:
            continue

        # Find matching fine result
        matching_fine = [
            r for r in fine_results
            if r.get('prompt_idx') == coarse_r['prompt_idx']
            and r.get('seed') == coarse_r['seed']
            and r.get('num_frames') == coarse_r['num_frames']
            and 'fvd' in r
        ]

        if not matching_fine:
            continue

        # Use best fine mode (lowest FVD)
        best_fine_fvd = min(r['fvd'] for r in matching_fine)

        # Label as unsafe if coarse degrades significantly
        is_unsafe = (coarse_r['fvd'] - best_fine_fvd) > delta_threshold

        labels.append({
            "prompt_idx": coarse_r['prompt_idx'],
            "seed": coarse_r['seed'],
            "num_frames": coarse_r['num_frames'],
            "fine_fvd": best_fine_fvd,
            "coarse_fvd": coarse_r['fvd'],
            "delta": coarse_r['fvd'] - best_fine_fvd,
            "is_unsafe": is_unsafe
        })

    # Save labels
    labels_file = results_dir / "unsafe_labels.jsonl"
    with open(labels_file, 'w') as f:
        for label in labels:
            f.write(json.dumps(label) + '\n')

    unsafe_count = sum(1 for l in labels if l['is_unsafe'])
    print(f"✓ Labeled {unsafe_count}/{len(labels)} segments as coarse-unsafe")
    print(f"✓ Saved to {labels_file}")

    return labels


def main():
    """Run E1 baseline experiment."""

    # Get mode from environment variable (for parallel job submission)
    mode = os.environ.get('MODE', 'c1-1-step2')

    print(f"Starting E1 baseline experiment: {mode}")

    # Configuration
    prompts = load_prompts(subset_size=100)
    seeds = [42, 123, 456]  # 3 seeds
    rollout_lengths = [10, 25, 50]  # 3 horizons

    output_dir = Path("/kaggle/working/results/e1_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiment
    results = run_baseline_experiment(
        mode=mode,
        prompts=prompts,
        seeds=seeds,
        rollout_lengths=rollout_lengths,
        output_dir=output_dir
    )

    # If this is the last job (c3-3-step4), compute unsafe labels
    if mode == 'c3-3-step4':
        compute_unsafe_labels(output_dir)

    print("\n✓ E1 baseline complete!")


if __name__ == "__main__":
    # Import here to avoid circular dependency
    from src.causal_rcm.inference import GenerationConfig
    main()
