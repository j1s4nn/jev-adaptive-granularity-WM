# F1 audit — endpoint, mode, and mechanism feasibility

**Status:** COMPLETE. **Gate verdict: NO-GO** for the c1-1/c3-3 adaptive-granularity
trade-off. **Date:** 2026-10-11. **Hardware:** Kaggle Tesla T4 (16 GB), fp16.
**Figure:** `docs/figures/f1_mode_pareto.png`.

## Purpose

Decide whether any compatible Causal-rCM fine/coarse operating-mode pair provides a
defensible quality–cost trade-off. If none does, the plan's dominance gate says to drop
the fine/coarse switching improvement claim and report a negative/diagnostic result.

## F1.1 — Existing Phase 5B timing audit

Source `scripts/f1_timing_audit.py` → `results/f1_timing_audit/f1_timing_audit.json`.

| Split | n | fine total (s) | coarse total (s) | fine DiT (s) | coarse DiT (s) | NFE f/c | fine faster |
|---|---:|---:|---:|---:|---:|---:|---:|
| calibration | 12 | 184.60 | 214.66 | 21.14 | 53.19 | 18/20 | 12/12 |
| held_out | 72 | 169.76 | 200.08 | 21.56 | 53.42 | 18/20 | 72/72 |

Phase 5B "fine" = c1-1 1-step regime `[4,1]`; "coarse" = c3-3 step-4. The 4-step
schedule cancels c3-3's structural efficiency, so the inversion is an operating-point
artifact — which is why F1.3 sweeps all regimes.

## F1.3 — clean-context mode cost sweep

Job `jobs/f1_mode_cost/` → `ajjisan/f1-mode-cost` (COMPLETE). 6 modes × 45/57/81 × 3 dev
prompts, seed 17, 54 videos. Audit `scripts/f1_mode_cost_audit.py` →
`results/f1_mode_cost/f1_mode_cost_audit.json`.

| mode | DiT (s) | VAE (s) | total (s) | NFE | VRAM (GB) | refL1 ↓ | s/NFE |
|---|---:|---:|---:|---:|---:|---:|---:|
| ref_c1_1_s4 | 59.42 | 155.7 | 227.8 | 64.0 | 11.32 | 0.0000 | 0.93 |
| fine_c1_1_s1 | **25.16** | 156.5 | 194.5 | 19.0 | 11.32 | **0.2019** | 1.32 |
| fine_c1_1_s2 | 36.73 | 155.7 | 205.5 | 34.0 | 11.32 | 0.1930 | 1.08 |
| coarse_c3_3_s4 | 62.05 | 155.7 | 232.1 | 21.3 | 11.32 | 0.2750 | 2.91 |
| coarse_c3_3_s2 | 40.63 | 155.6 | 210.5 | 12.7 | 11.32 | 0.2615 | 3.20 |
| coarse_c3_3_s1 | 29.72 | 155.9 | 197.6 | **8.3** | 11.32 | 0.2588 | 3.58 |

**Finding:** c3-3 has ~3× fewer NFEs but each forward is ~3× heavier (3 latent frames
per pass), so it is **not cheaper in wall-clock DiT time**. `fine_c1_1_s1` is both faster
and closer to the reference than every coarse mode. VAE decode (~156 s) dominates total
time and is mode-independent.

## F1.4 — independent quality endpoints

**F1.4a classical** (`scripts/f1_quality_classical.py` →
`results/f1_mode_cost/f1_quality_classical.json`):

| mode | temporal_warp ↓ | frame_diff ↓ | flow_mag | hist_corr ↑ | drift ↓ |
|---|---:|---:|---:|---:|---:|
| ref_c1_1_s4 | 0.0184 | 0.0124 | 0.361 | 0.9981 | 0.1143 |
| fine_c1_1_s1 | 0.0280 | 0.0195 | 0.728 | 0.9960 | 0.1524 |
| fine_c1_1_s2 | 0.0258 | 0.0183 | 0.630 | 0.9969 | 0.1243 |
| coarse_c3_3_s4 | 0.0294 | 0.0202 | 0.733 | 0.9984 | 0.1303 |
| coarse_c3_3_s2 | 0.0334 | 0.0235 | 1.051 | 0.9970 | 0.1426 |
| coarse_c3_3_s1 | 0.0336 | 0.0235 | 1.041 | 0.9966 | 0.1484 |

Coarse modes are **less temporally smooth** (higher warp and frame-diff).

**F1.4b CLIP** (`scripts/f1_quality_clip.py` → `results/f1_mode_cost/f1_quality_clip.json`):

| mode | subject_consistency ↑ | text_alignment ↑ | text_alignment_std ↓ |
|---|---:|---:|---:|
| ref_c1_1_s4 | 0.9929 | 0.2805 | 0.0063 |
| fine_c1_1_s1 | 0.9866 | 0.2971 | 0.0071 |
| fine_c1_1_s2 | 0.9894 | 0.2947 | 0.0070 |
| coarse_c3_3_s4 | 0.9908 | 0.2895 | 0.0085 |
| coarse_c3_3_s2 | 0.9888 | 0.3040 | 0.0079 |
| coarse_c3_3_s1 | 0.9882 | 0.3053 | 0.0068 |

Mixed and small: coarse s1/s2 have marginally higher text alignment; fine has higher
subject consistency and better temporal smoothness. No consistent coarse advantage.

## F1.3b — noisy-context cost

Job `jobs/f1_mode_cost_noisy/` → `ajjisan/f1-mode-cost-noisy` v2 (COMPLETE). 6 modes ×
57/81 × 3 dev prompts. Audit `scripts/f1_mode_cost_noisy_audit.py` →
`results/f1_mode_cost_noisy/f1_mode_cost_noisy_audit.json`.

| mode | DiT (s) | NFE | forwards | refL1 ↓ |
|---|---:|---:|---:|---:|
| ref_c1_1_s4 | 71.30 | 72 | 90 | 0.0000 |
| fine_c1_1_s1 | 30.52 | 21 | 39 | 0.2056 |
| **fine_c1_1_s1_noisy** | **15.70** | 21 | 22 | 0.2182 |
| coarse_c3_3_s1 | 34.95 | 9 | 15 | 0.2550 |
| coarse_c3_3_s1_noisy | 21.01 | 9 | 10 | 0.2464 |
| coarse_c3_3_s2_noisy | 35.84 | 14 | 15 | 0.2471 |

Noisy context (`context_from_last_step`) roughly halves DiT time for **both** families,
so it does not create a coarse advantage: `fine_c1_1_s1_noisy` (15.70 s, 0.218) still
beats `coarse_c3_3_s1_noisy` (21.01 s, 0.246).

## Gate verdict — NO-GO

1. c3-3 coarse is **never cheaper** in wall-clock DiT time (clean or noisy context).
2. No independent endpoint shows a **consistent coarse quality advantage**.
3. H1 already failed its locked test (Phase 5B residual AUROC 0.491).

Therefore the fine/coarse switching improvement claim is **dropped**; the project
delivers a negative/diagnostic study. A positive claim would require a different
backbone/operating-mode pair (out of scope here).

## Limitations

- 3 development prompts, 1 seed; a mode characterization, not a confirmatory test.
- The internal reference (c1-1 step-4) is in the fine family and may favor fine; the
  independent CLIP/classical metrics are the safeguard and do not reverse the verdict.
- CLIP text-image similarity on short prompts is noisy.
- VAE decode dominates total wall-clock and is mode-independent; the mode-dependent
  cost is DiT time.
- Secondary, non-switching finding: noisy context ≈ 2× DiT speed-up at ~+6% reference
  distance (the upstream authors' trick, replicated).

## Reproduction

```
python scripts/f1_timing_audit.py
python scripts/f1_mode_cost_audit.py
python scripts/f1_quality_classical.py
python scripts/f1_quality_clip.py
python scripts/f1_mode_cost_noisy_audit.py
python scripts/f1_pareto_figure.py
```
