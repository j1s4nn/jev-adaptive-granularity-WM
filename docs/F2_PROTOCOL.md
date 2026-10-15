# F2-lite frozen protocol (negative/diagnostic study)

**Version:** F2LITE-01, frozen 2026-10-11 before any F4-lite result. **Scope:** a
negative/diagnostic study of adaptive temporal-granularity switching in a causal video
world model (Causal-rCM / Wan2.1-1.3B-T2V, Kaggle T4). Text-to-video only; action
conditioning is out of scope.

## 1. Question and decision rule

Can online uncertainty-guided switching between fine (c1-1) and coarse (c3-3) generation
beat the best fixed policy at matched measured cost? F1 established there is no
quality–cost trade-off between the two families, so the pre-registered expectation is
**no**. This protocol exists to test that expectation once, honestly, on fresh prompts.

## 2. Primary and secondary endpoints

- **Primary (independent):** CLIP ViT-B/32 **subject consistency** (mean consecutive-frame
  cosine similarity) and **text alignment** (mean prompt–frame cosine similarity), plus the
  classical no-reference suite (temporal warp, frame-diff, dynamic degree, drift).
- **Secondary (internal diagnostic):** mean decoded-pixel L1 to the c1-1 step-4 reference.
  Flagged as possibly favoring the fine family; never used alone.
- **Cost (measured, not NFE alone):** DiT seconds, VAE seconds, total seconds, NFE,
  true forward count, peak VRAM, switch overhead. VAE decode is reported separately
  because it dominates and is mode-independent.

## 3. Effect direction and smallest meaningful effect

- Adaptive **passes** if it improves the primary independent endpoint at matched measured
  cost, or matches it at lower measured cost, versus the best fixed policy **and** beats a
  random same-budget control.
- Smallest meaningful effect (pre-declared): **≥ 5% relative** improvement in the primary
  independent endpoint at equal cost, or **≥ 15%** lower measured mode-dependent cost
  (DiT seconds) at equal primary endpoint. (F1 effect sizes are far below these.)
- Any difference smaller than this is reported as "no meaningful difference".

## 4. Signals (pre-generation only)

Residual (denoising residual of the last fine step), warp (flow-warp temporal error),
motion (flow magnitude), complexity (Sobel+var). Each has a measured probe cost; a signal
is only admissible if it is available **before** the next segment's generation.

## 5. Policies

1. `fixed_fine` — c1-1 step-2, `[4,1]`.
2. `fixed_coarse` — c3-3 step-4, 4-step.
3. `fine_noisy` — c1-1 step-2, `[4,1]`, `context_from_last_step`.
4. `coarse_noisy` — c3-3 step-2 noisy-ctx, `[4,1]`, `context_from_last_step`.
5. `random_matched` — same fine fraction as the adaptive policy, random schedule.
6. `adaptive` — residual/warp-thresholded switching (thresholds fit on development only).

Offline oracle (future-outcome upper bound) is reported for context only, never as a policy.

## 6. Data split and statistics

- **Fresh prompts:** 8 prompts selected from the VBench prompt pool, disjoint from the
  Phase 5B calibration and held-out manifests.
- Horizon 57 frames, seed 17. Segment = 3 latent frames; 4 scored segments per rollout.
- **Unit of analysis:** the prompt (n = 8). Report paired per-prompt differences and
  descriptive ranges; with n = 8 no confirmatory CI is claimed. This is a demonstration,
  not a powered test, and is labeled as such.

## 7. RQ gates

- **RQ1 (signal):** already failed its locked Phase 5B test; re-reported, not re-tested.
- **RQ2 (adaptive):** supported only if the adaptive policy passes §3. Expected: not supported.
- **RQ3 (recovery/hysteresis):** untested if the adaptive policy produces no switches;
  labeled unexercised, not successful.
- **RQ4 (monotonicity):** exploratory, no direction required.

## 8. Explicit non-claims

No KV-cache-preservation claim; no action-conditioned claim; no FVD/PSNR/LPIPS; no claim
that coarse is more perceptually faithful; no claim of a positive adaptive result unless
§3 is met on fresh prompts.

## 9. Provenance

Pin the rcm commit, package versions, job source hash, manifest, seeds, GPU type, kernel
version, and output hash. Save all videos and JSONL under `results/`. Keep model weights
and credentials out of git.
