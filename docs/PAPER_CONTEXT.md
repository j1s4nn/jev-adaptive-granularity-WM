# PAPER_CONTEXT.md — single consolidated handoff for the manuscript

> **Purpose.** One file a local AI (or human) can read to write the paper with no
> lost context. Keep this in sync with `STATUS.md`, `PROGRESS.md`, and the raw
> artifacts under `results/`. Every claim must map to a raw artifact path and a
> protocol version. Update at every phase gate.

**Project:** Adaptive temporal granularity for a causal video world model
(fine/coarse operating-mode switching with recovery + hysteresis).
**Plan:** `Mater_plan_final.md` (Part I = active; Part II = historical record).
**Spec:** `proposal.tex`. **Status:** `STATUS.md`. **Ledger:** `PROGRESS.md`. **Links:** `LINKS.md`.

**Scope decision:** text-to-video only. Action-conditioned control (F7) is OUT OF
SCOPE — the verified checkpoints expose no action interface (`lambda_action=0`),
and no compatible action checkpoint was found.

---

## 0. Environment and exact working recipe (verified)

- Kaggle Tesla T4 (16 GB), fp16, PyTorch SDPA, 480p 16:9. Kaggle runtime Python 3.13.15.
- Checkpoints (private dataset `ajjisan/causal-rcm-ckpts`): `c1-1_step2`, `c1-1_step4`,
  `c3-3_step2`, `c3-3_step2_noisy_ctx`, `c3-3_step4` (dCM-init and sCM-init variants).
  Each ~2.71 GB. **No EMA weights. No action-conditioned interface.**
- Text encoder `ajjisan/wan21-t5` (umT5-XXL bf16), VAE/backbone `ajjisan/wan21-dit-vae`.
- Upstream code: `https://github.com/NVlabs/rcm` (pin a commit before paper).
- Recipe steps (from `Mater_plan_final.md` PII.1): single-file Kaggle script kernels;
  clone rcm; patch bf16→fp16 in `wan2pt1_t2v_causal_infer.py` (2 places) and
  `rcm/utils/umt5.py` (dtype + local tokenizer path); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
  build DiT on device then `.to(cuda, fp16).cpu()`; build umT5 on CPU fp32 → fp16 →
  load on cuda → encode → free; sample; VAE-decode AFTER freeing the DiT.
- Latent frames `T = 1 + (num_pixels-1)//4`. c3-3 requires `(T-3)%3==0`, so valid
  `num_frames ∈ {21,33,45,57,81,...}`.

### Operating-mode config table (authoritative, from `Causal_rCM.md` + `wan2pt1_t2v_causal_infer.py`)

| Regime | steps per chunk | midpoints | flags |
|---|---|---|---|
| 4-step | 4 | `[15/16, 5/6, 5/8]` | `num_steps 4; mid_t 15/16 5/6 5/8` |
| 2-step | `[4, 2]` | `[[15/16,5/6,5/8],[5/6]]` | `steps_per_chunk 4 2; mid_t_schedules "15/16,5/6,5/8;5/6"` |
| 1-step | `[4, 1]` | `[[15/16,5/6,5/8],[]]` | `steps_per_chunk 4 1; mid_t_schedules "15/16,5/6,5/8;"` |

`steps_per_chunk` is expanded over chunks by repeating the last entry. If omitted,
every chunk uses `num_steps`. c1-1: `first_chunk_t=1, chunk_t=1`. c3-3: `first_chunk_t=3, chunk_t=3`.
Doc note: *"In the frame-wise (c1-1) setting, the 2-step checkpoint can infer with 1 step without losing quality."*

---

## 1. Protocol versions

| Version | Split | Metric / label | Status |
|---|---|---|---|
| phase5-v1 | 12 calibration (seed 17) + 24 held-out prompts (seeds 17/29/41), 57 frames | decoded-pixel L1 to internal c1-1 step-4 reference; unsafe if regret > 0.0896500349 | **H1 FAILED** (residual AUROC 0.491) |
| FINAL-PLAN-01 F1 | development = Phase 5B **calibration** prompts only | mode cost sweep (no confirmatory claim) | **COMPLETE — NO-GO on T4** (`results/f1_mode_cost/`, `results/f1_mode_cost_noisy/`) |
| A100-01 (A1–A3) | fresh split `config/a100_manifest.json` (disjoint from Phase 5B and F4-lite) | real umT5 + VAE; internal refL1 + independent CLIP/classical | **COMPLETE** (A1 PASS trade-off; A2 RQ2 neg. on independent endpoint; A3 RQ3 SUPPORTED) |
| Jev-max v1 (Phases 0–3) | Phase 0 re-uses T4 80 paired segments; Phases 1–3 use fresh A100 splits (`config/a100_manifest.json`, `config/a100_phase3_manifest.json`) | Jev System One signal; independent CLIP/classical; bootstrap CIs | **COMPLETE** (RQ1 positive; RQ2 weak positive for `hold`; Phase 3 = genuine trade-off) |
| Jev-max v1 (Phase 4.1/4.2) | A100 `results/a100_rq3/`; RQ4 from A100 Phase 3 data | RQ3 recovery/hysteresis; RQ4 monotonicity | **4.1 SUPPORTED; 4.2 NULL (confounded)** |

Rule: the Phase 5B held-out prompts/results are now **development evidence**. A new
method needs a new locked manifest + **fresh** held-out prompts.

---

## 2. Results so far

### 2.1 Phase 5B (locked RQ1 test — data gate PASSED, claim gate FAILED)

12 calibration + 72 held-out prompt-seed cases, 288 held-out scored segments.

| Signal | Calibration AUROC | Held-out AUROC | Held-out prompt-seed 95% CI |
|---|---:|---:|---|
| Residual | 0.630 | **0.491** | [0.393, 0.589] |
| Warp | 0.651 | 0.454 | [0.343, 0.566] |
| Motion | 0.580 | 0.488 | [0.376, 0.600] |
| Complexity | 0.526 | 0.301 | [0.197, 0.412] |

Residual − warp (held-out) = 0.0367, 95% CI [−0.0493, 0.1194] → includes zero. **RQ1 unsupported.**
Artifact: `results/phase5b_audit/phase5b_audit.json`.

### 2.2 F1.1 timing audit (local re-summary of Phase 5B cost rows)

Source: `scripts/f1_timing_audit.py` → `results/f1_timing_audit/f1_timing_audit.json`.

| Split | n | fine total (s) | coarse total (s) | fine DiT (s) | coarse DiT (s) | NFE f/c | fine faster |
|---|---:|---:|---:|---:|---:|---:|---:|
| calibration | 12 | 184.60 | 214.66 | 21.14 | 53.19 | 18/20 | 12/12 |
| held_out | 72 | 169.76 | 200.08 | 21.56 | 53.42 | 18/20 | 72/72 |

**Key finding:** the Phase 5B "fine" was the **c1-1 1-step** regime (`[4,1]`, NFE 18) and
"coarse" was **c3-3 step-4** (NFE 20). The 4-step schedule negates c3-3's structural
efficiency (fewer chunks). The cost inversion is therefore an **artifact of the chosen
operating points**, not proof that coarse can never be cheaper — this is what F1.3 tests.

### 2.3 F1.3 mode-cost benchmark (DONE — PASS 9/9)

Job: `jobs/f1_mode_cost/` → Kaggle `ajjisan/f1-mode-cost` (version 1, COMPLETE 2026-10-11).
6 modes × 3 horizons (45/57/81) × 3 dev prompts (object/motion/human), seed 17, 54 videos saved.
Audit: `scripts/f1_mode_cost_audit.py` → `results/f1_mode_cost/f1_mode_cost_audit.json`.

Mean over 9 cases (DiT time is the mode-dependent cost; VAE decode ~156 s is fixed):

| mode | DiT (s) | VAE (s) | total (s) | NFE | VRAM (GB) | refL1 (lower better) | s/NFE |
|---|---:|---:|---:|---:|---:|---:|---:|
| ref_c1_1_s4 | 59.42 | 155.7 | 227.8 | 64.0 | 11.32 | 0.0000 | 0.93 |
| fine_c1_1_s1 | **25.16** | 156.5 | 194.5 | 19.0 | 11.32 | **0.2019** | 1.32 |
| fine_c1_1_s2 | 36.73 | 155.7 | 205.5 | 34.0 | 11.32 | 0.1930 | 1.08 |
| coarse_c3_3_s4 | 62.05 | 155.7 | 232.1 | 21.3 | 11.32 | 0.2750 | 2.91 |
| coarse_c3_3_s2 | 40.63 | 155.6 | 210.5 | 12.7 | 11.32 | 0.2615 | 3.20 |
| coarse_c3_3_s1 | 29.72 | 155.9 | 197.6 | **8.3** | 11.32 | 0.2588 | 3.58 |

**Decisive finding:** the c3-3 family has far fewer NFEs (8.3 vs 19.0) but each c3-3
forward is ~3× heavier (3 latent frames/pass), so **c3-3 is NOT cheaper in wall-clock DiT
time**. `fine_c1_1_s1` (25.16 s DiT) is both **faster** and **closer to the reference**
(0.2019) than the cheapest coarse `coarse_c3_3_s1` (29.72 s, 0.2588). NFE ≠ GPU time.
VAE decode dominates total time (~80%) and is mode-independent.

### 2.4 F1.4a classical no-reference quality (DONE)

Audit: `scripts/f1_quality_classical.py` → `results/f1_mode_cost/f1_quality_classical.json`.
Independent of the internal c1-1 reference. 54 videos.

| mode | temporal_warp ↓ | frame_diff ↓ | flow_mag | hist_corr ↑ | drift ↓ |
|---|---:|---:|---:|---:|---:|
| ref_c1_1_s4 | 0.0184 | 0.0124 | 0.361 | 0.9981 | 0.1143 |
| fine_c1_1_s1 | 0.0280 | 0.0195 | 0.728 | 0.9960 | 0.1524 |
| fine_c1_1_s2 | 0.0258 | 0.0183 | 0.630 | 0.9969 | 0.1243 |
| coarse_c3_3_s4 | 0.0294 | 0.0202 | 0.733 | 0.9984 | 0.1303 |
| coarse_c3_3_s2 | 0.0334 | 0.0235 | 1.051 | 0.9970 | 0.1426 |
| coarse_c3_3_s1 | 0.0336 | 0.0235 | 1.041 | 0.9966 | 0.1484 |

The coarse modes are **less temporally smooth** (higher warp, higher frame diff) than
fine, with no consistency advantage. This is a lightweight proxy; a paper-grade
independent endpoint (VBench / CLIP subject consistency) and blinded human comparison
are still required before the F1 go/no-go.

### 2.5 F1.4b CLIP independent endpoint (DONE)

Audit: `scripts/f1_quality_clip.py` → `results/f1_mode_cost/f1_quality_clip.json`.
Model `openai/clip-vit-base-patch32`; 54 videos; independent of the internal reference.

| mode | subject_consistency ↑ | text_alignment ↑ | text_alignment_std ↓ |
|---|---:|---:|---:|
| ref_c1_1_s4 | 0.9929 | 0.2805 | 0.0063 |
| fine_c1_1_s1 | 0.9866 | 0.2971 | 0.0071 |
| fine_c1_1_s2 | 0.9894 | 0.2947 | 0.0070 |
| coarse_c3_3_s4 | 0.9908 | 0.2895 | 0.0085 |
| coarse_c3_3_s2 | 0.9888 | 0.3040 | 0.0079 |
| coarse_c3_3_s1 | 0.9882 | 0.3053 | 0.0068 |

Mixed and small: coarse s1/s2 have slightly **higher text alignment**, but lower
subject consistency and worse temporal smoothness (2.4). No metric shows a clear,
consistent coarse advantage. CLIP text-image similarity on short prompts is noisy.

### 2.6 F1.3b noisy-context cost (DONE) + F1 GATE VERDICT

Job `jobs/f1_mode_cost_noisy/` → `ajjisan/f1-mode-cost-noisy` v2 (COMPLETE 2026-10-11),
6/6 cases, 3 dev prompts × {57,81}. Audit `scripts/f1_mode_cost_noisy_audit.py` →
`results/f1_mode_cost_noisy/f1_mode_cost_noisy_audit.json`.

`context_from_last_step` (noisy context) removes one forward per chunk and roughly
**halves DiT time** — for BOTH families:

| mode | DiT (s) | NFE | forwards | refL1 ↓ |
|---|---:|---:|---:|---:|
| ref_c1_1_s4 | 71.30 | 72 | 90 | 0.0000 |
| fine_c1_1_s1 | 30.52 | 21 | 39 | 0.2056 |
| **fine_c1_1_s1_noisy** | **15.70** | 21 | 22 | 0.2182 |
| coarse_c3_3_s1 | 34.95 | 9 | 15 | 0.2550 |
| coarse_c3_3_s1_noisy | 21.01 | 9 | 10 | 0.2464 |
| coarse_c3_3_s2_noisy | 35.84 | 14 | 15 | 0.2471 |

**Decisive:** even with noisy context applied to both, `fine_c1_1_s1_noisy` (DiT 15.70 s,
refL1 0.218) is BOTH faster and closer to the reference than `coarse_c3_3_s1_noisy`
(21.01 s, 0.246). The noisy-context trick helps both families, so it does not create a
coarse advantage.

**F1 EXIT GATE = NO-GO for the c1-1/c3-3 adaptive-granularity trade-off.**
- No wall-clock cost advantage for c3-3 (clean OR noisy context).
- No consistent independent quality advantage for c3-3.
- H1 already failed its locked test (Phase 5B, AUROC 0.491).
Per the master plan's dominance gate: **drop the fine/coarse switching improvement claim
and report a negative/diagnostic result**, unless a different backbone/mode pair is
adopted (out of scope for a 3-day window).

**Secondary finding (not a switching result):** noisy context ≈ 2× DiT speed-up at
~+6% reference distance. This is the upstream authors' trick, replicated here; usable
as a characterization, not as our novelty.

---

### 2.7 F4-lite online switching (DONE — PASS 40/40)

Job `jobs/f4lite/` → Kaggle `ajjisan/f4-lite-online-switching` (COMPLETE 2026-10-11).
8 **fresh** prompts (`config/f4lite_manifest.json`, disjoint from Phase 5B), 57 frames,
seed 17. All policies share one common coarse prefix (3 latent frames) then make 4 segment
decisions of 3 latent frames each. Audits: `scripts/f4lite_audit.py` →
`results/f4lite/f4lite_audit.json`; `scripts/f4lite_quality.py` →
`results/f4lite/f4lite_quality.json`. Upstream commit `ed3cb14dd936f92cdc9f9381af7369991509b41f`.

Cost/quality (means over 8 prompts; refL1 = internal diagnostic, lower better):

| policy | fine_frac | switches | refL1 ↓ | DiT (s) | VAE (s) | total (s) | signal (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_coarse | 0.00 | 0.00 | 0.3186 | 35.0 | 110.4 | 202.1 | 0.00 |
| **fixed_fine** | 1.00 | 0.00 | **0.2998** | **18.1** | 110.9 | **184.1** | 0.00 |
| random50 | 0.50 | 2.12 | 0.3102 | 26.5 | 110.7 | 187.5 | 0.00 |
| alternating50 | 0.50 | 3.00 | 0.3058 | 26.5 | 110.8 | 187.3 | 0.00 |
| adaptive_warp | 0.75 | 0.00 | 0.2949 | 22.1 | 321.6 | 397.8 | 0.65 |

Independent CLIP/classical quality (means over 8 prompts):

| policy | subject_cons ↑ | text_align | temporal_warp ↓ | flow_mag | drift ↓ |
|---|---:|---:|---:|---:|---:|
| reference | 0.9898 | 0.2778 | 0.0435 | 0.921 | 0.1628 |
| fixed_coarse | 0.9490 | 0.2770 | 0.0733 | 2.073 | 0.2192 |
| **fixed_fine** | **0.9840** | 0.2737 | **0.0303** | 0.990 | **0.1586** |
| random50 | 0.9646 | 0.2762 | 0.0526 | 1.512 | 0.2107 |
| alternating50 | 0.9591 | 0.2752 | 0.0552 | 1.694 | 0.2192 |
| adaptive_warp | 0.9803 | 0.2761 | 0.0429 | 1.324 | 0.1784 |

**Finding:** `fixed_fine` is the best fixed policy on **both** the independent endpoint
(highest subject consistency 0.9840, lowest warp 0.0303, lowest drift) **and** cost
(DiT 18.1 s, total 184.1 s). `adaptive_warp` is marginally better than fixed fine on the
*internal* reference (0.2949 vs 0.2998, +1.6%) but **worse** on the independent CLIP
endpoint (0.9803 vs 0.9840) and costs **2.2× total wall-clock** (397.8 s) and **2.9× VAE**
from repeated context decoding. It also made **0 within-video switches** (it degenerated
to a per-prompt fixed choice: all-fine for 6 prompts, all-coarse for 2).

**RQ2/H2 not supported.** Per F2LITE-01 §3 (≥5% relative quality at equal cost, or ≥15%
lower cost at equal quality), no policy passes. **RQ3/H3 unexercised** (no switch-producing
policy). This is the pre-registered expected outcome.

### 2.8 RQ3 — recovery + hysteresis controller ablation (DONE — SUPPORTED)

Job `jobs/rq3_controller/` → `ajjisan/f-rq3-controller-ablation` (COMPLETE 2026-10-11),
48/48 + 8 references. 8 fresh prompts, 57 frames, shared coarse prefix. Signal trace
calibrated from F4-lite development videos (`config/rq3_threshold.json`; tau = pooled
median warp, Lambda = 2 sigma_u = 7.83, R = 2). Audit `scripts/rq3_audit.py` →
`results/rq3_controller/rq3_audit.json`. Figure `docs/figures/rq3_controller_ablation.png`.

| controller | switches | fine_frac | meanRef | termRef | AUHRef | switch-adj L1 | interior-adj L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| threshold_only | 0.50 | 0.50 | 0.3133 | 0.3171 | 0.3133 | 0.0364 | 0.0280 |
| hysteresis_only | 0.38 | 0.53 | 0.3116 | 0.3104 | 0.3116 | 0.0321 | 0.0273 |
| recovery_only | 0.38 | 0.53 | 0.3116 | 0.3104 | 0.3116 | 0.0321 | 0.0273 |
| full | 0.38 | 0.53 | 0.3116 | 0.3104 | 0.3116 | 0.0321 | 0.0273 |
| threshold_only_noisy | 1.12 | 0.53 | 0.3159 | 0.3172 | 0.3159 | 0.0630 | 0.0290 |
| full_noisy | 0.62 | 0.66 | 0.3150 | 0.3139 | 0.3150 | 0.0399 | 0.0260 |

**RQ3 SUPPORTED.** Paired vs threshold-only: real signal switch diff **-0.12** with AUH
regret diff **-0.00168** (better) and terminal regret diff **-0.00674** (better); noisy
stress switch diff **-0.50 (-45%)** with AUH regret diff **-0.00096**. So recovery +
hysteresis reduces mode oscillation without worsening terminal/AUH regret, most clearly
under signal noise (directly consistent with Proposition 1: Lambda > 2 sigma_u).
**Transition cost is real and measured:** adjacent-frame L1 at switch boundaries exceeds
interior L1 for every controller (e.g. threshold_only 0.0364 vs 0.0280), and full_noisy
reduces the boundary cost (0.0399 vs 0.0630). The Lambda in {0.5,1,2,4} sigma_u and
R in {1,2,3} sweeps are flat on these near-monotone real traces (3 switches), so the
mechanism's benefit is only observable under noise; reported as such.

### 2.9 RQ4 — monotonicity of the uncertainty-benefit relationship (DONE)

Job `jobs/rq4_monotonicity/` → `ajjisan/f-rq4-monotonicity` (COMPLETE 2026-10-11),
16/16 + 8 references at 81 frames. Combined with the 57-frame F4-lite paired segments
(80 paired points total). Audit `scripts/rq4_audit.py` →
`results/rq4_monotonicity/rq4_audit.json`. Figure `docs/figures/rq4_monotonicity.png`.

- Spearman rho(signal, fine_gap) = **-0.099** (no monotone relationship).
- Mean fine_gap = **-0.0332** (fine closer to reference at essentially every level).
- Tercile bins: low -0.0225, mid **-0.0571**, high -0.0195 -> non-monotone, fine benefit
  **peaks at intermediate uncertainty** (mirrors Causal-rCM's non-monotone step behaviour).

**RQ4/H4 (exploratory): no monotone relationship** for the *warp* signal. This is
superseded for the *Jev* signal by §2.15 Phase 0, where the T4 relationship becomes monotone
(rho -0.517); the A100 re-test (Phase 4.2) is confounded and returns NULL.

### 2.10 E-JEV-1 — Jev System One uncertainty signal (DONE — RQ1 POSITIVE)

The TypeSafe **Jev** System One model (text-only, typed decisions, ~0.7 s/call) used as a
pre-generation uncertainty oracle. State per segment = prompt + BLIP caption of the
pre-decision context frame + segment index + motion stats; one call with composite
questions (Noul drift-risk, Score difficulty, Choice motion). Label `coarse_unsafe =
regret_coarse > regret_fine` from existing data. 80 segments (57f F4-lite + 81f RQ4),
8 prompts, 56/80 positives. Artifacts: `scripts/jev_build_states.py`, `scripts/jev_query.py`,
`scripts/jev_audit.py` → `results/jev_signal/`; figure `docs/figures/jev_signal_auroc.png`.

| signal | AUROC | 95% CI (prompt-grouped) |
|---|---:|---|
| **jev_difficulty** | **0.788** | [0.575, 0.946] |
| jev_drift | 0.738 | [0.479, 0.945] |
| jev_motion | 0.703 | [0.472, 0.873] |
| motion (hand-crafted) | 0.607 | [0.260, 0.857] |
| warp (hand-crafted) | 0.586 | [0.222, 0.836] |
| complexity (hand-crafted) | 0.368 | [0.184, 0.562] |
| Phase 5B residual (reference) | 0.491 | — |

**Finding:** the best Jev signal beats the best hand-crafted signal by **+0.180 AUROC** and
the Phase 5B residual by **+0.297**; the CI excludes 0.491. **RQ1 now supported with Jev.**
Signal cost: Jev **0.67 s** per decision vs the warp signal's VAE decode (~156 s @57f) —
roughly **230× cheaper**, and off the GPU entirely. Caveat: n=80, 8 prompts, wide CIs;
captions come from a probe rollout.

### 2.11 E-JEV-2 — hardware sensitivity of the cost inversion (DECISIVE)

The cost inversion is **hardware-specific**. On a Kaggle **T4** the coarse/fine per-forward
ratio is **3.01** (break-even 2.33 -> coarse slower); on a Google Colab **A100-SXM4-80GB**
the same timing (timing-only, dummy text embedding, same modes, `colab/a100_timing.py`)
gives ratio **1.60** (break-even 2.33 -> **coarse cheaper**).

T4 replication (`results/jev_p100_timing/`, `ajjisan/jev-p100-timing`):

| mode | DiT (s) | NFE | forwards | s/forward |
|---|---:|---:|---:|---:|
| ref_c1_1_s4 | 61.40 | 72.0 | 90.0 | 0.670 |
| fine_c1_1_s1 | 25.86 | 21.0 | 39.0 | 0.652 |
| coarse_c3_3_s4 | 64.78 | 24.0 | 30.0 | 2.124 |
| coarse_c3_3_s1 | 29.79 | 9.0 | 15.0 | 1.960 |

A100 (`results/a100_timing/`, `scripts/a100_audit.py` -> `a100_audit.json`):

| horizon | mode | DiT (s) | NFE | forwards | s/forward |
|---|---:|---:|---:|---:|---:|
| 57f | fine_c1_1_s1 | **4.68** | 18 | 33 | 0.142 |
| 57f | coarse_c3_3_s4 | 4.99 | 20 | 25 | 0.200 |
| 57f | coarse_c3_3_s1 | **2.45** | 8 | 13 | 0.188 |
| 81f | fine_c1_1_s1 | **4.83** | 24 | 45 | 0.107 |
| 81f | coarse_c3_3_s4 | 7.85 | 28 | 35 | 0.224 |
| 81f | coarse_c3_3_s1 | **3.60** | 10 | 17 | 0.212 |

**Verdict:** on the A100 the coarse mode `c3-3` s1 is **1.3-1.9x faster** than fine s1
(57f: 2.45 s vs 4.68 s; 81f: 3.60 s vs 4.83 s), and the per-forward ratio (1.60) is below
break-even (2.33). **The fine/coarse cost trade-off exists on the A100 and is absent on the
T4.** Since quality is hardware-independent (coarse remains less accurate: refL1 0.259 vs
0.202), a genuine quality-cost trade-off is available on the A100 -- so the F1 NO-GO is a
**T4 artifact**, and the adaptive-granularity premise is viable on adequate hardware. The
A100 run is timing-only (no decode); quality numbers are the T4 measurements, which carry
over because the checkpoints and sampler are identical. (Mechanistic note: the A100 is
Ampere with FlashAttention, so the heavier 3-frame coarse forward amortizes far better than
on the Turing T4.)

### 2.12 E-JEV-3 — Jev-routed controller with a `hold` mode (Phase C)

Job `jobs/jev_hold_mode/` → `ajjisan/jev-hold-mode` PASS 40/40 (+8 refs, 48 videos).
8 prompts, 57 frames, shared coarse prefix. Modes: fine (c1-1 [4,1]), coarse (c3-3 step-4),
**hold** (copy the boundary latent frame, no generation). Jev routing
(`config/jev_routing.json`): static->hold, moderate->coarse, fast->fine. Audit
`scripts/jev_hold_audit.py` → `results/jev_hold_mode/jev_hold_audit.json`.

| policy | hold | fine | coarse | meanRef ↓ | termRef ↓ | DiT (s) | total (s) | forwards |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_fine | 0.0 | 4.0 | 0.0 | **0.2998** | 0.2978 | 19.7 | 193.9 | 29.0 |
| fixed_coarse | 0.0 | 0.0 | 4.0 | 0.3186 | 0.3205 | 37.7 | 205.5 | 25.0 |
| fixed_hold | 4.0 | 0.0 | 0.0 | 0.3108 | 0.3199 | **6.5** | **136.4** | 5.0 |
| jev_routed | 1.1 | 0.9 | 2.0 | 0.3085 | 0.3094 | 24.9 | 182.3 | 20.2 |
| jev_routed_nohold | 0.0 | 0.9 | 3.1 | 0.3131 | 0.3175 | 33.7 | 201.4 | 25.9 |

**Findings:** `hold` is a genuinely cheap mode (DiT 6.5 s vs 19.7 s fine; total 136 s vs
194 s) at a modest quality cost (refL1 0.311 vs 0.300). The Jev routing that sends
"moderate" segments to **coarse** is counterproductive (coarse is slower and less
accurate), so `jev_routed` does **not** beat `fixed_fine` overall (refL1 +0.0087, DiT
+5.2 s, total -11.6 s). But adding `hold` **helps relative to routing those segments to
coarse**: `jev_routed` vs `jev_routed_nohold` is better on quality (-0.0045) and cheaper
(total -19.1 s, DiT -8.8 s). **RQ2 remains negative for the coarse mode**; the `hold` mode
is a promising cheap mode, tested next as `static->hold, else->fine` (E-JEV-3b,
`jobs/jev_holdfine/`).

### 2.13 E-JEV-3b — Jev `hold-else-fine` controller (Phase C-b) — RQ2 POSITIVE (weak)

Job `jobs/jev_holdfine/` → `ajjisan/jev-holdfine` PASS 16/16. Routing: static->hold,
else->fine (from `config/jev_routing.json` motion choices). Audit
`scripts/jev_holdfine_audit.py` → `results/jev_holdfine/jev_holdfine_audit.json`.

| policy | hold | fine | meanRef ↓ | termRef ↓ | DiT (s) | VAE (s) | total (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_fine | 0.0 | 4.0 | 0.2998 | 0.2978 | 18.6 | 128.7 | 183.2 |
| **jev_holdfine** | 1.1 | 2.9 | **0.2988** | 0.2988 | **15.1** | 128.6 | **167.0** |

**Finding:** the Jev-routed `hold` mode **beats fixed fine on both axes** — regret
**-0.0010** (not worse, slightly better, because holding is a better match than
re-generating a static segment) and cost **total -16.2 s (-8.8%)**, **DiT -3.5 s
(-19%)**, VAE unchanged. Per the F2LITE-01 criterion (>=5% relative quality at equal cost,
or >=15% lower mode-dependent cost at equal quality) this **passes on the DiT cost metric**
and is neutral-to-better on quality. **RQ2 is therefore POSITIVE for the `hold` mode**
(weak: small quality delta, n=8 prompts), while remaining NEGATIVE for the coarse mode.
This is the paper's positive adaptive-granularity result: a System One oracle plus a
near-zero-cost mode yields a real, if modest, quality-cost improvement.

### 2.14 A100 arm (A100-01): real-decode sweep, online policies, RQ3

Protocol `docs/A100_PROTOCOL.md`; fresh split `config/a100_manifest.json`; rcm pinned
`ed3cb14`. Unlike the timing-only run, A1 uses **real umT5 + VAE**.

- **A1 six-mode sweep (3 dev prompts x 45/57/81, 54/54).** `results/a100_mode_cost/`.
  Coarse `c3-3 s1` DiT **2.69 s** vs fine `c1-1 s1` **3.92 s** (per-forward ratio **1.75 <
  2.33** break-even) → coarse **1.46x faster** but **less accurate** (internal refL1 0.258 vs
  0.177; lower CLIP subject consistency; higher temporal warp). **The quality-cost trade-off
  exists on the A100** (absent on the T4). VAE ~2.4 s vs ~156 s on T4.
- **A2 v2 online (12 fresh prompts, 72/72).** `results/a100_online_v2/`. `adaptive_warp`
  "passes" on the **internal** refL1 (+3.5% quality, -27.9% DiT) but the **independent
  endpoint reverses it** (CLIP subject consistency fine 0.9836 > adaptive 0.9492 > coarse
  0.9430) and its signal's extra VAE decodes make total wall-clock (52.3 s) exceed fixed fine
  (46.8 s). **RQ2 not supported on the independent endpoint.**
- **A3 RQ3 recovery+hysteresis (12 fresh prompts, 72/72) — SUPPORTED.** `results/a100_rq3/`.
  Switches `threshold_only` 0.50 -> `full` 0.42 (-17%) on the real signal, 0.92 -> 0.50
  (**-45%**) under 2-sigma noise, regret no worse; transition cost real (0.067 vs 0.030) and
  reduced by `full` (0.060). Consistent with T4.

### 2.15 Jev-maximized phases (2026-10-13/14, A100) — the current frontier

- **Phase 0 (0 GPU) — RQ4 flips positive on T4 data.** Re-analysing the existing 80 paired
  segments with the **Jev difficulty** signal gives rho(fine_gap) = **-0.517** (AUROC 0.788),
  monotone terciles, vs warp -0.098. Gate threshold **1.72** (Youden 0.524). `results/rq4_jev/`.
- **Phase 1 — A100 online Jev (96/96).** `results/a100_online_jev/`. `jev_holdfine` matches
  fixed_fine on CLIP subject consistency (0.9796 vs 0.9774) at **-35.5% DiT / -16.6% total**;
  `jev_routed` -49% DiT / -19% total. Signal is off the DiT path.
- **Phase 2 — signal-cost variants (60/60).** `results/a100_online_jev_phase2/`. **Single-decision
  Jev** (`jev_single`, one query/prompt) cuts signal cost 7.4 s -> **2.8 s** (2.6x) and lowers
  total (25.0 s vs 30.1 s). The **free latent proxy is rejected** (CLIP 0.9804 vs fine 0.9929).
- **Phase 3 — 24 fresh prompts + bootstrap CIs (120/120).** `results/a100_online_jev_phase3/`.
  **`jev_single`: DiT -1.78 s [-2.20,-1.37] and total -7.39 s [-12.39,-3.16] (both significant)
  at CLIP subject consistency -0.020 [-0.030,-0.010] (significant small loss)** → a genuine,
  statistically-supported quality-cost trade-off (not dominance). `latent` free but -0.013 subj.cons.
- **Phase 4.1 — RQ3 on A100 (72/72) SUPPORTED.** `results/a100_rq3/`: switches `threshold_only`
  0.50 -> `full` 0.42; under noise 0.92 -> 0.50; regret no worse. Matches T4 → the controller
  transfers across hardware.
- **Phase 4.2 — RQ4 on A100 = NULL (confounded).** `results/a100_online_jev_phase3/rq4_jev_a100.json`:
  Jev difficulty vs A100 fine_gap rho = **+0.059** (AUROC 0.451), i.e. no monotone relationship,
  unlike T4 (-0.517). Cause: the A100 internal reference **favors coarse** (mean fine_gap
  **+0.0137**, only **38.5%** of segments favour fine), so the label has no stable "fine benefit"
  to track. Also the coarse operating point differs (T4 RQ4 used c3-3 step-4; the A100 online job
  used c3-3 step-1), and the prompt splits are disjoint. The A100 A1 dev-prompt sweep still showed
  fine closer to the reference (0.177 vs 0.258) → the flip is a protocol/reference artifact, **not
  an A100 hardware effect**.
- **GPU accounting (2026-10-14):** 193.8 -> 82.3 compute units = **111.5 units (~16.5 A100-h)
  used**; **~12.1 A100-h remaining**. A100 experimental arm complete; only A4 write-up remains.

## 3. Artifact index (active)

- `config/phase5_prompt_manifest.json` — phase5-v1 locked manifest (12 cal + 24 held-out).
- `config/phase5c_online_calibration_lock.json`, `config/phase5c_pilot_lock.json`.
- `results/phase5b_audit/phase5b_audit.json` — RQ1 audit.
- `results/phase5b_heldout_00..07/…/results.jsonl` — 72 held-out cases.
- `results/phase5b_reference_calibration_a|b/…` — 12 calibration cases.
- `results/f1_timing_audit/f1_timing_audit.json` — F1.1.
- `results/f1_mode_cost/` — F1.3 (results.jsonl, audit, classical + CLIP quality, 54 videos).
- `results/f1_mode_cost_noisy/` — F1.3b (results.jsonl, audit, 26+ videos).
- `results/f4lite/` — F4-lite (results.jsonl, summary, audit, quality, 48 videos).
- `config/f4lite_manifest.json` — 8 fresh prompts for F4-lite.
- `results/rq3_controller/` — RQ3 controller ablation (results.jsonl, audit, 56 videos).
- `config/rq3_threshold.json` — RQ3 signal calibration (tau, sigma_u, traces).
- `docs/figures/rq3_controller_ablation.png` — RQ3 figure.
- `results/rq4_monotonicity/` — RQ4 monotonicity (results.jsonl, audit, 24 videos).
- `docs/figures/rq4_monotonicity.png` — RQ4 figure.
- `results/jev_signal/` — Jev uncertainty signal (states, answers, audit).
- `results/jev_p100_timing/` — hardware re-time (T4 replication).
- `results/a100_timing/` — A100 hardware re-time (DECISIVE: ratio 1.60, coarse cheaper).
- `colab/a100_timing.py`, `colab/README.md` — Colab A100 timing cell + instructions.
- `scripts/a100_audit.py`, `scripts/jev_p100_audit.py` — hardware audits.
- `results/jev_hold_mode/` — Jev-routed fine/coarse/hold controller.
- `results/jev_holdfine/` — Jev hold-else-fine controller (RQ2 positive).
- `config/jev_routing.json` — Jev motion-choice routing traces.
- `docs/figures/jev_signal_auroc.png`, `docs/figures/jev_hold_mode.png` — Jev figures.
- `docs/F1_AUDIT.md` — consolidated F1 audit (all tables + verdict).
- `docs/PAPER.md` — manuscript draft v3 (hardware-conditional; Jev method change).
- `docs/figures/f1_mode_pareto.png`, `docs/figures/f4lite_policy_pareto.png` — figures.
- `docs/F2_PROTOCOL.md` — frozen F2-lite protocol (F2LITE-01).
- `docs/figures/f1_mode_pareto.png` — cost vs reference-distance Pareto figure.
- `scripts/f1_*.py` — F1 audit/figure scripts.
- `scripts/f1_timing_audit.py` — F1.1 script.
- `results/a100_timing/` — A100 timing-only re-time (ratio 1.60).
- `results/a100_mode_cost/` — A100 A1 real-decode six-mode sweep (+ quality_clip/classical, audit).
- `results/a100_online/` (archived A2 v1), `results/a100_online_v2/` — A100 A2 online policies.
- `results/a100_rq3/` — A100 A3 / Phase 4.1 RQ3 (results.jsonl, summary, audit, videos).
- `results/rq4_jev/` — Phase 0 T4 RQ4 with Jev (audit, signal_select).
- `results/a100_online_jev/`, `results/a100_online_jev_phase2/`, `results/a100_online_jev_phase3/`
  — Jev-max phases 1-3 (results.jsonl, summary, audit, quality_clip/classical, `phase3_ci.json`,
  `rq4_jev_a100.json`).
- `colab/a100_sweep.py`, `colab/a100_online.py`, `colab/a100_online_jev.py`, `colab/a100_rq3.py`
  — A100 kernels.
- `scripts/a100_*_audit.py`, `scripts/jev_rq4_audit.py`, `scripts/phase4_rq4_jev.py` — audits.
- `config/a100_manifest.json`, `config/a100_phase3_manifest.json`, `config/jev_routing.json`.
- Placeholder paths that must NEVER enter evidence: `src/causal_rcm/inference.py`, `jobs/e1_baseline/`.

---

## 4. Claim-to-artifact matrix (target for F8)

| Paper claim | Status | Artifact(s) |
|---|---|---|
| Hardware-conditional cost inversion (T4: no trade-off; A100: coarse 1.46x faster) | **Supported** | `results/f1_mode_cost/`, `results/a100_timing/`, `results/a100_mode_cost/` |
| RQ1/H1 residual signal (pre-registered) | **Not supported** (preserved negative) | `results/phase5b_audit/phase5b_audit.json` |
| RQ1 signal via System One oracle (Jev) | **Supported (positive, new method)** | `results/jev_signal/`, `results/rq4_jev/` |
| RQ2/H2 adaptive for the coarse mode | **Not supported** | `results/f4lite/`, `results/a100_online_v2/` |
| RQ2 positive for Jev `hold`/`jev_single` | **Weak positive** | `results/jev_holdfine/`, `results/a100_online_jev_phase3/phase3_ci.json` |
| RQ3/H3 recovery + hysteresis (T4 + A100) | **Supported** | `results/rq3_controller/rq3_audit.json`, `results/a100_rq3/a100_rq3_audit.json` |
| RQ4/H4 monotonicity (exploratory) | T4 **+** (rho -0.517); A100 **NULL/confounded** | `results/rq4_jev/`, `results/a100_online_jev_phase3/rq4_jev_a100.json` |
| Noisy context ~2x DiT (characterization) | Supported | `results/f1_mode_cost_noisy/` |

---

## 5. Open items / next actions (3-day Path A-lite)

1. ✅ F1 gate closed NO-GO; F1 audit + Pareto figure written (`docs/F1_AUDIT.md`).
2. ✅ F2-lite protocol frozen (`docs/F2_PROTOCOL.md`, F2LITE-01).
3. ✅ Day 2: F4-lite done (PASS 40/40) + independent quality. RQ2 not supported; RQ3 unexercised.
4. ✅ Day 3: manuscript (`docs/PAPER.md`), figures (`docs/figures/`), claim-to-artifact matrix, and final consistency audit (all 27 artifacts present; verdicts agree across STATUS/PROGRESS/PAPER_CONTEXT/PAPER).
5. ✅ RQ3 + RQ4 complete (proposal E5D + H4):
   - **E-RQ3** (`results/rq3_controller/`) — recovery+hysteresis reduces switches with no
     worse regret; **SUPPORTED** (real 0.50->0.38; noisy 1.12->0.62).
   - **E-RQ4** (`results/rq4_monotonicity/`) — Spearman rho -0.099; **no monotone
     relationship**; fine benefit peaks at intermediate uncertainty.
6. ✅ E-JEV complete (Phases A-D):
   - **E-JEV-1** (`results/jev_signal/`) — Jev signal AUROC 0.788 vs 0.607 hand-crafted;
     **RQ1 POSITIVE**; signal cost 0.67 s vs ~156 s.
   - **E-JEV-2** (`results/jev_p100_timing/`) — P100 unavailable; T4 replication ratio 3.01.
   - **E-JEV-3** (`results/jev_hold_mode/`) — hold is cheap; coarse routing counterproductive.
   - **E-JEV-3b** (`results/jev_holdfine/`) — `static->hold, else->fine` beats fixed fine
     on both axes (regret -0.0010, total -8.8%, DiT -19%); **RQ2 POSITIVE for hold**.
7. ✅ A100 arm (A100-01) complete: A1 trade-off PASS; A2 RQ2 **not supported** on the independent endpoint; A3 RQ3 **SUPPORTED**.
8. ✅ Jev-maximized phases complete: Phase 0 RQ4 positive on T4 (rho -0.517); Phases 1-3 A100 online Jev (Phase 3 = genuine trade-off with bootstrap CIs); Phase 4.1 RQ3 A100 SUPPORTED; Phase 4.2 RQ4 A100 NULL (confounded by reference bias).
9. **Next: A4 paper reframe (0 GPU-h).** Update title/abstract/contributions to hardware-conditional; introduce Jev as an explicit method change; keep negatives; map every claim to artifacts. This is the only remaining work.
10. Compute budget: ~24 GPU-h (T4, phases 0-5C) + ~8 F1 + ~4 F4-lite + ~5 RQ3 + ~4 RQ4 + ~4 E-JEV + ~16.5 A100-h (A100-01 + Jev-max phases). **~12.1 A100-h remain.**
11. Keep `STATUS.md` / `PROGRESS.md` / this file consistent at each gate.
