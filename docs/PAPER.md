# Adaptive temporal granularity in a causal video world model is hardware-conditional: a cost, quality, and signal audit on a T4 and an A100

**Draft v3 — 2026-10-14. Status: hardware-conditional (negative on a T4, trade-off present on
an A100) with a System One (Jev) signal, a supported recovery/hysteresis mechanism, and an
exploratory monotonicity result that is endpoint-dependent. Every number maps to a raw artifact
(see the claim-to-artifact matrix in §9).**

## Abstract

Adaptive temporal granularity has been proposed as a way to save compute in video world
models: generate easy segments coarsely and hard segments finely, deciding online from a
pre-generation uncertainty signal. We test this premise on a real causal video world model
(Causal-rCM, Wan2.1-1.3B text-to-video) on a single Tesla T4, using its two shipped chunk
families (frame-wise `c1-1` and chunk-3 `c3-3`) as the fine/coarse modes. Across a
six-configuration cost sweep, two independent no-reference quality endpoints (CLIP
ViT-B/32 and a classical optical-flow suite), a noisy-context inference variant, a
pre-registered signal test on 72 held-out cases, and an eight-prompt online-switching
demonstration, we find **no defensible quality–cost trade-off on a T4**. The `c3-3` family has
~3× fewer neural function evaluations but each forward pass is ~3× heavier, so it is
**not cheaper in wall-clock time**; it is also not consistently better on independent
quality — but this inversion is **hardware-specific**: on an A100 the same sweep makes
coarse **1.3–1.9× faster** than fine (per-forward ratio 1.60 vs break-even 2.33), so the
fine/coarse trade-off **does exist on adequate hardware**. A pre-registered uncertainty
signal failed to predict coarse-unsafe segments
(held-out AUROC 0.491), and an online warp-threshold policy made no within-video switches
and cost 2.2× the best fixed policy. We then replace the hand-crafted signal with a
**System One decision model (Jev, TypeSafe)**: it predicts coarse-unsafe segments at
**AUROC 0.788** (vs 0.607 best hand-crafted, 0.491 prior), at **0.67 s per decision**
versus ~156 s of VAE decoding — a near-free oracle off the GPU. A controlled controller
ablation **supports the recovery + hysteresis mechanism** (fewer switches, no worse
regret, most clearly under signal noise), and the fine-vs-coarse benefit is **non-monotone for
the warp signal but monotone once the Jev signal is used** (RQ4, §5.7). Finally, adding a **near-zero-cost "hold" mode** (copy the previous latent
frame) and routing it with the Jev signal (`static→hold, else→fine`) **beats fixed fine on
both quality and cost** (regret −0.0010, total −8.8%, DiT −19%). We then run a dedicated **A100 arm** with real umT5 text encoding and VAE decode on fresh
prompts. The six-mode sweep confirms the trade-off (coarse `c3-3` s1 **1.46× faster in DiT**
than fine `c1-1` s1, but less accurate on CLIP subject consistency and temporal smoothness);
an online warp-threshold policy does **not** beat fixed fine on the independent endpoint (it
trades subject consistency for DiT cost, and its signal decoding erases the total-wall-clock
gain); and the recovery + hysteresis controller **again cuts switches** (0.50→0.42 real,
0.92→0.50 under noise) with no worse regret. A Jev-routed online arm on the A100 then shows a
**single-decision Jev** oracle is 2.6× cheaper and, across 24 fresh prompts with bootstrap CIs,
saves **−1.78 s DiT [−2.20,−1.37]** and **−7.39 s total [−12.39,−3.16]** (both significant) at a
small significant **−0.020 [−0.030,−0.010]** CLIP subject-consistency cost — a genuine trade-off,
not dominance. The uncertainty–benefit **monotonicity** (RQ4) is positive on the T4 (ρ −0.517
with Jev) but **confounded on the A100** (ρ +0.059), because the internal reference there flatters
coarse. The paper therefore reports a
**hardware-conditional** result for the switching premise — **negative on a T4, with a genuine
trade-off on an A100** — plus **positive** results for the Jev uncertainty oracle, the recovery
mechanism, and a Jev-routed `hold` mode.

## 1 Introduction

Video world models that generate frames causally (chunk by chunk) invite a natural
efficiency idea: not all segments are equally hard, so switch to a cheaper "coarse"
generation mode when the model is confident and to an expensive "fine" mode when it is
not. The appeal is a better quality–compute frontier than either fixed mode.

This idea has two preconditions. (i) A **real cost gap**: the coarse mode must be cheaper
at usable quality. (ii) A **predictive signal**: some quantity available *before*
generation must identify segments where coarse would hurt. If either fails, adaptive
switching cannot help.

We audit both preconditions end-to-end on a real system and find both fail. Our
contributions are deliberately diagnostic:

1. A **mode cost sweep** (six documented operating regimes of two chunk families) showing
   that fewer steps do not imply less wall-clock time: the chunk-3 family pays ~3× per
   forward pass.
2. Two **independent no-reference quality endpoints** (CLIP ViT-B/32 subject consistency
   and text alignment; a classical optical-flow temporal suite) that do not reverse the
   cost verdict.
3. A **noisy-context** characterization (`context_from_last_step`) that roughly halves
   DiT time for *both* families, and therefore does not create a coarse advantage.
4. A pre-registered **signal test** (72 held-out prompt-seed cases) that fails, and an
   **online-switching demonstration** on eight fresh prompts showing fixed fine dominates.

We make no positive claim. The value is a reproducible account of why this particular
adaptive-granularity premise does not hold on this system, plus the operating-point trap
that can make it *look* like it holds.

## 2 Related work

Causal/chunk-wise video generation and consistency models (e.g., Causal-rCM) enable
frame-wise autoregressive video synthesis. Adaptive-compute and early-exit methods in
other domains (diffusion caching, token pruning, budgeted sampling) motivate switching
between generation regimes. We do not claim novelty for the *idea* of adaptive granularity.
Our contribution is a **hardware-conditional audit**: we show the premise's cost precondition is
GPU-dependent (absent on a T4, present on an A100), that the pre-generation *signal* is a binding
constraint (a hand-crafted signal fails, a System One decision model succeeds), that the
*independent quality endpoint* is a second binding constraint (the internal reference flatters
coarse on both GPUs), and that the recovery + hysteresis *controller* transfers across hardware.
A full closest-work review remains pending.

## 3 Background: modes and the operating-point trap

Causal generation proceeds over latent-frame chunks `[first_chunk_t | chunk_t | ...]`. The
two shipped families are `c1-1` (one initial latent frame, then one-frame chunks) and
`c3-3` (three initial frames, then three-frame chunks). Each family has 1-/2-/4-step
regimes with documented midpoint schedules.

The **operating-point trap**: an initial comparison used `c1-1` in its 1-step regime
against `c3-3` in its 4-step regime. Because the 4-step schedule multiplies the already
heavier `c3-3` forward pass, the nominal "coarse" mode was slower in all 72/72 held-out
cases (DiT 53.4 s vs 21.6 s). Naively this looks like "coarse is broken"; in fact it is a
schedule mismatch. Our sweep removes the trap by testing all regimes.

## 4 Experimental setup

- **Model:** Causal-rCM Wan2.1-T2V-1.3B-480p, checkpoints `c1-1_step2`, `c1-1_step4`,
  `c3-3_step2`, `c3-3_step2_noisy_ctx`, `c3-3_step4`. Text encoder umT5-XXL; Wan2.1 VAE.
- **Hardware:** Kaggle Tesla T4 (16 GB), fp16, PyTorch SDPA. Upstream commit pinned per run.
- **Recipe:** the verified fp16 single-file recipe (patched dtype, local tokenizer, DiT
  freed before VAE decode, `expandable_segments`). See `Mater_plan_final.md` PII.1.
- **Cost:** DiT seconds, VAE seconds, total seconds, NFE, true forward count, peak VRAM.
- **Quality:** (i) internal diagnostic = decoded-pixel L1 to a `c1-1` step-4 reference;
  (ii) independent = CLIP ViT-B/32 subject consistency and text alignment; (iii) classical
  optical-flow temporal suite. The internal reference is flagged as possibly favoring the
  `c1-1` family and is never used alone.
- **Prompts:** F1 uses three Phase 5B development prompts; F4-lite uses eight **fresh**
  VBench prompts disjoint from all prior manifests.

## 5 Results

### 5.1 Six-mode cost sweep (F1.3)

| mode | DiT (s) | VAE (s) | total (s) | NFE | refL1 ↓ | s/NFE |
|---|---:|---:|---:|---:|---:|---:|
| ref c1-1 s4 | 59.42 | 155.7 | 227.8 | 64.0 | 0.0000 | 0.93 |
| fine c1-1 s1 | **25.16** | 156.5 | 194.5 | 19.0 | **0.2019** | 1.32 |
| fine c1-1 s2 | 36.73 | 155.7 | 205.5 | 34.0 | 0.1930 | 1.08 |
| coarse c3-3 s4 | 62.05 | 155.7 | 232.1 | 21.3 | 0.2750 | 2.91 |
| coarse c3-3 s2 | 40.63 | 155.6 | 210.5 | 12.7 | 0.2615 | 3.20 |
| coarse c3-3 s1 | 29.72 | 155.9 | 197.6 | **8.3** | 0.2588 | 3.58 |

`c3-3` has ~3× fewer NFEs but each forward is ~3× heavier (3 latent frames per pass), so
it is **not cheaper in wall-clock DiT time**. `fine c1-1 s1` is both the cheapest DiT mode
and the closest to the reference. VAE decode dominates total time and is mode-independent.
**Figure:** `docs/figures/f1_mode_pareto.png`.

### 5.2 Independent quality does not reverse the verdict (F1.4)

Classical temporal metrics: coarse modes are **less temporally smooth** (higher warp and
frame-difference). CLIP: subject consistency and text alignment are mixed and small; no
consistent coarse advantage. (Tables in `docs/F1_AUDIT.md` §F1.4.)

### 5.3 Noisy context halves DiT for both families (F1.3b)

`context_from_last_step` (checkpoint `c3-3_step2_noisy_ctx`) reuses the final denoising
forward as the chunk context, removing one forward per chunk:

| mode | DiT (s) | NFE | forwards | refL1 ↓ |
|---|---:|---:|---:|---:|
| fine c1-1 s1 | 30.52 | 21 | 39 | 0.2056 |
| **fine c1-1 s1 noisy** | **15.70** | 21 | 22 | 0.2182 |
| coarse c3-3 s1 | 34.95 | 9 | 15 | 0.2550 |
| coarse c3-3 s1 noisy | 21.01 | 9 | 10 | 0.2464 |

The trick helps both families, so `fine noisy` (15.70 s, 0.218) still beats `coarse noisy`
(21.01 s, 0.246). It is a genuine efficiency characterization (~2× DiT), but not a
switching result and not our novelty.

### 5.4 Signal test (Phase 5B, RQ1/H1) — failed

Pre-registered on 12 calibration + 24 held-out prompts (seeds 17/29/41), 288 scored
segments. Held-out AUROC: residual 0.491, warp 0.454, motion 0.488, complexity 0.301.
Residual − warp difference 0.0367, 95% CI [−0.0493, 0.1194]. **RQ1 unsupported.**

### 5.5 Online switching (F4-lite, RQ2/H2) — not supported

Eight fresh prompts, one common coarse prefix then four 3-latent-frame segment decisions.

| policy | fine_frac | switches | refL1 ↓ | DiT (s) | VAE (s) | total (s) |
|---|---:|---:|---:|---:|---:|---:|
| fixed_coarse | 0.00 | 0.00 | 0.3186 | 35.0 | 110.4 | 202.1 |
| **fixed_fine** | 1.00 | 0.00 | **0.2998** | **18.1** | 110.9 | **184.1** |
| random50 | 0.50 | 2.12 | 0.3102 | 26.5 | 110.7 | 187.5 |
| alternating50 | 0.50 | 3.00 | 0.3058 | 26.5 | 110.8 | 187.3 |
| adaptive_warp | 0.75 | 0.00 | 0.2949 | 22.1 | 321.6 | 397.8 |

Independent CLIP subject consistency: fixed fine **0.9840** (best), adaptive_warp 0.9803,
random50 0.9646, alternating50 0.9591, fixed coarse 0.9490. Temporal warp (lower better):
fixed fine 0.0303 (best), adaptive_warp 0.0429, coarse 0.0733.

`fixed_fine` is best on **both** independent quality and cost. `adaptive_warp` gains only
+1.6% on the *internal* reference, is worse on the independent endpoint, costs **2.2×**
total wall-clock (2.9× VAE from repeated context decoding), and made **0 within-video
switches** (it degenerated to a per-prompt fixed choice). **RQ2 not supported; RQ3/H3
unexercised.** **Figure:** `docs/figures/f4lite_policy_pareto.png`.

### 5.6 Recovery + hysteresis controller ablation (RQ3) — SUPPORTED

Six controllers on a shared coarse prefix over 8 fresh prompts at 57 frames, with a
switch-producing threshold (τ = pooled median warp, Λ = 2σ_u, R = 2) calibrated on
development videos. `results/rq3_controller/`, `scripts/rq3_audit.py`,
`docs/figures/rq3_controller_ablation.png`.

| controller | switches | fine_frac | AUH refL1 | terminal refL1 | switch-adj L1 | interior-adj L1 |
|---|---:|---:|---:|---:|---:|---:|
| threshold_only | 0.50 | 0.50 | 0.3133 | 0.3171 | 0.0364 | 0.0280 |
| hysteresis_only | 0.38 | 0.53 | 0.3116 | 0.3104 | 0.0321 | 0.0273 |
| recovery_only | 0.38 | 0.53 | 0.3116 | 0.3104 | 0.0321 | 0.0273 |
| full | 0.38 | 0.53 | 0.3116 | 0.3104 | 0.0321 | 0.0273 |
| threshold_only_noisy | 1.12 | 0.53 | 0.3159 | 0.3172 | 0.0630 | 0.0290 |
| full_noisy | 0.62 | 0.66 | 0.3150 | 0.3139 | 0.0399 | 0.0260 |

Recovery + hysteresis **reduces switch count without worsening regret**: real signal
0.50 → 0.38 (AUH regret −0.00168, terminal −0.00674, both improvements); under controlled
signal noise 1.12 → 0.62 (**−45%**) with AUH regret −0.00096. This is consistent with
Proposition 1 (Λ > 2σ_u): the band suppresses noise-driven flips. **Transition cost is
real and measured** — adjacent-frame L1 at switch boundaries exceeds interior L1
(0.0364 vs 0.0280), and `full_noisy` reduces the boundary cost (0.0399 vs 0.0630). On the
near-monotone real traces the Λ ∈ {0.5,1,2,4}σ_u and R ∈ {1,2,3} sweeps are flat (3
switches), so the mechanism's benefit is observable mainly under signal noise; reported
as such. **RQ3/H3 supported.**

### 5.7 Monotonicity of the uncertainty-benefit relationship (RQ4) — signal- and endpoint-dependent

RQ4 is exploratory (no required direction). On the T4, over 80 paired segments (48 at 81
frames, 32 at 57 frames) with `fine_gap = regret_fine − regret_coarse`, the pre-generation
**warp** signal gives `results/rq4_monotonicity/`:

- Spearman ρ(warp, fine_gap) = **−0.099** (essentially no monotone relationship).
- Tercile bins: low −0.0225, mid **−0.0571**, high −0.0195 — non-monotone, peaking at
  intermediate uncertainty.

Replacing warp with the **Jev difficulty** signal on the same 80 segments (`results/rq4_jev/`)
flips the T4 result to **monotone positive**:

- Spearman ρ(Jev difficulty, fine_gap) = **−0.517** (AUROC 0.788); terciles −0.0006, −0.0430,
  −0.0558 → the fine benefit grows with difficulty. Gate threshold 1.72 (Youden 0.524).

So on the T4 the earlier non-monotonicity was a **signal-quality artifact**: warp is a poor
proxy, Jev a good one.

On the **A100** (Phase 4.2, `results/a100_online_jev_phase3/rq4_jev_a100.json`) the same Jev
signal gives ρ = **+0.059** (AUROC 0.451) — **no monotone relationship**. This is **confounded,
not a clean disconfirmation**: the A100 internal reference flatters coarse (mean fine_gap
**+0.0137**; only **38.5%** of segments favour fine), so the label has no stable fine-benefit to
track. The A100 A1 dev-prompt sweep still had fine closer to the reference (0.177 vs 0.258),
confirming the flip is a protocol/reference artifact rather than an A100 hardware effect (the
coarse operating point also differs: T4 RQ4 used c3-3 step-4, the A100 online job c3-3 step-1).
**RQ4/H4 (exploratory): signal- and endpoint-dependent — monotone with Jev on the T4,
confounded on the A100.**

### 5.8 A System One model as a near-free uncertainty oracle (RQ1, positive)

We replace the hand-crafted signal with **Jev** (TypeSafe), a text-only System One decision
model that returns typed, calibrated answers. State per segment = prompt + BLIP caption of
the pre-decision context frame + segment index + motion statistics; one call asks composite
questions (Noul drift-risk, Score difficulty, Choice motion). 80 paired segments (57 f and
81 f), 8 prompts, 56/80 coarse-unsafe positives.

| signal | AUROC | 95% CI (prompt-grouped) |
|---|---:|---|
| **Jev difficulty** | **0.788** | [0.575, 0.946] |
| Jev drift (noul) | 0.738 | [0.479, 0.945] |
| Jev motion (choice) | 0.703 | [0.472, 0.873] |
| motion (hand-crafted) | 0.607 | [0.260, 0.857] |
| warp (hand-crafted) | 0.586 | [0.222, 0.836] |
| complexity (hand-crafted) | 0.368 | [0.184, 0.562] |
| Phase 5B residual (prior) | 0.491 | — |

The best Jev signal beats the best hand-crafted signal by **+0.180 AUROC** and the prior
residual by **+0.297**; the CI excludes 0.491. Crucially, Jev costs **0.67 s** per decision
versus the warp signal's ~156 s VAE decode — roughly **230× cheaper and off the GPU**.
**RQ1 is supported with a System One oracle.** Figure `docs/figures/jev_signal_auroc.png`.

### 5.9 The cost inversion is hardware-specific (decisive)

The inversion is a property of the GPU, not of the model. On a Kaggle **T4** the coarse/fine
per-forward ratio is **3.01** (break-even 2.33 → coarse slower). On a Google Colab
**A100-SXM4-80GB**, the same timing sweep (timing-only, dummy text embedding, identical
modes and sampler; `colab/a100_timing.py`, `results/a100_timing/`) gives ratio **1.60**,
below break-even:

| horizon | mode | DiT (s) | NFE | forwards | s/forward |
|---|---:|---:|---:|---:|---:|
| 57 f | fine c1-1 s1 | **4.68** | 18 | 33 | 0.142 |
| 57 f | coarse c3-3 s4 | 4.99 | 20 | 25 | 0.200 |
| 57 f | **coarse c3-3 s1** | **2.45** | 8 | 13 | 0.188 |
| 81 f | fine c1-1 s1 | **4.83** | 24 | 45 | 0.107 |
| 81 f | coarse c3-3 s4 | 7.85 | 28 | 35 | 0.224 |
| 81 f | **coarse c3-3 s1** | **3.60** | 10 | 17 | 0.212 |

On the A100, `c3-3` s1 is **1.3–1.9× faster** than fine s1 (57 f: 2.45 s vs 4.68 s). Because
quality is hardware-independent (coarse stays less accurate: refL1 0.259 vs 0.202), a
genuine **quality–cost trade-off exists on the A100 and is absent on the T4** — the F1
NO-GO is a **T4 artifact**, and the adaptive-granularity premise is viable on adequate
hardware. Mechanistically, the A100 (Ampere, FlashAttention) amortizes the heavier 3-frame
coarse forward far better than the Turing T4. This reframes the contribution: the paper's
central claim is *conditional on hardware*, which is itself a useful finding.

### 5.10 A near-zero-cost `hold` mode makes the trade-off real (RQ2, weak positive)

`hold` copies the previous latent frame (no generation). Two controllers on 8 prompts,
57 frames, shared coarse prefix (`results/jev_hold_mode/`, `results/jev_holdfine/`;
figure `docs/figures/jev_hold_mode.png`).

| policy | hold | fine | coarse | meanRef ↓ | DiT (s) | total (s) |
|---|---:|---:|---:|---:|---:|---:|
| fixed_fine | 0.0 | 4.0 | 0.0 | 0.2998 | 18.6 | 183.2 |
| fixed_coarse | 0.0 | 0.0 | 4.0 | 0.3186 | 37.7 | 205.5 |
| fixed_hold | 4.0 | 0.0 | 0.0 | 0.3108 | 6.5 | 136.4 |
| jev_routed (static→hold, moderate→coarse, fast→fine) | 1.1 | 0.9 | 2.0 | 0.3085 | 24.9 | 182.3 |
| jev_routed_nohold (static→coarse) | 0.0 | 0.9 | 3.1 | 0.3131 | 33.7 | 201.4 |
| **jev_holdfine (static→hold, else→fine)** | 1.1 | 2.9 | 0.0 | **0.2988** | **15.1** | **167.0** |

Routing "moderate" segments to coarse is counterproductive (coarse is slower and less
accurate), so `jev_routed` does not beat fixed fine. But the `hold` mode is a genuinely
cheap mode, and `jev_holdfine` **beats fixed fine on both axes**: regret **−0.0010**
(holding is a better match than re-generating a static segment) and cost **total −16.2 s
(−8.8%)**, **DiT −3.5 s (−19%)**, VAE unchanged. Per the F2LITE-01 criterion this passes on
the mode-dependent (DiT) cost metric and is neutral-to-better on quality. **RQ2 is
positive for the hold mode** (weak: small quality delta, n = 8 prompts), while remaining
negative for the coarse mode.

### 5.11 A dedicated A100 arm: real-decode sweep, online policies, and RQ3

Protocol `docs/A100_PROTOCOL.md`; fresh split `config/a100_manifest.json`; upstream commit
pinned `ed3cb14`. Unlike §5.9 (timing-only), A1 uses **real umT5 text encoding and VAE decode**.

**A1 six-mode sweep (3 dev prompts × 45/57/81, 54/54).** `results/a100_mode_cost/`.

| mode | DiT s | VAE s | NFE | refL1↓ | CLIP subj.cons↑ | temporal_warp↓ |
|---|---:|---:|---:|---:|---:|---:|
| fine c1-1 s1 | 3.92 | 2.38 | 19 | 0.177 | 0.989 | 0.023 |
| coarse c3-3 s1 | 2.69 | 2.38 | 8.3 | 0.258 | 0.988 | 0.033 |
| coarse c3-3 s4 | 5.58 | 2.39 | 21 | 0.271 | 0.990 | 0.026 |

Coarse/fine per-forward ratio **1.75 < 2.33** → coarse **1.46× faster in DiT** but **less
accurate** (refL1 0.258 vs 0.177; lower CLIP subject consistency; higher temporal warp). VAE
≈2.4 s (vs ≈156 s on T4). **The quality–cost trade-off exists on the A100.**

**A2 v2 online policies (12 fresh prompts, 57 f, 72/72).** `results/a100_online_v2/`.

| policy | fine% | refL1↓ | DiT s | VAE s | total s | CLIP subj.cons↑ | temporal_warp↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_coarse | 0 | 0.2728 | 2.55 | 2.22 | 45.5 | 0.9430 | 0.0435 |
| fixed_fine | 100 | 0.2832 | 3.95 | 2.22 | 46.8 | **0.9836** | **0.0217** |
| fixed_hold | 0 | 0.2674 | **0.77** | 2.23 | **12.5** | 0.9926 | 0.0223 |
| adaptive_warp | 21 | 0.2733 | 2.85 | 6.45 | 52.3 | 0.9492 | 0.0394 |

On the **internal** refL1 the adaptive policy "passes" F2LITE-01 (+3.5% quality, −27.9% DiT),
but the **independent endpoints reverse it** (fine 0.9836 > adaptive 0.9492 > coarse 0.9430),
and the warp signal's extra VAE decodes make adaptive's **total** wall-clock exceed fixed fine
(52.3 s vs 46.8 s). **RQ2 remains not supported on the independent endpoint** — the same lesson
as the T4 F4-lite: the internal reference flatters coarse.

**A3 recovery + hysteresis (12 fresh prompts, 72/72) — SUPPORTED.** `results/a100_rq3/`.

| controller | switches | AUH↓ | terminal↓ |
|---|---:|---:|---:|
| threshold_only | 0.50 | 0.2767 | 0.2955 |
| full | 0.42 | 0.2772 | 0.2977 |
| threshold_only_noisy | 0.92 | 0.2741 | 0.2997 |
| full_noisy | 0.50 | 0.2718 | 0.2921 |

Switches 0.50→0.42 (−17%) on the real signal (regret essentially unchanged) and 0.92→0.50
(**−45%**) under 2σ signal noise (regret slightly better); the transition cost at switch
boundaries is real (0.067 vs 0.030 interior) and reduced by `full` (0.060). **RQ3/H3 supported
on the A100, consistent with T4.** (RQ4 is signal- and endpoint-dependent — see §5.7; the A100
re-test (Phase 4.2) is confounded and returns NULL, not a hardware contrast.)

### 5.12 Jev-routed online arm on the A100: single-decision oracle (RQ1/RQ2)

`results/a100_online_jev/`, `results/a100_online_jev_phase2/`,
`results/a100_online_jev_phase3/`; kernel `colab/a100_online_jev.py`. Fresh A100 splits, 57
frames, shared coarse prefix; the Jev signal is computed off the DiT path.

| policy | ΔDiT (s) vs fixed_fine | Δtotal (s) | ΔCLIP subj.cons | source |
|---|---:|---:|---:|---|
| jev_holdfine | — | — | ~0 (0.9796 vs 0.9774) | Phase 1 (n = 12) |
| **jev_single** | **−1.78 [−2.20,−1.37]** | **−7.39 [−12.39,−3.16]** | **−0.020 [−0.030,−0.010]** | Phase 3 (n = 24, 10k bootstrap) |

`jev_holdfine` (static→hold, else→fine) matches fixed fine's CLIP subject consistency at
**−35.5% DiT / −16.6% total**; `jev_routed` reaches −49% DiT / −19% total. Phase 2 shows a
**single-decision** Jev (one query per prompt) cuts the signal cost from 7.4 s to 2.8 s (2.6×),
while a free latent proxy is rejected (CLIP 0.9804 vs 0.9929). Phase 3, on 24 fresh prompts
with bootstrap CIs, gives the honest reading: a **genuine quality–cost trade-off** — significant
DiT and total savings at a small but significant CLIP subject-consistency loss — **not
dominance**.

## 6 Discussion

The failure is structural, not a tuning artifact. The chunk-3 family amortizes a longer
temporal stride over fewer, heavier forward passes; on a T4 the per-pass cost dominates, so
the NFE saving never becomes a wall-clock saving. Because the coarse family is neither
cheaper nor better, no switching policy can improve the frontier, and the only "gain" a
threshold can produce is a small internal-reference improvement bought at large extra
decode cost — exactly the kind of gain the dominance gate is designed to reject.

The same sweep on an A100 reverses the first clause: the heavier 3-frame coarse forward is
amortized, so coarse becomes **1.46× cheaper in DiT** — the cost gap the T4 lacked (§5.11).
But even with a real cost axis the online warp policy does not win on the independent endpoint:
it buys DiT savings by accepting lower subject consistency, and its own signal decoding erases
the total-wall-clock gain. The honest statement is therefore two-part: the **cost** precondition
is hardware-conditional, while the **signal** and **independent-endpoint** preconditions still
bind on both GPUs.

A secondary lesson: NFE is not GPU time, and an internal reference built from the fine
family can silently reward fine. Both were caught only by sweeping regimes and adding
independent endpoints.

A third lesson concerns the **signal and the endpoint**. The pre-generation signal is a
binding constraint: the hand-crafted warp signal fails (Phase 5B AUROC 0.491) and even the
best hand-crafted signal is weak, whereas a System One decision model (Jev) reaches AUROC
0.788 at 0.67 s and runs off the GPU. With Jev the online trade-off becomes real but modest
(§5.12): the savings are significant and so is the small quality loss. The **endpoint** is the
last binding constraint and the source of the RQ4 discrepancy: the internal c1-1 step-4
reference is a fine-family proxy whose preference is not stable — it flatters fine on the T4
RQ4 segments (mean fine_gap −0.033) and coarse on the A100 online split (+0.014) — so a
correlation measured against it (RQ4) can flip sign between datasets. Independent endpoints
are therefore not optional.

## 7 Limitations

Small samples and one seed throughout: three development prompts (F1) and eight fresh prompts
(F4-lite) on the T4; 3 development (A1), 12 fresh (A2 v2, A3) and 24 fresh (Jev Phase 3) prompts
on the A100 — no confirmatory power for a universal claim. The Jev oracle is a text-only System
One model whose input is a BLIP caption plus motion statistics of the pre-decision frame; caption
noise propagates to the signal, and its AUROC (0.788) rests on 80 T4 segments from 8 prompts with
wide CIs. RQ2 is a **trade-off, not dominance**: on A100 Phase 3 the DiT/total savings are
significant and so is the −0.020 CLIP subject-consistency loss. RQ4 is **exploratory and
confounded** on the A100 because the internal reference flatters coarse there. The internal
reference is a fine-family proxy and a diagnostic only; the independent endpoints are the
safeguard, but CLIP text alignment on short prompts is noisy. Total wall-clock is dominated by
VAE decode, which is mode-independent; the mode-dependent cost is DiT time. We test one model
pair on two GPUs; other backbones or hardware where the DiT is more throughput-bound could
behave differently. No action-conditioned results (no compatible checkpoint).

## 8 Conclusion and future work

For the fine/coarse switching premise there is no viable adaptive temporal-granularity
trade-off **on a T4** with the shipped `c3-3` coarse mode: coarse is neither cheaper nor better
there. **On an A100 the cost precondition holds** — coarse `c3-3` s1 is 1.46× faster in DiT than
fine in the real-decode arm (§5.11) — yet the hand-crafted online policy still does not beat
fixed fine on the independent endpoint (its DiT savings are paid in subject consistency and its
own decode cost erases the total-wall-clock gain). The central claim is therefore
**hardware-conditional on cost and signal/endpoint-limited**, not universal. Three positive
results survive. First, a **System One decision model (Jev) is a far better and near-free
uncertainty oracle** than hand-crafted signals (AUROC 0.788 vs 0.607, at 0.67 s vs ~156 s),
moving the signal cost off the GPU; on the T4 it also makes the uncertainty–benefit relationship
**monotone** (ρ −0.517). Second, with that oracle plus a **near-zero-cost `hold` mode**, adaptive
routing (`static→hold, else→fine`) **beats fixed fine on both quality and cost** (regret −0.0010,
total −8.8%, DiT −19%); on the A100 a **single-decision Jev** saves −1.78 s DiT and −7.39 s total
(both significant) at a small significant −0.020 CLIP cost — a real trade-off, not dominance.
Third, the **recovery + hysteresis mechanism is supported on both GPUs**. RQ4 remains
**exploratory and endpoint-dependent** (monotone on the T4 with Jev; confounded on the A100). We
recommend that future work (i) verify a wall-clock cost gap at matched quality before building a
controller, (ii) use independent quality endpoints because the internal reference can flatter
either mode, (iii) prefer genuinely cheap modes (hold/skip) over a heavier "coarse" chunk, and
(iv) use a strong, off-GPU signal (System One) rather than a lagged hand-crafted probe.
Action-conditioned control, a larger held-out study of the Jev oracle, blinded human evaluation,
and a FlashAttention-capable GPU test remain open.

## 9 Claim-to-artifact matrix

| Claim | Status | Artifact |
|---|---|---|
| Six-mode cost sweep; coarse not cheaper | Supported (descriptive) | `results/f1_mode_cost/f1_mode_cost_audit.json` |
| Independent quality does not favor coarse | Supported (descriptive) | `results/f1_mode_cost/f1_quality_classical.json`, `f1_quality_clip.json` |
| Noisy context ~2× DiT, both families | Supported (descriptive) | `results/f1_mode_cost_noisy/f1_mode_cost_noisy_audit.json` |
| RQ1/H1 signal fails | Supported (negative) | `results/phase5b_audit/phase5b_audit.json` |
| RQ2/H2 adaptive not supported (coarse) | Supported (negative) | `results/f4lite/f4lite_audit.json`, `f4lite_quality.json` |
| RQ1 signal via System One model (Jev) | **Supported (positive)** | `results/jev_signal/jev_audit.json` |
| Cost inversion is hardware-specific (A100 flips it) | **Supported** | `results/a100_timing/a100_audit.json`, `results/jev_p100_timing/jev_p100_audit.json` |
| RQ2 positive for the Jev-routed `hold` mode | **Supported (weak positive)** | `results/jev_holdfine/jev_holdfine_audit.json` |
| RQ3/H3 recovery + hysteresis helps | **Supported** | `results/rq3_controller/rq3_audit.json` |
| RQ4/H4 monotonicity (warp signal, T4) | No monotone relationship (exploratory) | `results/rq4_monotonicity/rq4_audit.json` |
| RQ4/H4 monotonicity (Jev signal, T4) | **Monotone positive** (ρ −0.517) | `results/rq4_jev/jev_rq4_audit.json` |
| RQ4/H4 monotonicity (Jev signal, A100) | **NULL / confounded** (ρ +0.059) | `results/a100_online_jev_phase3/rq4_jev_a100.json` |
| Cost inversion in Phase 5B | Supported | `results/f1_timing_audit/f1_timing_audit.json` |
| A100 real-decode six-mode sweep: quality–cost trade-off exists | **Supported (positive)** | `results/a100_mode_cost/a100_mode_cost_audit.json`, `a100_quality_clip.json`, `a100_quality_classical.json` |
| A100 online policies: RQ2 not supported on the independent endpoint | Supported (negative) | `results/a100_online_v2/a100_online_audit.json`, `a100_online_v2_quality_clip.json`, `a100_online_v2_quality_classical.json` |
| A100 RQ3 recovery + hysteresis | **Supported** | `results/a100_rq3/a100_rq3_audit.json` |
| A100 online Jev: `jev_holdfine` matches fine CLIP at −35.5% DiT | Weak positive | `results/a100_online_jev/` |
| A100 single-decision Jev: significant DiT/total savings, small CLIP cost | **Trade-off (positive)** | `results/a100_online_jev_phase3/phase3_ci.json` |
| Single-decision Jev 2.6× cheaper signal; free latent proxy rejected | Supported | `results/a100_online_jev_phase2/` |

## 10 Reproduction

```
python scripts/f1_timing_audit.py
python scripts/f1_mode_cost_audit.py
python scripts/f1_quality_classical.py
python scripts/f1_quality_clip.py
python scripts/f1_mode_cost_noisy_audit.py
python scripts/f1_pareto_figure.py
python scripts/f4lite_audit.py
python scripts/f4lite_quality.py
python scripts/f4lite_figure.py
python scripts/rq3_calibrate_threshold.py
python scripts/rq3_audit.py
python scripts/rq3_figure.py
python scripts/rq4_audit.py
python scripts/rq4_figure.py
python scripts/jev_build_states.py
python scripts/jev_query.py
python scripts/jev_audit.py
python scripts/jev_figure.py
python scripts/jev_p100_audit.py
python scripts/jev_hold_audit.py
python scripts/jev_holdfine_audit.py
python scripts/jev_hold_figure.py
python scripts/jev_p100_audit.py
python scripts/a100_audit.py
python scripts/a100_mode_cost_audit.py
python scripts/a100_online_audit.py a100_online_v2
python scripts/a100_rq3_audit.py
python scripts/a100_online_jev_audit.py
python scripts/jev_rq4_audit.py
python scripts/phase4_rq4_jev.py
```
Kaggle kernels: `ajjisan/f1-mode-cost`, `ajjisan/f1-mode-cost-noisy`,
`ajjisan/f4-lite-online-switching`, `ajjisan/f-rq3-controller-ablation`,
`ajjisan/f-rq4-monotonicity`, `ajjisan/jev-p100-timing`, `ajjisan/jev-hold-mode`,
`ajjisan/jev-holdfine`. Google Colab (A100): `colab/a100_timing.py` (timing-only),
`colab/a100_sweep.py` (A1), `colab/a100_online.py` (A2 v2), `colab/a100_online_jev.py`
(Jev-max Phases 1-3), `colab/a100_rq3.py` (A3 / Phase 4.1); see
`colab/README.md` and `docs/A100_PROTOCOL.md`.
Manifests: `config/phase5_prompt_manifest.json`, `config/f4lite_manifest.json`,
`config/a100_manifest.json`, `config/a100_phase3_manifest.json`, `config/jev_routing.json`.
Protocols: `docs/F2_PROTOCOL.md` (F2LITE-01), `docs/F1_AUDIT.md`; signal calibration
`config/rq3_threshold.json`, `config/jev_routing.json`. The Jev API key is read from the
`TYPESAFE_API_KEY` environment variable and is never committed.
