# A100-01 frozen protocol (hardware-conditional arm)

**Version:** A100-01, frozen 2026-10-12 before any A100 result. **Scope:** a
hardware-conditional re-test of the adaptive temporal-granularity study
(Causal-rCM / Wan2.1-1.3B-T2V) on a Google Colab **A100-SXM4-80GB**. Text-to-video
only; action conditioning remains out of scope.

**Why this arm exists.** The F1 T4 sweep found no quality-cost trade-off: `c3-3`
coarse had ~3x fewer NFE but ~3x heavier forwards, so it was not cheaper in wall-clock
DiT time, and independent endpoints showed no coarse quality advantage. The E-JEV-2
re-time (`results/a100_timing/`) then showed the per-forward coarse/fine ratio is
**hardware-specific**: T4 = 3.01 (break-even 2.33 -> coarse slower), A100 = **1.60**
(< break-even -> coarse **1.3-1.9x faster**). A100-01 tests whether that flipped cost
axis produces a genuine, claimable quality-cost trade-off and a positive adaptive result.

**This is a new hardware arm. It must never overwrite the preserved T4 evidence**
(Phase 5B residual AUROC 0.491, F4-lite, F1 NO-GO, RQ3, RQ4, JEV). T4 artifacts stay
the comparison baseline.

## 1. Hardware and provenance

- **GPU:** Google Colab A100-SXM4-80GB (Ampere, FlashAttention-capable), fp16.
- **Backbone:** Causal-rCM Wan2.1-T2V-1.3B-480p; checkpoints `c1-1_step2`,
  `c1-1_step4`, `c3-3_step2`, `c3-3_step4` (private dataset `ajjisan/causal-rcm-ckpts`).
- **Text/VAE:** `ajjisan/wan21-t5` (umT5-XXL), `ajjisan/wan21-dit-vae`.
- **Upstream code:** `https://github.com/NVlabs/rcm` pinned at
  `ed3cb14dd936f92cdc9f9381af7369991509b41f` (unchanged since F4-lite; verified
  2026-10-12 via `git ls-remote`).
- **Recipe:** the verified fp16 recipe (patched dtype in
  `wan2pt1_t2v_causal_infer.py` and `rcm/utils/umt5.py`, local tokenizer,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, DiT freed before VAE decode).

## 2. Difference from the timing-only A100 run

`results/a100_timing/a100_audit.json` used a **dummy text embedding and no VAE decode**.
A100-01 uses **real umT5 text encoding and real VAE decode**, so both the mode-dependent
DiT cost and the total wall-clock (VAE-dominated on T4) are measured on the A100.

## 3. Modes (identical to F1.3 for direct comparability)

| mode | ckpt | first/chunk | steps per chunk | schedule |
|---|---|---|---|---|
| ref_c1_1_s4 | c1-1_step4 | 1/1 | 4 | `15/16 5/6 5/8` |
| fine_c1_1_s1 | c1-1_step2 | 1/1 | [4,1] | `15/16,5/6,5/8;` |
| fine_c1_1_s2 | c1-1_step2 | 1/1 | [4,2] | `15/16,5/6,5/8;5/6` |
| coarse_c3_3_s4 | c3-3_step4 | 3/3 | 4 | `15/16 5/6 5/8` |
| coarse_c3_3_s2 | c3-3_step2 | 3/3 | [4,2] | `15/16,5/6,5/8;5/6` |
| coarse_c3_3_s1 | c3-3_step2 | 3/3 | [4,1] | `15/16,5/6,5/8;` |

## 4. Phases and gates

- **A1 (mode cost + quality).** 6 modes x horizons {45,57,81} x the 3 F1 development
  prompts, seed 17. Same prompts as F1 so T4-vs-A100 is a clean hardware comparison.
  Real T5 + VAE. **Gate:** does at least one coarse mode beat the best fine mode on
  wall-clock DiT at usable quality, with an independent endpoint not reversing it?
- **A2 (online RQ2).** Fresh prompts (this manifest, held-out split), 57 frames, shared
  coarse prefix. Policies: fixed fine/coarse/hold, Jev-routed (`static->hold, else->fine`),
  random matched, offline oracle. Independent endpoint + measured cost Pareto.
  **Gate (F2LITE-01):** adaptive improves the independent endpoint at matched cost, or
  matches it at >=15% lower DiT; must beat random same-budget.
- **A3 (RQ3 + RQ4).** Recovery/hysteresis ablation and uncertainty-benefit monotonicity
  on A100 traces.
- **A4 (paper).** Reframe to hardware-conditional; Jev introduced explicitly as a method
  change (not in `proposal.tex`); novelty refresh; claim-to-artifact matrix.

## 5. Endpoints and cost

- **Primary (independent):** CLIP ViT-B/32 subject consistency and text alignment; the
  classical no-reference suite (temporal warp, frame-diff, flow magnitude, drift).
- **Secondary (internal diagnostic):** decoded-pixel L1 to the c1-1 step-4 reference.
  Flagged as possibly favoring the fine family; never used alone.
- **Cost:** DiT s, VAE s, total s, NFE, true forward count, peak VRAM. VAE reported
  separately because it dominates total and is mode-independent.

## 6. Statistics and anti-leakage

- Unit of analysis = prompt. A1 is a characterization (3 dev prompts), not confirmatory.
- A2/A3 use the fresh split in `config/a100_manifest.json` (disjoint from Phase 5B and
  F4-lite); thresholds fit on calibration only.
- Paired per-prompt differences; descriptive ranges where n is small.

## 7. Explicit non-claims

No action-conditioned claim; no KV-cache-preservation claim; no FVD/PSNR/LPIPS; no claim
that coarse is more perceptually faithful. T4 negatives remain reported as negatives.

## 9. A1 result (2026-10-12) — PASS, trade-off confirmed on A100

`colab/a100_sweep.py` on A100-SXM4-80GB, real umT5 + VAE, 3 dev prompts x 45/57/81 x 6
modes, seed 17 = 54/54 complete. Artifacts `results/a100_mode_cost/` (results.jsonl,
summary, a100_quality_classical.json, a100_quality_clip.json, 54 videos on the VM).
Audit `scripts/a100_mode_cost_audit.py` -> `results/a100_mode_cost/a100_mode_cost_audit.json`.

Means over 9 cases (DiT is the mode-dependent cost; VAE ~2.4 s, vs ~156 s on T4):

| mode | DiT s | VAE s | total s | NFE | refL1 | CLIP subj.cons | text_align | temporal_warp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ref_c1_1_s4 | 9.10 | 2.52 | 18.40 | 64.0 | 0.0000 | 0.9914 | 0.3021 | 0.0215 |
| fine_c1_1_s1 | 3.92 | 2.38 | 12.95 | 19.0 | 0.1769 | 0.9890 | 0.2974 | 0.0226 |
| fine_c1_1_s2 | 5.62 | 2.38 | 14.68 | 34.0 | 0.1697 | 0.9907 | 0.2977 | 0.0213 |
| coarse_c3_3_s4 | 5.58 | 2.39 | 14.71 | 21.3 | 0.2712 | 0.9903 | 0.2987 | 0.0262 |
| coarse_c3_3_s2 | 3.65 | 2.39 | 12.59 | 12.7 | 0.2571 | 0.9880 | 0.2997 | 0.0331 |
| coarse_c3_3_s1 | **2.69** | 2.38 | **11.66** | **8.3** | 0.2575 | 0.9880 | 0.3010 | 0.0329 |

**Verdict (PASS):** coarse/fine per-forward ratio **1.75 < break-even 2.33** -> coarse
`c3-3 s1` is **1.46x faster in wall-clock DiT** than `fine c1-1 s1` (2.69 vs 3.92 s), while
being **less accurate** (internal refL1 0.258 vs 0.177; lower CLIP subject consistency;
higher temporal warp). This is a **genuine quality-cost trade-off on the A100** — the
precondition that was absent on the T4 (where fine dominated both axes). Independent
endpoints agree on direction (fine smoother/more consistent; coarse slightly higher text
alignment). The paper's central claim is therefore **hardware-conditional**, not universal.

## 8. Provenance / artifacts (target paths)

- `results/a100_mode_cost/` (A1): `results.jsonl`, `a100_mode_cost_summary.json`,
  `a100_quality_clip.json`, `a100_quality_classical.json`, `videos/`.
- `results/a100_online/` (A2), `results/a100_rq3/`, `results/a100_rq4/` (A3).
- `colab/a100_sweep.py` (A1 kernel), `config/a100_manifest.json` (fresh split).

## 10. A2/A3 status and session blocker (2026-10-12)

**A2 v1 (wrong coarse) — executed, RQ2 not supported as configured.** `colab/a100_online.py`
initially used the expensive c3-3 **step-4** coarse (DiT 4.43 s > fine 3.82 s) and the T4 warp
threshold, so `adaptive_warp` chose fine 92% of the time and matched `fixed_fine`. Archived
`results/a100_online/` (84 videos) for comparison.

**A2 v2 (corrected) — COMPLETE (72/72 PASS, 2026-10-13).** coarse = cheap c3-3 **step-1**,
warp threshold refit on A100 signals to **-4.0458**. `results/a100_online_v2/` (results.jsonl,
a100_online_v2_summary.json, a100_online_audit.json, a100_online_v2_quality_clip.json,
a100_online_v2_quality_classical.json, 84 videos). Means over 12 fresh prompts:

| policy | fine% | refL1 | DiT s | VAE s | total s | NFE | CLIP subj.cons | temporal_warp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_coarse | 0 | 0.2728 | 2.55 | 2.22 | 45.5 | 8.0 | 0.9430 | 0.0435 |
| fixed_fine | 100 | 0.2832 | 3.95 | 2.22 | 46.8 | 16.0 | **0.9836** | **0.0217** |
| fixed_hold | 0 | 0.2674 | **0.77** | 2.23 | **12.5** | 4.0 | 0.9926 | 0.0223 |
| random50 | 50 | 0.2789 | 3.24 | 2.22 | 46.2 | 12.0 | 0.9588 | 0.0345 |
| alternating50 | 50 | 0.2784 | 3.25 | 2.23 | 46.8 | 12.0 | 0.9589 | 0.0344 |
| adaptive_warp | 21 | 0.2733 | 2.85 | 6.45 | 52.3 | 9.7 | 0.9492 | 0.0394 |

**Internal refL1:** adaptive beats fixed_fine by +3.5% and uses -27.9% DiT -> `f2lite01_pass=true`.
**Independent endpoints reverse it:** CLIP subject consistency fine 0.9836 > adaptive 0.9492 >
coarse 0.9430; classical temporal_warp fine 0.022 < adaptive 0.039. Adaptive's **total** (52.3 s)
exceeds fine (46.8 s) because the warp signal costs extra VAE decodes (6.45 s vs 2.22 s).
**Verdict: RQ2 is NOT supported on the independent endpoint.** The A100 quality-cost trade-off is
real (A1) and coarse/adaptive capture DiT savings, but those savings are paid in subject
consistency, and the online warp signal's own cost erases the total-wall-clock gain.

**A3 (RQ3 recovery/hysteresis ablation) — COMPLETE (72/72 PASS, 2026-10-13).** `colab/a100_rq3.py`,
signal from the A2 v2 `fixed_fine` videos (tau -4.93, sigma_u 4.06, lambda 8.13, R=2, noise
2*sigma_u). `results/a100_rq3/`. Means over 12 fresh prompts:

| controller | switches | fine% | AUH | terminal | switch_adj | interior_adj |
|---|---:|---:|---:|---:|---:|---:|
| threshold_only | 0.50 | 0.50 | 0.2767 | 0.2955 | 0.0672 | 0.0305 |
| hysteresis_only | 0.42 | 0.52 | 0.2772 | 0.2977 | 0.0601 | 0.0303 |
| recovery_only | 0.42 | 0.52 | 0.2772 | 0.2977 | 0.0601 | 0.0303 |
| full | 0.42 | 0.52 | 0.2772 | 0.2977 | 0.0601 | 0.0303 |
| threshold_only_noisy | 0.92 | 0.48 | 0.2741 | 0.2997 | 0.0793 | 0.0326 |
| full_noisy | 0.50 | 0.56 | 0.2718 | 0.2921 | 0.0551 | 0.0315 |

**Verdict (SUPPORTED):** recovery+hysteresis reduces switches 0.50 -> 0.42 (-17%) on the real
signal with no worse AUH/terminal regret, and 0.92 -> 0.50 (**-45%**) under 2-sigma signal noise
with slightly better regret. Transition cost is real (switch 0.067 vs interior 0.030) and reduced
by the full controller (0.060). Consistent with the T4 RQ3 result. (RQ4 is handled in §11; the
A100 re-test (Phase 4.2) is confounded and returns NULL, not a hardware contrast.)

**Session stability (resolved 2026-10-13).** The mid-run drops were a client-side
network/runtime-socket problem. With a stable high-speed connection, A2 v2 completed in one
~82 min session (72/72). Google Drive mount is NOT available on CLI-provisioned runtimes
(`ValueError: mount failed`), so persistence is via downloading partial results. All T4
artifacts untouched.

**GPU accounting (2026-10-13).** Balance 193.8 -> 147.8 compute units = **46.0 units used
~= 6.8 A100-h** at 6.77 units/hr; **remaining ~21.8 A100-h**.

**Remaining work.** A3 (RQ3/RQ4 on A100; kernel `colab/a100_rq3.py` ready) ~1.5 h; A4 paper
reframe 0 GPU-h. Estimated **~2-3 A100-hours** to finish. Never overwrite the T4 artifacts or
the archived A2 v1.

## 11. Jev-maximized phases and A4 status (2026-10-13/14)

After A3 the project ran a Jev-maximized extension (all on A100 except Phase 0). Recorded here
for provenance; full numbers are in `docs/PAPER_CONTEXT.md` §2.15 and `STATUS.md`.

- **Phase 0 (0 GPU, T4 data):** RQ4 becomes **positive** with the Jev difficulty signal
  (rho -0.517, AUROC 0.788, gate 1.72). `results/rq4_jev/`.
- **Phase 1 (A100, 96/96):** online Jev policies. `jev_holdfine` matches fixed_fine CLIP subject
  consistency (0.9796 vs 0.9774) at -35.5% DiT / -16.6% total; `jev_routed` -49% DiT / -19%
  total. `results/a100_online_jev/`.
- **Phase 2 (60/60):** signal-cost variants; single-decision `jev_single` is a 2.6x cheaper signal
  (7.4 s -> 2.8 s); the free latent proxy is rejected. `results/a100_online_jev_phase2/`.
- **Phase 3 (24 fresh prompts, 120/120):** `jev_single` DiT **-1.78 s [-2.20,-1.37]**, total
  **-7.39 s [-12.39,-3.16]** (both significant) at CLIP subject consistency **-0.020
  [-0.030,-0.010]** (significant small loss) -> a genuine quality-cost trade-off, not dominance.
  `results/a100_online_jev_phase3/`.
- **Phase 4.1 (72/72) — RQ3 on A100 SUPPORTED**, matches T4 (`results/a100_rq3/`).
- **Phase 4.2 — RQ4 on A100 NULL (confounded).** Jev difficulty vs A100 fine_gap rho **+0.059**
  (AUROC 0.451). The A100 internal reference **favours coarse** (mean fine_gap +0.0137; only
  38.5% favour fine), so the label is biased; the flip is a protocol/reference artifact, not an
  A100 hardware effect (the A1 dev-prompt sweep still had fine closer, 0.177 vs 0.258). The coarse
  operating point also differs (T4 RQ4 = c3-3 **step-4**; A100 online = c3-3 **step-1**) and the
  prompt splits are disjoint. `results/a100_online_jev_phase3/rq4_jev_a100.json`.

**GPU accounting (2026-10-14).** 193.8 -> 82.3 compute units = **111.5 units (~16.5 A100-h)
used**; **~12.1 A100-h remaining**.

**Remaining work.** **A4 paper reframe only (0 GPU-h).** No further experiments are required.
Never overwrite the T4 artifacts or the archived A2 v1.
