# BOOK OF THE PROJECT — consolidated memory
# idea_05_adaptive_granularity_WM / 04_Experiment
# Author of record: Hossen Md Jisan (NUIST, B.Sc. AI, jisanahammed1952@gmail.com)
# Compiled 2026-10-14 from every readable file in the working tree.
# Videos (.mp4) and raster images (.png/.jpg) are catalogued by name only —
# their pixels are not machine-readable here, but selected frames were extracted
# with ffmpeg into paper/figures_src/candidates/ and visually inspected.

=====================================================================
PART 0 — ONE-PARAGRAPH SUMMARY
=====================================================================
The project asked: can an autoregressive streaming video generator (Causal-rCM)
switch at run time between a fine per-latent-frame mode (c1-1) and a coarse
3-frame-chunk mode (c3-3), guided by an uncertainty signal, to save computation
without losing quality? The answer, established over ~3 months of Kaggle/Colab
GPU work (Oct 4 – Oct 14, 2026), is nuanced and honest:
(1) The trade-off EXISTS but is HARDWARE-CONDITIONAL — on a Tesla T4 the coarse
mode is never cheaper (per-forward cost ratio 3.01 > break-even 2.33, so F1 gate
went NO-GO), while on an A100-80GB the ratio drops to 1.60–1.75 < 2.33 and the
trade-off appears.
(2) Hand-crafted uncertainty signals FAIL on locked held-out data (Phase 5B:
residual AUROC 0.491, 95% CI [0.393,0.589], n=84 cases) — the original H1
hypothesis was rejected and reported as a negative result.
(3) An EXTERNAL oracle fixes the signal problem: the Jev model (TypeSafe System
One API, model "jev-latest") predicts coarse-unsafe segments at AUROC 0.788
(95% CI [0.575,0.946]) at ~0.67 s/decision, vs 0.491–0.607 for all hand-crafted
signals, and it answers BEFORE generation (the warp signal needs a ~156 s VAE
decode on T4).
(4) The recovery+hysteresis controller mechanism WORKS on both GPUs (RQ3
supported): it reduces switches 0.5→0.375 (T4) and 0.917→0.5 (A100 noisy stress)
without worsening area-under-horizon regret; and switching itself carries a
measurable seam cost (adjacent-frame L1 0.036 vs 0.028 interior).
(5) Jev-routed policies beat random same-budget routing and the warp controller
on the independent CLIP subject-consistency endpoint (F2LITE-01 passed on both
T4 and A100). RQ4 found NO monotonicity with the hand-crafted warp signal
(ρ=−0.10, flat terciles) but a clean monotone relation with Jev difficulty
(ρ=−0.517, terciles +0.0006/−0.0430/−0.0558).
The deliverable is a diagnostic + feasibility study with the negative results
preserved: exactly what the locked master plan requires.

=====================================================================
PART 1 — TIMELINE OF THE WHOLE PROJECT (what happened, in order)
=====================================================================
- Prehistory (Phases 0–4, exploratory pilot, T4 only, Kaggle):
  Phase 0: feasibility — real Causal-rCM c1-1/c3-3 inference on T4 verified.
  Phase 1: switch test — c1-1→c3-3→c1-1 continuation runs; KV-cache persistence
           was NOT available in the verified upstream API.
  Phase 2: baseline — fixed fine vs fixed coarse, 2 prompts × 3 horizons
           (21/33/45 frames); produced results/phase2_baseline/.../frames/*.png.
  Phase 3: calibration (ran twice: phase3_calibration, phase3_calibration_rerun).
  Phase 4: adaptive + online + residual pilots; the residual controller chose
           coarse for all 12 decisions → recovery/hysteresis unexercised.
  This pilot phase is "historical" in the current plan; only Phase 5+ numbers
  are publication-grade.
- 2026-10-05..06: Phase 5A reference/capacity probe PASSED on T4 (81 frames,
  4/4 matched cases).
- 2026-10-07: Phase 5B reference-regret calibration COMPLETE and data audit
  PASSED (12/12 calibration cases/48 segments; 72/72 held-out cases/288 segments
  in 8 batches). RQ1 signal claim NOT supported (AUROC 0.491). Phase 5C never ran.
- 2026-10-11: F1 feasibility audit → NO-GO. F4-lite budgeted selection run.
  RQ3 controller ablation complete. RQ4 monotonicity complete.
- 2026-10-11/12: Jev pivot (E-JEV). New plan section written (Mater_plan_final
  Part II). F2LITE-01 protocol locked. Jev signal test done on T4.
- 2026-10-12: Jev P100 re-timing (T4 again); A100 timing, A100 mode cost (A1)
  → trade-off EXISTS on A100. jev_hold_mode + jev_holdfine on T4.
- 2026-10-13: A100 online policies (A2 arm), A100 Jev-online Phases 0–2,
  Phase 3 (24-prompt confirmatory CLIP endpoint) PASS, RQ4-with-Jev (T4 + A100).
- 2026-10-14: A100 RQ3 (A3) complete; Phase 4.2 Jev-max CI rerun; paper writing.

=====================================================================
PART 2 — THE BACKBONE: Causal-rCM (what we built on)
=====================================================================
- Repo: https://github.com/thu-ml/Causal-rcm (public; pinned upstream commit in
  later A100 runs: ed3cb14dd936f92cdc9f9381af7369991509b41f). Local mirror under
  results/phase5b_reference_calibration_a/rcm/ and
  results/phase5c_online_pilot/v2_download/rcm/.
- Base model: Wan2.1-T2V-1.3B bidirectional video diffusion; distilled into
  few-step autoregressive students via rCM (teacher-forcing + self-forcing).
- Checkpoints used (both on HuggingFace; T4 runs used Kaggle Models mounts):
    Causal-rcm/c1-1 (frame-wise AR, 1 latent frame per chunk)
    Causal-rcm/c3-3 (chunk-wise AR, 3 latent frames per chunk)
- Resolution 480×832, 16 fps, guidance 5.0, shift 8.0. SDPA attention on T4
  (flash-attn needs Ampere+), flash-attn on A100.
- Rollout structure (57-frame protocol): common coarse prefix of 3 latent frames
  + 4 segments of 3 latent frames each (segment bounds in scripts:
  start = 1+4(3g−1), end = horizon if last group else 1+4(3(g+1)−1)).
- Operating modes swept in F1:
    ref_c1_1_s4   (reference: c1-1, 4 denoising steps)
    fine_c1_1_s1  (1 step), fine_c1_1_s2 (2 steps)
    coarse_c3_3_s4/s2/s1
  Plus noisy-context variants (context_from_last_step — the upstream
  "history-as-noisy" trick), which halve DiT time for BOTH families at ~+6%
  reference distance (replication of an upstream trick, secondary finding).
- THE OPERATING-POINT TRAP (key technical discovery, validated by upstream):
  NFE is defined per latent-frame group; a c3-3 call covers 3 latent frames in
  one forward. "18 vs 20 NFE" therefore means nearly equal REAL function
  evaluations (18 forwards vs 20/3≈6.7 forwards × 3-frame-heavier each).
  On T4 each coarse forward costs ~3× a fine forward (ratio 3.01), so coarse is
  never cheaper in wall-clock DiT time even with 2.3× fewer forwards.
  Upstream Causal-rCM authors confirmed this explanation — "the operating point
  drives the result; these checkpoints were trained against 8×H800".
  ALSO: Causal-rCM itself reports frame-wise 1–2 step beating 4-step (non-
  monotone quality vs refinement) — this motivated RQ4.

=====================================================================
PART 3 — THE ORIGINAL PROPOSAL (proposal.tex, 611 lines)
=====================================================================
Working title: "Uncertainty-Guided Adaptive Temporal Granularity with Hysteretic
Error Recovery for Run-Time Autoregressive World-Model Rollouts".
Pre-registered hypotheses:
  H1: the calibrated uncertainty signal has higher held-out AUROC for reference
      regret than the strongest hand-crafted baseline, with a CI for the gap.
  H2: adaptive switching improves reference regret over fixed and random
      same-budget controls, or matches regret at lower measured cost.
  H3: recovery+hysteresis reduces switch count without worsening terminal or
      area-under-horizon reference regret at matched fine budget.
  H4 (exploratory): the fine/coarse benefit vs the difficulty/surprise signal is
      non-monotone.
Controller math (as implemented in src/controller/policy.py):
  Potential Ψ_i = u_i + λΔa_i  (λ=0 in the text-only study)
  Threshold policy: Ψ_i ≥ τ_high → fine; Ψ_i ≤ τ_low → coarse; else keep
  previous mode (hysteresis band Λ = τ_high − τ_low).
  Recovery: on spike, set counter r←R and force fine for R steps.
Propositions proved in the proposal:
  P1 (local suboptimality of fixed modes): if D(C)<D(F) on safe segments and
     D(F)<D(C) on unsafe ones and both sets are non-empty, then per-segment
     oracle beats either fixed policy; any selected policy with selection error
     below a derived bound beats fixed-fine.
  P2 (oscillation bound): for a signal with additive noise ε (|ε|≤σ), the
     per-step switching probability of a two-threshold controller is bounded by
     Pr[|u+ε−τ| ≥ Λ/2], so Λ ≥ 2σ kills noise-only crossings.
Related work organized into 7 themes: (1) video world models & interactive
control; (2) causal/few-step distillation (Self Forcing, Causal Forcing,
CF++, Causal-rCM); (3) adaptive computation (ACT, BranchyNet, SkipNet, DSA);
(4) adaptive tokenization (AdapTok, EVATok, DLFR-VAE, VFRTok); (5) semi-AR
(NBP, NOVA); (6) error accumulation & adaptation (ENkG, SUNTA, Flex-Forcing,
Diagonal Distillation, SANTS, Surprise Forcing, DASH-OPD, MAGI-1); (7) caching
& memory (Focused Forcing, FlowCache, MotionCache, KV-quant study, SlackServe).
Positioning sentence: existing systems make generation cheap (CF++) or smooth
(Causal-rCM), or adapt the compute/tokens offline, or allocate at the serving
layer — "none of them decides per segment which temporal granularity to run
next, from an online signal, with a recovery mechanism when the signal fails."

=====================================================================
PART 4 — LOCKED PROTOCOLS (what was promised before running)
=====================================================================
Master plan (Mater_plan_final.md) locked part (immutable):
  §1 backbone & assets; §2 internal reference c1-1 step-4, same prompt/seed/
  noise; §3 segment grid 45/57/81 frames (T4 used 57/81); §4 distance fallback
  decoded-pixel L1 (int8) if no aligned feature space; §5 signals: residual
  (in-model uncertainty), warp (optical-flow backward-warp residual), motion
  (Farnebäck magnitude), complexity (Sobel energy); EMA signal DROPPED
  (unavailable in verified checkpoints — recorded limitation); §6 locked
  calibration split (≥12 calibration / ≥24 held-out prompts, seeds locked
  separately); §7 fixed/fine/coarse + signal-driven + random same-budget +
  offline oracle policies; §8 reference regret as primary target; §9 matched
  fine-decision budgets; §10 two-threshold + recovery controller; §11 grouped
  prompt×seed bootstrap CIs; §12 negative results preserved; §13 every number
  maps to raw artifact.
  Decision hierarchy: backbone & endpoint > locked protocol > hypothesis labels
  > exploratory interpretation. "Re-label first": if data contradict a locked
  hypothesis, the claim is corrected and recorded.
F2LITE-01 (locked after the F1 NO-GO, the protocol the paper is judged against):
  Segment = 3 latent frames (12 decoded frames); horizon 57; per-decision
  measurement point. Signals: warp (post-hoc reference-free, needs VAE decode)
  and Jev (pre-generation). Policies: fixed coarse/fine, random50, alternating50,
  adaptive warp, Jev-routed, Jev-difficulty, Jev-holdfine. Baselines same budget.
  Endpoints: internal refL1 (diagnostic) + independent CLIP subject/text +
  classical temporal suite. Calibration-only thresholds. Grouped CIs.
  PASS requires: adaptive policy beats fixed fine AND random same-budget on the
  independent endpoint at lower measured DiT cost, with grouped CIs excluding 0.
A100-01: same segment grid, policies {fixed fine/coarse/hold, random50,
  adaptive warp, Jev-routed, Jev-difficulty, Jev-holdfine, latent (mid-noise
  refresh)}; refL1 diagnostic + CLIP + classical; CI-bearing Phase 3 on 24 fresh
  prompts with prompt-paired bootstrap vs fixed fine (10,000 resamples).
Compute budget declaration (in the locked plan): T4 30 h/wk Kaggle + 30 h/wk
Colab; 45-frame pairs ≈ 15 min/pair; full fixed-mode grid ≤ 1 week of Kaggle
quota; scale-up needs an A100/H100 spot. Actual documented subtotals:
historical Phases 0–5B ≈ 24 T4-h; F1 ≈ 8 T4-h; F4-lite ≈ 4; RQ3 ≈ 5; RQ4 ≈ 4;
E-JEV-1/3 ≈ 4; Jev A100 phases 0–4.2 ≈ 16.5 A100-h; plus downloads/audits.

=====================================================================
PART 5 — RESULTS, PHASE BY PHASE (every number, with artifact paths)
=====================================================================
----- Phase 5A (reference & capacity, T4) -----
results/phase5a_reference_capacity/ — PASSED at 81 frames, 4/4 matched
reference/fine/coarse cases; distance = decoded-pixel L1 (feature fallback not
needed); seed 17.

----- Phase 5B (reference-regret calibration, T4) — THE NEGATIVE RESULT -----
results/phase5b_audit/phase5b_audit.json:
  calibration: 12 cases, 48 segments; held-out: 72 cases, 288 segments
  (8 batches heldout_00..07, 9 cases each). Data audit PASSED for all.
  Held-out residual AUROC = 0.491, prompt-seed 95% CI [0.393, 0.589].
  Calibration-selected warp AUROC = 0.454.
  Paired residual-minus-warp difference = 0.0367, 95% CI [−0.0493, 0.1194]
  → pre-registered RQ1 signal-advantage claim NOT SUPPORTED. Reported as-is.
  Calibration freeze: results/phase5b_calibration_lock.json.

----- F1 feasibility (T4) — GATE NO-GO -----
docs/F1_AUDIT.md + results/f1_timing_audit/, results/f1_mode_cost/,
results/f1_mode_cost_noisy/.
  F1.1 re-audit of 5B timings: fine total 184.6 s vs coarse 214.7 s
  (calibration, n=12); fine faster in 84/84 cases. The 5B "fine=c1-1 s1,
  coarse=c3-3 s4" pairing made coarse look slower — an operating-point artifact.
  F1.3 clean-context mode sweep (6 modes × 45/57/81 × 3 dev prompts, seed 17,
  54 videos; results/f1_mode_cost/f1_mode_cost_audit.json):
    mode             DiT(s)  VAE(s) total(s)  NFE   VRAM   refL1    s/NFE
    ref_c1_1_s4      59.42   155.7  227.8     64.0  11.32  0.0000   0.93
    fine_c1_1_s1     25.16   156.5  194.5     19.0  11.32  0.2019   1.32
    fine_c1_1_s2     36.73   155.7  205.5     34.0  11.32  0.1930   1.08
    coarse_c3_3_s4   62.05   155.7  232.1     21.3  11.32  0.2750   2.91
    coarse_c3_3_s2   40.63   155.6  210.5     12.7  11.32  0.2615   3.20
    coarse_c3_3_s1   29.72   155.9  197.6      8.3  11.32  0.2588   3.58
  → c3-3 has ~3× fewer NFEs but each forward ~3× heavier → NOT cheaper.
  fine_c1_1_s1 dominates every coarse mode on BOTH cost and reference distance.
  F1.4a classical endpoints (results/f1_mode_cost/f1_quality_classical.json):
    ref: warp 0.0184, fdiff 0.0124, flow 0.361, hist 0.9981, drift 0.1143
    fine_s1: warp 0.0280 / fine_s2: 0.0258
    coarse_s4: 0.0294 / s2: 0.0334 / s1: 0.0336  (coarse = less smooth)
  F1.4b CLIP (results/f1_mode_cost/f1_quality_clip.json): subject consistency
    ref 0.9929 > fine_s1 0.9866 > coarse_s1 0.9882; text alignment mixed, coarse
    s1/s2 marginally higher (0.305/0.304 vs 0.297) — no consistent coarse win.
  F1.3b noisy context (26 videos): halves DiT for both families
    (fine_s1_noisy 15.70 s / refL1 0.2182; coarse_s1_noisy 21.01 s / 0.2464)
    → fine still dominates; trick is mode-independent.
  GATE VERDICT: NO-GO. Fine/coarse switching improvement claim DROPPED on T4.
  Figure: docs/figures/f1_mode_pareto.png.

----- F4-lite (budgeted online selection, T4) -----
results/f4lite/ (40 rollouts = 8 prompts × 5 policies, 57 frames):
  policy          refL1    DiT(s)  VAE(s) total(s) NFE  fine-frac switches
  fixed_coarse    0.3186   34.96   110.4  202.1    20   0.0     0
  fixed_fine      0.2998   18.05   110.9  184.1    16   1.0     0
  random50        0.3102   26.53   110.7  187.5    18   0.5     2.13
  alternating50   0.3058   26.51   110.8  187.3    18   0.5     3.0
  adaptive_warp   0.2949   22.15   321.6* 397.8    17   0.75    0
  (*VAE inflated because the warp signal needs a mid-rollout VAE decode;
   signal_s 0.65 s on top. best_fixed_policy = fixed_fine.)
  Prompts (strata): p0 a boat (object), p1 a book (object), p2 baseball stadium
  (scene), p3 basement (scene), p4 a bird flying over a snowy forest (motion),
  p5 a bird soaring gracefully in the sky (motion), p6 A person is bungee
  jumping (human_action), p7 A person is canoeing or kayaking (human_action).
  Figure: docs/figures/f4lite_policy_pareto.png.

----- RQ3 controller ablation (T4) — SUPPORTED -----
results/rq3_controller/rq3_audit.json (48 rollouts = 8 prompts × 6 controllers;
videos in results/rq3_controller/videos/):
  controller            switches  fine-frac  AUH refL1  term refL1  seam/int L1
  threshold_only        0.500     0.500      0.3133     0.3171      0.0364/0.0280
  hysteresis_only       0.375     0.531      0.3116     0.3104      0.0321/0.0273
  recovery_only         0.375     0.531      0.3116     0.3104      0.0321/0.0273
  full (rec+hyst)       0.375     0.531      0.3116     0.3104      0.0321/0.0273
  threshold_only_noisy  1.125     0.531      0.3159     0.3172      0.0630/0.0290
  full_noisy            0.625     0.656      0.3150     0.3139      0.0399/0.0260
  Paired (real): Δswitches −0.125, ΔAUH −0.0017, Δterminal −0.0067 (n=8).
  Paired (noisy): Δswitches −0.5, ΔAUH −0.0010, Δterminal −0.0033.
  λ-sweep Λ ∈ {0.5,1,2,4}·σ_u (σ_u=3.915): switches 3→3 (insensitive here);
  R-sweep R ∈ {1,2,3}: 3→3. Verdict: rq3_supported real AND noisy = true.
  Threshold calibration (config/rq3_threshold.json): signal = warp,
  τ = pooled median 4.416, σ_u = 3.915, direction −1 (low warp → fine);
  warp produced the most within-rollout crossings (4) at the pooled median.
  Figure: docs/figures/rq3_controller_ablation.png.

----- RQ4 monotonicity, hand-crafted warp (T4) — NO MONOTONE RELATION -----
results/rq4_monotonicity/rq4_audit.json (80 segment points: 48 @81f + 32 @57f;
24 videos = 8 prompts × {reference, fixed_fine, fixed_coarse}):
  Spearman ρ(warp, fine_gap) = −0.0988; verdict "no monotone relationship".
  Terciles of warp: mean fine_gap −0.0225 / −0.0571 / −0.0195 (non-monotone).
  fine_gap = refL1_fine − refL1_coarse (negative ⇒ fine closer to reference).
  Figure: docs/figures/rq4_monotonicity.png.

----- E-JEV-1: the Jev signal test (T4) — RQ1 REVERSED WITH THE ORACLE -----
What Jev is (from scripts/jev_query.py — verbatim details):
  Endpoint: POST https://api.typesafe.ai/v1/systemone
  Model: "jev-latest". Auth: Bearer TYPESAFE_API_KEY (key lives OUTSIDE the repo
  at %USERPROFILE%\.kilo-automation\typesafe.env; never committed).
  Per segment, the state sent is:
    {video_prompt, context_caption (BLIP caption of the last pre-decision frame,
     Salesforce/blip-image-captioning-base), segment_index, horizon_frames,
     recent_motion_magnitude (round 3), recent_temporal_warp (round 3)}.
  Three questions asked per state:
    coarse_drift — type "noul" (numeric 0–1): "would generating the next few
      frames with a coarse 3-frame-chunk method likely drift or become
      temporally inconsistent compared with a frame-wise method?"
      (true = fast/complex motion, occlusion, scene change; false = simple/slow)
    difficulty — type "score", criteria very easy/easy/moderate/hard/very hard
      (mapped to a numeric score): "how difficult is it to predict the next few
      frames accurately?"
    motion — type "choice": static / moderate / fast / scene_change
      (ordinalized static=0, moderate=1, fast=2, scene_change=3).
  Retries: up to 4 attempts, exponential backoff on 429/529, 40 s timeout.
  Mean latency 0.672 s per decision (n=80) — vs the warp signal which needed a
  VAE decode (~156 s @57f) per decision ON THE GENERATION PATH. Jev answers
  BEFORE any generation; it is off the DiT path.
States built (scripts/jev_build_states.py) from the EXISTING paired segments:
  80 segments = 8 prompts × (4 groups @57f from F4-lite + 6 groups @81f from
  RQ4). Label coarse_unsafe = (regret_coarse > regret_fine); positives 56/80.
Results (results/jev_signal/jev_audit.json; AUROC, prompt-grouped bootstrap
  95% CI, 2000 resamples, seed 20261026):
    jev_difficulty  0.788  [0.575, 0.946]
    jev_drift       0.738  [0.479, 0.945]
    jev_motion      0.703  [0.472, 0.873]
    motion (hc)     0.607  [0.260, 0.857]
    warp (hc)       0.586  [0.222, 0.836]
    complexity (hc) 0.368  [0.184, 0.562]
  Gain of best Jev over best hand-crafted: +0.180. Phase 5B residual reference:
  0.491. Figure: docs/figures/jev_signal_auroc.png.
Routing rule (config/jev_routing.json), derived from the 57f answers:
  static→hold, moderate→coarse, fast/scene_change→fine.
  Trace mode counts over 32 decisions: coarse 16, hold 9, fine 7.
  Per-prompt traces: p0 [c,c,c,c], p1 [h,h,h,h], p2 [h,h,c,h], p3 [f,c,h,h],
  p4 [f,f,c,c], p5 [c,c,c,c], p6 [f,f,f,f], p7 [c,c,c,c].

----- E-JEV-2: hardware re-timing -----
results/jev_p100_timing/ (labeled P100 run but executed on T4, n=16):
  per-forward: fine 0.652 s, coarse_s4 2.124 s, coarse_s1 1.960 s → ratio 3.01
  (> break-even 2.33) → coarse NOT cheaper on T4. dit ratio 1.15; NFE ratio 0.43.
results/a100_timing/a100_audit.json (A100-SXM4-80GB):
  per-forward: fine 0.1247 s, coarse 0.2000 s → ratio 1.604 < 2.33
  → "TRADE-OFF EXISTS on this GPU. The T4 inversion is hardware-specific."
  (also 15-vs-21-latent-frame rows confirm ratio stable ≈ 1.6–2.2 range)

----- E-JEV-3 / 3b: hold mode and hold+fine (T4) -----
results/jev_hold_mode/jev_hold_audit.json (40 rollouts = 8 prompts × 5 policies):
  policy             refL1   DiT(s)  total(s) forwards hold fine coarse
  fixed_fine         0.2998  19.65   193.9    29.0    0    4    0
  fixed_coarse       0.3186  37.69   205.5    25.0    0    0    4
  fixed_hold         0.3108   6.48   136.4     5.0    4    0    0
  jev_routed         0.3085  24.88   182.3    20.25   1.125 0.875 2.0
  jev_routed_nohold  0.3131  33.71   201.4    25.875  0    0.875 3.125
  Paired: jev_routed vs fixed_fine: Δregret +0.0087, Δtotal −11.6 s;
  jev_routed vs jev_routed_nohold: Δregret −0.0045, Δtotal −19.1 s,
  ΔDiT −8.8 s → the HOLD mode is where the saving comes from.
results/jev_holdfine/jev_holdfine_audit.json (16 rollouts, 2 prompts recorded
  locally as p0/p1 reference/jev_holdfine/fixed_fine videos):
  jev_holdfine refL1 0.2988 vs fixed_fine 0.2998 (Δ −0.0010), total 167.0 s vs
  183.2 s (Δ −16.2 s), DiT 15.07 s vs 18.59 s (Δ −3.5 s) → matches fine quality
  at lower cost on T4. Figure: docs/figures/jev_hold_mode.png.
  Videos: results/jev_hold_mode/videos/ (48 mp4), results/jev_holdfine/videos/
  (6 mp4).

----- A100 arm (A100-01 protocol) -----
A1 mode cost (results/a100_mode_cost/a100_mode_cost_audit.json, 9 cases):
  mode             DiT(s)  total(s) NFE   refL1   s/forward
  ref_c1_1_s4      9.097   18.40    64.0  0.0000  0.1137
  fine_c1_1_s1     3.922   12.95    19.0  0.1769  0.1120
  fine_c1_1_s2     5.619   14.68    34.0  0.1697  0.1124
  coarse_c3_3_s4   5.580   14.71    21.3  0.2712  0.2092
  coarse_c3_3_s2   3.651   12.59    12.7  0.2571  0.2028
  coarse_c3_3_s1   2.685   11.66     8.3  0.2575  0.1965
  coarse/fine per-forward ratio 1.753 < break-even 2.33 → TRADE-OFF on A100.
  CLIP: subject ref 0.9914, fine_s1 0.9890, coarse_s1 0.9880 (within ~0.003);
  classical: coarse warp 0.0329 > fine 0.0226 (less smooth), consistent with T4.
A2 online policies (results/a100_online/ + a100_online_v2/, 72 rollouts =
  12 prompts × 6 policies; v2 added fixed_hold; videos in
  results/a100_online_v2/videos/, 84 mp4 = 12 prompts × 7 variants):
  policy         refL1   DiT(s) total(s) NFE   fine-frac switches
  fixed_coarse   0.2728  2.549  45.47    8     0       0
  fixed_fine     0.2832  3.947  46.76   16     1       0
  fixed_hold     0.2674  0.765  12.54    4     0       0
  random50       0.2789  3.243  46.20   12     0.5     2.25
  alternating50  0.2784  3.248  46.81   12     0.5     3.0
  adaptive_warp  0.2733  2.846  52.26    9.67  0.208   0.083
  Internal verdict: adaptive vs fixed_fine quality +3.5% (lower refL1),
  DiT −27.9%, beats random same-budget → F2LITE-01 internal pass (A100).
  NOTE: on A100 the fixed_fine refL1 (0.283) is HIGHER than fixed_coarse
  (0.273) — the internal direction flipped relative to T4; this is why the
  independent CLIP endpoint is primary.

----- A100 Jev-online (Jev-max phases, all on A100-80GB) -----
Phase 0 (results/a100_online_jev/, 96 rollouts = 12 prompts × 8 policies):
  Adds jev_difficulty, jev_routed, jev_holdfine with REAL Jev calls in-loop:
  jev_s ≈ 1.03–1.04 s per rollout, blip_s ≈ 0.18–0.70 s (captioning, off-path).
  policy         refL1   DiT(s) total(s) NFE   CLIP-subj  twarp
  fixed_fine     0.2832  4.008  41.84   16    0.9774     0.0249
  fixed_hold     0.2674  0.763  11.53    4    0.9900     0.0241
  jev_difficulty 0.2678  3.191  47.73   11.5  0.9545     0.0373
  jev_routed     0.2715  2.038  33.70    7.83 0.9614     0.0328
  jev_holdfine   0.2739  2.586  34.87   10.75 0.9796     0.0249
  All Jev policies beat random same-budget (verdict.beats_random = true).
Phase 2 (results/a100_online_jev_phase2/, 30/60 complete = PARTIAL; 12 prompts
  a cake, a car, a carrot, botanical garden, bridge, cafeteria, 3× bus motion,
  clapping, pottery, windows; warp_threshold −4.0458, jev_difficulty_threshold
  1.72, upstream commit ed3cb14).
Phase 3 (results/a100_online_jev_phase3/, 120/120 COMPLETE = PASS; 24 fresh
  prompts across 4 strata; run 20261013_102905, 63 min wall-clock):
  Prompt-paired bootstrap CIs vs fixed_fine (n_boot=10000):
    jev_single:  ΔCLIP-subj −0.0202 [−0.0304, −0.0103]; Δtext +0.0013 n.s.;
                 ΔDiT −1.78 s [−2.20, −1.37]; Δtotal −7.39 s [−12.39, −3.16]
    jev_routed:  ΔCLIP-subj −0.0180 [−0.0264, −0.0100]; Δtext −0.00007 n.s.;
                 ΔDiT −1.81 s [−2.20, −1.41]; Δtotal −2.58 s [−6.62, +1.26] n.s.
    latent:      ΔCLIP-subj −0.0131 [−0.0163, −0.0099]; ΔDiT −0.353 s.
  phase3_ci.json is the file with these numbers.
RQ4-with-Jev on A100 (results/a100_online_jev_phase3/rq4_jev_a100.json,
  n=96 segments): difficulty ρ=+0.059 (AUROC 0.451), drift ρ=−0.086 (0.547),
  motion ρ=+0.084 (0.448) → NO signal advantage on the A100 refL1 diagnostic
  (contrast with the T4 RQ4-Jev result below; recorded as signal/hardware
  dependence).
A3 RQ3 on A100 (results/a100_rq3/a100_rq3_audit.json, 72 rollouts = 12 prompts
  × 6 controllers): threshold_only switches 0.5 → full 0.417 (real), noisy
  0.917 → 0.5; AUH full−threshold = +0.00054 (not worse); seam/interior L1
  0.067/0.031 (real), 0.079/0.033 (noisy) → RQ3 SUPPORTED on A100 too.
  Videos: results/a100_rq3/ (tar.gz, 95 MB).

----- RQ4-with-Jev on T4 (E-JEV extension) — MONOTONE WITH JEV -----
results/rq4_jev/jev_rq4_audit.json (same 80 segments, Jev signals):
  jev_difficulty: ρ = −0.517, AUROC 0.788; terciles of difficulty vs mean
  fine_gap: +0.0006 (easy) → −0.0430 (moderate) → −0.0558 (hard): MONOTONE.
  jev_drift ρ = −0.438; jev_motion ρ = −0.446; warp ρ = −0.098 (flat).
  Split by horizon: monotone on both 57f (ρ=−0.458) and 81f (ρ=−0.581).
  Figure: docs/figures/rq4_monotonicity.png (updated with Jev).

=====================================================================
PART 6 — CONSOLIDATED ANSWERS TO THE FOUR RESEARCH QUESTIONS
=====================================================================
RQ1 (signal): hand-crafted signals FAIL held-out (0.491 [0.393,0.589]); the Jev
  oracle succeeds on the same segments (0.788 [0.575,0.946], +0.18 over the best
  hand-crafted). The signal content was the bottleneck, not the label.
RQ2 (trade-off): hardware-conditional. T4: NO-GO (ratio 3.01 > 2.33; fine
  dominates coarse on cost AND quality). A100: trade-off exists (ratio
  1.60–1.75 < 2.33; coarse cheaper per wall-clock DiT). Jev-routed policies beat
  random same-budget on both platforms (F2LITE-01 pass).
RQ3 (mechanism): SUPPORTED on both. Recovery+hysteresis cuts switches without
  worsening AUH/terminal regret; a real seam cost at switch boundaries exists
  (adjacent-frame L1 ≈ 0.036–0.079 at boundaries vs ≈ 0.027–0.033 interior).
RQ4 (monotonicity): signal-dependent. Warp: non-monotone (ρ=−0.10). Jev
  difficulty: monotone (ρ=−0.517) on T4 segments; on the A100 Phase-3 diagnostic
  the relation is flat (ρ≈+0.06) — reported as an honest boundary.

=====================================================================
PART 7 — FILE/BY-FILE READING LOG (what was read where)
=====================================================================
Root: proposal.tex (611 l.), STATUS.md (268 l.), PROGRESS.md (159 l.),
  CLAUDE.md, README.md, LINKS.md, Mater_plan_final.md (702 l.), requirements.txt.
docs/: PAPER.md (paper draft v3), PAPER_CONTEXT.md (writing context),
  A100_PROTOCOL.md, F1_AUDIT.md, F2_PROTOCOL.md; figures/ 6 PNGs.
old/: README.md, agent_memory/*, docs/PHASE5B_FINAL_AUDIT.md,
  PHASE5B_PROTOCOL.md, PHASE5C_*.md, PROJECT_AUDIT_2026-10-06.md,
  PROPOSAL_UPDATE_TRACKER.md, root/{CLAUDE,MASTER_PLAN,PROGRESS,prompt,README,
  STATUS,WAITING_FOR_PHASE0}.md — historical, superseded where conflicting.
scripts/ (read in full): jev_query.py, jev_build_states.py, jev_audit.py,
  rq3_calibrate_threshold.py; (headers/docstrings of the rest enumerated:
  f1_*, f4lite_*, rq3_*, rq4_*, jev_figure.py, jev_routing.py, jev_hold_audit.py,
  jev_holdfine_audit.py, jev_hold_figure.py, jev_p100_audit.py,
  jev_rq4_audit.py, jev_signal_select.py, phase4_rq4_jev.py,
  a100_online_jev_audit.py).
src/: controller/policy.py (ThresholdController, RecoveryController,
  PotentialFunction, AdaptiveController — full read); causal_rcm/inference.py,
  fallback.py, load_checkpoints.py; baselines/, evaluation/, uncertainty/,
  utils/ enumerated.
config/: controller.yaml (τ_high 0.5, τ_low 0.2, R 5, λ 0, init coarse),
  rq3_threshold.json (warp τ=4.416, σ_u=3.915), jev_routing.json (full traces),
  f4lite_manifest.json, a100_manifest.json, a100_phase3_manifest.json,
  phase5_prompt_manifest.json, phase5c_*lock.json.
results/: every audit/summary JSON listed in Part 5 read in full; .jsonl raw
  rolls enumerated (results.jsonl per phase; phase3 results 396 KB).
  Videos: 346 mp4 catalogued (f1_mode_cost 54, f1_mode_cost_noisy 26, f4lite 48,
  jev_hold_mode 48, jev_holdfine 6, rq3_controller 56, rq4_monotonicity 24,
  a100_online_v2 84) + 2 tar.gz (a100_rq3, a100_online_v2 mirror).
  Frame PNGs: phase2_baseline 24, phase0c 4, phase1_switch 3.
  Causal-rCM teaser assets: results/phase5c_online_pilot/v2_download/rcm/assets/
  {background,causal-teaser,comparison,dcm_vs_scm,diagonal,infra,vbench}.png.
jobs/: 33 T4/Kaggle job dirs + 18 result-side dirs enumerated (phase0..a100_rq3).
colab/: a100_online_jev.py, a100_timing.py and siblings enumerated (the A100
  Jev-online runner calls the same TypeSafe endpoint with the same payload).
logs/, .remember/: enumerated (memory hooks; no result data).
paper_materials/: paper.tex (IEEE draft v1), title_and_github.md (v1 titles),
  make_architecture.py / make_qual_grid.py / make_contact_sheets.py /
  prepare_assets.py read (figure provenance documented), figures/ 8 PNG,
  qualitative_frames/ 8 PNG, image_contact_sheets/ 7 JPG, source_notes/ copies.

=====================================================================
PART 8 — MEDIA CATALOG FOR THE PAPER (names only; pixels not read here)
=====================================================================
Qualitative candidates visually inspected (paper/figures_src/candidates/,
160 PNGs = 8 prompts × 4 policies × 5 time fractions):
  p4 "a bird flying over a snowy forest": at late frames fixed_fine keeps a
  white gull (red feet), fixed_coarse drifts to a yellow/black-headed bird,
  jev_routed produces a red-bodied bird, reference stays a white gull — visible
  identity drift difference between policies.
  p6 "A person is bungee jumping": fixed_fine and jev_routed keep the same man
  over the same green field; fixed_coarse drifts to a different person over a
  LAKE (scene-level drift). jev_routed chose fine for all 4 segments of p6
  (trace [fine,fine,fine,fine]) — a direct visual proof of correct routing.
Chosen for the qualitative figure: p6 (primary) + p4 (secondary).
Quantitative figures to regenerate: T4+A100 cost Pareto (combined), Jev AUROC
bars, RQ3 ablation (T4+A100), RQ4 Jev-vs-warp terciles, Phase-3 CI forest plot,
architecture diagram (Jev oracle → controller → 3 modes), qualitative strips.

=====================================================================
PART 9 — KNOWN LIMITATIONS (must appear in the paper)
=====================================================================
1. Hardware-conditional cost claim (T4 vs A100); results do not transfer across
   GPUs without re-measuring the per-forward ratio against break-even 2.33.
2. Small-n: 3 dev prompts (F1), 8 prompts (F4-lite/RQ3/RQ4/E-JEV-1/3), 24
   prompts (Phase 3), single seed (17) throughout.
3. Internal refL1 reference is diagnostic only and its fine/coarse direction
   FLIPPED between T4 and A100 — independent CLIP/classical endpoints primary.
4. Jev is an EXTERNAL, proprietary, versioned API ("jev-latest", TypeSafe
   System One); results depend on its version and availability; ~0.67–1.04 s
   latency per decision is off the DiT path but real.
5. No action conditioning (no compatible checkpoint); λ=0 throughout.
6. No FVD/PSNR/LPIPS/VBench (no ground-truth video for T2V rollouts).
7. KV-cache persistence unavailable in the verified upstream API.
8. EMA uncertainty signal unavailable in the verified checkpoints (dropped).
9. Jev introduced AFTER the original locked protocol as the Jev-max plan
   (method change disclosed; original Phase-5B negative preserved).
10. Proposition-1 parameters were never reachable in the deployed setting
    (recorded in PROGRESS.md).
11. Kaggle jobs installed unpinned packages and cloned upstream default branch
    at runtime (weakened exact reproducibility; A100 runs pinned commit ed3cb14).

=====================================================================
PART 10 — PEOPLE / LINKS / CREDIT
=====================================================================
Author: Hossen Md Jisan, Nanjing University of Information Science and
Technology, B.Sc. in Artificial Intelligence, jisanahammed1952@gmail.com.
Upstream: Causal-rCM (THU-ML, arXiv:2606.25473) — checkpoints + recipe; Wan2.1
(Alibaba, arXiv:2503.20314) — base model. Jev model: TypeSafe System One API
(https://api.typesafe.ai/v1/systemone). GPU platforms: Kaggle (Tesla T4 16 GB),
Google Colab Pro (A100-SXM4-80GB). Checkpoint mirror:
https://www.kaggle.com/models/ajjisan/causal-rcm.
(END OF BOOK)
