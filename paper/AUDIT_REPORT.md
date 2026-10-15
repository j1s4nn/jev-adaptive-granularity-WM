# Research Paper Integrity and Submission-Readiness Audit

**Manuscript:** *Jev-Routed Temporal Granularity with Recovery and Hysteresis for Causal Video Generation*
**Canonical source:** `04_Experiment/paper/main_updated2.tex` (11 pp.)
**Date:** 2026-10-15
**Scope:** arXiv submission readiness; possible IEEE TCSVT submission

> Canonical manuscript is `main_updated2.tex`. The earlier `main.tex` is retained but is **not** the current source. Every number below was traced to a locked artifact under `results/`, a config under `config/`, or the project ledger (`STATUS.md`, `PROGRESS.md`, `paper/book.md`). No experiment, result, or reference was fabricated.

---

## Phase 1 — Structural & mathematical audit (discrepancy table)

| # | Location | Exact claim | Conflicting evidence | Severity | Verification | Correction | Status |
|---|---|---|---|---|---|---|---|
| D1 | Table III (`tab:costpre`), row "T4, clean context" | fine `1.32`, coarse `3.58`, ratio `3.01` s/fwd | `1.32`/`3.58` are **s/NFE** (`25.155/19`, `29.721/8.333`; labelled "s/NFE" in `book.md:208-214`, `docs/F1_AUDIT.md`), whose ratio is **2.69**, not 3.01. The `3.01` per-forward ratio comes from `results/jev_p100_timing` (fine `0.652`, coarse `1.960`). Table mixed two metrics. | High | Confirmed in `jev_p100_audit.json`, `f1_mode_cost_audit.json` | Row rewritten to per-forward `0.65` / `1.96`, relabelled "T4, timing arm", sources added to caption; body text changed from "s per NFE" to "s per forward" | **Fixed** |
| D2 | §III, eq. (1) text | "NFE 19 vs. 8.33 **at 57 frames**, and 21 vs. 9 … the break-even ratio is 2.33" | `19`/`8.33` is the **45/57/81 mean** (57 f is `18/8`); `19/8.33 = 2.28`, only `21/9 = 2.33` | Medium | `f1_mode_cost_audit.json` `nfe_by_mode` | Sentence rewritten: 2.28 (clean, 45/57/81) and 2.33 (noisy); "conservative 2.33" stated | **Fixed** |
| D3 | Abstract / §I findings / §VI.A | coarse family has "**about 3× fewer** NFEs" | one-step NFE pair is `19` vs `8.33` = **2.28×** (3.0× holds only for the 4-step pair `64` vs `21.3`) | Medium | `f1_mode_cost_audit.json` | Changed to "roughly 2.3× fewer NFEs at the one-step regimes (8.33 vs. 19)" | **Fixed** |
| D4 | §VI.A | "Independent endpoints **agree** … with **no consistent CLIP advantage**" | `f1_quality_clip.json`: coarse s1 subject consistency **0.9882 > 0.9866** (fine s1) and text alignment **0.3053 > 0.2971**; the CLIP endpoint does not favour fine | Medium | `f1_quality_clip.json`; older `main.tex` had disclosed these numbers and `main_updated2.tex` deleted them | Reworded to "Independent endpoints are **mixed**" with the explicit numbers restored; cost/refL1 conclusion kept | **Fixed** |
| D5 | §VI.A | coarse warp "**0.0334–0.0336**" | `f1_quality_classical.json`: coarse family spans **0.0294–0.0336** (s4 = 0.0294) | Low | `f1_quality_classical.json` | Range corrected to 0.0294–0.0336 | **Fixed** |
| D6 | §IV.C | Jev rule + gate "calibrated on **development** answers" | `jev_routing.json`, `jev_signal_select.json`, `jev_audit.json` all derive from the **same 8 prompts / 80 segments** whose AUROC 0.788 is reported → calibration on the evaluation set | High | `config/jev_routing.json` (source = `jev_signal/jev_answers.jsonl`), `jev_signal_select.json` n=80 | Wording changed to "fitted on the eight-prompt Jev audit set"; explicit calibration-on-evaluation-set caveat added to §IV.C and to Limitation (ii) | **Fixed** |
| D7 | §II.E "Positioning" | "To our knowledge **none** decides per segment, online, which temporal granularity to run next from a signal computed before generation" | LongScape (cited) routes **online variable-length chunk granularity** for the next chunk; Flex-Forcing selects chunking at test time | Medium | arXiv:2509.21790 (LongScape), arXiv:2607.03509 (Flex-Forcing) | Narrowed to "none does **all three together**: pre-generation *difficulty* signal (not action priors / global budget) + recovery mechanism + cross-hardware cost audit" | **Fixed** |

**Verified-correct (no change):** all of Tables I, II, IV, V, VI, VII, VIII and IX; the RQ3/RQ4 numbers; the phase-3 bootstrap CIs; the AUROC table; the T4 trap (`53.4` vs `21.6` s, 72/72); the `-0.0010` hold-fine regret; `0.672` s oracle latency; `~156` s VAE; `230×` ratio.

---

## Phase 2 — Reproducibility / evidence audit

- **Fully traced (reproducible):** Tables I–II (`f1_mode_cost`, `a100_mode_cost`), IV (`f4lite_audit`+`f4lite_quality`), V (`jev_hold_mode`, `jev_holdfine`), VI (`a100_online_v2` + CLIP/classical), VII (`a100_online_jev`), VIII (`phase3_ci`, 10 000 resamples), IX (`rq3_controller`, `a100_rq3`), AUROC table (`phase5b_audit`, `jev_audit`), RQ4 (`rq4_monotonicity`, `rq4_jev`, `rq4_jev_a100`). Locked manifests, calibration locks (`phase5b_calibration_lock.json`, `phase5c_*.json`) and per-batch SHA-256s are present. A100 runs pin `ed3cb14`.
- **Timing splits:** T4 3.01 is from `jev_p100_timing` (n=16, 2 prompts × {57,81}); A100 1.60 from `a100_timing`; 1.75 from `a100_mode_cost`; 2.94 from `f1_mode_cost_noisy` — all per-forward (`dit/forwards`).
- **Prompt/seed splits:** Phase-5B (12 cal + 24 held-out prompts, seeds 17/29/41 → 72 cases/288 segments); F4-lite 8 fresh; A100-01 12; A100-Phase3 24. Disjointness asserted in the manifests.
- **Unverified / not directly in artifacts:**
  - "only **38.5 %** of A100 segments favour fine" (§VI.F) — not stored in `rq4_jev_a100.json` (tercile means imply ≈0.0137; the fraction is not recorded).
  - Confirmatory subject-consistency CIs use **n=23**, not 24 (`phase3_ci.json`); one prompt drops out of the CLIP arm. DiT/total use n=24.
  - `config` writes videos at `FPS=8`; the paper's "16 fps" is the model rate. Not contradictory, but worth one clarifying word.
- **Version drift:** `main_updated2.tex` silently dropped (vs `main.tex`) Limitation (viii) about unpinned Kaggle packages, the sentence "All splits, thresholds, and manifests were frozen before the corresponding evaluation runs", the abstract's "three positive results" list, and the explicit CLIP numbers (D4). D4 was restored; the others are authorial choices flagged for your decision.

---

## Phase 3 — Literature & novelty (targeted search, **not exhaustive**)

| Paper | Actual method / setting | Overlap | Genuine difference | Claim status |
|---|---|---|---|---|
| **DSA** (CVPR-F 2026, arXiv:2606.04432) | Learned confidence head → dynamic denoising **steps** per frame, AR video, Wan-1.3B, chunk-3 | Adaptive per-segment compute in AR video, learned signal, same base model | Changes **steps**, not temporal granularity; internal jointly-trained head; no hardware audit, no seam/hysteresis | "closest published point" is **supported**; distinction is fair |
| **LongScape** (arXiv:2509.21790) | Action-prior variable-length chunking + router picking next chunk's expert/length | **Online next-chunk granularity routing** | Embodied/action-conditioned; no pre-generation difficulty signal, no hysteresis/recovery, no cost audit | The old "none decides per segment online" claim was **too broad** → narrowed (D7) |
| **Flex-Forcing** (ICML 2026, arXiv:2607.03509) | Flexible chunking over frames + timesteps, device-budget aware | Test-time chunk-size control | Global/budget-driven, brute-force/search, not per-segment pre-generation signal | "device-budget-aware" is **supported** |
| **DASH-OPD** (arXiv:2607.29078) | Drift/recovery evidence + two thresholds, **hysteresis**, LLM OPD | Two-threshold hysteresis band, recovery direction | Domain is agentic on-policy distillation, not inference-time video granularity | "reuses the same classical band" is **supported** |
| **Causal-rCM** (arXiv:2606.25473), **Causal Forcing/++** (arXiv:2602.02214), **Diagonal Distillation** (ICLR 2026, arXiv:2603.09488) | Teacher/self-forcing distillation; frame vs chunk regimes; asymmetric early-heavy step schedule | Provided checkpoints; frame/chunk trade-off framing | Do not adapt granularity online | Descriptions **accurate** |

All 35 `references_ieee.bib` entries resolve to real, correctly described works (checked the load-bearing ones). **No fabricated reference found.** Do not describe the search as exhaustive.

---

## Phase 4 — Submission readiness

- **A. Technical correctness — good after fixes.** The only real defect was the D1 metric conflation; D2/D3 were precision errors; the conclusions are robust under all interpretations (coarse is slower on T4 either way).
- **B. Novelty — defensible after D7.** The three-precondition framing and the hardware-conditional cost inversion are the strongest contributions and are not pre-empted.
- **C. Experimental strength — moderate.** 8/12/24-prompt arms, single seed, internal reference that flips direction; adequate for a *feasibility/audit* paper, not for a universal claim (the paper already says so).
- **D. Reproducibility — strong** (locked manifests, hashes, pinned commits), with the caveats in Phase 2.
- **E. arXiv — ready** after the applied fixes.
- **F. IEEE TCSVT — plausible but borderline.** The negative/feasibility framing and small-n may draw reviewer pushback; the fixes below raise it to submission quality.

**Prioritized list**
- **P0 (blocking):** none remaining.
- **P1 (mandatory, applied):** D1, D2, D3, D4, D6, D7.
- **P2 (optional/recommended):** D5 (applied); record the 38.5 % figure; note n=23 for confirmatory CLIP; reconcile the "frozen before runs" sentence and Limitation (viii) from `main.tex`; delete the unused `figures/fig_t4_online.pdf`; align the title with `title_and_github.md`.

---

## Phase 5 — Corrections applied

Edited only `paper/main_updated2.tex` (8 edits, all evidence-backed): Table III row + caption; §III break-even sentence; §I and §VI.A NFE wording; §VI.A endpoint/warp wording; §IV.B/§IV.C Jev-calibration wording; Limitation (ii); §II.E positioning.

**No numerical result was changed to force consistency.** The headline per-forward ratio `3.01` and all AUROC/policy/timing values are untouched; the Table III fix aligns a cell to its documented per-forward source.

**Regenerated artifacts** (MiKTeX `latexmk -pdf` + bibtex, clean): `main_updated2.pdf` (11 pp., 3 277 168 B), `main_updated2.aux/.bbl/.blg/.log/.out/.fls/.fdb_latexmk`. Consistency check: **35 bibitems, 0 undefined citations/references, 0 overfull boxes**, one cosmetic "text page 9 contains only floats" warning.

**Length reduction (12 → 11 pp.):** redundant prose removed without dropping any verified number, table value, or negative result — condensed the system-overview and positioning sentences, the oracle-risk and noisy-variant clauses, the Jev-route paragraph, the Discussion's practitioner paragraph, the Reproducibility section, and the Conclusion's future-work list. All key values re-verified present after the cuts (`3.01`, `1.96`/`0.65`, `2.28`, `0.0294`, `0.9866`/`0.9882`, `0.491`, `0.788`, `-0.0010`, `-0.0202`, `1.78`, `7.39`, `288`); obsolete `3.58`/`0.0334` remain absent.

---

## Unresolved / needs your decision (no fabrication)

1. Independent held-out Jev evaluation — the gate is fitted on the reported set; only a **new prompt set** can certify it (plan, not run).
2. 38.5 %-favour-fine figure and n=23 CLIP count — record or drop.
3. Whether to restore `main.tex`'s frozen-splits sentence and Limitation (viii).
4. Confirmatory arm is single-seed; blinded human evaluation and FVD/PSNR/LPIPS remain explicitly out of scope.
