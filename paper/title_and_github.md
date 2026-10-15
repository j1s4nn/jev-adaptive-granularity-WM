# Paper Title Proposals & GitHub Naming — Jev / Adaptive-Granularity Study

## 5 candidate titles (all mention the Jev model)

1. **Jev-Guided Adaptive Temporal Granularity for Autoregressive Video World Models: A Hardware-Conditional Feasibility Study** ✅ **RECOMMENDED FOR THE PAPER**
   - Why: names the method (Jev), the mechanism (adaptive temporal granularity), the domain (AR video world models), and the honest headline finding (hardware-conditional). Reviewers reward the "feasibility study" framing because the negative T4 result is preserved inside it.
2. *When Does Coarse Pay? Adaptive Granularity with the Jev System One Oracle in Autoregressive Video Generation*
   - Catchy question-lead; emphasizes the cost finding. Slightly less formal.
3. *Per-Segment Mode Routing with the Jev Model: Uncertainty-Guided Fine/Coarse/Hold Switching for Streaming Video Diffusion*
   - Method-forward; good if the committee cares more about the controller than the diagnosis.
4. *A Hardware-Conditional Trade-off in Chunked Autoregressive Video Diffusion: Jev-Guided Granularity Control with Hysteretic Recovery*
   - Leads with the systems finding; mentions the RQ3 mechanism (hysteresis/recovery).
5. *From Failed Signals to an Oracle: Jev-Based Uncertainty Routing for Adaptive Temporal Granularity in Causal Video World Models*
   - Tells the honest story (hand-crafted signals failed → oracle succeeded). Bold framing; may suit a workshop more than TCSVT/TMM.

## Chosen for the paper
**Title 1** — *Jev-Guided Adaptive Temporal Granularity for Autoregressive Video World Models: A Hardware-Conditional Feasibility Study.*

## GitHub repository
- **Repo name:** `jev-adaptive-granularity`
- **Short description (GitHub "About"):**
  > Official code, locked protocols, and audit artifacts for "Jev-Guided Adaptive Temporal Granularity for Autoregressive Video World Models." Per-segment fine/coarse/hold mode routing for Causal-rCM (Wan2.1-1.3B) driven by the Jev System One oracle, with a recovery+hysteresis controller, two-GPU (T4/A100) feasibility audits, and full negative-result preservation.
- **Topics:** `video-generation`, `world-models`, `autoregressive-diffusion`, `causal-rcm`, `adaptive-computation`, `uncertainty`, `jev`, `efficient-inference`

## One-line pitch (abstract preview)
Adaptive per-segment granularity in autoregressive video world models is real but conditional: it requires a GPU where coarse chunks are actually cheaper (A100, not T4), a signal that predicts coarse-unsafe segments before generation (the Jev System One oracle, AUROC 0.788 where hand-crafted signals score 0.49–0.61), and an endpoint that reflects perceived quality — and when those hold, a simple recovery+hysteresis controller beats fixed and random same-budget baselines.
