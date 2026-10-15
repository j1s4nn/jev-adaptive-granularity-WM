# A100 hardware re-time — step-by-step (human-in-the-loop)

Goal: run the same fine/coarse timing sweep on an **A100-class GPU** in Google Colab
to test whether the T4 cost inversion is hardware-specific.

You do this; the agent cannot drive Colab. It should take ~20–30 minutes.

## Step 1 — Open Colab
1. Go to https://colab.research.google.com
2. Sign in as **jisanahammed1952@gmail.com** (top-right account button).

## Step 2 — Pick the GPU
1. Menu **Runtime → Change runtime type**.
2. **Hardware accelerator → GPU**.
3. Look at the **GPU type** dropdown. If you see **A100**, select it. If you only see
   **T4 / L4 / V100**, select the best available and **tell the agent which one** — the
   test still runs (any non-T4 GPU answers the question).
4. Click **Save**.

## Step 3 — Get your Kaggle token (no new API needed)
- Open this file on your laptop: `C:\Users\jisan\.kaggle\kaggle.json`.
- Copy the value of **`key`** (it starts with `KGAT_...`). That is your Kaggle token.

## Step 4 — Paste and run the script
1. Open the file `colab/a100_timing.py` (in this project) and **copy all of it**.
2. In Colab, click **+ Code** to add a cell, then **paste** the whole script.
3. In the pasted text, replace `PASTE_KAGGLE_TOKEN_HERE` with your `KGAT_...` token.
4. Press the **▶ Run** button (or Ctrl+Enter).
5. If Colab warns about running a cell not authored by Google, click **Run anyway**.

## Step 5 — Wait
- It installs packages, clones the repo, and downloads 3 checkpoints (~8 GB). Expect
  **15–25 minutes**. Keep the browser tab open (Colab disconnects if idle too long).

## Step 6 — Send the result back
At the end you will see a table and a line like:

```
RESULT_JSON: {"gpu": "...", "ratio": 3.01, ...}
```

**Copy that `RESULT_JSON:` line (and the table above it) and paste it back to the agent.**

## If something fails
- **"No GPU available" / disconnect**: Runtime → Disconnect and delete runtime, then re-run.
- **A100 not offered**: use whatever GPU is offered and tell the agent which.
- **Checkpoint download error**: confirm the token has no spaces and starts with `KGAT_`.
- **Out of memory**: tell the agent; the script can be reduced to one horizon.

## What this decides
- If the coarse/fine per-forward ratio drops **below 2.33** on the A100, the fine/coarse
  trade-off exists on that hardware → the project's positive path reopens.
- If it stays **above 2.33**, the inversion is fundamental to the `c3-3` design, not the T4,
  and we finalize the paper with the Jev + hold-mode positives.

## A100 arm kernels (agent-driven via the Colab CLI)

The timing run above was human-in-the-loop. The full A100 arm was later driven by the agent
through the **Colab CLI** (installed in WSL2; OAuth completed once). Kernels:

- `colab/a100_sweep.py` — **A1** six-mode cost + quality sweep (real umT5 + VAE), 54 videos.
- `colab/a100_online.py` — **A2 v2** online policies (fixed coarse/fine/hold, random50,
  alternating50, adaptive_warp) on the fresh A100-01 split; writes `a100_online_v2`.
- `colab/a100_rq3.py` — **A3** recovery + hysteresis controller ablation (signal derived from
  the A2 v2 `fixed_fine` videos).

Results: `results/a100_mode_cost/` (trade-off confirmed, per-forward ratio **1.75**),
`results/a100_online_v2/` (RQ2 not supported on the independent endpoint), `results/a100_rq3/`
(RQ3 supported). Protocol and full accounting: `docs/A100_PROTOCOL.md`.

Operational notes: Google Drive mount is **not** available on CLI-provisioned runtimes
(`ValueError: mount failed`); persistence is by downloading partial results. Large
(>~30 MB) uploads to the runtime proxy fail on SSL, so upload small files individually.
The A100 runtime can be reclaimed mid-job; a stable network and in-kernel execution help.
