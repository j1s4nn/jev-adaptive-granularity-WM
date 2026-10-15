# Extract sample frames from jev_hold_mode videos for qualitative figure selection.
# Writes candidate frames to paper/figures_src/candidates/
import subprocess, os, sys

ROOT = r"C:\Users\jisan\OneDrive\Desktop\research\idea_05_adaptive_granularity_WM\04_Experiment"
VID = os.path.join(ROOT, "results", "jev_hold_mode", "videos")
OUT = os.path.join(ROOT, "paper", "figures_src", "candidates")
os.makedirs(OUT, exist_ok=True)

variants = ["reference", "fixed_fine", "fixed_coarse", "jev_routed"]
prompts = [f"p{i}" for i in range(8)]
# sample 5 time points across the clip
times_frac = [0.05, 0.25, 0.5, 0.75, 0.97]

def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", path],
                       capture_output=True, text=True)
    return float(r.stdout.strip())

for p in prompts:
    for v in variants:
        f = os.path.join(VID, f"{p}_{v}.mp4")
        if not os.path.exists(f):
            print("MISSING", f); continue
        d = duration(f)
        for k, fr in enumerate(times_frac):
            t = max(0.0, min(d - 0.05, d * fr))
            out = os.path.join(OUT, f"{p}_{v}_t{k}.png")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", f,
                            "-frames:v", "1", out], check=True)
print("done")
