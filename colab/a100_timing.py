# =====================================================================
#  A100 hardware re-time for the adaptive-granularity project
#  Paste this WHOLE file into a single Google Colab cell and run it.
#  Runtime must be set to an A100 GPU (Runtime > Change runtime type).
#  It prints a table and a final line:  RESULT_JSON: {...}
#  Copy that final line back to the agent.
#
#  BEFORE RUNNING: replace PASTE_KAGGLE_TOKEN_HERE below with your Kaggle
#  API token (the same KGAT_... value in %USERPROFILE%\.kaggle\kaggle.json).
#  Your token is only used inside your own Colab session.
# =====================================================================
import os, subprocess, sys, time, json, zipfile, shutil

KAGGLE_USERNAME = "ajjisan"
KAGGLE_KEY = "PASTE_KAGGLE_TOKEN_HERE"   # <-- replace with your KGAT_... token

print("=" * 60)
subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv", shell=True)

# ---- install deps ----
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "einops", "transformers", "sentencepiece", "ftfy", "regex", "tqdm",
                "torchvision", "fvcore", "omegaconf", "attrs", "pyyaml", "iopath",
                "termcolor", "pynvml", "pandas", "loguru", "safetensors",
                "cloudpickle", "dill", "matplotlib", "packaging", "kaggle"], check=True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---- clone rcm ----
if not os.path.exists("/content/rcm"):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/NVlabs/rcm.git", "/content/rcm"], check=True)
print("rcm commit:", subprocess.check_output(["git", "-C", "/content/rcm", "rev-parse", "HEAD"], text=True).strip())

# ---- download the 3 DiT checkpoints (only what timing needs) ----
os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
os.environ["KAGGLE_KEY"] = KAGGLE_KEY
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"], check=True)
# also write kaggle.json so every kaggle CLI version picks it up
_kdir = os.path.expanduser("~/.kaggle")
os.makedirs(_kdir, exist_ok=True)
with open(os.path.join(_kdir, "kaggle.json"), "w") as fh:
    json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, fh)
try:
    os.chmod(os.path.join(_kdir, "kaggle.json"), 0o600)
except Exception:
    pass

CKPT = "/content/ckpts"
os.makedirs(CKPT, exist_ok=True)
WANT = {
    "fine_c1_1_s1": "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step2.pt",
    "coarse_c3_3_s4": "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step4.pt",
    "coarse_c3_3_s1": "Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step2.pt",
}
EXPECTED = 2838292823  # bytes per causal checkpoint (from the Kaggle dataset listing)


def size_ok(p):
    return os.path.exists(p) and abs(os.path.getsize(p) - EXPECTED) < 5000


def move_nested(name, dst):
    for root, _, files in os.walk(CKPT):
        for f in files:
            p = os.path.join(root, f)
            if f == name and p != dst:
                shutil.move(p, dst)


# 1) try single-file download for each checkpoint, verifying size
for key, name in WANT.items():
    dst = os.path.join(CKPT, name)
    if size_ok(dst):
        continue
    if os.path.exists(dst):
        os.remove(dst)
    for src in (f"rcm_ckpts/{name}", name):
        subprocess.run(["kaggle", "datasets", "download", "-d", "ajjisan/causal-rcm-ckpts",
                        "-f", src, "-p", CKPT, "--force"], capture_output=True, text=True)
        move_nested(name, dst)
        if size_ok(dst):
            break
    print("single-file", name, "ok" if size_ok(dst) else "FAILED",
          (os.path.getsize(dst) if os.path.exists(dst) else None))

# 2) fallback: download the whole dataset zip and extract just the 3 files
if not all(size_ok(os.path.join(CKPT, n)) for n in WANT.values()):
    print("falling back to whole-dataset download (larger, ~12 GB)...")
    DS = "/content/ds"
    os.makedirs(DS, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", "ajjisan/causal-rcm-ckpts", "-p", DS, "--force"], check=True)
    zpath = os.path.join(DS, "causal-rcm-ckpts.zip")
    with zipfile.ZipFile(zpath) as z:
        for key, name in WANT.items():
            dst = os.path.join(CKPT, name)
            if size_ok(dst):
                continue
            with z.open(f"rcm_ckpts/{name}") as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out, 1024 * 1024 * 8)
            print("extracted", name, os.path.getsize(dst))

if not all(size_ok(os.path.join(CKPT, n)) for n in WANT.values()):
    for key, name in WANT.items():
        p = os.path.join(CKPT, name)
        print("MISSING/BAD:", name, os.path.getsize(p) if os.path.exists(p) else "absent", "expected", EXPECTED)
    raise SystemExit("checkpoint download failed - send this output to the agent")
print("all 3 checkpoints verified:", [os.path.getsize(os.path.join(CKPT, n)) for n in WANT.values()])

# ---- patch rcm for fp16 ----
P = "/content/rcm/rcm/inference/wan2pt1_t2v_causal_infer.py"
s = open(P).read()
s = s.replace('TENSOR_KWARGS = {"device": "cuda", "dtype": torch.bfloat16}',
              'TENSOR_KWARGS = {"device": "cuda", "dtype": torch.float16}')
s = s.replace(".to(dtype=torch.bfloat16).cuda()", ".to(dtype=torch.float16).cuda()")
open(P, "w").write(s)

sys.path.insert(0, "/content/rcm")
import torch
from imaginaire.lazy_config import instantiate
from rcm.inference.wan2pt1_t2v_causal_infer import (
    DIT_CONFIGS, TENSOR_KWARGS, build_few_step_t_steps, build_few_step_schedules_per_chunk,
    expand_steps_per_chunk, parse_mid_t_schedules, make_block_pattern, block_span,
    load_dit_weights, RECTIFIED_FLOW_T_SCALING)
from rcm.utils.model_utils import init_weights_on_device
from rcm.utils.blockmask import AttnMaskSpec
from rcm.utils.kv_cache import CausalInferenceState, KVCacheMode
from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
TENSOR_KWARGS["dtype"] = torch.float16

GPU = torch.cuda.get_device_name()
print("torch", torch.__version__, "| gpu", GPU)
MID = [15/16, 5/6, 5/8]
W, H = VIDEO_RES_SIZE_INFO["480p"]["16:9"]
LH, LW = H // 8, W // 8            # 480p 16:9 latent grid (spatial_compression_factor=8)
LC = 16
SEED = 17
MODES = {
    "fine_c1_1_s1":   {"ckpt": "fine_c1_1_s1",   "first": 1, "chunk": 1, "steps": [4, 1], "sched": "15/16,5/6,5/8;", "default": 4},
    "coarse_c3_3_s4": {"ckpt": "coarse_c3_3_s4", "first": 3, "chunk": 3, "steps": None,    "sched": "",              "default": 4},
    "coarse_c3_3_s1": {"ckpt": "coarse_c3_3_s1", "first": 3, "chunk": 3, "steps": [4, 1],  "sched": "15/16,5/6,5/8;", "default": 1},
}

@torch.no_grad()
def sampler(net, init_noise, t_steps, per_chunk, counts, cond, first, chunk, gen):
    B, C, T, Hh, Ww = init_noise.shape
    frame_tokens, num_blocks, bp = make_block_pattern(T, Hh, Ww, first, chunk, net.get_spatial_patch_size())
    ones = torch.ones(B, 1, device=init_noise.device, dtype=torch.float64)
    mx = max(counts)
    noises = [torch.randn_like(init_noise, dtype=torch.float32, generator=gen) for _ in range(mx)]
    kv = net.allocate_kv_caches(max_len=T * frame_tokens)
    fwd = 0
    for i in range(num_blocks):
        fs, fe, _ = block_span(bp, i)
        attn = AttnMaskSpec(mode="block_causal", pattern=bp, q_block_offset=i)
        ts = per_chunk[i] if per_chunk else t_steps
        x = init_noise[:, :, fs:fe].to(torch.float64) * ts[0]
        for si, (tc, tn) in enumerate(zip(ts[:-1], ts[1:])):
            tcur = (tc * ones * RECTIFIED_FLOW_T_SCALING).to(**TENSOR_KWARGS)
            st = CausalInferenceState(mode=KVCacheMode.READONLY, kv_caches=kv, pattern=bp, block_cursor=i, fast_infer=True)
            v = net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS), timesteps_B_T=tcur, **cond, inference_state=st, attn_meta=attn).float()
            fwd += 1
            x = (1 - tn) * (x - tc * v.to(torch.float64)) + tn * noises[si][:, :, fs:fe]
        zero = torch.zeros(B, 1, **TENSOR_KWARGS)
        ap = CausalInferenceState(mode=KVCacheMode.APPEND, kv_caches=kv, pattern=bp, block_cursor=i, fast_infer=True)
        net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS), timesteps_B_T=zero, **cond, inference_state=ap, attn_meta=attn)
        fwd += 1
    del kv
    torch.cuda.empty_cache()
    return fwd

def run(mode_key, latent_frames):
    cfg = MODES[mode_key]
    with init_weights_on_device():
        net = instantiate(DIT_CONFIGS["1.3B"]).eval()
    load_dit_weights(net, os.path.join(CKPT, WANT[cfg["ckpt"]]))
    net.to(**TENSOR_KWARGS).cpu()
    torch.cuda.empty_cache()
    _, blocks, _ = make_block_pattern(latent_frames, LH, LW, cfg["first"], cfg["chunk"], net.get_spatial_patch_size())
    counts = expand_steps_per_chunk(cfg["steps"], blocks, cfg["default"])
    per_chunk = build_few_step_schedules_per_chunk(counts, cfg["steps"], 1600, MID, parse_mid_t_schedules(cfg["sched"]), "cuda")
    t_steps = build_few_step_t_steps(4, 1600, MID, "cuda")
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    noise = torch.randn(1, LC, latent_frames, LH, LW, dtype=torch.float32, device="cuda", generator=gen)
    cond = {"crossattn_emb": torch.zeros(1, 512, 4096, dtype=torch.float16, device="cuda")}  # dummy text emb
    net.cuda()
    torch.cuda.synchronize(); t0 = time.time()
    fwd = sampler(net, noise, t_steps, per_chunk, counts, cond, cfg["first"], cfg["chunk"], gen)
    torch.cuda.synchronize(); dit = time.time() - t0
    del net, noise
    torch.cuda.empty_cache()
    return {"mode": mode_key, "latent_frames": latent_frames, "dit_s": dit, "nfe": int(sum(counts)),
            "forwards": fwd, "s_per_forward": dit / fwd}

rows = []
for lf in (15, 21):
    for mk in MODES:
        r = run(mk, lf)
        rows.append(r)
        print(f"h{lf} {mk:<16} nfe={r['nfe']:>2} fwd={r['forwards']:>3} dit={r['dit_s']:6.2f}s s/fwd={r['s_per_forward']:.3f}")

fine = [r for r in rows if r["mode"] == "fine_c1_1_s1"]
cs1 = [r for r in rows if r["mode"] == "coarse_c3_3_s1"]
import statistics
fp = statistics.fmean(r["s_per_forward"] for r in fine)
cp = statistics.fmean(r["s_per_forward"] for r in cs1)
print("=" * 60)
print(f"GPU: {GPU}")
print(f"fine s/forward={fp:.3f}  coarse s/forward={cp:.3f}  ratio={cp/fp:.2f}")
print("RESULT_JSON: " + json.dumps({"gpu": GPU, "rows": rows, "fine_s_per_fwd": fp, "coarse_s_per_fwd": cp, "ratio": cp / fp}))
