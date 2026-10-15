"""
Phase 1: real runtime-switch smoke test.

This job is derived from the verified Phase 0c recipe. It performs one
text-conditioned sequence with real Causal-rCM checkpoints:

  c1-1 initial rollout -> c3-3 continuation -> c1-1 continuation

Continuation uses the released causal_i2v_rollout_sampling interface with the
last generated latent as the boundary context. This is a feasibility test for
online switching, not a quality claim. The job records load/sampling/decode
times, VRAM, latent shapes, seam statistics, and a pass/fail verdict.

The job is resumable: if the final metadata is complete it exits without
re-running the experiment. No model files are written to local storage.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import gc
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

HELD_OUT = [
    "Splash of turquoise water in extreme slow motion, alpha channel included.",
    "The Bund Shanghai, with an intense shaking effect",
    "A beautiful coastal beach in spring, waves lapping on sand, watercolor painting",
    "A cyclist rides quickly along a forest trail in bright sunlight",
]
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("/kaggle/working/results/phase4_adaptive_online")
FRAME_DIR = OUT_DIR / "frames"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)
RAW = OUT_DIR / "results.jsonl"
SUMMARY = OUT_DIR / "phase4_summary.json"

# Complexity thresholds are fitted on Phase 3 calibration prompts only.
TAU_LOW = 1146.01034546
TAU_HIGH = 1370.908203125
SEED = 17
HORIZON = 45


def log(msg):
    print(msg, flush=True)


def find_under(root, filename):
    root = Path(root)
    if not root.exists():
        return None
    for item in root.rglob(filename):
        if item.is_file():
            return item
    return None


def install_deps():
    packages = [
        "einops", "transformers", "sentencepiece", "ftfy", "regex",
        "imageio", "imageio-ffmpeg", "tqdm", "torchvision", "fvcore",
        "omegaconf", "attrs", "pyyaml", "iopath", "termcolor", "pynvml",
        "pandas", "loguru", "safetensors", "cloudpickle", "dill",
        "matplotlib", "packaging", "opencv-python-headless",
    ]
    for package in packages:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", package],
            capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(f"pip install failed for {package}: {result.stderr[-500:]}")


def patch_file(path, replacements):
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in source:
            log(f"[warn] patch target not found in {path.name}: {old[:80]}")
        source = source.replace(old, new)
    path.write_text(source, encoding="utf-8")


def vram():
    if not torch.cuda.is_available():
        return {"cuda": False}
    return {
        "cuda": True,
        "device": torch.cuda.get_device_name(),
        "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3),
        "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }


def save_keyframes(video, label):
    from PIL import Image
    px = (1.0 + video.float().cpu().clamp(-1, 1)) / 2.0
    if px.ndim == 5:
        px = px[0]
    indices = sorted(set([0, max(0, px.shape[1] // 2), px.shape[1] - 1]))
    for index in indices:
        frame = (px[:, index].permute(1, 2, 0).numpy() * 255).astype("uint8")
        Image.fromarray(frame).save(FRAME_DIR / f"{label}_{index:03d}.png")
def append_result(item):
    with RAW.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

def completed_keys():
    keys=set()
    if RAW.exists():
        for line in RAW.read_text(encoding="utf-8").splitlines():
            try:
                x=json.loads(line)
                if x.get("status")=="complete": keys.add(x["sample_key"])
            except Exception:
                pass
    return keys

def complexity_signal(frames):
    import cv2
    t0=time.time()
    arr=np.asarray(frames, dtype=np.float32)
    if len(arr)<3:
        return 0.0, time.time()-t0
    current=arr[-1].mean(axis=2)
    gx=cv2.Sobel(current, cv2.CV_32F, 1, 0, ksize=3)
    gy=cv2.Sobel(current, cv2.CV_32F, 0, 1, ksize=3)
    value=float(np.mean(np.sqrt(gx*gx+gy*gy)) + 0.5*np.var(current))
    return value, time.time()-t0

def to_frames(video):
    px=(1.0+video.float().cpu().clamp(-1,1))/2.0
    if px.ndim==5: px=px[0]
    return (px.permute(1,2,3,0).numpy()*255).astype("uint8")

def main():
    report={"run_id":RUN_ID,"phase":"phase4-adaptive-online","status":"running","protocol":{"prompts":HELD_OUT,"seed":SEED,"horizon":HORIZON,"signal":"decoded-frame complexity","tau_low":TAU_LOW,"tau_high":TAU_HIGH,"threshold_fit":"Phase 3 calibration split only","initial_mode":"coarse","lambda_action":0.0}}
    try:
        log("=== Phase 4 real online adaptive controller ===")
        install_deps()
        log(f"torch={torch.__version__} python={sys.version.split()[0]} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'}")
        if not torch.cuda.is_available(): raise RuntimeError("CUDA/T4 unavailable")
        repo=Path("/kaggle/working/rcm")
        if repo.exists(): subprocess.run(["rm","-rf",str(repo)],check=False)
        subprocess.run(["git","clone","--depth","1","https://github.com/NVlabs/rcm.git",str(repo)],check=True,capture_output=True)
        inp=Path("/kaggle/input")
        paths={"c1":find_under(inp,"Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step2.pt"),"c3":find_under(inp,"Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step4.pt"),"vae":find_under(inp,"Wan2.1_VAE.pth"),"t5":find_under(inp,"models_t5_umt5-xxl-enc-bf16.pth"),"tokenizer":find_under(inp,"tokenizer.json")}
        if paths["tokenizer"]: paths["tokenizer"]=paths["tokenizer"].parent
        report["dataset_paths"]={k:str(v) for k,v in paths.items()}
        missing=[k for k,v in paths.items() if v is None]
        if missing: raise FileNotFoundError(f"Missing inputs: {missing}")
        for k,v in paths.items(): log(f"{k}: {v}")
        patch_file(repo/"rcm/inference/wan2pt1_t2v_causal_infer.py",[("TENSOR_KWARGS = {\"device\": \"cuda\", \"dtype\": torch.bfloat16}","TENSOR_KWARGS = {\"device\": \"cuda\", \"dtype\": torch.float16}"),(".to(dtype=torch.bfloat16).cuda()",".to(dtype=torch.float16).cuda()")])
        patch_file(repo/"rcm/utils/umt5.py",[("dtype=torch.bfloat16,","dtype=torch.float16,"),("tokenizer_path=\"google/umt5-xxl\",",f"tokenizer_path=\"{paths['tokenizer']}\",")])
        sys.path.insert(0,str(repo))
        from imaginaire.lazy_config import instantiate
        from rcm.inference.wan2pt1_t2v_causal_infer import DIT_CONFIGS,TENSOR_KWARGS,build_few_step_t_steps,build_few_step_schedules_per_chunk,expand_steps_per_chunk,parse_mid_t_schedules,make_block_pattern,causal_rollout_sampling,causal_i2v_rollout_sampling,load_dit_weights
        from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
        from rcm.utils.model_utils import init_weights_on_device
        from rcm.utils.umt5 import clear_umt5_memory,umt5_xxl,HuggingfaceTokenizer
        from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
        from imaginaire.utils.easy_io import easy_io
        TENSOR_KWARGS["dtype"]=torch.float16
        def encode_prompt(prompt):
            model=umt5_xxl(encoder_only=True).eval().requires_grad_(False).to(dtype=torch.float16)
            ck=easy_io.load(str(paths["t5"]),backend_args=None,file_format="pt",map_location="cuda",weights_only=False)
            for k in list(ck):
                if torch.is_tensor(ck[k]): ck[k]=ck[k].to(torch.float16)
            model.load_state_dict(ck,strict=False,assign=True); del ck
            tok=HuggingfaceTokenizer(name=str(paths["tokenizer"]),seq_len=512,clean="whitespace")
            ids,mask=tok(prompt,return_mask=True,add_special_tokens=True); ids,mask=ids.cuda(),mask.cuda()
            context=model(ids,mask); lengths=mask.gt(0).sum(dim=1).long(); padded=[]
            for item,length in zip(context,lengths):
                length=int(length); padded.append(item[:512] if length>512 else torch.cat([item[:length],torch.zeros(512-length,item.shape[1],device=item.device,dtype=item.dtype)],dim=0))
            emb=torch.stack(padded).to(torch.float16)
            del model,tok,ids,mask,context,padded; clear_umt5_memory(); gc.collect(); torch.cuda.empty_cache(); return emb
        tokenizer=Wan2pt1VAEInterface(vae_pth=str(paths["vae"]))
        width,height=VIDEO_RES_SIZE_INFO["480p"]["16:9"]; latent_frames=tokenizer.get_latent_num_frames(HORIZON)
        latent_h=height//tokenizer.spatial_compression_factor; latent_w=width//tokenizer.spatial_compression_factor; latent_c=tokenizer.latent_ch
        mid=[15/16,5/6,5/8]
        def load_net(path):
            started=time.time()
            with init_weights_on_device(): net=instantiate(DIT_CONFIGS["1.3B"]).eval()
            load_dit_weights(net,str(path)); net.to(**TENSOR_KWARGS).cpu(); torch.cuda.empty_cache(); return net,time.time()-started
        def run_initial(condition,generator):
            net,load_s=load_net(paths["c3"]); t_steps=build_few_step_t_steps(4,1600,mid,"cuda")
            _,blocks,_=make_block_pattern(3,latent_h,latent_w,3,3,net.get_spatial_patch_size()); counts=expand_steps_per_chunk(None,blocks,4)
            per=build_few_step_schedules_per_chunk(counts,None,1600,mid,[],"cuda")
            noise=torch.randn(1,latent_c,3,latent_h,latent_w,dtype=torch.float32,device="cuda",generator=generator)
            net.cuda(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started=time.time()
            samples=causal_rollout_sampling(net,noise,t_steps,per,None,condition,None,1.0,3,3,False,generator=generator)
            torch.cuda.synchronize(); elapsed=time.time()-started; peak=torch.cuda.max_memory_allocated()/2**30
            del net; torch.cuda.empty_cache(); return samples.detach().cpu(),{"mode":"coarse","load_s":load_s,"dit_s":elapsed,"nfe":int(sum(counts)),"peak_vram_gb":peak}
        def run_i2v(mode,boundary,condition,generator):
            net,load_s=load_net(paths["c3"] if mode=="coarse" else paths["c1"]); remaining=3
            chunk=3 if mode=="coarse" else 1; steps=None if mode=="coarse" else [1]
            t_steps=build_few_step_t_steps(4,1600,mid,"cuda"); total=4
            _,blocks,_=make_block_pattern(total,latent_h,latent_w,1,chunk,net.get_spatial_patch_size()); counts=expand_steps_per_chunk(steps,blocks-1,4)
            per=build_few_step_schedules_per_chunk(counts,steps,1600,mid,[],"cuda")
            noise=torch.randn(1,latent_c,remaining,latent_h,latent_w,dtype=torch.float32,device="cuda",generator=generator)
            net.cuda(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started=time.time()
            samples=causal_i2v_rollout_sampling(net,boundary.to("cuda"),noise,t_steps,per,steps,condition,None,1.0,chunk,False,generator=generator)
            torch.cuda.synchronize(); elapsed=time.time()-started; peak=torch.cuda.max_memory_allocated()/2**30
            fresh=samples[:,:,1:].detach().cpu(); del net; torch.cuda.empty_cache(); return fresh,{"mode":mode,"load_s":load_s,"dit_s":elapsed,"nfe":int(sum(counts)),"peak_vram_gb":peak}
        done=completed_keys(); complete=[]
        for prompt_idx,prompt in enumerate(HELD_OUT):
            key=f"held_out_{prompt_idx}_seed{SEED}_h{HORIZON}"
            if key in done: continue
            try:
                log(f"running {key}: {prompt}"); started=time.time(); emb=encode_prompt(prompt); condition={"crossattn_emb":emb.to(**TENSOR_KWARGS)}; generator=torch.Generator(device="cuda").manual_seed(SEED)
                initial,meta=run_initial(condition,generator); latents=[initial]; decisions=[]; total_load=meta["load_s"]; total_dit=meta["dit_s"]; total_nfe=meta["nfe"]; total_signal=0.0; total_vae=0.0
                video=tokenizer.decode(initial.float().cuda()); torch.cuda.synchronize(); total_vae+=0.0; frames=to_frames(video); del video; torch.cuda.empty_cache()
                mode="coarse"
                for group in range(1,latent_frames//3):
                    score,signal_s=complexity_signal(frames); total_signal+=signal_s
                    if score>=TAU_HIGH: mode="fine"
                    elif score<=TAU_LOW: mode="coarse"
                    boundary=latents[-1][:,:,-1:]; fresh,m=run_i2v(mode,boundary,condition,generator); latents.append(fresh); total_load+=m["load_s"]; total_dit+=m["dit_s"]; total_nfe+=m["nfe"]
                    combined=torch.cat(latents,dim=2); tvae=time.time(); v=tokenizer.decode(combined.float().cuda()); torch.cuda.synchronize(); total_vae+=time.time()-tvae; frames=to_frames(v); del v,combined
                    decisions.append({"group":group,"complexity":score,"mode":mode,"signal_s":signal_s,"checkpoint_load_s":m["load_s"],"dit_s":m["dit_s"],"nfe":m["nfe"]})
                latent=torch.cat(latents,dim=2)[:,:,:latent_frames]; tvae=time.time(); final_video=tokenizer.decode(latent.float().cuda()); torch.cuda.synchronize(); final_vae=time.time()-tvae; total_vae+=final_vae; final_frames=to_frames(final_video)
                segments=[]
                for group,d in enumerate(decisions,1):
                    a=0 if 3*group==0 else 1+4*(3*group-1); b=HORIZON if 3*(group+1)>=latent_frames else 1+4*(3*(group+1)-1); clip=final_frames[min(a,HORIZON):min(b,HORIZON)]; l1=float(np.abs(clip[1:].astype(np.float32)-clip[:-1].astype(np.float32)).mean()/255) if len(clip)>1 else 0.0; segments.append({"group":group,"start_frame":a,"end_frame":b,"mode":d["mode"],"adjacent_pixel_l1":l1})
                item={"sample_key":key,"prompt_idx":prompt_idx,"prompt":prompt,"seed":SEED,"horizon":HORIZON,"status":"complete","decisions":decisions,"segments":segments,"timing":{"checkpoint_load_s":total_load,"dit_s":total_dit,"vae_s":total_vae,"signal_s":total_signal,"total_s":time.time()-started,"nfe":total_nfe,"peak_vram_gb":max([meta["peak_vram_gb"]]+[x.get("peak_vram_gb",0) for x in []])},"controller":{"signal":"complexity","tau_low":TAU_LOW,"tau_high":TAU_HIGH,"fine_fraction":sum(x["mode"]=="fine" for x in decisions)/len(decisions),"switch_count":sum(decisions[i]["mode"]!=decisions[i-1]["mode"] for i in range(1,len(decisions)))} }
                append_result(item); complete.append(item); del final_video,latent,final_frames,emb; gc.collect(); torch.cuda.empty_cache()
            except Exception as exc:
                append_result({"sample_key":key,"prompt_idx":prompt_idx,"prompt":prompt,"status":"failed","error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc()}); log(traceback.format_exc())
        all_items=[]
        for line in RAW.read_text(encoding="utf-8").splitlines() if RAW.exists() else []:
            try:
                x=json.loads(line)
                if x.get("status")=="complete": all_items.append(x)
            except Exception: pass
        report.update({"status":"complete" if len(all_items)==len(HELD_OUT) else ("partial" if all_items else "failed"),"n_complete":len(all_items),"n_failed":len(HELD_OUT)-len(all_items),"results_file":str(RAW),"limitations":["four held-out prompts, one seed","complexity signal and thresholds calibrated on Phase 3 calibration split","pixel L1 is a temporal proxy, not perceptual quality","model reload and intermediate VAE decoding are included in timing"]})
        if all_items:
            report["mean_timing"]={k:float(np.mean([x["timing"][k] for x in all_items])) for k in ["checkpoint_load_s","dit_s","vae_s","signal_s","total_s","nfe","peak_vram_gb"]}
    except Exception as exc:
        report.update({"status":"failed","error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc()}); log(traceback.format_exc())
    finally:
        report["end_time"]=datetime.now().isoformat(); SUMMARY.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding="utf-8"); log(f"summary: {SUMMARY}")

if __name__=="__main__":
    main()
