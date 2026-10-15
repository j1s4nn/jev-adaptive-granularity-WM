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
TAU_LOW = 1.0250263810157776
TAU_HIGH = 1.0357918739318848
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
        "matplotlib", "packaging",
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

def to_frames(video):
    px=(1.0+video.float().cpu().clamp(-1,1))/2.0
    if px.ndim==5: px=px[0]
    return (px.permute(1,2,3,0).numpy()*255).astype("uint8")

def main():
    report={"run_id":RUN_ID,"phase":"phase4-residual-adaptive-online","status":"running","protocol":{"prompts":HELD_OUT,"seed":SEED,"horizon":HORIZON,"signal":"denoising residual between adjacent predictions in preceding coarse segment","tau_low":TAU_LOW,"tau_high":TAU_HIGH,"threshold_fit":"Phase 3 calibration split only","initial_mode":"coarse","lambda_action":0.0,"recovery_length":0}}
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
        from rcm.inference.wan2pt1_t2v_causal_infer import DIT_CONFIGS,TENSOR_KWARGS,build_few_step_t_steps,build_few_step_schedules_per_chunk,expand_steps_per_chunk,make_block_pattern,block_span,load_dit_weights,RECTIFIED_FLOW_T_SCALING
        from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
        from rcm.utils.model_utils import init_weights_on_device
        from rcm.utils.umt5 import clear_umt5_memory,umt5_xxl,HuggingfaceTokenizer
        from rcm.utils.blockmask import AttnMaskSpec
        from rcm.utils.kv_cache import CausalInferenceState,KVCacheMode
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
        @torch.no_grad()
        def sample_blocks(mode,condition,generator,boundary=None):
            """Verified causal sampler with Phase 3 residual measured in-flight."""
            is_initial=boundary is None
            net,load_s=load_net(paths["c3"] if mode=="coarse" else paths["c1"])
            chunk=3 if mode=="coarse" else 1
            first=3 if is_initial else 1
            total=3 if is_initial else 4
            steps=None if mode=="coarse" else ([4,1] if is_initial else [1])
            t_steps=build_few_step_t_steps(4,1600,mid,"cuda")
            _,nblocks,pattern=make_block_pattern(total,latent_h,latent_w,first,chunk,net.get_spatial_patch_size())
            n_generated=nblocks if is_initial else nblocks-1
            counts=expand_steps_per_chunk(steps,n_generated,4)
            per=build_few_step_schedules_per_chunk(counts,steps,1600,mid,[],"cuda")
            noise_t=total if is_initial else total-1
            noise=torch.randn(1,latent_c,noise_t,latent_h,latent_w,dtype=torch.float32,device="cuda",generator=generator)
            net.cuda(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started=time.time()
            frame_tokens,_,_=make_block_pattern(total,latent_h,latent_w,first,chunk,net.get_spatial_patch_size())
            kv=net.allocate_kv_caches(max_len=total*frame_tokens); blocks=[]; residuals=[]
            ones=torch.ones(1,1,device="cuda",dtype=torch.float64)
            if not is_initial:
                prefill_pattern=pattern
                prefill_attn=AttnMaskSpec(mode="block_causal",pattern=prefill_pattern,q_block_offset=0)
                prefill=CausalInferenceState(mode=KVCacheMode.APPEND,kv_caches=kv,pattern=prefill_pattern,block_cursor=0,fast_infer=True)
                net(x_B_C_T_H_W=boundary.to(**TENSOR_KWARGS),timesteps_B_T=torch.zeros(1,1,**TENSOR_KWARGS),**condition,inference_state=prefill,attn_meta=prefill_attn)
                step_noises=[torch.randn(noise.shape,device=noise.device,dtype=torch.float32,generator=generator) for _ in range(max(counts))]
                # `boundary` is kept on CPU between chunks; move it back to
                # the active CUDA device before concatenating with the sampled
                # GPU blocks below.
                blocks.append(boundary.to(device="cuda", dtype=torch.float64))
                block_indices=range(1,nblocks)
            else:
                step_noises=[torch.randn(noise.shape,device=noise.device,dtype=torch.float32,generator=generator) for _ in range(max(counts))]
                block_indices=range(nblocks)
            for i in block_indices:
                fs,fe,_=block_span(pattern,i)
                attn=AttnMaskSpec(mode="block_causal",pattern=pattern,q_block_offset=i)
                schedule=per[i if is_initial else i-1]
                # In a continuation call, `noise` contains only the newly
                # generated frames while the block pattern also includes the
                # one-frame boundary prefix. Align the random-noise slice with
                # the prefix-free tensor, as the upstream causal_i2v sampler
                # does with its `frame_start - 1` offset. Using `fs:fe` here
                # drops one frame and breaks the Euler update for c3-3.
                noise_start=fs if is_initial else fs-1
                x=noise[:,:,noise_start:noise_start+(fe-fs)].to(torch.float64)*schedule[0]
                previous_v=None; residual=None
                for step_idx,(tc,tn) in enumerate(zip(schedule[:-1],schedule[1:])):
                    ts=(tc*ones*RECTIFIED_FLOW_T_SCALING).to(**TENSOR_KWARGS)
                    state=CausalInferenceState(mode=KVCacheMode.READONLY,kv_caches=kv,pattern=pattern,block_cursor=i,fast_infer=True)
                    current_v=net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS),timesteps_B_T=ts,**condition,inference_state=state,attn_meta=attn).float()
                    if previous_v is not None: residual=float((current_v-previous_v).abs().mean().item())
                    previous_v=current_v
                    noise_slice=step_noises[step_idx][:,:,noise_start:noise_start+(fe-fs)]
                    x=(1-tn)*(x-tc*current_v.to(torch.float64))+tn*noise_slice
                zero=torch.zeros(1,1,**TENSOR_KWARGS)
                append=CausalInferenceState(mode=KVCacheMode.APPEND,kv_caches=kv,pattern=pattern,block_cursor=i,fast_infer=True)
                net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS),timesteps_B_T=zero,**condition,inference_state=append,attn_meta=attn)
                blocks.append(x); residuals.append(residual)
            torch.cuda.synchronize(); elapsed=time.time()-started; peak=torch.cuda.max_memory_allocated()/2**30
            latent=torch.cat(blocks,dim=2).detach().cpu(); del kv,net; torch.cuda.empty_cache()
            if not is_initial: latent=latent[:,:,1:]
            return latent,residuals,{"mode":mode,"load_s":load_s,"dit_s":elapsed,"nfe":int(sum(counts)),"peak_vram_gb":peak}
        def run_initial(condition,generator):
            return sample_blocks("coarse",condition,generator)
        def run_i2v(mode,boundary,condition,generator):
            return sample_blocks(mode,condition,generator,boundary)
        done=completed_keys(); complete=[]
        for prompt_idx,prompt in enumerate(HELD_OUT):
            key=f"held_out_{prompt_idx}_seed{SEED}_h{HORIZON}"
            if key in done: continue
            try:
                log(f"running {key}: {prompt}"); started=time.time(); ttext=time.time(); emb=encode_prompt(prompt); text_encode_s=time.time()-ttext; condition={"crossattn_emb":emb.to(**TENSOR_KWARGS)}; generator=torch.Generator(device="cuda").manual_seed(SEED)
                initial,initial_res,meta=run_initial(condition,generator); latents=[initial]; decisions=[]; total_load=meta["load_s"]; total_dit=meta["dit_s"]; total_nfe=meta["nfe"]; total_signal=0.0; total_vae=0.0; peaks=[meta["peak_vram_gb"]]
                last_coarse=float(initial_res[-1] or 0.0); mode="coarse"
                for group in range(1,latent_frames//3):
                    signal_start=time.time(); score=last_coarse; signal_s=time.time()-signal_start; total_signal+=signal_s
                    if score>=TAU_HIGH: mode="fine"
                    elif score<=TAU_LOW: mode="coarse"
                    boundary=latents[-1][:,:,-1:]; fresh,chunk_res,m=run_i2v(mode,boundary,condition,generator); latents.append(fresh); total_load+=m["load_s"]; total_dit+=m["dit_s"]; total_nfe+=m["nfe"]; peaks.append(m["peak_vram_gb"])
                    if mode=="coarse" and chunk_res and chunk_res[-1] is not None: last_coarse=float(chunk_res[-1])
                    decisions.append({"group":group,"residual":score,"mode":mode,"recovery_counter":0,"signal_s":signal_s,"checkpoint_load_s":m["load_s"],"dit_s":m["dit_s"],"nfe":m["nfe"],"next_residual":last_coarse})
                latent=torch.cat(latents,dim=2)[:,:,:latent_frames]; tvae=time.time(); final_video=tokenizer.decode(latent.float().cuda()); torch.cuda.synchronize(); total_vae=time.time()-tvae; final_frames=to_frames(final_video)
                segments=[]
                for group,d in enumerate(decisions,1):
                    a=0 if 3*group==0 else 1+4*(3*group-1); b=HORIZON if 3*(group+1)>=latent_frames else 1+4*(3*(group+1)-1); clip=final_frames[min(a,HORIZON):min(b,HORIZON)]; l1=float(np.abs(clip[1:].astype(np.float32)-clip[:-1].astype(np.float32)).mean()/255) if len(clip)>1 else 0.0; segments.append({"group":group,"start_frame":a,"end_frame":b,"mode":d["mode"],"adjacent_pixel_l1":l1})
                item={"sample_key":key,"prompt_idx":prompt_idx,"prompt":prompt,"seed":SEED,"horizon":HORIZON,"status":"complete","decisions":decisions,"segments":segments,"timing":{"text_encode_s":text_encode_s,"checkpoint_load_s":total_load,"dit_s":total_dit,"vae_s":total_vae,"signal_s":total_signal,"total_s":time.time()-started,"nfe":total_nfe,"peak_vram_gb":max(peaks)},"controller":{"signal":"denoising residual","tau_low":TAU_LOW,"tau_high":TAU_HIGH,"recovery_length":0,"fine_fraction":sum(x["mode"]=="fine" for x in decisions)/len(decisions),"switch_count":sum(decisions[i]["mode"]!=decisions[i-1]["mode"] for i in range(1,len(decisions)))}}
                append_result(item); complete.append(item); del final_video,latent,final_frames,emb; gc.collect(); torch.cuda.empty_cache()
            except Exception as exc:
                append_result({"sample_key":key,"prompt_idx":prompt_idx,"prompt":prompt,"status":"failed","error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc()}); log(traceback.format_exc())
        all_items=[]
        for line in RAW.read_text(encoding="utf-8").splitlines() if RAW.exists() else []:
            try:
                x=json.loads(line)
                if x.get("status")=="complete": all_items.append(x)
            except Exception: pass
        report.update({"status":"complete" if len(all_items)==len(HELD_OUT) else ("partial" if all_items else "failed"),"n_complete":len(all_items),"n_failed":len(HELD_OUT)-len(all_items),"results_file":str(RAW),"limitations":["four held-out prompts, one seed","residual thresholds calibrated on Phase 3 calibration split","pixel L1 is a temporal proxy, not perceptual quality","fine-mode steps retain the latest residual observed in a coarse segment"]})
        if all_items:
            report["mean_timing"]={k:float(np.mean([x["timing"][k] for x in all_items])) for k in ["text_encode_s","checkpoint_load_s","dit_s","vae_s","signal_s","total_s","nfe","peak_vram_gb"]}
    except Exception as exc:
        report.update({"status":"failed","error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc()}); log(traceback.format_exc())
    finally:
        report["end_time"]=datetime.now().isoformat(); SUMMARY.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding="utf-8"); log(f"summary: {SUMMARY}")

if __name__=="__main__":
    main()
