"""Phase 4: held-out online residual-threshold controller evaluation on T4.

Runs a real c3-3 initial chunk followed by three causal c1-1/c3-3 continuation
decisions. Residual thresholds are fixed from Phase 3 calibration prompts.
The paired Phase 3 held-out fixed-mode videos are used for descriptive quality
and offline oracle comparisons; they are never used to tune the controller.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import gc, json, shutil, subprocess, sys, time, traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("/kaggle/working/results/phase4_adaptive")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW = OUT_DIR / "results.jsonl"
SUMMARY = OUT_DIR / "phase4_summary.json"
HELD_OUT = [
    "Splash of turquoise water in extreme slow motion, alpha channel included.",
    "The Bund Shanghai, with an intense shaking effect",
    "A beautiful coastal beach in spring, waves lapping on sand, watercolor painting",
    "A cyclist rides quickly along a forest trail in bright sunlight",
]
SEED, HORIZON = 17, 45
# Residual Q25/Q75 fitted on the 12 Phase 3 calibration observations only.
TAU_LOW, TAU_HIGH = 1.02502638, 1.04670611

def log(x): print(x, flush=True)
def find_under(root, filename):
    return next((p for p in Path(root).rglob(filename) if p.is_file()), None)
def install_deps():
    for p in ["einops","transformers","sentencepiece","ftfy","regex","imageio","imageio-ffmpeg","tqdm","torchvision","fvcore","omegaconf","attrs","pyyaml","iopath","termcolor","pynvml","pandas","loguru","safetensors","cloudpickle","dill","matplotlib","packaging","opencv-python-headless"]:
        r=subprocess.run([sys.executable,"-m","pip","install","--quiet",p],capture_output=True,text=True)
        if r.returncode: raise RuntimeError(f"pip install failed ({p}): {r.stderr[-500:]}")
def patch_file(path, replacements):
    s=Path(path).read_text(encoding="utf-8")
    for a,b in replacements:
        if a not in s: log(f"[warn] patch target missing: {a[:60]}")
        s=s.replace(a,b)
    Path(path).write_text(s,encoding="utf-8")
def write(item):
    with RAW.open("a",encoding="utf-8") as f: f.write(json.dumps(item,ensure_ascii=False)+"\n")
def done_keys():
    out=set()
    if RAW.exists():
        for line in RAW.read_text(encoding="utf-8").splitlines():
            try:
                x=json.loads(line)
                if x.get("status")=="complete": out.add(x["sample_key"])
            except Exception: pass
    return out
def segment_bounds(num_frames, latent_frames, group_idx):
    sl=3*group_idx; el=min(3*(group_idx+1),latent_frames)
    sp=0 if sl==0 else 1+4*(sl-1)
    ep=num_frames if el==latent_frames else 1+4*(el-1)
    return min(sp,num_frames),min(ep,num_frames)
def main():
    report={"run_id":RUN_ID,"phase":"phase4-adaptive-controller","status":"running","thresholds":{"signal":"previous coarse-chunk denoising residual","tau_low":TAU_LOW,"tau_high":TAU_HIGH,"fit_split":"Phase 3 calibration prompts only"}}
    try:
        install_deps()
        log(f"torch={torch.__version__} python={sys.version.split()[0]} gpu={torch.cuda.get_device_name() if torch.cuda.is_available() else 'none'}")
        if not torch.cuda.is_available(): raise RuntimeError("CUDA/T4 unavailable")
        repo=Path("/kaggle/working/rcm")
        if repo.exists(): shutil.rmtree(repo)
        subprocess.run(["git","clone","--depth","1","https://github.com/NVlabs/rcm.git",str(repo)],check=True,capture_output=True)
        inp=Path("/kaggle/input")
        paths={"c1":find_under(inp,"Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step2.pt"),"c3":find_under(inp,"Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step4.pt"),"vae":find_under(inp,"Wan2.1_VAE.pth"),"t5":find_under(inp,"models_t5_umt5-xxl-enc-bf16.pth"),"tokenizer":find_under(inp,"tokenizer.json")}
        if paths["tokenizer"]: paths["tokenizer"]=paths["tokenizer"].parent
        report["dataset_paths"]={k:str(v) for k,v in paths.items()}
        missing=[k for k,v in paths.items() if v is None]
        if missing: raise FileNotFoundError(f"Missing Kaggle inputs: {missing}")
        for k,v in paths.items(): log(f"{k}: {v}")
        patch_file(repo/"rcm/inference/wan2pt1_t2v_causal_infer.py",[("TENSOR_KWARGS = {\"device\": \"cuda\", \"dtype\": torch.bfloat16}","TENSOR_KWARGS = {\"device\": \"cuda\", \"dtype\": torch.float16}"),(".to(dtype=torch.bfloat16).cuda()",".to(dtype=torch.float16).cuda()")])
        patch_file(repo/"rcm/utils/umt5.py",[("dtype=torch.bfloat16,","dtype=torch.float16,"),("tokenizer_path=\"google/umt5-xxl\",",f"tokenizer_path=\"{paths['tokenizer']}\",")])
        sys.path.insert(0,str(repo)); sys.path.insert(0,"/kaggle/working")
        from imaginaire.lazy_config import instantiate
        from rcm.inference.wan2pt1_t2v_causal_infer import DIT_CONFIGS,TENSOR_KWARGS,build_few_step_t_steps,build_few_step_schedules_per_chunk,expand_steps_per_chunk,parse_mid_t_schedules,make_block_pattern,block_span,load_dit_weights,RECTIFIED_FLOW_T_SCALING,causal_i2v_rollout_sampling
        from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
        from rcm.utils.model_utils import init_weights_on_device
        from rcm.utils.umt5 import clear_umt5_memory,umt5_xxl,HuggingfaceTokenizer
        from rcm.utils.blockmask import AttnMaskSpec
        from rcm.utils.kv_cache import CausalInferenceState,KVCacheMode
        from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
        from imaginaire.utils.easy_io import easy_io
        TENSOR_KWARGS["dtype"]=torch.float16
        from src.controller.policy import AdaptiveController,ControllerConfig

        def encode_prompt(prompt):
            model=umt5_xxl(encoder_only=True).eval().requires_grad_(False).to(dtype=torch.float16)
            ck=easy_io.load(str(paths["t5"]),backend_args=None,file_format="pt",map_location="cuda",weights_only=False)
            for k in list(ck):
                if torch.is_tensor(ck[k]): ck[k]=ck[k].to(torch.float16)
            model.load_state_dict(ck,strict=False,assign=True); del ck
            tok=HuggingfaceTokenizer(name=str(paths["tokenizer"]),seq_len=512,clean="whitespace")
            ids,mask=tok(prompt,return_mask=True,add_special_tokens=True); ids,mask=ids.cuda(),mask.cuda()
            context=model(ids,mask); length=int(mask.gt(0).sum(dim=1)[0]); emb=context[:, :min(length,512)]
            if emb.shape[1]<512: emb=torch.cat([emb,torch.zeros(1,512-emb.shape[1],emb.shape[2],device="cuda",dtype=emb.dtype)],dim=1)
            emb=emb.to(torch.float16)
            del model,tok,ids,mask,context; clear_umt5_memory(); gc.collect(); torch.cuda.empty_cache()
            return emb
        tokenizer=Wan2pt1VAEInterface(vae_pth=str(paths["vae"]))
        width,height=VIDEO_RES_SIZE_INFO["480p"]["16:9"]
        latent_frames=tokenizer.get_latent_num_frames(HORIZON)
        lh,lw,lc=height//tokenizer.spatial_compression_factor,width//tokenizer.spatial_compression_factor,tokenizer.latent_ch
        mid=[15/16,5/6,5/8]; t_steps=build_few_step_t_steps(4,1600,mid,"cuda")
        def load_net(mode):
            ckpt=paths["c3"] if mode=="coarse" else paths["c1"]
            t=time.time()
            with init_weights_on_device(): net=instantiate(DIT_CONFIGS["1.3B"]).eval()
            load_dit_weights(net,str(ckpt)); net.to(**TENSOR_KWARGS).cpu(); torch.cuda.empty_cache()
            return net,time.time()-t
        @torch.no_grad()
        def sample_initial(prompt_condition, generator):
            mode="coarse"; net,load_s=load_net(mode); first=3; chunk=3
            _,nblocks,bp=make_block_pattern(3,lh,lw,first,chunk,net.get_spatial_patch_size())
            counts=expand_steps_per_chunk(None,nblocks,4); sched=build_few_step_schedules_per_chunk(counts,None,1600,mid,[],"cuda")
            noise=torch.randn(1,lc,3,lh,lw,dtype=torch.float32,device="cuda",generator=generator)
            net.cuda(); torch.cuda.synchronize(); t=time.time()
            frame_tokens,_,_=make_block_pattern(3,lh,lw,first,chunk,net.get_spatial_patch_size())
            ones=torch.ones(1,1,device="cuda",dtype=torch.float64); kv=net.allocate_kv_caches(max_len=3*frame_tokens); blocks=[]; records=[]
            for i in range(nblocks):
                fs,fe,_=block_span(bp,i); attn=AttnMaskSpec(mode="block_causal",pattern=bp,q_block_offset=i); x=noise[:,:,fs:fe].double()*t_steps[0]; prev=None
                for tc,tn in zip(t_steps[:-1],t_steps[1:]):
                    ts=(tc*ones*RECTIFIED_FLOW_T_SCALING).to(**TENSOR_KWARGS); state=CausalInferenceState(mode=KVCacheMode.READONLY,kv_caches=kv,pattern=bp,block_cursor=i,fast_infer=True)
                    v=net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS),timesteps_B_T=ts,**prompt_condition,inference_state=state,attn_meta=attn).float()
                    if prev is not None: residual=float((v-prev).abs().mean().item())
                    prev=v; x=(1-tn)*(x-tc*v.double())+tn*torch.randn_like(x,dtype=torch.float32,generator=generator)[:,:,fs:fe]
                zero=torch.zeros(1,1,**TENSOR_KWARGS); append=CausalInferenceState(mode=KVCacheMode.APPEND,kv_caches=kv,pattern=bp,block_cursor=i,fast_infer=True)
                net(x_B_C_T_H_W=x.to(**TENSOR_KWARGS),timesteps_B_T=zero,**prompt_condition,inference_state=append,attn_meta=attn); blocks.append(x); records.append(residual if 'residual' in locals() else None)
            torch.cuda.synchronize(); dit_s=time.time()-t; del kv,net; torch.cuda.empty_cache()
            return torch.cat(blocks,dim=2).detach().cpu(),records,{"mode":mode,"load_s":load_s,"dit_s":dit_s,"nfe":int(sum(counts))}
        @torch.no_grad()
        def sample_continuation(mode,boundary,condition,generator):
            net,load_s=load_net(mode); remaining=3; chunk=3 if mode=="coarse" else 1; steps=None if mode=="coarse" else [1]
            nt=4; schedules=build_few_step_t_steps(nt,1600,mid,"cuda"); total=1+remaining
            _,nb,bp=make_block_pattern(total,lh,lw,1,chunk,net.get_spatial_patch_size()); counts=expand_steps_per_chunk(steps,nb-1,nt); per=build_few_step_schedules_per_chunk(counts,steps,1600,mid,[],"cuda")
            noise=torch.randn(1,lc,remaining,lh,lw,dtype=torch.float32,device="cuda",generator=generator)
            net.cuda(); torch.cuda.synchronize(); t=time.time()
            samples=causal_i2v_rollout_sampling(net,boundary.to("cuda"),noise,schedules,per,steps,condition,None,1.0,chunk,False,generator=generator)
            torch.cuda.synchronize(); dit_s=time.time()-t
            # The causal API returns the boundary as its first latent.
            fresh=samples[:,:,1:].detach().cpu(); del net; torch.cuda.empty_cache()
            return fresh,[],{"mode":mode,"load_s":load_s,"dit_s":dit_s,"nfe":int(sum(counts))}
        completed=done_keys(); out=[]
        for idx,prompt in enumerate(HELD_OUT):
            key=f"held_out_{idx}_seed{SEED}_h{HORIZON}"
            if key in completed: continue
            try:
                log(f"adaptive rollout {key}: {prompt}"); started=time.time(); emb=encode_prompt(prompt); cond={"crossattn_emb":emb.to(**TENSOR_KWARGS)}; gen=torch.Generator(device="cuda").manual_seed(SEED)
                initial,initial_res,initial_meta=sample_initial(cond,gen); latents=[initial]; boundary=initial[:,:,-1:]; last_coarse=float(initial_res[-1] or 0.0)
                cfg=ControllerConfig(tau_high=TAU_HIGH,tau_low=TAU_LOW,recovery_length=0,lambda_action=0.0,initial_mode="coarse"); controller=AdaptiveController(cfg); records=[]; total_nfe=initial_meta["nfe"]; total_load=initial_meta["load_s"]; total_dit=initial_meta["dit_s"]
                for group in range(1,latent_frames//3):
                    score=last_coarse; chosen=controller.step(score); mode="fine" if chosen=="fine" else "coarse"
                    fresh,chunk_res,meta=sample_continuation(mode,boundary,cond,gen); latents.append(fresh); boundary=fresh[:,:,-1:]
                    total_nfe+=meta["nfe"]; total_load+=meta["load_s"]; total_dit+=meta["dit_s"]
                    if mode=="coarse" and chunk_res and chunk_res[-1] is not None: last_coarse=float(chunk_res[-1])
                    records.append({"group":group,"signal_residual":score,"mode":mode,"switch_count":controller.switch_count,"checkpoint_load_s":meta["load_s"],"dit_s":meta["dit_s"],"nfe":meta["nfe"]})
                latent=torch.cat(latents,dim=2)[:,:,:latent_frames]
                t=time.time(); video=tokenizer.decode(latent.float().cuda()); torch.cuda.synchronize(); vae_s=time.time()-t
                px=(1+video.float().cpu().clamp(-1,1))/2
                if px.ndim==5: px=px[0]
                frames=(px.permute(1,2,3,0).numpy()*255).astype(np.uint8)
                seg=[]
                for group in range(1,latent_frames//3):
                    a,b=segment_bounds(HORIZON,latent_frames,group); f=frames[a:b]
                    l1=float(np.abs(f[1:].astype(np.float32)-f[:-1].astype(np.float32)).mean()/255) if len(f)>1 else 0.0
                    seg.append({"group":group,"start_frame":a,"end_frame":b,"adjacent_pixel_l1":l1,"mode":records[group-1]["mode"]})
                item={"sample_key":key,"split":"held_out","prompt_idx":idx,"prompt":prompt,"seed":SEED,"horizon":HORIZON,"status":"complete","initial_mode":"coarse","thresholds":{"tau_low":TAU_LOW,"tau_high":TAU_HIGH},"decisions":records,"segments":seg,"controller_stats":controller.get_statistics(),"timing":{"checkpoint_load_s":total_load,"dit_s":total_dit,"vae_s":vae_s,"signal_s":0.0,"total_s":time.time()-started,"nfe":total_nfe,"peak_vram_gb":torch.cuda.max_memory_allocated()/2**30},"proxy":"adjacent decoded-pixel temporal L1; not perceptual/reference quality"}
                write(item); out.append(item); del video,latent,frames,emb; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                write({"sample_key":key,"split":"held_out","prompt_idx":idx,"prompt":prompt,"status":"failed","error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc()}); log(traceback.format_exc())
        # Aggregate resumable outputs and pair with Phase 3 fixed outputs if available locally as an input dataset later.
        all_items=[]
        for line in RAW.read_text(encoding="utf-8").splitlines() if RAW.exists() else []:
            try:
                x=json.loads(line)
                if x.get("status")=="complete": all_items.append(x)
            except Exception: pass
        report.update({"status":"complete" if len(all_items)==len(HELD_OUT) else ("partial" if all_items else "failed"),"protocol":{"held_out_prompts":HELD_OUT,"seed":SEED,"horizon":HORIZON,"decision_unit":"3 latent frames; 3 decisions after initial coarse chunk","signal":"latest available residual from preceding coarse chunk; held constant through fine spans","recovery_length":0,"lambda_action":0.0,"threshold_source":"Phase 3 calibration split Q25/Q75 residual values","fixed_baselines":"paired Phase 3 fixed-mode outputs; not regenerated here","limitations":["four held-out prompts, one seed","residual is calibrated on coarse trajectories and held stale during fine spans","temporal pixel L1 is motion/content dependent, not perceptual quality","initial coarse segment is shared protocol context","oracle is offline and uses both fixed-mode outcomes"]},"n_complete":len(all_items),"n_failed_or_incomplete":len(HELD_OUT)-len(all_items),"results_file":str(RAW),"mean_timing":{k:float(np.mean([x["timing"][k] for x in all_items])) for k in ["checkpoint_load_s","dit_s","vae_s","total_s","nfe","peak_vram_gb"] if all_items}})
    except Exception as e:
        report.update({"status":"failed","error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc()}); log(traceback.format_exc())
    finally:
        report["end_time"]=datetime.now().isoformat(); SUMMARY.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding="utf-8"); log(f"summary: {SUMMARY}")
if __name__=="__main__": main()
