import os
from huggingface_hub import HfApi, snapshot_download

api = HfApi()
for repo in ["worstcoder/rcm-Wan", "worstcoder/Wan"]:
    print("=== REPO:", repo, flush=True)
    for f in api.list_repo_tree(repo, recursive=True):
        print(getattr(f, "path", f), getattr(f, "size", None), flush=True)

out = "/kaggle/working/rcm_ckpts"
snapshot_download(
    "worstcoder/rcm-Wan",
    local_dir=out,
    cache_dir="/kaggle/temp/hf_cache",
    allow_patterns=["*SF-DMD*c1-1_step2.pt", "*SF-DMD*c1-1_step4.pt",
                    "*SF-DMD*c3-3*", "*chunkwise*"],
)
snapshot_download(
    "worstcoder/Wan",
    local_dir=out,
    cache_dir="/kaggle/temp/hf_cache",
    allow_patterns=["umT5_wan_negative_emb.pt"],
)
os.system("find " + out + " -type f -exec du -h {} +")
