import os
from huggingface_hub import snapshot_download

snapshot_download(
    "Wan-AI/Wan2.1-T2V-1.3B",
    local_dir="/kaggle/working/wan21_t5",
    cache_dir="/kaggle/temp/hf_cache",
    allow_patterns=["models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/*"],
)
os.system("du -sh /kaggle/working/wan21_t5/*")
os.system("find /kaggle/working/wan21_t5 -type f")
