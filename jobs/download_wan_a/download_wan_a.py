import os
from huggingface_hub import snapshot_download, list_repo_files

print(list_repo_files("Wan-AI/Wan2.1-T2V-1.3B"))
snapshot_download(
    "Wan-AI/Wan2.1-T2V-1.3B",
    local_dir="/kaggle/working/wan21_dit_vae",
    cache_dir="/kaggle/temp/hf_cache",
    allow_patterns=["diffusion_pytorch_model.safetensors",
                    "Wan2.1_VAE.pth", "config.json"],
)
os.system("du -sh /kaggle/working/wan21_dit_vae/*")
