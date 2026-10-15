"""
Causal-rCM Wrapper Package
===========================

INPUT: Kaggle datasets (causal-rcm-ckpts, wan21-t5, wan21-dit-vae)
OUTPUT: Inference interface for c1-1 (frame-wise) and c3-3 (chunk-wise) checkpoints

Modules:
- load_checkpoints.py: Mount Kaggle datasets, load matched pair
- inference.py: Single-step generation, expose internals (EMA, latents, KV)
- fallback.py: PyTorch SDPA fallback if FlashAttention-2 unavailable (T4 Turing)
"""

from .load_checkpoints import load_causal_rcm_checkpoints
from .inference import CausalRCMInference
from .fallback import check_flash_attention, get_attention_backend

__all__ = [
    'load_causal_rcm_checkpoints',
    'CausalRCMInference',
    'check_flash_attention',
    'get_attention_backend'
]
