"""
FlashAttention-2 Fallback for T4 (Turing)
=========================================

T4 has compute capability 7.5 (Turing), but FlashAttention-2 requires 8.0+ (Ampere).
This module detects FA2 availability and falls back to PyTorch SDPA.

INPUT: CUDA device info
OUTPUT: Attention backend selection
"""

import torch
from typing import Literal


def check_flash_attention() -> bool:
    """
    Check if FlashAttention-2 is available and compatible.

    Returns:
        True if FA2 can be used, False otherwise
    """
    if not torch.cuda.is_available():
        return False

    # Check compute capability
    compute_cap = torch.cuda.get_device_capability()
    compute_cap_num = compute_cap[0] + compute_cap[1] / 10

    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"Compute capability: {compute_cap[0]}.{compute_cap[1]}")

    # FlashAttention-2 requires Ampere (8.0) or newer
    if compute_cap_num < 8.0:
        print("WARNING: FlashAttention-2 requires compute capability ≥8.0 (Ampere+)")
        print(f"         Current device is {compute_cap_num} (Turing/Volta)")
        return False

    # Try importing
    try:
        import flash_attn
        print(f"FlashAttention-2 available: version {flash_attn.__version__}")
        return True
    except ImportError:
        print("FlashAttention-2 not installed")
        return False


def get_attention_backend() -> Literal['flash_attn', 'sdpa', 'math']:
    """
    Select best available attention backend.

    Returns:
        'flash_attn': Use FlashAttention-2 (fastest, Ampere+ only)
        'sdpa': Use PyTorch Scaled Dot Product Attention (good fallback)
        'math': Use naive attention (slowest, always works)
    """
    if check_flash_attention():
        return 'flash_attn'

    # Check if SDPA is available (PyTorch 2.0+)
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        print("Using PyTorch SDPA (scaled_dot_product_attention)")
        return 'sdpa'

    print("WARNING: Falling back to naive attention (slow)")
    return 'math'


def patch_rcm_attention(model, backend: str = 'auto'):
    """
    Patch Causal-rCM model to use specified attention backend.

    Args:
        model: Loaded Causal-rCM model
        backend: 'auto', 'flash_attn', 'sdpa', or 'math'
    """
    if backend == 'auto':
        backend = get_attention_backend()

    # This is a placeholder - actual patching depends on rcm codebase structure
    # In Phase 0, we'll inspect the model and implement the actual patch

    if backend == 'sdpa':
        print("Patching model to use PyTorch SDPA...")
        # Replace attention modules with SDPA-based ones
        # model.transformer.attn = SDPAAttention(...)
        pass
    elif backend == 'math':
        print("WARNING: Using naive attention - expect slower inference")
        pass

    return model


class SDPAAttention(torch.nn.Module):
    """
    SDPA-based attention module (fallback for T4).
    This replaces FlashAttention-2 when unavailable.
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.qkv = torch.nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x, causal_mask=None):
        B, L, D = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Use PyTorch 2.0+ SDPA
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(causal_mask is None)  # Assume causal if no mask provided
        )

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(attn_out)

        return out
