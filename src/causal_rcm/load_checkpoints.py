"""
Load Causal-rCM checkpoints from Kaggle datasets
================================================

INPUT:
- /kaggle/input/causal-rcm-ckpts/
- /kaggle/input/wan21-t5/
- /kaggle/input/wan21-dit-vae/

OUTPUT:
- Loaded c1-1 (frame-wise K=1) checkpoint
- Loaded c3-3 (chunk-wise K=3) checkpoint
- VAE decoder
- T5 text encoder
- Checkpoint metadata (sha256, size, config)
"""

import os
import hashlib
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional


def safe_torch_load(path, map_location, verbose=True):
    """torch.load with weights_only fallback (torch>=2.6 defaults changed)."""
    try:
        return torch.load(str(path), map_location=map_location)
    except Exception as e:
        if verbose:
            print(f"    [warn] torch.load failed ({type(e).__name__}), retrying weights_only=False")
        return torch.load(str(path), map_location=map_location, weights_only=False)


def find_file_recursive(root: Path, filename: str) -> Optional[Path]:
    """Find a file by exact name anywhere under root (Kaggle nests files in subfolders)."""
    if root is None or not root.exists():
        return None
    for f in root.rglob(filename):
        if f.is_file():
            return f
    return None


def find_kaggle_input_paths() -> Dict[str, Path]:
    """
    Find actual mount paths in /kaggle/input (names may vary).
    NEVER hardcode paths - Kaggle renames them.
    """
    kaggle_input = Path("/kaggle/input")

    paths = {
        'rcm_ckpts': None,
        't5': None,
        'vae': None
    }

    if not kaggle_input.exists():
        return paths

    # Kaggle mounts private datasets beneath nested paths such as
    # /kaggle/input/datasets/<owner>/<dataset>/..., so immediate-child
    # inspection is insufficient. Identify dataset roots from known files and
    # retain the containing directory for recursive checkpoint lookup.
    marker_files = {
        'rcm_ckpts': 'Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step2.pt',
        't5': 'models_t5_umt5-xxl-enc-bf16.pth',
        'vae': 'Wan2.1_VAE.pth',
    }
    for kind, marker in marker_files.items():
        matches = list(kaggle_input.rglob(marker))
        if matches:
            paths[kind] = matches[0].parent

    # Fall back to name-based discovery for partial datasets and local tests.
    if paths['rcm_ckpts'] is None or paths['t5'] is None or paths['vae'] is None:
        for item in kaggle_input.rglob('*'):
            if not item.is_dir():
                continue
            name = item.name.lower()
            if paths['rcm_ckpts'] is None and ('rcm' in name or 'ckpt' in name):
                paths['rcm_ckpts'] = item
            elif paths['t5'] is None and 't5' in name:
                paths['t5'] = item
            elif paths['vae'] is None and ('vae' in name or 'dit' in name):
                paths['vae'] = item

    return paths


def compute_sha256(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_causal_rcm_checkpoints(
    device: str = 'cuda',
    checkpoint_type: str = 'both',  # 'c1-1', 'c3-3', or 'both'
    step_count: int = 2,  # 1, 2, or 4
    verify_hashes: bool = False
) -> Dict:
    """
    Load Causal-rCM checkpoints from Kaggle datasets.

    Args:
        device: 'cuda' or 'cpu'
        checkpoint_type: Which checkpoints to load
        step_count: Denoising steps (1, 2, or 4)
        verify_hashes: Compute SHA256 for integrity check

    Returns:
        Dict with:
            - 'c1_1': Frame-wise checkpoint (if requested)
            - 'c3_3': Chunk-wise checkpoint (if requested)
            - 'vae': VAE decoder
            - 't5': T5 text encoder
            - 'metadata': Checkpoint info
    """
    paths = find_kaggle_input_paths()

    if paths['rcm_ckpts'] is None:
        raise FileNotFoundError(
            "Causal-rCM checkpoints not found in /kaggle/input. "
            "Ensure dataset is mounted in kernel-metadata.json"
        )

    # Map step count to checkpoint filenames
    c1_1_files = {
        1: 'Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step1.pt',  # May not exist
        2: 'Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step2.pt',
        4: 'Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c1-1_step4.pt'
    }

    c3_3_files = {
        2: 'Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step2.pt',
        4: 'Causal_rCM_Wan2.1_T2V_1.3B_480p_TF-dCM-init_SF-DMD_c3-3_step4.pt'
    }

    result = {'metadata': {}}

    print(f"Loading Causal-rCM checkpoints (step={step_count})...")
    print(f"RCM path: {paths['rcm_ckpts']}")

    # Load frame-wise checkpoint
    if checkpoint_type in ['c1-1', 'both']:
        c1_1_file = c1_1_files.get(step_count)
        if c1_1_file:
            c1_1_path = find_file_recursive(paths['rcm_ckpts'], c1_1_file)
            if c1_1_path is not None:
                print(f"Loading c1-1 (frame-wise, K=1, S={step_count}) from {c1_1_path.name}...")
                result['c1_1'] = safe_torch_load(c1_1_path, map_location=device)
                result['metadata']['c1_1'] = {
                    'path': str(c1_1_path),
                    'size_mb': c1_1_path.stat().st_size / (1024**2),
                    'sha256': compute_sha256(c1_1_path) if verify_hashes else None
                }
                print(f"  Loaded: {result['metadata']['c1_1']['size_mb']:.1f} MB")
            else:
                print(f"  WARNING: {c1_1_file} not found (searched recursively), skipping c1-1")

    # Load chunk-wise checkpoint
    if checkpoint_type in ['c3-3', 'both']:
        c3_3_file = c3_3_files.get(step_count)
        if c3_3_file:
            c3_3_path = find_file_recursive(paths['rcm_ckpts'], c3_3_file)
            if c3_3_path is not None:
                print(f"Loading c3-3 (chunk-wise, K=3, S={step_count}) from {c3_3_path.name}...")
                result['c3_3'] = safe_torch_load(c3_3_path, map_location=device)
                result['metadata']['c3_3'] = {
                    'path': str(c3_3_path),
                    'size_mb': c3_3_path.stat().st_size / (1024**2),
                    'sha256': compute_sha256(c3_3_path) if verify_hashes else None
                }
                print(f"  Loaded: {result['metadata']['c3_3']['size_mb']:.1f} MB")
            else:
                print(f"  WARNING: {c3_3_file} not found (searched recursively), skipping c3-3")

    # Load VAE
    if paths['vae']:
        vae_path = find_file_recursive(paths['vae'], 'Wan2.1_VAE.pth')
        if vae_path is not None:
            print(f"Loading VAE decoder...")
            result['vae'] = safe_torch_load(vae_path, map_location=device)
            result['metadata']['vae'] = {
                'path': str(vae_path),
                'size_mb': vae_path.stat().st_size / (1024**2)
            }
            print(f"  Loaded: {result['metadata']['vae']['size_mb']:.1f} MB")
        else:
            print(f"  WARNING: Wan2.1_VAE.pth not found")

    # Load T5 text encoder
    if paths['t5']:
        t5_path = find_file_recursive(paths['t5'], 'models_t5_umt5-xxl-enc-bf16.pth')
        if t5_path is not None:
            print(f"Loading T5 text encoder...")
            result['t5'] = safe_torch_load(t5_path, map_location=device)
            result['metadata']['t5'] = {
                'path': str(t5_path),
                'size_mb': t5_path.stat().st_size / (1024**2)
            }
            print(f"  Loaded: {result['metadata']['t5']['size_mb']:.1f} MB")
        else:
            print(f"  WARNING: models_t5_umt5-xxl-enc-bf16.pth not found")

    return result


def list_available_checkpoints() -> Dict[str, list]:
    """List all checkpoint files in Kaggle datasets."""
    paths = find_kaggle_input_paths()

    available = {
        'rcm_ckpts': [],
        't5': [],
        'vae': []
    }

    if paths['rcm_ckpts'] and paths['rcm_ckpts'].exists():
        available['rcm_ckpts'] = [
            f.name for f in paths['rcm_ckpts'].rglob('*.pt')
            if f.is_file() and f.stat().st_size > 1024**2  # > 1MB
        ]

    if paths['t5'] and paths['t5'].exists():
        available['t5'] = [f.name for f in paths['t5'].rglob('*.pth') if f.is_file()]

    if paths['vae'] and paths['vae'].exists():
        available['vae'] = [f.name for f in paths['vae'].rglob('*.pth') if f.is_file()]

    return available
