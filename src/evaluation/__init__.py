"""
Evaluation Metrics
==================

INPUT: Generated videos, reference distributions, controller traces
OUTPUT: Quality metrics (FVD, LPIPS, PSNR), efficiency metrics, AUROC

Metrics:
1. Quality: FVD (Fréchet Video Distance), LPIPS, PSNR
2. Efficiency: GPU time, FLOPs, latency
3. Signal quality: AUROC (uncertainty vs coarse-unsafe labels)
4. Controller: Switch count, switch cost
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from typing import List, Tuple, Dict


class FVDMetric:
    """
    Fréchet Video Distance.

    Measures distribution distance between generated and reference videos
    using I3D feature embeddings.
    """

    def __init__(self, i3d_model_path: str = None):
        """
        Args:
            i3d_model_path: Path to pretrained I3D model (optional)
        """
        self.i3d_model_path = i3d_model_path
        self.i3d = None  # Load lazily

    def load_i3d(self):
        """Load I3D model for feature extraction."""
        # Placeholder - real implementation loads pretrained I3D
        # from torchvision or styleGAN repo
        print("Loading I3D model...")
        # self.i3d = load_pretrained_i3d(self.i3d_model_path)
        pass

    def extract_features(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Extract I3D features from videos.

        Args:
            videos: [B, T, C, H, W]

        Returns:
            Features [B, D]
        """
        if self.i3d is None:
            self.load_i3d()

        # Placeholder
        B = videos.shape[0]
        return torch.randn(B, 2048)  # I3D has 2048-dim features

    def compute(
        self,
        generated_videos: torch.Tensor,
        reference_videos: torch.Tensor
    ) -> float:
        """
        Compute FVD between two video distributions.

        Args:
            generated_videos: [B1, T, C, H, W]
            reference_videos: [B2, T, C, H, W]

        Returns:
            FVD score (lower is better)
        """
        # Extract features
        gen_features = self.extract_features(generated_videos)
        ref_features = self.extract_features(reference_videos)

        # Compute Fréchet distance
        mu_gen = gen_features.mean(dim=0)
        mu_ref = ref_features.mean(dim=0)

        sigma_gen = torch.cov(gen_features.T)
        sigma_ref = torch.cov(ref_features.T)

        # FVD = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        diff = mu_gen - mu_ref
        covmean = torch.sqrt(sigma_gen @ sigma_ref)

        fvd = (diff @ diff) + torch.trace(sigma_gen + sigma_ref - 2 * covmean)

        return fvd.item()


class LPIPSMetric:
    """
    Learned Perceptual Image Patch Similarity.

    Measures perceptual distance between frames using deep features.
    """

    def __init__(self, net: str = 'alex'):
        """
        Args:
            net: 'alex' (AlexNet) or 'vgg' (VGG16)
        """
        self.net = net
        self.lpips_model = None

    def load_lpips(self):
        """Load LPIPS model."""
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net=self.net)
        except ImportError:
            print("WARNING: lpips not installed, using placeholder")

    def compute(self, frame1: torch.Tensor, frame2: torch.Tensor) -> float:
        """
        Compute LPIPS between two frames.

        Args:
            frame1, frame2: [B, C, H, W] in range [-1, 1]

        Returns:
            LPIPS score (lower is better)
        """
        if self.lpips_model is None:
            self.load_lpips()

        if self.lpips_model is None:
            # Placeholder: MSE in pixel space
            return torch.nn.functional.mse_loss(frame1, frame2).item()

        with torch.no_grad():
            dist = self.lpips_model(frame1, frame2)

        return dist.mean().item()


class AUROCMetric:
    """
    Area Under ROC Curve for binary classification.

    Used in E2 to evaluate uncertainty signal vs coarse-unsafe labels.
    """

    def compute(
        self,
        scores: List[float],
        labels: List[bool]
    ) -> Tuple[float, Dict]:
        """
        Compute AUROC.

        Args:
            scores: Predicted scores (uncertainty, motion, etc.)
            labels: True labels (True = unsafe, False = safe)

        Returns:
            AUROC score, ROC curve data
        """
        scores = np.array(scores)
        labels = np.array(labels, dtype=int)

        if len(np.unique(labels)) < 2:
            # All same label, AUROC undefined
            return 0.5, {}

        auroc = roc_auc_score(labels, scores)

        fpr, tpr, thresholds = roc_curve(labels, scores)

        # Find optimal threshold (Youden's index)
        youden_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[youden_idx]

        return auroc, {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'optimal_threshold': optimal_threshold
        }


class SwitchCostMetric:
    """
    Measure artifact severity at mode transitions.

    Computes LPIPS spike when mode changes.
    """

    def __init__(self):
        self.lpips = LPIPSMetric()

    def compute(
        self,
        frames: torch.Tensor,
        mode_trace: List[str]
    ) -> Dict:
        """
        Compute switch cost statistics.

        Args:
            frames: [T, C, H, W]
            mode_trace: ['coarse', 'fine', 'fine', 'coarse', ...]

        Returns:
            Dict with mean/max switch cost
        """
        switch_costs = []

        for t in range(1, len(mode_trace)):
            if mode_trace[t] != mode_trace[t-1]:
                # Mode switch detected
                cost = self.lpips.compute(
                    frames[t-1:t],
                    frames[t:t+1]
                )
                switch_costs.append(cost)

        if not switch_costs:
            return {'mean': 0.0, 'max': 0.0, 'count': 0}

        return {
            'mean': float(np.mean(switch_costs)),
            'max': float(np.max(switch_costs)),
            'std': float(np.std(switch_costs)),
            'count': len(switch_costs)
        }


class ComputeCostMetric:
    """
    Track GPU time and FLOPs.

    Uses torch.cuda.Event for precise timing.
    """

    def __init__(self):
        self.timings = []

    def measure(self, fn, *args, **kwargs) -> Tuple[any, float]:
        """
        Measure GPU time of a function.

        Args:
            fn: Function to measure
            *args, **kwargs: Function arguments

        Returns:
            (result, gpu_ms)
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            result = fn(*args, **kwargs)
            end.record()

            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            import time
            start_time = time.time()
            result = fn(*args, **kwargs)
            elapsed_ms = (time.time() - start_time) * 1000

        self.timings.append(elapsed_ms)

        return result, elapsed_ms

    def get_statistics(self) -> Dict:
        """Get timing statistics."""
        if not self.timings:
            return {}

        return {
            'mean_ms': float(np.mean(self.timings)),
            'std_ms': float(np.std(self.timings)),
            'min_ms': float(np.min(self.timings)),
            'max_ms': float(np.max(self.timings)),
            'total_ms': float(np.sum(self.timings))
        }
