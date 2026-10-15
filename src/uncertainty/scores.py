"""
Uncertainty Score Computation
==============================

INPUT: Generated frames, model internals (EMA predictions, past latents)
OUTPUT: Uncertainty scores (u_i^EMA, u_i^res, u_i^warp, u_i composite)

Three components:
1. u_i^EMA: ||v_θ(x_t, x^<i) - v_θ^-(x_t, x^<i)|| (student vs EMA divergence)
2. u_i^res: ||pred_i - pred_{i-1}|| (self-consistency)
3. u_i^warp: ||x_{i-1} - W(x_{i-2}, flow)|| (optical flow prediction error)

Composite: u_i = g_φ(u_EMA, u_res, u_warp) where g_φ can be:
- Mean: (u_EMA + u_res + u_warp) / 3
- Max: max(u_EMA, u_res, u_warp)
- Learned MLP (optional, requires calibration)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class EMAScore:
    """
    u_i^EMA: Student vs EMA model divergence

    Measures prediction disagreement between online student (v_θ)
    and its exponential moving average (v_θ^-).
    """

    def __init__(self, norm: str = 'l2'):
        """
        Args:
            norm: 'l2' or 'l1'
        """
        self.norm = norm

    def compute(
        self,
        student_pred: torch.Tensor,
        ema_pred: torch.Tensor
    ) -> float:
        """
        Compute EMA divergence score.

        Args:
            student_pred: Student model prediction [B, C, H, W]
            ema_pred: EMA model prediction [B, C, H, W]

        Returns:
            Scalar uncertainty score
        """
        diff = student_pred - ema_pred

        if self.norm == 'l2':
            score = torch.norm(diff, p=2).item()
        else:
            score = torch.norm(diff, p=1).item()

        # Normalize by number of elements
        score = score / diff.numel()

        return score


class ResidualScore:
    """
    u_i^res: Temporal self-consistency

    Measures how much the prediction changes between consecutive steps.
    Low residual → stable generation, high residual → model uncertainty.
    """

    def __init__(self, norm: str = 'l2'):
        self.norm = norm

    def compute(
        self,
        pred_current: torch.Tensor,
        pred_previous: torch.Tensor
    ) -> float:
        """
        Compute residual score.

        Args:
            pred_current: Current step prediction [B, C, H, W]
            pred_previous: Previous step prediction [B, C, H, W]

        Returns:
            Scalar uncertainty score
        """
        if pred_previous is None:
            # First step has no previous, return 0
            return 0.0

        diff = pred_current - pred_previous

        if self.norm == 'l2':
            score = torch.norm(diff, p=2).item()
        else:
            score = torch.norm(diff, p=1).item()

        score = score / diff.numel()

        return score


class WarpScore:
    """
    u_i^warp: Optical flow prediction error

    Warp previous frame using flow, compare with current frame.
    Large mismatch → complex motion or model uncertainty.
    """

    def __init__(self, flow_backend: str = 'raft'):
        """
        Args:
            flow_backend: 'raft' (accurate but slow) or 'farneback' (fast)
        """
        self.flow_backend = flow_backend

    def compute_flow(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute optical flow between two frames.

        Args:
            frame1: [B, C, H, W]
            frame2: [B, C, H, W]

        Returns:
            Flow field [B, 2, H, W]
        """
        # Placeholder - real implementation would use:
        # - torchvision.models.optical_flow.raft_large (accurate)
        # - cv2.calcOpticalFlowFarneback (fast)

        B, C, H, W = frame1.shape
        # Dummy flow for now
        flow = torch.zeros(B, 2, H, W, device=frame1.device)

        return flow

    def warp_frame(
        self,
        frame: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        """
        Warp frame according to flow field.

        Args:
            frame: [B, C, H, W]
            flow: [B, 2, H, W]

        Returns:
            Warped frame [B, C, H, W]
        """
        B, C, H, W = frame.shape

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=frame.device),
            torch.arange(W, device=frame.device),
            indexing='ij'
        )

        grid = torch.stack([grid_x, grid_y], dim=0).float()  # [2, H, W]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]

        # Apply flow
        grid = grid + flow

        # Normalize to [-1, 1]
        grid[:, 0] = 2.0 * grid[:, 0] / (W - 1) - 1.0
        grid[:, 1] = 2.0 * grid[:, 1] / (H - 1) - 1.0

        # Permute to [B, H, W, 2] for grid_sample
        grid = grid.permute(0, 2, 3, 1)

        # Warp
        warped = F.grid_sample(
            frame,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return warped

    def compute(
        self,
        frame_current: torch.Tensor,
        frame_previous: torch.Tensor,
        frame_prev_prev: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute warp prediction error.

        Args:
            frame_current: Current frame [B, C, H, W]
            frame_previous: Previous frame [B, C, H, W]
            frame_prev_prev: Frame before previous [B, C, H, W] (optional)

        Returns:
            Scalar uncertainty score
        """
        if frame_prev_prev is None:
            # Need at least 2 past frames for flow
            return 0.0

        # Compute flow from t-2 to t-1
        flow = self.compute_flow(frame_prev_prev, frame_previous)

        # Warp t-1 using flow (predict where pixels will go)
        warped = self.warp_frame(frame_previous, flow)

        # Compare warped prediction with actual current frame
        diff = frame_current - warped
        score = torch.norm(diff, p=2).item() / diff.numel()

        return score


class CompositeUncertainty:
    """
    Composite uncertainty: u_i = g_φ(u_EMA, u_res, u_warp)

    Aggregation strategies:
    - 'mean': Simple average
    - 'max': Worst-case (most conservative)
    - 'weighted': Learned weights
    - 'mlp': Small MLP (requires training on E2 labels)
    """

    def __init__(
        self,
        ema_score: EMAScore,
        residual_score: ResidualScore,
        warp_score: WarpScore,
        aggregation: str = 'mean',
        weights: Optional[Dict[str, float]] = None
    ):
        self.ema_score = ema_score
        self.residual_score = residual_score
        self.warp_score = warp_score
        self.aggregation = aggregation
        self.weights = weights or {'ema': 1.0, 'res': 1.0, 'warp': 1.0}

    def compute(
        self,
        student_pred: torch.Tensor,
        ema_pred: torch.Tensor,
        pred_current: torch.Tensor,
        pred_previous: Optional[torch.Tensor],
        frame_current: torch.Tensor,
        frame_previous: torch.Tensor,
        frame_prev_prev: Optional[torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute all uncertainty components and aggregate.

        Returns:
            Dict with individual scores and composite
        """
        u_ema = self.ema_score.compute(student_pred, ema_pred)
        u_res = self.residual_score.compute(pred_current, pred_previous)
        u_warp = self.warp_score.compute(frame_current, frame_previous, frame_prev_prev)

        # Aggregate
        if self.aggregation == 'mean':
            u_composite = (u_ema + u_res + u_warp) / 3.0
        elif self.aggregation == 'max':
            u_composite = max(u_ema, u_res, u_warp)
        elif self.aggregation == 'weighted':
            u_composite = (
                self.weights['ema'] * u_ema +
                self.weights['res'] * u_res +
                self.weights['warp'] * u_warp
            ) / sum(self.weights.values())
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return {
            'u_ema': u_ema,
            'u_res': u_res,
            'u_warp': u_warp,
            'u_composite': u_composite
        }


class UncertaintyMLPAggregator(nn.Module):
    """
    Learned MLP for uncertainty aggregation.

    Trained in E2 to predict coarse-unsafe labels from (u_EMA, u_res, u_warp).
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, u_ema: float, u_res: float, u_warp: float) -> float:
        """
        Compute composite uncertainty.

        Args:
            u_ema, u_res, u_warp: Individual scores

        Returns:
            Composite score in [0, 1]
        """
        x = torch.tensor([[u_ema, u_res, u_warp]], dtype=torch.float32)
        return self.net(x).item()
