"""
Baseline Controllers
====================

Motion-driven and complexity-driven adaptive controllers for comparison (E5).

These are ablations/baselines that use different signals than uncertainty:
1. Motion: Switch based on optical flow magnitude
2. Complexity: Switch based on scene complexity (edge density, entropy)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Literal


class MotionController:
    """
    Motion-driven adaptive controller (LongScape/MotionCache-style).

    Uses optical flow magnitude as switching signal:
    - High motion → fine mode (needs temporal precision)
    - Low motion → coarse mode (stable scene)
    """

    def __init__(
        self,
        tau_high: float = 2.0,
        tau_low: float = 0.5,
        initial_mode: Literal['fine', 'coarse'] = 'coarse'
    ):
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.current_mode = initial_mode

    def compute_motion_score(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor
    ) -> float:
        """
        Compute motion score from optical flow magnitude.

        Args:
            frame1, frame2: [B, C, H, W]

        Returns:
            Mean flow magnitude
        """
        # Placeholder - real implementation uses RAFT or Farneback
        # For now, use frame difference as proxy
        diff = torch.abs(frame2 - frame1)
        motion_score = diff.mean().item()

        return motion_score

    def decide(
        self,
        frame_current: torch.Tensor,
        frame_previous: torch.Tensor
    ) -> str:
        """
        Decide mode based on motion.

        Args:
            frame_current, frame_previous: [B, C, H, W]

        Returns:
            'fine' or 'coarse'
        """
        motion = self.compute_motion_score(frame_previous, frame_current)

        if motion >= self.tau_high:
            self.current_mode = 'fine'
        elif motion <= self.tau_low:
            self.current_mode = 'coarse'

        return self.current_mode


class ComplexityController:
    """
    Complexity-driven adaptive controller (EVATok/AdapTok-style).

    Uses scene complexity as switching signal:
    - High complexity (edges, textures) → fine mode
    - Low complexity (smooth regions) → coarse mode
    """

    def __init__(
        self,
        tau_high: float = 0.3,
        tau_low: float = 0.1,
        initial_mode: Literal['fine', 'coarse'] = 'coarse'
    ):
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.current_mode = initial_mode

    def compute_complexity_score(self, frame: torch.Tensor) -> float:
        """
        Compute scene complexity score.

        Uses:
        - Edge density (Sobel filter response)
        - Spatial entropy

        Args:
            frame: [B, C, H, W]

        Returns:
            Complexity score
        """
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=frame.dtype, device=frame.device)
        sobel_y = sobel_x.t()

        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(frame.shape[1], 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(frame.shape[1], 1, 1, 1)

        edges_x = F.conv2d(frame, sobel_x, padding=1, groups=frame.shape[1])
        edges_y = F.conv2d(frame, sobel_y, padding=1, groups=frame.shape[1])

        edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
        edge_density = edge_magnitude.mean().item()

        # Spatial entropy (simplified)
        # Divide frame into patches, compute variance
        B, C, H, W = frame.shape
        patch_size = 16
        patches = frame.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patch_var = patches.var(dim=(-2, -1)).mean().item()

        # Combine
        complexity = edge_density + 0.5 * patch_var

        return complexity

    def decide(self, frame: torch.Tensor) -> str:
        """
        Decide mode based on scene complexity.

        Args:
            frame: [B, C, H, W]

        Returns:
            'fine' or 'coarse'
        """
        complexity = self.compute_complexity_score(frame)

        if complexity >= self.tau_high:
            self.current_mode = 'fine'
        elif complexity <= self.tau_low:
            self.current_mode = 'coarse'

        return self.current_mode


class RandomController:
    """
    Random switching baseline (sanity check).

    Switches randomly with probability p at each step.
    Expected switch count = p * T
    """

    def __init__(self, switch_prob: float = 0.5):
        self.switch_prob = switch_prob
        self.current_mode = 'coarse'

    def decide(self) -> str:
        """Random mode decision."""
        if np.random.rand() < self.switch_prob:
            self.current_mode = 'fine' if self.current_mode == 'coarse' else 'coarse'

        return self.current_mode
