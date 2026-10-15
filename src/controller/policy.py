"""
Adaptive Granularity Controller
================================

INPUT: Uncertainty score u_i, optional action delta Δa_i
OUTPUT: Mode decision (fine K=1 or coarse K=3)

Components:
1. Potential function: Ψ_i = u_i + λΔa_i (Eq. 6 from proposal)
2. Threshold policy: τ_high, τ_low with hysteresis (Eq. 7)
3. Recovery counter: r_i forces fine mode after spikes (Eq. 8)
"""

import torch
from typing import Optional, Literal
from dataclasses import dataclass


@dataclass
class ControllerConfig:
    """Configuration for adaptive controller."""
    tau_high: float = 0.5  # High threshold (force fine if Ψ_i ≥ τ_high)
    tau_low: float = 0.2   # Low threshold (allow coarse if Ψ_i ≤ τ_low)
    recovery_length: int = 5  # R: steps to stay fine after spike
    lambda_action: float = 0.0  # λ: weight for action control term (0 = text-only)
    initial_mode: Literal['fine', 'coarse'] = 'coarse'  # Start mode


class ThresholdController:
    """
    Two-threshold policy with hysteresis.

    Mode switching rules (Eq. 7 from proposal):
    - If Ψ_i ≥ τ_high → switch to fine (c1-1)
    - If Ψ_i ≤ τ_low → switch to coarse (c3-3)
    - Otherwise → keep current mode (hysteresis band)
    """

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.current_mode = config.initial_mode

    def decide(self, potential: float) -> str:
        """
        Decide mode based on potential.

        Args:
            potential: Ψ_i = u_i + λΔa_i

        Returns:
            'fine' or 'coarse'
        """
        if potential >= self.config.tau_high:
            # High uncertainty → force fine mode
            self.current_mode = 'fine'
        elif potential <= self.config.tau_low:
            # Low uncertainty → allow coarse mode
            self.current_mode = 'coarse'
        # else: stay in current mode (hysteresis)

        return self.current_mode


class RecoveryController:
    """
    Recovery + hysteresis controller (Eq. 8 from proposal).

    After uncertainty spike (Ψ_i ≥ τ_high):
    1. Set recovery counter r_i ← R
    2. Force fine mode for R steps
    3. Decrement counter each step
    4. Return to threshold policy when r_i = 0
    """

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.threshold_controller = ThresholdController(config)
        self.recovery_counter = 0

    def decide(self, potential: float) -> str:
        """
        Decide mode with recovery mechanism.

        Args:
            potential: Ψ_i = u_i + λΔa_i

        Returns:
            'fine' or 'coarse'
        """
        # Check if spike detected
        if potential >= self.config.tau_high:
            # Start recovery period
            self.recovery_counter = self.config.recovery_length
            return 'fine'

        # If in recovery, stay fine and decrement
        if self.recovery_counter > 0:
            self.recovery_counter -= 1
            return 'fine'

        # Otherwise use threshold policy
        return self.threshold_controller.decide(potential)

    def get_recovery_state(self) -> int:
        """Get current recovery counter value."""
        return self.recovery_counter


class PotentialFunction:
    """
    Compute control potential: Ψ_i = u_i + λΔa_i (Eq. 6 from proposal).

    For text-only (no action): λ=0, so Ψ_i = u_i
    """

    def __init__(self, lambda_action: float = 0.0):
        self.lambda_action = lambda_action

    def compute(
        self,
        uncertainty: float,
        action_delta: Optional[float] = None
    ) -> float:
        """
        Compute potential function.

        Args:
            uncertainty: u_i (composite uncertainty score)
            action_delta: Δa_i (action magnitude change, optional)

        Returns:
            Ψ_i = u_i + λΔa_i
        """
        potential = uncertainty

        if action_delta is not None and self.lambda_action > 0:
            potential += self.lambda_action * action_delta

        return potential


class AdaptiveController:
    """
    Full adaptive granularity controller.

    Combines:
    - Potential function (visual + optional action)
    - Threshold policy with hysteresis
    - Recovery counter for stability
    """

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.potential_fn = PotentialFunction(config.lambda_action)
        self.recovery_controller = RecoveryController(config)

        # Statistics tracking
        self.step_count = 0
        self.switch_count = 0
        self.prev_mode = config.initial_mode

        self.history = {
            'potentials': [],
            'modes': [],
            'recovery_counters': [],
            'uncertainties': []
        }

    def step(
        self,
        uncertainty: float,
        action_delta: Optional[float] = None
    ) -> str:
        """
        Execute one controller step.

        Args:
            uncertainty: u_i (composite uncertainty)
            action_delta: Δa_i (optional)

        Returns:
            Mode decision: 'fine' or 'coarse'
        """
        # Compute potential
        potential = self.potential_fn.compute(uncertainty, action_delta)

        # Decide mode
        mode = self.recovery_controller.decide(potential)

        # Track switch
        if mode != self.prev_mode:
            self.switch_count += 1

        # Update history
        self.history['potentials'].append(potential)
        self.history['modes'].append(mode)
        self.history['recovery_counters'].append(
            self.recovery_controller.get_recovery_state()
        )
        self.history['uncertainties'].append(uncertainty)

        self.prev_mode = mode
        self.step_count += 1

        return mode

    def get_statistics(self) -> dict:
        """Get controller statistics."""
        return {
            'total_steps': self.step_count,
            'switch_count': self.switch_count,
            'switch_rate': self.switch_count / max(1, self.step_count),
            'mean_potential': sum(self.history['potentials']) / max(1, len(self.history['potentials'])),
            'mean_uncertainty': sum(self.history['uncertainties']) / max(1, len(self.history['uncertainties'])),
            'fine_fraction': sum(1 for m in self.history['modes'] if m == 'fine') / max(1, len(self.history['modes']))
        }

    def reset(self):
        """Reset controller state."""
        self.step_count = 0
        self.switch_count = 0
        self.prev_mode = self.config.initial_mode
        self.recovery_controller.recovery_counter = 0
        self.history = {
            'potentials': [],
            'modes': [],
            'recovery_counters': [],
            'uncertainties': []
        }
