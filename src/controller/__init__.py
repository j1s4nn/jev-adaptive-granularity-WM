"""
Controller Package
==================

INPUT: Uncertainty scores, optional action deltas
OUTPUT: Mode decisions (fine K=1 or coarse K=3)

Modules:
- policy.py: Threshold controller + recovery mechanism
"""

from .policy import (
    ControllerConfig,
    ThresholdController,
    RecoveryController,
    PotentialFunction,
    AdaptiveController
)

__all__ = [
    'ControllerConfig',
    'ThresholdController',
    'RecoveryController',
    'PotentialFunction',
    'AdaptiveController'
]
