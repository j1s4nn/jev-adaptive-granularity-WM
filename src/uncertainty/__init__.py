"""
Uncertainty Package
===================

INPUT: Model predictions, generated frames
OUTPUT: Uncertainty scores for adaptive granularity controller

Modules:
- scores.py: EMA, residual, warp uncertainty components
- calibrate.py: AUROC evaluation against coarse-unsafe labels (E2)
"""

from .scores import (
    EMAScore,
    ResidualScore,
    WarpScore,
    CompositeUncertainty,
    UncertaintyMLPAggregator
)

__all__ = [
    'EMAScore',
    'ResidualScore',
    'WarpScore',
    'CompositeUncertainty',
    'UncertaintyMLPAggregator'
]
