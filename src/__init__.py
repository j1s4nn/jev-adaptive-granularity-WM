"""
Uncertainty-Guided Adaptive Temporal Granularity
Research project for interactive video world models

Packages:
- causal_rcm: Wrapper for NVlabs/rcm inference
- uncertainty: Pre-generation uncertainty scores (EMA, residual, warp)
- controller: Threshold policy + recovery mechanism
- baselines: Motion/complexity-driven controllers
- evaluation: Metrics (FVD, LPIPS, AUROC, switch cost)
- utils: Logging, tree generation, figure styling
"""

__version__ = "1.0.0"
__author__ = "Hossen Md Jisan"
