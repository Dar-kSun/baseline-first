"""Honest baselines. See `base.Baseline` for the interface."""

from baseline_first.baselines.base import Baseline
from baseline_first.baselines.global_mean import GlobalMean
from baseline_first.baselines.hvg_ridge import HVGRidge

__all__ = ["Baseline", "GlobalMean", "HVGRidge"]
