"""Objective-only, immutable scaling for ACP S1--S3.

Physical accounting belongs in alternative_metrics, never in reward shaping.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Mapping

ACP_TO_REWARD_MODE = {
    "ACP S1": "economy_primary",
    "ACP S2": "energy_primary",
    "ACP S3": "resource_primary",
}
MODE_TO_METRIC = {
    "economy_primary": ("net_cost", -1.0),
    "energy_primary": ("net_energy", -1.0),
    "resource_primary": ("resource_output", 1.0),
}
REWARD_VERSION = "objective-only-tanh-v2"


class MetricUnavailableError(ValueError):
    """An objective is missing a verified physical accounting contract."""


@dataclass(frozen=True)
class RewardSpec:
    reward_mode: str
    scale: float
    metric_version: str
    version: str = REWARD_VERSION

    def __post_init__(self):
        if self.reward_mode not in MODE_TO_METRIC:
            raise ValueError(f"Unknown alternative reward mode: {self.reward_mode}")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Reward scale must be finite, strictly positive and frozen")
        if not self.metric_version:
            raise ValueError("A versioned metric accounting contract is required")
        if self.version != REWARD_VERSION:
            raise ValueError(f"Unsupported shaping version: {self.version}")


@dataclass(frozen=True)
class RewardBreakdown:
    metric: str
    metric_value: float
    reference_value: float
    scale: float
    normalized_delta: float
    raw_reward: float
    shaped_reward: float

    def to_dict(self):
        return asdict(self)


def alternative_reward(spec: RewardSpec, metrics: Mapping, reference: Mapping) -> RewardBreakdown:
    key, direction = MODE_TO_METRIC[spec.reward_mode]
    current, baseline = metrics.get(key), reference.get(key)
    if current is None or baseline is None:
        raise MetricUnavailableError(f"{key} has not been verified for this environment")
    current, baseline = float(current), float(baseline)
    if not math.isfinite(current) or not math.isfinite(baseline):
        raise FloatingPointError(f"Nonfinite raw {key}: value={current}, reference={baseline}")
    delta = (current - baseline) / spec.scale
    if not math.isfinite(delta):
        raise FloatingPointError("Nonfinite reward normalization")
    raw = direction * delta
    return RewardBreakdown(key, current, baseline, spec.scale, delta, raw, math.tanh(raw))
