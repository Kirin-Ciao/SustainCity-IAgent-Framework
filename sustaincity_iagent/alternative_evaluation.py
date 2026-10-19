"""Paired deterministic post-training checks without changing a saved policy.

This evaluates held-out samples from the original training input distribution.
It does not claim convergence, impose collection floors, or predict HA/MA/LA.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math

import numpy as np
from stable_baselines3 import PPO

from .alternative_factory import make_alternative_env, sha256_file
from .alternative_metrics import ROUTE_GROUPS
from .alternative_rewards import ACP_TO_REWARD_MODE, MODE_TO_METRIC

FOUR_METRICS = ("ghg", "net_cost", "net_energy", "resource_output")


def _model_samples(model_path, config, seeds):
    """Load once and use one environment for every predetermined held-out seed."""
    env = make_alternative_env(config["archetype"], config["acp"],
                               config["scale"], config.get("data_file"))
    try:
        model = PPO.load(str(model_path), env=env, device="cpu")
        samples = []
        total_steps = saturated_steps = 0
        maximum_residual = 0.
        numeric_valid = True
        zero_collection_seeds, zero_sorting_seeds = [], []
        for seed in seeds:
            try:
                obs, reset_info = env.reset(seed=seed)
                reference = deepcopy(reset_info["reference"])
                rewards = []
                minimum_collection = minimum_sorting = 1.
                sample_residual = 0.
                for step in range(20):
                    action, _ = model.predict(obs.copy(), deterministic=True)
                    action = np.asarray(action, dtype=np.float32)
                    if action.shape != (29,) or not np.isfinite(action).all():
                        raise FloatingPointError("Policy action is not a finite 29-vector")
                    obs, reward, terminated, truncated, info = env.step(action)
                    ip = np.asarray(info["ip"], dtype=float)
                    if ip.shape != (29,) or not np.isfinite(ip).all() or (ip < 0).any() or (ip > 1).any():
                        raise FloatingPointError("Invalid policy IP bounds")
                    for start, end, waste, _ in ROUTE_GROUPS:
                        if abs(float(ip[start:end].sum()) - 1.) > 1e-6:
                            raise FloatingPointError(f"Invalid {waste} route simplex")
                    if not env.observation_space.contains(obs):
                        raise FloatingPointError("Observation outside the new policy Box")
                    metrics = info["metrics"]
                    for key in FOUR_METRICS:
                        value = metrics[key]
                        if value is not None and not math.isfinite(float(value)):
                            raise FloatingPointError(f"Nonfinite {key}")
                    residual = float(metrics["primary_mass_relative_residual"])
                    if not math.isfinite(residual) or residual > 1e-6:
                        raise FloatingPointError("Primary waste routing mass conservation failed")
                    if not math.isfinite(float(reward)) or abs(float(reward)) > 1.0000001:
                        raise FloatingPointError("Invalid shaped reward")
                    if truncated or bool(terminated) != (step == 19):
                        raise FloatingPointError("Policy rollout did not preserve the 20-step horizon")
                    rewards.append(float(reward))
                    total_steps += 1
                    saturated_steps += int(abs(float(reward)) > .95)
                    sample_residual = max(sample_residual, residual)
                    minimum_collection = min(minimum_collection, float(ip[0]))
                    minimum_sorting = min(minimum_sorting, float(ip[1]))
                maximum_residual = max(maximum_residual, sample_residual)
                if minimum_collection == 0.:
                    zero_collection_seeds.append(seed)
                if minimum_sorting == 0.:
                    zero_sorting_seeds.append(seed)
                samples.append({
                    "seed": seed, "ip": ip.tolist(), "metrics": deepcopy(metrics),
                    "reference": reference,
                    "reward_stats": {"mean": float(np.mean(rewards)), "minimum": min(rewards),
                                     "maximum": max(rewards),
                                     "saturation_fraction_abs_reward_gt_095":
                                         sum(abs(r) > .95 for r in rewards) / 20},
                    "minimum_collection": minimum_collection,
                    "minimum_sorting": minimum_sorting,
                    "primary_mass_max_relative_residual": sample_residual,
                    "numeric_valid": True,
                })
            except (ValueError, FloatingPointError, RuntimeError) as exc:
                numeric_valid = False
                samples.append({"seed": seed, "numeric_valid": False,
                                "error_type": type(exc).__name__, "error": str(exc)})
        valid_samples = [s for s in samples if s["numeric_valid"]]
        means = {}
        for key in FOUR_METRICS:
            values = [s["metrics"][key] for s in valid_samples]
            present = [float(v) for v in values if v is not None]
            if present and len(present) != len(values):
                numeric_valid = False
            means[key] = math.fsum(present) / len(present) if present else None
        return {
            "model_path": str(Path(model_path).resolve()), "model_sha256": sha256_file(model_path),
            "num_timesteps": int(model.num_timesteps), "samples": samples,
            "four_metric_mean_per_tonne": means,
            "numeric_valid": numeric_valid and len(valid_samples) == len(seeds),
            "validated_steps": total_steps,
            "saturation_fraction_abs_reward_gt_095": saturated_steps / total_steps if total_steps else None,
            "primary_mass_max_relative_residual": maximum_residual,
            "zero_collection_seeds": zero_collection_seeds,
            "zero_sorting_seeds": zero_sorting_seeds,
            "metric_version": env.calculator.contract["version"],
            "metric_status": deepcopy(env.calculator.contract["metric_status"]),
            "registry_sha256": env.calculator.contract["registry_sha256"],
            "data_sha256": env.calculator.contract["data_sha256"],
        }
    finally:
        env.close()


def evaluate_saved_policies(initial_model_path, final_model_path, config):
    """Compare initial and final policies on the same explicit held-out seeds.

    A review outcome retains the trained model and requests scientific assessment;
    it never extends training, changes a reward scale, or adds a hidden objective.
    """
    evaluation = config.get("evaluation") or {}
    count = evaluation.get("heldout_episodes", 16)
    first_seed = evaluation.get("seed_start", 88000)
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("heldout_episodes must be a positive integer")
    if not isinstance(first_seed, int) or isinstance(first_seed, bool) or first_seed < 0:
        raise ValueError("seed_start must be a nonnegative integer")
    if evaluation.get("deterministic", True) is not True or evaluation.get("horizon", 20) != 20:
        raise ValueError("Post-evaluation requires deterministic full 20-step policy rollouts")
    seeds = list(range(first_seed, first_seed + count))
    mode = config.get("reward_mode") or ACP_TO_REWARD_MODE[config["acp"]]
    target, direction = MODE_TO_METRIC[mode]
    initial = _model_samples(initial_model_path, config, seeds)
    final = _model_samples(final_model_path, config, seeds)
    before = initial["four_metric_mean_per_tonne"][target]
    after = final["four_metric_mean_per_tonne"][target]
    improvement = direction * (after - before) if before is not None and after is not None else None
    valid = initial["numeric_valid"] and final["numeric_valid"]
    saturation = final["saturation_fraction_abs_reward_gt_095"]
    reasons = []
    if not valid:
        reasons.append("Numeric, IP, observation, horizon, or primary mass checks require review")
    if improvement is None or improvement <= 0:
        reasons.append("Paired held-out mean did not improve the selected target")
    if saturation is None or saturation >= .1:
        reasons.append("Final policy reward saturation is at least 10% of held-out steps")
    flags = []
    if final["zero_collection_seeds"]:
        flags.append("Zero collection occurs; inspect unconstrained single-objective policy behavior")
    if final["zero_sorting_seeds"]:
        flags.append("Zero sorting occurs; inspect unconstrained single-objective policy behavior")
    return {
        "schema_version": 1, "status": "passed" if not reasons else "scientific_review_required",
        "evaluation_version": "paired-heldout-deterministic-v1",
        "archetype": config["archetype"], "acp": config["acp"],
        "reward_version": config.get("reward_version"), "scale": float(config["scale"]),
        "target": target, "direction": "minimize" if direction < 0 else "maximize",
        "target_mean_improvement": improvement, "numeric_valid": valid,
        "seeds": seeds, "deterministic": True, "horizon": 20,
        "initial": initial, "final": final,
        "acceptance_rule": "Target mean improves, final saturation < 0.10, and all numeric/IP/primary-mass checks pass",
        "scientific_review_reasons": reasons, "configuration_review_flags": flags,
        "units": {"ghg": "kgCO2e/tMSW", "net_cost": "USD2020/tMSW",
                  "net_energy": "MJ/tMSW", "resource_output": "kg products/tMSW"},
        "limitations": ["Single training seed and paired held-out sampling do not establish convergence",
                        "Economic accounting uses verified archetype contracts; authorized missing-price substitutions retain source provenance; unknown values are never silently filled",
                        "Primary route conservation does not imply total downstream product mass conservation",
                        "Collection/sorting flags do not impose hidden service floors or climate priority",
                        "Static validated MEE background; this is not an HA/MA/LA future prediction"],
    }


__all__ = ["evaluate_saved_policies"]
