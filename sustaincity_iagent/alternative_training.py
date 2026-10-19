"""Offline S1--S3 PPO runs with update-boundary, stateful checkpoints.

Legacy train.py is intentionally untouched. Checkpoints are local trusted
artifacts; model-only .zip files cannot resume this runner's environment state.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import torch
from filelock import FileLock, Timeout as FileLockTimeout
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .alternative_env import AlternativeMSWEnv, canonical_archetype
from .alternative_factory import load_contract
from .alternative_rewards import ACP_TO_REWARD_MODE, RewardSpec, MODE_TO_METRIC, REWARD_VERSION
from .alternative_launch import git_metadata, shared_launch_preflight, StartupPreflightError

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS = ("ghg", "gross_cost", "economic_benefits", "net_cost", "energy_input",
           "energy_output", "net_energy", "resource_output", "primary_mass_relative_residual")
MONITOR_KEYS = ("rewards", "needs_reset", "episode_returns", "episode_lengths",
                "episode_times", "total_steps", "current_reset_info")
VEC_KEYS = ("buf_obs", "buf_dones", "buf_rews", "buf_infos", "reset_infos", "_seeds", "_options")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_prefix(path, size):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        remaining = size
        while remaining:
            block = handle.read(min(1024 * 1024, remaining))
            if not block: raise ValueError("Checkpoint log prefix is truncated")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(f"Not serializable: {type(value)}")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                    default=_json_value, allow_nan=False), encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 5: raise
            time.sleep(.01 * (attempt + 1))


def _runtime():
    versions = {p: importlib.metadata.version(p) for p in ("stable-baselines3", "gymnasium", "numpy", "torch", "pandas", "openpyxl")}
    return {"python": sys.version, "versions": versions, "device": "cpu"}


def prepare_config(config):
    config = copy.deepcopy(config)
    readiness = config.get("readiness", {})
    status = readiness.get("status") if isinstance(readiness, dict) else readiness
    if status not in ("ready", "verified", "pilot_ready", "pass"):
        raise ValueError(f"Training readiness gate is closed: {readiness}")
    config["archetype"] = canonical_archetype(config["archetype"])
    mode = config.get("reward_mode", ACP_TO_REWARD_MODE.get(config.get("acp")))
    if mode is None: raise ValueError("A registered ACP S1--S3 is required")
    config["reward_mode"] = mode
    config.setdefault("reward_version", REWARD_VERSION)
    if config["reward_version"] != REWARD_VERSION:
        raise ValueError("Frozen reward_version differs from the implemented reward")
    if config.get("acp") and ACP_TO_REWARD_MODE.get(config["acp"]) != mode:
        raise ValueError("ACP label and reward mode differ")
    data = Path(config["data_file"])
    if not data.is_absolute(): data = REPO_ROOT / data
    data = data.resolve()
    config["data_file"] = str(data)
    actual_data_sha = sha256(data)
    if config.get("data_sha256") != actual_data_sha:
        raise ValueError("Training-data SHA256 mismatch or missing frozen hash")
    contract = load_contract(config["archetype"], str(data))
    if config.get("registry_sha256") != contract.get("registry_sha256"):
        raise ValueError("Product-registry SHA256 mismatch or missing frozen hash")
    if config.get("metric_version") != contract.get("version"):
        raise ValueError("Frozen metric version differs from loaded contract")
    if config.get("run_kind", "formal") == "formal":
        gate = config.get("pilot_gate")
        if not gate:
            raise ValueError("Formal training requires its frozen CHN pilot acceptance record")
        gate_path = Path(gate["path"])
        if not gate_path.is_absolute(): gate_path = REPO_ROOT / gate_path
        if sha256(gate_path) != gate["sha256"]:
            raise ValueError("Frozen pilot acceptance record hash mismatch")
        evidence = json.loads(gate_path.read_text(encoding="utf-8"))
        if evidence["registry_sha256"] != config["registry_sha256"] or evidence["metric_version"] != config["metric_version"] or evidence["reward_version"] != config["reward_version"]:
            raise ValueError("Pilot evidence is incompatible with the formal metrics/reward")
        if evidence["objectives"][config["acp"]]["status"] != "passed":
            raise ValueError("Selected objective has not passed its short-pilot gate")
        if config["archetype"] == "Permissive" and config["acp"] == "ACP S1":
            own_evidence = evidence.get("archetype_objectives", {}).get("Permissive|ACP S1", {})
            if own_evidence.get("status") != "passed":
                raise ValueError("Permissive ACP S1 requires its economic-substitution pilot acceptance")
    objective = MODE_TO_METRIC[mode][0]
    if contract["metric_status"].get(objective) != "verified":
        raise ValueError(f"{objective} is not verified for {config['archetype']}")
    spec = RewardSpec(mode, float(config["scale"]), config["metric_version"])
    target = int(config["total_timesteps"])
    if target <= 0 or target != config["total_timesteps"]: raise ValueError("Invalid timestep target")
    if not isinstance(config["seed"], int): raise ValueError("An integer seed is required")
    config.setdefault("torch_num_threads", 1)
    config.setdefault("progress_interval_timesteps", 256)
    if not isinstance(config["progress_interval_timesteps"], int) or config["progress_interval_timesteps"] <= 0:
        raise ValueError("Progress interval must be a positive integer")
    config.setdefault("checkpoint_rollouts", max(1, int(config.get("checkpoint_interval_timesteps", 10240)) // int(config.get("ppo", {}).get("n_steps", 2048))))
    if int(config["checkpoint_rollouts"]) <= 0: raise ValueError("Checkpoint interval must be positive")
    ppo = config.setdefault("ppo", {})
    ppo.setdefault("n_steps", 2048)
    ppo.setdefault("batch_size", 64)
    if int(ppo["n_steps"]) <= 1 or int(ppo["batch_size"]) <= 1:
        raise ValueError("PPO rollout and batch sizes must exceed one")
    if ppo["n_steps"] % ppo["batch_size"] != 0:
        raise ValueError("Batch size must divide rollout size")
    if "checkpoint_interval_timesteps" in config and config["checkpoint_interval_timesteps"] != config["checkpoint_rollouts"] * ppo["n_steps"]:
        raise ValueError("Checkpoint interval must be an exact multiple of complete rollouts")
    if config.get("normalization", "none") != "none":
        raise ValueError("VecNormalize is not part of this frozen experiment contract")
    if ppo.get("device", "cpu") != "cpu":
        raise ValueError("This validated runner uses CPU; GPU needs an independent recovery validation")
    return config, contract, spec


def _signature(config):
    files = ("alternative_env.py", "alternative_metrics.py", "alternative_rewards.py",
             "alternative_factory.py", "alternative_training.py", "alternative_evaluation.py",
             "alternative_launch.py")
    code = {name: sha256(Path(__file__).parent / name) for name in files}
    original_engine = Path(__file__).parent / "env" / f"ENV_{config['archetype']}.py"
    code[f"env/{original_engine.name}"] = sha256(original_engine)
    material = {"config": config, "runtime": _runtime(), "code_sha256": code}
    material["compatibility_sha256"] = hashlib.sha256(
        json.dumps(material, sort_keys=True, default=_json_value).encode()).hexdigest()
    material.update(git_metadata(REPO_ROOT))
    return material


class PlannedPause(Exception):
    """Stop only after a complete PPO update, preserving the configured horizon."""


class RunCallback(BaseCallback):
    def __init__(self, run_dir, config, signature, *, resume=False, pause_after_timesteps=None):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.config = config
        self.signature = signature
        self.pause_after = pause_after_timesteps
        self.started = time.monotonic()
        self.elapsed_prior_seconds = 0.
        self.last_diagnostics = {}
        self.checkpoint_event = "none"
        self.resumed = bool(resume)
        self.pending = None
        self.latest_checkpoint = None
        self.samples = []
        self.saturation_count = 0
        self.validation_steps = 0
        self.metric_ranges = {k: [None, None] for k in METRICS}
        self.handles = {}
        self.writers = {}
        fields = {
            "trajectory.csv": ["timesteps", "episode_step", "reward", "raw_reward", "normalized_delta",
                               "reference_value", "scale", "saturated", *METRICS,
                               "collection", "sorting",
                               *[f"IP{i:02d}" for i in range(1, 30)], "products_json", "routes_json"],
            "episodes.csv": ["timesteps", "length", "reward", "elapsed_seconds", *METRICS],
            "rollouts.csv": ["run_id", "acp", "archetype", "seed", "metric_version", "reward_version", "scale",
                             "timesteps", "updates", "elapsed_seconds", "mean_episode_reward", "episode_reward_window",
                             "fps", "explained_variance", "explained_variance_status", "reward_mean", "reward_min", "reward_max",
                             "saturation_fraction", *[k + "_mean" for k in METRICS],
                             "approx_kl", "clip_fraction", "entropy_loss", "value_loss", "policy_gradient_loss", "loss"],
        }
        for name, keys in fields.items():
            path = self.run_dir / name
            exists = path.exists()
            if exists and not resume: raise FileExistsError(path)
            h = path.open("a" if resume else "w", newline="", encoding="utf-8")
            self.handles[name] = h
            self.writers[name] = csv.DictWriter(h, fieldnames=keys, extrasaction="ignore")
            if not exists: self.writers[name].writeheader()

    def _flush(self):
        for handle in self.handles.values(): handle.flush()

    def close(self):
        for handle in self.handles.values(): handle.close()

    def elapsed_seconds(self):
        return self.elapsed_prior_seconds + time.monotonic() - self.started

    def publish_progress(self, status="running", *, emit=False, event="progress", **extra):
        elapsed = self.elapsed_seconds()
        steps = int(self.model.num_timesteps)
        nominal = int(self.config["total_timesteps"])
        aligned = math.ceil(nominal / self.model.n_steps) * self.model.n_steps
        fps = steps / elapsed if steps and elapsed > 0 else None
        record = {"run_id": self.run_dir.name, "acp": self.config["acp"],
            "archetype": self.config["archetype"], "seed": self.config["seed"],
            "status": status, "event": event, "resumed": self.resumed,
            "pid": __import__("os").getpid(), "updated_at_epoch": time.time(),
            "timesteps": steps, "target": nominal, "aligned_target": aligned,
            "percent_complete": min(100., steps / aligned * 100.),
            "elapsed_seconds": elapsed, "fps": fps,
            "eta_seconds": max(0., aligned - steps) / fps if fps else None,
            "latest_checkpoint": str(self.latest_checkpoint) if self.latest_checkpoint else None,
            "checkpoint_event": self.checkpoint_event, **self.last_diagnostics, **extra}
        write_json(self.run_dir / "progress.json", record)
        with (self.run_dir / "progress_history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps(record, default=_json_value, allow_nan=False) + "\n")
        if emit:
            def fmt(value): return f"{value:.5g}" if value is not None else "pending"
            checkpoint = Path(record["latest_checkpoint"]).name if record["latest_checkpoint"] else "none"
            eta = record["eta_seconds"]
            eta_text = f"{int(eta // 60):02d}:{int(eta % 60):02d}" if eta is not None else "pending"
            print(f"{record['acp']} | {record['archetype']} | seed={record['seed']} | "
                  f"{steps}/{nominal} ({record['percent_complete']:.1f}%) | "
                  f"ep_reward={fmt(record.get('mean_episode_reward'))} | "
                  f"KL={fmt(record.get('approx_kl'))} | PGL={fmt(record.get('policy_gradient_loss'))} | "
                  f"fps={fmt(fps)} ETA={eta_text} | checkpoint={checkpoint}/{record['checkpoint_event']} | {status}", flush=True)
        return record

    def _on_step(self):
        info = self.locals["infos"][0]
        metrics = info["metrics"]
        reward = float(self.locals["rewards"][0])
        parts = info["reward_components"]
        ip = np.asarray(info["ip"], dtype=float)
        if not np.isfinite(ip).all() or ip.shape != (29,) or (ip < 0).any() or (ip > 1).any():
            raise FloatingPointError("Invalid pilot/production IP configuration")
        if not math.isfinite(reward) or abs(reward) > 1.0000001:
            raise FloatingPointError("Invalid shaped reward")
        if metrics["primary_mass_relative_residual"] > 1e-6:
            raise FloatingPointError("Primary mass conservation failed")
        for key in METRICS:
            value = metrics[key]
            if value is not None:
                value = float(value)
                if not math.isfinite(value): raise FloatingPointError(f"Nonfinite {key}")
                low, high = self.metric_ranges[key]
                self.metric_ranges[key] = [value if low is None else min(low, value),
                                           value if high is None else max(high, value)]
        saturated = abs(reward) > .95
        self.validation_steps += 1
        self.saturation_count += int(saturated)
        engine = self.training_env.envs[0].env
        # DummyVecEnv resets finished episodes before callback dispatch.
        episode_step = info.get("episode", {}).get("l", engine.current_step)
        self.writers["trajectory.csv"].writerow({"timesteps": self.num_timesteps,
            "episode_step": episode_step, "reward": reward,
            "raw_reward": parts["raw_reward"], "normalized_delta": parts["normalized_delta"],
            "reference_value": parts["reference_value"], "scale": parts["scale"],
            "saturated": int(saturated), **{k: metrics[k] for k in METRICS},
            "collection": ip[0], "sorting": ip[1],
            **{f"IP{i:02d}": value for i, value in enumerate(ip, 1)},
            "products_json": json.dumps(metrics["products"], sort_keys=True),
            "routes_json": json.dumps(metrics["route_mass_per_tonne"], sort_keys=True)})
        self.samples.append({"reward": reward, "saturated": float(saturated), **{k: metrics[k] for k in METRICS}})
        if "episode" in info:
            episode = info["episode"]
            self.writers["episodes.csv"].writerow({"timesteps": self.num_timesteps,
                "length": episode["l"], "reward": episode["r"], "elapsed_seconds": episode.get("t"),
                **{k: metrics[k] for k in METRICS}})
        if self.num_timesteps % self.config["progress_interval_timesteps"] == 0:
            self._flush()
            progress = self.publish_progress()
            write_json(self.run_dir / "status.json", progress)
        return True

    def _on_rollout_end(self):
        rewards = [x["reward"] for x in self.samples]
        self.pending = {"timesteps": self.num_timesteps,
            "reward_mean": float(np.mean(rewards)), "reward_min": min(rewards), "reward_max": max(rewards),
            "saturation_fraction": float(np.mean([x["saturated"] for x in self.samples]))}
        for key in METRICS:
            values = [x[key] for x in self.samples if x[key] is not None]
            self.pending[key + "_mean"] = float(np.mean(values)) if values else None
        self.samples = []

    def _record_updated_rollout(self):
        if self.pending is None: return
        row = {**self.pending, "updates": self.model._n_updates,
               "elapsed_seconds": self.elapsed_seconds(), "run_id": self.run_dir.name,
               **{key: self.config[key] for key in ("acp", "archetype", "seed", "metric_version", "reward_version", "scale")}}
        episodes = self.model.ep_info_buffer
        row["mean_episode_reward"] = float(np.mean([episode["r"] for episode in episodes])) if episodes else None
        row["episode_reward_window"] = len(episodes)
        row["fps"] = self.num_timesteps / row["elapsed_seconds"] if row["elapsed_seconds"] > 0 else None
        for key in ("approx_kl", "clip_fraction", "entropy_loss", "value_loss", "policy_gradient_loss", "loss"):
            value = self.model.logger.name_to_value.get("train/" + key)
            row[key] = float(value) if value is not None else None
            if row[key] is not None and not math.isfinite(row[key]):
                raise FloatingPointError(f"Nonfinite PPO diagnostic: {key}")
        explained = self.model.logger.name_to_value.get("train/explained_variance")
        row["explained_variance"] = float(explained) if explained is not None and math.isfinite(float(explained)) else None
        row["explained_variance_status"] = "defined" if row["explained_variance"] is not None else "undefined_or_not_reported"
        self.last_diagnostics = {key: row[key] for key in ("mean_episode_reward", "approx_kl", "policy_gradient_loss", "value_loss", "entropy_loss", "explained_variance", "explained_variance_status")}
        self.writers["rollouts.csv"].writerow(row)
        self.pending = None
        self._flush()
        write_json(self.run_dir / "status.json", self.publish_progress(emit=True, event="ppo_update"))

    def save_checkpoint(self):
        self._flush()
        destination = self.run_dir / "checkpoints" / f"step_{self.num_timesteps:09d}"
        if destination.exists():
            self.latest_checkpoint = destination
            self.checkpoint_event = "restored"
            return destination
        temporary = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex[:8])
        temporary.mkdir(parents=True)
        monitor = self.training_env.envs[0]
        state = {"environment": monitor.env.export_state(),
            "monitor": {key: copy.deepcopy(getattr(monitor, key)) for key in MONITOR_KEYS},
            "monitor_elapsed": time.time() - monitor.t_start,
            "vec_env": {key: copy.deepcopy(getattr(self.training_env, key)) for key in VEC_KEYS},
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
            "callback": {"validation_steps": self.validation_steps, "saturation_count": self.saturation_count,
                         "metric_ranges": self.metric_ranges,
                         "elapsed_prior_seconds": self.elapsed_seconds(),
                         "last_diagnostics": self.last_diagnostics},
            "log_offsets": {name: (self.run_dir / name).stat().st_size for name in self.handles},
            "log_prefix_sha256": {name: sha256(self.run_dir / name) for name in self.handles},
            "num_timesteps": self.num_timesteps,
            "n_updates": self.model._n_updates}
        self.model.save(temporary / "model.zip")
        with (temporary / "state.pkl").open("wb") as handle: pickle.dump(state, handle, protocol=5)
        manifest = {"checkpoint_version": 1, "boundary": "after_complete_ppo_update",
                    "timesteps": self.num_timesteps, "n_updates": self.model._n_updates,
                    "signature": self.signature, "run_dir": str(self.run_dir),
                    "files": {name: sha256(temporary / name) for name in ("model.zip", "state.pkl")}}
        write_json(temporary / "manifest.json", manifest)
        temporary.replace(destination)
        self.latest_checkpoint = destination
        self.checkpoint_event = "saved"
        write_json(self.run_dir / "latest_checkpoint.json", {"path": str(destination), "timesteps": self.num_timesteps,
                                                            "manifest_sha256": sha256(destination / "manifest.json")})
        write_json(self.run_dir / "status.json", self.publish_progress(event="checkpoint_saved"))
        return destination

    def _on_rollout_start(self):
        self._record_updated_rollout()
        if self.num_timesteps:
            rollout_number = self.num_timesteps // self.model.n_steps
            if rollout_number % int(self.config["checkpoint_rollouts"]) == 0 or (
                    self.pause_after is not None and self.num_timesteps >= self.pause_after):
                self.save_checkpoint()
            if self.pause_after is not None and self.num_timesteps >= self.pause_after:
                raise PlannedPause()

    def _on_training_end(self):
        self._record_updated_rollout()
        self.save_checkpoint()


def _load_checkpoint(checkpoint, signature, env):
    checkpoint = Path(checkpoint).resolve()
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    if manifest["boundary"] != "after_complete_ppo_update": raise ValueError("Unsafe checkpoint boundary")
    if manifest["signature"]["compatibility_sha256"] != signature["compatibility_sha256"]:
        raise ValueError("Checkpoint config/data/code/runtime signature mismatch; start a separately versioned run")
    for name, expected in manifest["files"].items():
        if sha256(checkpoint / name) != expected: raise ValueError(f"Damaged checkpoint file: {name}")
    with (checkpoint / "state.pkl").open("rb") as handle: state = pickle.load(handle)
    monitor = env.envs[0]
    monitor.env.import_state(state["environment"])
    for key, value in state["monitor"].items(): setattr(monitor, key, copy.deepcopy(value))
    monitor.t_start = time.time() - state["monitor_elapsed"]
    model = PPO.load(checkpoint / "model.zip", env=env, device="cpu", force_reset=False)
    # PPO.load seeds its new VecEnv during setup, so restore pending seeds/options
    # after that setup as well as the active physical and Monitor states.
    for key, value in state["vec_env"].items(): setattr(env, key, copy.deepcopy(value))
    if model._last_obs is None or model.num_timesteps != manifest["timesteps"]:
        raise ValueError("PPO checkpoint did not retain observation/timestep state")
    if model._n_updates != state["n_updates"]: raise ValueError("Optimizer update count mismatch")
    run_dir = Path(manifest["run_dir"]).resolve()
    if checkpoint.parent.parent != run_dir or checkpoint.parent.name != "checkpoints":
        raise ValueError("Checkpoint does not belong to its declared run directory")
    latest = json.loads((run_dir / "latest_checkpoint.json").read_text(encoding="utf-8"))
    if Path(latest["path"]).resolve() != checkpoint:
        raise ValueError("Only the latest checkpoint can resume this run; older checkpoints require a separate fork")
    if latest["manifest_sha256"] != sha256(checkpoint / "manifest.json"):
        raise ValueError("Latest checkpoint manifest hash mismatch")
    if set(state["log_offsets"]) != {"trajectory.csv", "episodes.csv", "rollouts.csv"}:
        raise ValueError("Checkpoint declares unexpected log files")
    # A crash may have appended an uncommitted rollout; restore exact log prefix.
    for name, offset in state["log_offsets"].items():
        path = run_dir / name
        if not isinstance(offset, int) or offset < 0: raise ValueError("Invalid log checkpoint offset")
        if path.stat().st_size < offset: raise ValueError(f"Missing checkpoint log prefix: {name}")
        if sha256_prefix(path, offset) != state["log_prefix_sha256"][name]:
            raise ValueError(f"Checkpoint log prefix hash mismatch: {name}")
    for name, offset in state["log_offsets"].items():
        path = run_dir / name
        with path.open("r+b") as handle: handle.truncate(offset)
    return model, state, run_dir, checkpoint


def validate_completed(run_dir, *, strict_current=True):
    run_dir = Path(run_dir)
    completed = json.loads((run_dir / "completed.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config, contract, spec = prepare_config(config)
    signature = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if strict_current and signature["compatibility_sha256"] != _signature(config)["compatibility_sha256"]:
        raise ValueError("Completed run is incompatible with current code/data/config/runtime")
    checkpoint = Path(completed["checkpoint"])
    for name, expected in completed["files"].items():
        if sha256(run_dir / name) != expected: raise ValueError(f"Completed artifact hash mismatch: {name}")
    validation_env = AlternativeMSWEnv(config["archetype"], config["data_file"], spec, contract,
                                      mass_range=config.get("mass_range", (.5, 2.)),
                                      max_steps=config.get("max_episode_steps", 20))
    try:
        model = PPO.load(run_dir / "model.zip", env=validation_env, device="cpu")
    finally:
        validation_env.close()
    if model.num_timesteps < int(config["total_timesteps"]) or model.num_timesteps != completed["actual_timesteps"]:
        raise ValueError("Saved model has not completed the frozen timestep target")
    if not checkpoint.is_dir(): raise ValueError("Full final checkpoint is missing")
    checkpoint_manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    if checkpoint_manifest["signature"]["compatibility_sha256"] != signature["compatibility_sha256"]:
        raise ValueError("Final checkpoint and run signatures differ")
    if checkpoint_manifest["timesteps"] != model.num_timesteps:
        raise ValueError("Final checkpoint timestep count differs")
    for name, expected in checkpoint_manifest["files"].items():
        if sha256(checkpoint / name) != expected: raise ValueError(f"Final checkpoint hash mismatch: {name}")
    return completed


def _run_training_owned(config, output_root, *, resume_from=None, pause_after_timesteps=None):
    """Train one configured IA; return its unique run directory, including on pause."""
    config, contract, spec = prepare_config(config)
    signature = _signature(config)
    torch.set_num_threads(int(config["torch_num_threads"]))
    env = DummyVecEnv([lambda: Monitor(AlternativeMSWEnv(config["archetype"], config["data_file"], spec, contract,
                  mass_range=config.get("mass_range", (.5, 2.)), max_steps=config.get("max_episode_steps", 20)))])
    root = Path(output_root)
    if not root.is_absolute(): root = REPO_ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    state = None
    if resume_from:
        candidate = Path(resume_from).resolve()
        candidate_manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
        candidate_run = Path(candidate_manifest["run_dir"]).resolve()
        if candidate.parent.parent != candidate_run or candidate.parent.name != "checkpoints":
            raise ValueError("Checkpoint does not belong to its declared run directory")
        if candidate_manifest["signature"]["compatibility_sha256"] != signature["compatibility_sha256"]:
            raise ValueError("Checkpoint config/data/code/runtime signature mismatch; start a separately versioned run")
        # A completed run is read-only, even when the caller selected an older
        # checkpoint. Never truncate its verified log history before this check.
        if (candidate_run / "completed.json").exists():
            validate_completed(candidate_run)
            env.close()
            return candidate_run
        model, state, run_dir, checkpoint = _load_checkpoint(resume_from, signature, env)
    else:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        label = config.get("acp", config["reward_mode"]).replace(" ", "_")
        run_id = f"{stamp}_{config['archetype']}_{label}_seed{config['seed']}_{uuid.uuid4().hex[:8]}"
        run_dir = root / run_id
        run_dir.mkdir()
        write_json(run_dir / "config.json", config)
        write_json(run_dir / "manifest.json", signature)
        ppo = dict(config["ppo"])
        policy = ppo.pop("policy", "MlpPolicy")
        ppo.pop("seed", None)
        ppo.pop("device", None)
        ppo.pop("verbose", None)
        model = PPO(policy, env, seed=config["seed"], device="cpu", verbose=0, **ppo)
        model.save(run_dir / "initial_model.zip")
    callback = RunCallback(run_dir, config, signature, resume=bool(resume_from), pause_after_timesteps=pause_after_timesteps)
    callback.init_callback(model)
    if state is not None:
        for key, value in state["callback"].items(): setattr(callback, key, copy.deepcopy(value))
        callback.latest_checkpoint = checkpoint
        callback.checkpoint_event = "restored"
    segment = len(list((run_dir / "sb3_segments").glob("segment_*"))) if (run_dir / "sb3_segments").exists() else 0
    model.set_logger(configure(str(run_dir / "sb3_segments" / f"segment_{segment:03d}"), ["csv"]))
    if state is not None:
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
    write_json(run_dir / "status.json", {"status": "running", "timesteps": model.num_timesteps,
                                        "target": config["total_timesteps"]})
    callback.publish_progress(emit=True, event="resumed" if resume_from else "started")
    start = time.monotonic()
    try:
        remaining = int(config["total_timesteps"]) - model.num_timesteps
        if remaining > 0:
            model.learn(remaining, callback=callback, reset_num_timesteps=False, log_interval=1)
        elif state is None or callback.latest_checkpoint is None:
            raise ValueError("Reached target without a complete, stateful checkpoint")
        if model.num_timesteps < int(config["total_timesteps"]): raise RuntimeError("Training stopped before target")
        if callback.validation_steps != model.num_timesteps:
            raise ValueError("Validated trajectory count differs from PPO timesteps")
        callback._flush()
        checkpoint = callback.latest_checkpoint
        shutil.copy2(checkpoint / "model.zip", run_dir / "model.zip")
        shutil.copy2(checkpoint / "model.zip", run_dir / "final_model.zip")
        saved = PPO.load(run_dir / "model.zip", env=env, device="cpu")
        if saved.num_timesteps != model.num_timesteps: raise ValueError("Final load verification failed")
        validation = {"status": "passed", "steps": callback.validation_steps,
                      "metric_ranges": callback.metric_ranges,
                      "saturation_fraction": callback.saturation_count / callback.validation_steps,
                      "checks": ["finite_metrics_and_reward", "29_IP_bounds", "primary_mass_conservation", "model_load_and_timesteps"],
                      "limitations": ["Single-seed short runs do not establish convergence or stochastic robustness",
                                      "Primary waste routing conservation does not imply all downstream product mass is conserved"]}
        write_json(run_dir / "validation.json", validation)
        callback.publish_progress("evaluating", emit=True, event="heldout_evaluation")
        from .alternative_evaluation import evaluate_saved_policies
        acceptance = evaluate_saved_policies(run_dir / "initial_model.zip", run_dir / "model.zip", config)
        write_json(run_dir / "acceptance.json", acceptance)
        completed = {"status": "completed", "actual_timesteps": model.num_timesteps,
                     "scientific_acceptance": acceptance["status"],
                     "target_timesteps": config["total_timesteps"], "checkpoint": str(checkpoint),
                     "elapsed_this_segment_seconds": time.monotonic() - start,
                     "files": {name: sha256(run_dir / name) for name in
                               ("model.zip", "initial_model.zip", "final_model.zip", "config.json", "manifest.json", "validation.json", "acceptance.json", "trajectory.csv", "episodes.csv", "rollouts.csv")}}
        write_json(run_dir / "completed.json", completed)
        write_json(run_dir / "status.json", completed)
        callback.publish_progress("completed", emit=True, event="completed", scientific_acceptance=acceptance["status"])
    except PlannedPause:
        write_json(run_dir / "status.json", {"status": "paused_at_complete_update", "timesteps": model.num_timesteps,
            "target": config["total_timesteps"], "latest_checkpoint": str(callback.latest_checkpoint)})
        callback.publish_progress("paused_at_complete_update", emit=True, event="paused")
    except BaseException as exc:
        write_json(run_dir / "error_summary.json", {"error_type": type(exc).__name__, "error": str(exc),
            "timesteps": model.num_timesteps, "latest_checkpoint": str(callback.latest_checkpoint) if callback.latest_checkpoint else None,
            "traceback": traceback.format_exc()})
        write_json(run_dir / "status.json", {"status": "failed", "timesteps": model.num_timesteps,
            "target": config["total_timesteps"], "error": str(exc),
            "latest_checkpoint": str(callback.latest_checkpoint) if callback.latest_checkpoint else None})
        callback.publish_progress("failed", emit=True, event="failed", error=str(exc))
        raise
    finally:
        callback.close()
        env.close()
    return run_dir


class TrainingBusyError(RuntimeError):
    """The same frozen training configuration already has an active writer."""


def run_training(config, output_root, *, resume_from=None, pause_after_timesteps=None):
    """Own a configuration lock before creating, restoring or truncating logs."""
    normalized, _, _ = prepare_config(config)
    identity = _signature(normalized)["compatibility_sha256"]
    lock_root = REPO_ROOT / "experiments" / "alternative_primary_objectives" / "logs" / "training_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(lock_root / f"{identity}.lock", timeout=0):
            return _run_training_owned(normalized, output_root, resume_from=resume_from,
                                       pause_after_timesteps=pause_after_timesteps)
    except FileLockTimeout as error:
        raise TrainingBusyError("Same frozen training configuration is already active; refusing duplicate training or live-log rewind") from error


def status_summary(output_root):
    rows = []
    for path in sorted(Path(output_root).glob("*/status.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        progress_path = path.parent / "progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
        age = max(0., time.time() - progress["updated_at_epoch"]) if progress.get("updated_at_epoch") else None
        rows.append({**progress, "run_id": path.parent.name, "status": item["status"],
                     "timesteps": item.get("actual_timesteps", item.get("timesteps")),
                     "target": item.get("target_timesteps", item.get("target")),
                     "latest_checkpoint": item.get("latest_checkpoint") or item.get("checkpoint") or progress.get("latest_checkpoint"),
                     "error": item.get("error"), "scientific_acceptance": item.get("scientific_acceptance"),
                     "last_update_age_seconds": age,
                     "stale_or_interrupted": bool(age is not None and age > 30 and item["status"] in ("running", "evaluating"))})
    return rows


def display_status(rows):
    if not rows:
        print("No local formal training runs yet.", flush=True)
        return
    for row in rows:
        def fmt(value): return f"{value:.5g}" if value is not None else "pending"
        eta = row.get("eta_seconds")
        checkpoint = row.get("latest_checkpoint")
        print(f"{row.get('acp', '?')} | {row.get('archetype', '?')} | seed={row.get('seed', '?')} | "
              f"{row['timesteps']}/{row['target']} ({fmt(row.get('percent_complete'))}%) | "
              f"ep_reward={fmt(row.get('mean_episode_reward'))} KL={fmt(row.get('approx_kl'))} "
              f"PGL={fmt(row.get('policy_gradient_loss'))} | fps={fmt(row.get('fps'))} "
              f"ETA={fmt(eta)}s | checkpoint={Path(checkpoint).name if checkpoint else 'none'} | "
              f"{row['status']}{'/resumed' if row.get('resumed') else ''}"
              f"{'/stale' if row.get('stale_or_interrupted') else ''}", flush=True)


def record_launch_failure(error, *, phase, config_path=None):
    """Print and persist the entire exception chain, including pre-run failures."""
    trace = traceback.format_exc()
    print(trace, file=sys.stderr, flush=True)
    directory = REPO_ROOT / "experiments/alternative_primary_objectives/logs/launch_failures"
    path = directory / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:8] + ".json")
    report = {"status": "failed", "phase": phase, "error_type": type(error).__name__,
              "error": str(error), "traceback": trace, "config_path": str(config_path) if config_path else None,
              "python_executable": sys.executable, "cwd": str(REPO_ROOT),
              "startup_report": getattr(error, "report", None)}
    try:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
        print(f"Full launch traceback saved: {path}", file=sys.stderr, flush=True)
        return str(path)
    except OSError as log_error:
        print(f"Cannot write launch error file: {log_error}; complete traceback printed above.", file=sys.stderr, flush=True)
        return None


def _report_startup(report):
    directory = REPO_ROOT / "experiments/alternative_primary_objectives/logs"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "startup_preflight.json", report)
    print(f"Startup passed | Python={report['python_executable']} | cwd={report['cwd']} | "
          f"import subprocess=passed | Git metadata={report['git_metadata_status']}", flush=True)
    if report["git_metadata_status"] != "available":
        print("Git is optional: commit metadata unavailable; code/data hashes are still recorded.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--output-root", default="experiments/alternative_primary_objectives/runs")
    train.add_argument("--resume")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--startup-test", action="store_true", help="Check paths and an import subprocess without learning")
    train.add_argument("--pause-after", type=int)
    batch = commands.add_parser("batch")
    batch.add_argument("--config-dir", required=True)
    batch.add_argument("--output-root", default="experiments/alternative_primary_objectives/runs")
    batch.add_argument("--archetype", action="append")
    batch.add_argument("--acp", action="append", choices=list(ACP_TO_REWARD_MODE))
    batch.add_argument("--dry-run", action="store_true")
    batch.add_argument("--startup-test", action="store_true", help="Check paths and an import subprocess without learning")
    batch.add_argument("--resume", action="store_true")
    batch.add_argument("--pause-after", type=int, help="Pause a single selected IA at an update boundary for launch testing")
    status = commands.add_parser("status")
    status.add_argument("--output-root", default="experiments/alternative_primary_objectives/runs")
    status.add_argument("--watch", action="store_true")
    status.add_argument("--interval", type=float, default=2.)
    status.add_argument("--iterations", type=int)
    status.add_argument("--json", action="store_true")
    verify = commands.add_parser("validate")
    verify.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "status":
        output = Path(args.output_root)
        if not output.is_absolute(): output = REPO_ROOT / output
        if args.interval < .2 or not math.isfinite(args.interval): raise ValueError("Monitor interval must be finite and >=0.2 seconds")
        count = 0
        try:
            while True:
                rows = status_summary(output)
                if args.json: print(json.dumps(rows, indent=2, ensure_ascii=False), flush=True)
                else: display_status(rows)
                count += 1
                if not args.watch or args.iterations is not None and count >= args.iterations: break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
        return 0
    if args.command == "validate":
        print(json.dumps(validate_completed(args.run_dir), indent=2))
        return 0
    if args.command == "train":
        try:
            report = shared_launch_preflight(REPO_ROOT, args.output_root)
            config_path = Path(args.config)
            if not config_path.is_absolute(): config_path = REPO_ROOT / config_path
            config = json.loads(config_path.read_text(encoding="utf-8"))
            normalized, _, _ = prepare_config(config)
            _signature(normalized)
            _report_startup(report)
            if args.dry_run or args.startup_test:
                print(json.dumps({"status": "ready", "config": normalized, "learning_started": False}, indent=2))
            else:
                print(run_training(config, args.output_root, resume_from=args.resume, pause_after_timesteps=args.pause_after))
            return 0
        except Exception as error:
            record_launch_failure(error, phase="single_launch", config_path=args.config)
            return 2
    output = Path(args.output_root)
    if not output.is_absolute(): output = REPO_ROOT / output
    outcomes = []
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute(): config_dir = REPO_ROOT / config_dir
    try:
        report = shared_launch_preflight(REPO_ROOT, output, config_dir)
        names = json.loads((config_dir / "batch_manifest.json").read_text(encoding="utf-8"))["configs"]
        prepared = []
        for name in names:
            path = config_dir / name
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not {"archetype", "acp", "data_file", "total_timesteps"} <= raw.keys() or raw.get("run_kind") == "pilot": continue
            if args.archetype and canonical_archetype(raw["archetype"]) not in {canonical_archetype(a) for a in args.archetype}: continue
            if args.acp and raw["acp"] not in args.acp: continue
            config, _, _ = prepare_config(raw)
            _signature(config)
            prepared.append((path, config))
        if not prepared: raise ValueError("No training configurations matched the requested selection")
        if args.pause_after is not None and (args.pause_after <= 0 or len(prepared) != 1):
            raise ValueError("A short launch pause requires a positive timestep cap and exactly one selected IA")
        report["selected_config_count"] = len(prepared)
        _report_startup(report)
    except Exception as error:
        trace_file = record_launch_failure(error, phase="shared_batch_startup")
        aborted = [{"status": "aborted_shared_startup", "reason": str(error),
                    "traceback_file": trace_file, "training_tasks_started": 0}]
        try:
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / ("batch_plan.json" if args.dry_run or args.startup_test else "batch_summary.json"), aborted)
        except OSError:
            pass  # The primary complete traceback was already printed/persisted.
        return 2
    for path, config in prepared:
        try:
            if args.dry_run or args.startup_test:
                outcomes.append({"config": str(path), "status": "ready"})
                continue
            matching = []
            for previous in output.glob("*/config.json"):
                if json.loads(previous.read_text(encoding="utf-8")) == config: matching.append(previous.parent)
            successful = [run for run in matching if (run / "completed.json").exists()]
            if successful:
                validate_completed(successful[-1])
                outcomes.append({"config": str(path), "status": "skipped_completed", "run_dir": str(successful[-1])})
                continue
            resume = None
            if args.resume:
                candidates = [(run / "latest_checkpoint.json") for run in matching if (run / "latest_checkpoint.json").exists()]
                if candidates: resume = json.loads(candidates[-1].read_text(encoding="utf-8"))["path"]
            run = run_training(config, output, resume_from=resume, pause_after_timesteps=args.pause_after)
            state = json.loads((run / "status.json").read_text(encoding="utf-8"))
            outcomes.append({"config": str(path), "status": state["status"], "run_dir": str(run)})
        except TrainingBusyError as exc:
            outcomes.append({"config": str(path), "status": "skipped_active", "reason": str(exc)})
            print(json.dumps(outcomes[-1]), flush=True)
        except Exception as exc:
            trace_file = record_launch_failure(exc, phase="training_task", config_path=path)
            outcomes.append({"config": str(path), "status": "blocked_or_failed", "reason": str(exc),
                             "traceback_file": trace_file})
            print(json.dumps(outcomes[-1]), flush=True)
            if isinstance(exc, OSError):
                print("Fatal filesystem/process launch error; remaining batch tasks were not started.", flush=True)
                break
    if not outcomes:
        raise ValueError("No training configurations matched the requested selection")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / ("batch_plan.json" if args.dry_run or args.startup_test else "batch_summary.json"), outcomes)
    print(json.dumps(outcomes, indent=2, ensure_ascii=False))
    return 2 if any(x["status"] == "blocked_or_failed" for x in outcomes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
