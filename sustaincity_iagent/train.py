import os
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from importlib import import_module, resources

from .config import IAS_TO_REWARD_MODE, SUPPORTED_ARCHETYPES, DEFAULT_TRAIN_DATA_FILE


class SaveMetricsCallback(BaseCallback):
    """Callback that saves basic training metrics to a CSV file."""

    def __init__(self, save_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.metrics = {k: [] for k in [
            "timesteps",
            "ep_rew_mean",
            "approx_kl",
            "entropy_loss",
            "explained_variance",
            "loss",
            "policy_gradient_loss",
            "std",
            "value_loss",
        ]}

    def _on_step(self) -> bool:
        if self.model is None:
            return True

        logger = self.model.logger
        step = self.num_timesteps
        self.metrics["timesteps"].append(step)

        def get_val(key):
            try:
                return float(logger.name_to_value[f"train/{key}"])
            except KeyError:
                return np.nan

        for key in self.metrics.keys():
            if key == "timesteps":
                continue
            self.metrics[key].append(get_val(key))

        return True

    def _on_training_end(self) -> None:
        import pandas as pd

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.save_path, index=False)


def _load_env_class(archetype: str, custom_env_module: Optional[str] = None):
    """Return the MSWEnv class for a given archetype.

    Parameters
    ----------
    archetype
        One of JPN, EU, US, CHN, IND, Permissive.
    custom_env_module
        Optional dotted-path to a user-defined env module that exposes MSWEnv.
    """
    if custom_env_module is not None:
        module = import_module(custom_env_module)
        return module.MSWEnv

    arch = archetype.upper()
    if arch not in SUPPORTED_ARCHETYPES:
        raise ValueError(
            f"Unsupported archetype '{archetype}'. "
            f"Supported: {sorted(SUPPORTED_ARCHETYPES)}"
        )

    module_name = f"sustaincity_iagent.env.ENV_{arch}"
    module = import_module(module_name)
    return module.MSWEnv


def train_msw_agent(
    archetype: str,
    ias: str,
    *,
    reward_mode: Optional[str] = None,
    train_data_path: Optional[str] = None,
    total_timesteps: int = 200_000,
    ent_coef: float = 0.005,
    output_dir: Optional[str] = None,
    custom_env_module: Optional[str] = None,
    print_summary: bool = True,
) -> str:
    """Train a PPO agent for the MSW intelligent-management environment.

    Parameters
    ----------
    archetype
        One of JPN, EU, US, CHN, IND, Permissive.
    ias
        Intelligent-agent scenario label, e.g. 'Scenario I'.
    reward_mode
        Internal reward_mode string. If None, inferred from `ias`.
    train_data_path
        Path to the Excel file providing training data. If None, use the
        built-in example `data/train_data.xlsx` shipped with the package.
    total_timesteps
        PPO training timesteps.
    ent_coef
        Entropy coefficient passed to PPO.
    output_dir
        Directory to store the trained model and metrics. If None, defaults
        to `training_results_<reward_mode>` under the current working dir.
    custom_env_module
        Optional dotted-path to a custom ENV module exposing MSWEnv.

    Returns
    -------
    model_path : pathlib.Path
        Path to the saved agent (.zip).
    """

    if reward_mode is None:
        try:
            reward_mode = IAS_TO_REWARD_MODE[ias]
        except KeyError as exc:
            raise ValueError(
                f"Unknown IAS '{ias}'. Please choose from: "
                f"{', '.join(IAS_TO_REWARD_MODE.keys())} "
                "or explicitly pass a reward_mode."
            ) from exc

    if train_data_path is None:
        # Use package-embedded example
        base = Path(resources.files("sustaincity_iagent"))
        train_data_path = base / DEFAULT_TRAIN_DATA_FILE
    else:
        train_data_path = Path(train_data_path)

    if not train_data_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_data_path}")

    if output_dir is None:
        output_dir = Path.cwd() / f"training_results_{reward_mode}"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cls = _load_env_class(archetype, custom_env_module=custom_env_module)
    env = env_cls(str(train_data_path), reward_mode=reward_mode)

    metrics_path = output_dir / f"{reward_mode}_metrics.csv"
    callback = SaveMetricsCallback(metrics_path, verbose=1)

    model = PPO("MlpPolicy", env, ent_coef=ent_coef, verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model_path = output_dir / f"{reward_mode}_agent.zip"
    model.save(model_path)

    if print_summary:

        try:
            obs, info = env.reset()
        except Exception:

            obs = env.reset()
            info = {}

        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated


        try:
            env_module_name = type(env).__module__
            env_module = __import__(env_module_name,
                                    fromlist=["print_intervention_values"])

            if hasattr(env_module, "print_intervention_values"):
                print("\n最终干预方案：")
                env_module.print_intervention_values(obs)
            else:
                print("\n[提示] 当前 ENV 模块中未找到 print_intervention_values，跳过干预方案打印。")

        except Exception as e:

            print(f"\n[提示] 打印干预方案时出现问题：{e}")


        print("\n测试信息：", info)
        print("累计奖励：", total_reward)

    return str(model_path)


__all__ = ["train_msw_agent"]
