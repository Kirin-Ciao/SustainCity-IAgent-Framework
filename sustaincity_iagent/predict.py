from pathlib import Path
from typing import Optional, Dict, Any

import copy
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from importlib import import_module, resources

from .config import (
    IAS_TO_REWARD_MODE,
    SUPPORTED_ARCHETYPES,
    DEFAULT_TRAIN_DATA_FILE,
    DEFAULT_COUNTRY_DATA_FILE,
)


def _load_env_class(archetype: str, custom_env_module: Optional[str] = None):
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


def _get_default_train_data_path() -> Path:
    base = Path(resources.files("sustaincity_iagent"))
    return base / DEFAULT_TRAIN_DATA_FILE


def _get_default_country_data_path() -> Path:
    base = Path(resources.files("sustaincity_iagent"))
    return base / DEFAULT_COUNTRY_DATA_FILE


def _load_model(
    archetype: str,
    ias: str,
    reward_mode: str,
    model_path: Optional[str],
    use_builtin: bool,
) -> PPO:
    if use_builtin:
        # packaged pre-trained agents live in IAgents/<arch>/<Scenario_*.zip>
        base = Path(resources.files("sustaincity_iagent"))
        agent_file = base / "IAgents" / archetype / f"{ias.replace(' ', '_')}.zip"
        if not agent_file.exists():
            raise FileNotFoundError(
                f"Built-in agent not found: {agent_file}. "
                "Check archetype/IAS or set use_builtin=False and provide model_path."
            )
        model_path = agent_file
    else:
        if model_path is None:
            # default: assume user trained with train_msw_agent (same reward_mode)
            model_path = (
                Path.cwd()
                / f"training_results_{reward_mode}"
                / f"{reward_mode}_agent.zip"
            )
        else:
            model_path = Path(model_path)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return PPO.load(str(model_path))


import pandas as pd  

def _prepare_country_dataframe(country_data_path, sheet_name=None) -> pd.DataFrame:

    df = pd.read_excel(country_data_path, sheet_name=sheet_name)

    rename_map = {}


    if "archetype" not in df.columns:
        if "baseline_mode" in df.columns:
            rename_map["baseline_mode"] = "archetype"

    if "country" not in df.columns:
        if "iso3c" in df.columns:
            rename_map["iso3c"] = "country"

    if rename_map:
        df = df.rename(columns=rename_map)

    missing = [c for c in ["country", "archetype"] if c not in df.columns]
    if missing:
        raise ValueError(
            "country_data.xlsx is missing required columns: "
            + ", ".join(missing)
            + " (accepted names: country/iso3c and archetype/baseline_mode)"
        )

    return df



def _comp_nudge(action: np.ndarray, comp: Dict[str, float]) -> np.ndarray:
    """Small post-perturbation of the raw action based on waste composition.

    This is a simplified, documented version of the heuristic used in the paper.
    It is intentionally conservative; users can always bypass it by turning it off.
    """
    a = action.astype(np.float32).copy()

    s = 0.12  # global scale
    # Example: adjust biomass-related actions when food fraction is high/low
    start = 3
    z_food = float(comp["food"]) - 0.1  
    a[start + 2] += 1.00 * s * z_food
    a[start + 3] += 0.80 * s * z_food
    a[start + 0] -= 0.60 * s * z_food
    a[start + 1] -= 0.20 * s * z_food

    return np.clip(a, -0.2, 0.2)


def predict_msw_scenario(
    archetype: str,
    ias: str,
    ssp: str,
    *,
    reward_mode: Optional[str] = None,
    country_data_path: Optional[str] = None,
    sheet_name: Optional[str] = None,
    model_path: Optional[str] = None,
    use_builtin_agent: bool = True,
    save_to: Optional[str] = None,
    custom_env_module: Optional[str] = None,
    apply_comp_nudge: bool = True,
) -> pd.DataFrame:
    

    if reward_mode is None:
        try:
            reward_mode = IAS_TO_REWARD_MODE[ias]
        except KeyError as exc:
            raise ValueError(
                f"Unknown IAS '{ias}'. Please choose from: "
                f"{', '.join(IAS_TO_REWARD_MODE.keys())} "
                "or explicitly pass a reward_mode."
            ) from exc

    if country_data_path is None:
        country_data_path = _get_default_country_data_path()
    else:
        country_data_path = Path(country_data_path)
    if sheet_name is None:
        sheet_name = ssp

    if not country_data_path.exists():
        raise FileNotFoundError(f"country_data.xlsx not found: {country_data_path}")

    df = _prepare_country_dataframe(country_data_path, sheet_name=sheet_name)

    env_cls = _load_env_class(archetype, custom_env_module=custom_env_module)
    env = env_cls(str(_get_default_train_data_path()), reward_mode=reward_mode)

    model = _load_model(archetype, ias, reward_mode, model_path, use_builtin_agent)

    results = []
    comp_cols = ["food", "glass", "metal", "paper", "plastic", "rubber", "wood", "yard", "other"]

    for _, row in df.iterrows():
        total_mass = float(row["total_mass"])
        composition = {c: float(row[c]) for c in comp_cols}

        env.total_mass = total_mass
        env.composition = copy.deepcopy(composition)
        env.composition_vector = np.array([composition[c] for c in comp_cols], dtype=np.float32)

        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)

        if apply_comp_nudge:
            action = _comp_nudge(action, composition)

        obs, reward, terminated, truncated, info = env.step(action)

        def fmt(x: Any):
            return None if x is None else float(x)

        result = {
            "country": row.get("country"),
            "archetype": archetype,
            "ssp": ssp,
            "ias": ias,
            "reward_mode": reward_mode,
            "total_mass": total_mass,
            "total_reward": fmt(info.get("episode", {}).get("r", reward)),
            "buffer": fmt(info.get("buffer")),
            "biomass_score": fmt(info.get("biomass_score")),
            "incineration_score": fmt(info.get("incineration_score")),
            "incineration_norm": fmt(info.get("incineration_norm")),
            "fallback_on": info.get("fallback_on"),
        }
        results.append(result)

    result_df = pd.DataFrame(results)

    # 如果指定了保存路径，就存一份；否则使用默认命名并保存到当前工作目录
    if save_to is None:
        filename = f"country_predict_results_{reward_mode}_{archetype}_{ssp}.xlsx"
        save_path = Path.cwd() / filename
    else:
        save_path = Path(save_to)

    try:
        result_df.to_excel(save_path, index=False)
        print(f"预测结果已保存到：{save_path}")
    except Exception as e:
        print(f"[提示] 无法保存预测结果到 {save_path}：{e}")

    return result_df


__all__ = ["predict_msw_scenario"]
