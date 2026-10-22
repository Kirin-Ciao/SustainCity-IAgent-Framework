# Alternative primary objectives: ACP S1–S3

This research branch of SustainCity-IAgent-Framework provides **18 pretrained PPO agents** and the training implementation for three alternative primary objectives in municipal solid waste management. It builds on the existing `MSW` branch; the original ACP I–IX environments and pretrained agents are retained unchanged.

| Objective | Primary metric | Direction | Unit per tonne MSW |
|---|---|---|---|
| ACP S1 — Economy-primary | Net cost = gross cost − recovered-product economic benefits | Minimize | USD2020/t |
| ACP S2 — Energy-primary | Net energy balance = energy input − energy output | Minimize; negative means surplus | MJ/t |
| ACP S3 — Resource-primary | Physical resource-product output; excludes energy products and Cu-equivalent aggregation | Maximize | kg/t |

Each reward uses an episode-fixed reference, a frozen positive scale and a single-objective tanh transform.

## Pretrained agents

| Archetype | Economy | Energy | Resources |
|---|---|---|---|
| JPN | [ACP S1](experiments/alternative_primary_objectives/models/JPN/ACP_S1.zip) | [ACP S2](experiments/alternative_primary_objectives/models/JPN/ACP_S2.zip) | [ACP S3](experiments/alternative_primary_objectives/models/JPN/ACP_S3.zip) |
| EU | [ACP S1](experiments/alternative_primary_objectives/models/EU/ACP_S1.zip) | [ACP S2](experiments/alternative_primary_objectives/models/EU/ACP_S2.zip) | [ACP S3](experiments/alternative_primary_objectives/models/EU/ACP_S3.zip) |
| US | [ACP S1](experiments/alternative_primary_objectives/models/US/ACP_S1.zip) | [ACP S2](experiments/alternative_primary_objectives/models/US/ACP_S2.zip) | [ACP S3](experiments/alternative_primary_objectives/models/US/ACP_S3.zip) |
| CHN | [ACP S1](experiments/alternative_primary_objectives/models/CHN/ACP_S1.zip) | [ACP S2](experiments/alternative_primary_objectives/models/CHN/ACP_S2.zip) | [ACP S3](experiments/alternative_primary_objectives/models/CHN/ACP_S3.zip) |
| IND | [ACP S1](experiments/alternative_primary_objectives/models/IND/ACP_S1.zip) | [ACP S2](experiments/alternative_primary_objectives/models/IND/ACP_S2.zip) | [ACP S3](experiments/alternative_primary_objectives/models/IND/ACP_S3.zip) |
| Permissive | [ACP S1](experiments/alternative_primary_objectives/models/Permissive/ACP_S1.zip) | [ACP S2](experiments/alternative_primary_objectives/models/Permissive/ACP_S2.zip) | [ACP S3](experiments/alternative_primary_objectives/models/Permissive/ACP_S3.zip) |

Every archive is the unchanged **formal final model**, trained with seed **1729** for **200,704 actual steps** (200,000 nominal steps rounded to full PPO rollouts). The [model manifest](experiments/alternative_primary_objectives/models/manifest.json) records SHA256, configuration path, objective version, frozen scale and original run identifier for each file. `Permissive` is the sixth WMS archetype; it is not an alias for the IND environment.

## Installation from this branch

Use Python 3.12 and a separate virtual environment. Training was performed with Stable-Baselines3 2.7.0, Gymnasium 1.1.1 and PyTorch 2.8.0 on CPU. The recorded dependency versions are in `experiments/alternative_primary_objectives/configs/requirements-pinned.txt`; they describe the training environment rather than a cross-platform lock file.

```powershell
git clone --branch alternative-primary-objectives https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework.git
Set-Location SustainCity-IAgent-Framework
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
# CPU PyTorch, matching the training backend:
& .\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -r .\experiments\alternative_primary_objectives\configs\requirements-pinned.txt
& .\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Use this source checkout in editable mode: it supplies the additional runtime modules and product registry. The existing PyPI release is not a distribution of these new agents. No package version or PyPI release is created by this branch.

## Load a pretrained policy

The branch preserves byte-level checksums for the released models and training contracts.

Run from the repository root. This example only loads the model and performs a deterministic 20-step episode; it does not update the policy.

```python
import json
from pathlib import Path
from stable_baselines3 import PPO
from sustaincity_iagent.alternative_factory import make_alternative_env

root = Path("experiments/alternative_primary_objectives")
config = json.loads((root / "configs/JPN_s1.json").read_text(encoding="utf-8"))
env = make_alternative_env(config["archetype"], config["acp"], config["scale"])
model = PPO.load(root / "models/JPN/ACP_S1.zip", device="cpu")
observation, info = env.reset(seed=88000)
for _ in range(20):
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
metrics = info["metrics"]
env.close()
```

The adapter uses 39-dimensional observations and the original 29-dimensional continuous action space. Fixed-composition inputs can be supplied through the adapter's documented `reset(options={"composition": ..., "total_mass": ...})` interface. Policy mass must remain within the calibrated 0.5–2 tonne range.

## Historical training records

The [training-log index](experiments/alternative_primary_objectives/training_logs/README.md) links each released agent to its original formal run, raw diagnostic CSVs, monitoring history and configuration metadata. The index documents path-only privacy edits and original/published SHA256 checksums.

## Train, monitor and resume

The 18 formal configuration files record the parameters used to train the released models. Training reuses the six existing MEE ENV modules and `train_data.xlsx` from `MSW`; the additional adapter implements the ACP S1–S3 rewards, net-metric accounting and compatible training inputs.

```powershell
$exp = '.\experiments\alternative_primary_objectives'
# Check all 18 configurations and child-process imports; no learning:
& "$exp\run_training.ps1" -DryRun
# Train all 18 into NEW local runs (only run when desired):
& "$exp\run_training.ps1"
# Or select one combination:
& "$exp\run_training.ps1" -Archetype JPN -ACP 'ACP S1'
# Continue an interrupted training run created by this checkout:
& "$exp\run_training.ps1" -Resume
# In another terminal, monitor the local training output:
& "$exp\check_training.ps1" -Watch
```

The Python CLI is `python -m sustaincity_iagent.alternative_training batch --config-dir experiments/alternative_primary_objectives/configs --output-root experiments/alternative_primary_objectives/runs`. Add `--dry-run` for the preflight check or `--resume` to restore an eligible latest checkpoint. Published final archives are under `models/`; new training writes under `runs/` and does not replace these archives. Historical training logs are available in [training_logs](experiments/alternative_primary_objectives/training_logs/README.md). Checkpoint bundles are not included; a final model archive alone is not an interrupted-run resume bundle.

| PPO parameter | Value |
|---|---|
| Policy / architecture | MlpPolicy, pi/vf [64, 64] |
| Nominal timesteps / rollout | 200000 / 2048 |
| Batch size / epochs | 64 / 10 |
| Learning rate | 3e-4 |
| Gamma / GAE lambda | 0.99 / 0.95 |
| Clip range / entropy coefficient | 0.2 / 0.1 |
| Episode horizon / normalization | 20 decision steps / no VecNormalize |
| Seed / device | 1729 / CPU, one Torch thread |

Training continuously records episode reward, approximate KL and policy-gradient loss, saves checkpoints at complete update boundaries, and validates the final policy.

The repository's existing Apache-2.0 license continues to apply. Original default-branch documentation is available on the `MSW` branch.
