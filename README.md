# SustainCity-IAgent-Framework
Sustainable City Intelligent-Agent Decision Framework, initially developed for municipal solid waste management and designed to be extensible to broader urban sustainability applications.

---

## 1. Overview

`SustainCity-IAgent-Framework` is a Python package that provides:

- **Reinforcement-learning (RL) environments** for six municipal solid waste (MSW) management archetypes  
  (`JPN`, `EU`, `US`, `CHN`, `IND`, `Permissive`);
- A **training interface** to learn intelligent agents (IAgents) under different intelligent-agent scenarios (IAS);
- A **prediction interface** to directly apply **pre-trained IAgents** or user-trained agents to country-level data.

The package was developed for the experiments in our MSW intelligent management study.  
Although the initial case is MSW, the design and APIs are intended to be extensible to other urban sustainability applications.

---

## 2. Installation

### 2.1 From PyPI (recommended for users)

To install the package from PyPI, run:

```bash
pip install sustaincity-iagent
```

This command installs:

- `sustaincity_iagent` (this package);
- `gymnasium`, `stable-baselines3`, `pandas`, `numpy`, `matplotlib` and other required libraries.

> **Note:** The package has been tested with Python 3.11.  
> We recommend using a fresh virtual environment (e.g. `conda` or `venv`) before installation.

### 2.2 From source (for developers)

To work with the latest development version or modify the framework locally, clone the repository and install it in **editable** mode:

```bash
git clone https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework.git
cd SustainCity-IAgent-Framework
python -m pip install -e .
```

The `-e` flag (editable mode) allows you to modify the source code in this folder and immediately test the changes without reinstalling the package.

## 3. Package structure

After installation, the package layout is (simplified):

```text
sustaincity_iagent/
  __init__.py
  config.py                 # IAS–reward mapping, archetype & SSP lists
  train.py                  # training API
  predict.py                # prediction API
  env/
    ENV_JPN.py
    ENV_EU.py
    ENV_US.py
    ENV_CHN.py
    ENV_IND.py
    ENV_Permissive.py
  IAgents/
    JPN/
      Scenario_I.zip
      ...
      Scenario_IX.zip
    EU/
      ...
    US/
    CHN/
    IND/
    Permissive/
  data/
    train_data.xlsx # example training data
    country_data.xlsx       # country-level predicted MSW amounts and fractions across SSPs
```

At the top level of the repository, you will also find:

- `pyproject.toml` – packaging configuration;  
- `LICENSE` – Apache 2.0 license;  
- `README.md` – this documentation file.

## 4. Intelligent-Agent Scenarios (IAS) and reward modes

Internally, each Intelligent-Agent Scenario (IAS) is implemented as a **reward mode** in the RL environments.  
The mapping is defined in `sustaincity_iagent.config.IAS_TO_REWARD_MODE`:

| IAS label     | Reward mode string           |
|--------------|------------------------------|
| Scenario I   | `carbon`                     |
| Scenario II  | `carbon_cost`                |
| Scenario III | `carbon_energy`              |
| Scenario IV  | `carbon_resource`            |
| Scenario V   | `carbon_biomass_led`         |
| Scenario VI  | `carbon_incineration_led`    |
| Scenario VII | `ssp3_weighted_carbon`       |
| Scenario VIII| `ssp4_weighted_multi`        |
| Scenario IX  | `ssp5_cost_service_led`      |

Users only need to choose the IAS label (e.g. `"Scenario VIII"`); the framework automatically translates it into the corresponding internal `reward_mode` string.

The list of supported archetypes and SSPs is also defined in `config.py`:

- `SUPPORTED_ARCHETYPES = ["JPN", "EU", "US", "CHN", "IND", "Permissive"]`
- `SUPPORTED_SSP = ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]`

These constants are exposed to the user and can be imported as:

```python
from sustaincity_iagent.config import IAS_TO_REWARD_MODE, SUPPORTED_ARCHETYPES, SUPPORTED_SSP
```

## 5. Quick start

### 5.1 Training your own intelligent agent

This example trains a PPO agent for the **CHN archetype** under **Scenario VIII**  
(IAS = Scenario VIII → reward mode `ssp4_weighted_multi`), using the example training data
packaged in `data/train_data_example.xlsx`.

We recommend running the example from a normal working directory (not inside the repository),
after you have installed the package via:

- `pip install sustaincity-iagent`, or
- `python -m pip install -e .` (from the cloned repository).

```python
from sustaincity_iagent import train_msw_agent
from sustaincity_iagent.config import IAS_TO_REWARD_MODE

# Optional: inspect the IAS → reward_mode mapping
print(IAS_TO_REWARD_MODE["Scenario VIII"])  # 'ssp4_weighted_multi'

model_path = train_msw_agent(
    archetype="CHN",          # one of: JPN, EU, US, CHN, IND, Permissive
    ias="Scenario VIII",      # IAS label, see Section 4
    # reward_mode=None,       # usually let the function infer this from IAS
    # train_data_path=None,   # use packaged example train_data_example.xlsx
    total_timesteps=200_000,  # training steps (adjust as needed)
    ent_coef=0.005,           # PPO entropy coefficient
    # output_dir=None,        # default: ./training_results_<reward_mode>
    # custom_env_module=None, # advanced: custom ENV module (optional)
    print_details=True        # print detailed information after training (default True)
)

print("Trained model saved at:", model_path)
```

**What this call does:**

1. **Selects the environment** based on `archetype`  
   e.g. `archetype="CHN"` → `sustaincity_iagent.env.ENV_CHN.MSWEnv`.
2. **Maps the IAS to an internal reward mode** using `IAS_TO_REWARD_MODE`.
3. **Loads training data**:
   - If `train_data_path` is `None`, the built-in example `data/train_data_example.xlsx`
     is used (for demonstration and testing);
   - You can provide your own training data file (see Section 5.1.1).
4. **Trains a PPO agent** with Stable-Baselines3 for the specified number of timesteps,
   and logs training metrics via a callback.
5. **Saves the trained model** as a `.zip` file and returns its path.
6. If `print_details=True` (default), the function:
   - prints a textual overview of the intervention scheme structure; and
   - runs one deterministic evaluation episode with the trained policy; and
   - prints the final intervention decision, episode info, and cumulative reward.

---

#### 5.1.1 Training data – using your own file

You can replace the packaged example training data with your own file, as long as the
structure (columns, units, semantics) is consistent with `data/train_data_example.xlsx`:

```python
model_path = train_msw_agent(
    archetype="CHN",
    ias="Scenario II",
    train_data_path=r"C:\path\to\my_train_data.xlsx",  # custom training data
    total_timesteps=300_000,
)
```

**Important:**

- Each row in your custom `train_data.xlsx` should represent one sample
  (e.g., a country, a city, or a scenario);
- Column names and units should follow the same convention as
  `train_data_example.xlsx`, as described in the paper and supplementary materials.

---

#### 5.1.2 Controlling detailed logging

The training interface provides a simple switch to control how much detail is printed
after training:

- `print_details=True` (default)  
  - prints:
    - a structural description of the intervention scheme; and
    - one evaluation episode’s final intervention decision, info, and cumulative reward.
- `print_details=False`  
  - leaves only the regular Stable-Baselines3 training logs
    (e.g., `rollout/ep_rew_mean`) on the console.

Example:

```python
# Quiet training: no post-training intervention summary is printed
model_path = train_msw_agent(
    archetype="CHN",
    ias="Scenario VIII",
    total_timesteps=200_000,
    print_details=False,
)
```

---

### 5.2 Using pre-trained IAgents for prediction

The package ships with pre-trained PPO agents (IAgents) covering all combinations of:

- 6 archetypes: `JPN`, `EU`, `US`, `CHN`, `IND`, `Permissive`; and  
- 9 IAS: Scenario I – Scenario IX.

These models are stored as `.zip` files under `IAgents/<archetype>/Scenario_X.zip`, e.g.:

- `IAgents/CHN/Scenario_VIII.zip`
- `IAgents/JPN/Scenario_I.zip`
- …

You can directly apply these pre-trained IAgents to country-level data to generate
optimal intervention portfolios for a given SSP pathway:

```python
from sustaincity_iagent import predict_msw_scenario

df = predict_msw_scenario(
    archetype="CHN",        # matches env/IAgents subdirectory
    ias="Scenario VIII",    # IAS label
    ssp="SSP4",             # sheet name in country_data.xlsx
    # use_pretrained=True,  # default: True (use packaged IAgents)
    # country_data_path=None,  # default: data/country_data.xlsx in the package
)

# Save prediction results to Excel
save_file = r"C:\Users\YourName\country_predict_results_ssp4_weighted_multi_CHN_SSP4.xlsx"
df.to_excel(save_file, index=False)
print("Prediction results saved to:", save_file)
```

**What this call does:**

1. **Reads country-level data** from `country_data.xlsx` (or a user-provided file):
   - Uses the specified `ssp` (e.g., `"SSP4"`) as the sheet name;
   - Accepts the following column names:
     - country identifier: `country` or `iso3c` (internally normalised to `country`);
     - archetype label: `archetype` or `baseline_mode`
       (internally normalised to `archetype`).
2. For each row (country/region), **selects the appropriate archetype** and constructs
   the corresponding environment.
3. **Loads the appropriate pre-trained IAgent** based on `archetype` and `ias`,
   e.g. `CHN + Scenario VIII` → `IAgents/CHN/Scenario_VIII.zip`.
4. Runs the agent in the environment to generate the optimal intervention portfolio
   and associated indicators.
5. Aggregates the results for all rows into a `pandas.DataFrame` and returns it.

---

#### 5.2.1 Using your own trained model for prediction

If you prefer to use a model you have just trained (instead of the packaged IAgents),
you can pass its path to `predict_msw_scenario`:

```python
from pathlib import Path
from sustaincity_iagent import train_msw_agent, predict_msw_scenario

# 1. Train your own agent
model_path = train_msw_agent(
    archetype="CHN",
    ias="Scenario VIII",
    total_timesteps=200_000,
)

# 2. Use this model for country-level prediction
df = predict_msw_scenario(
    archetype="CHN",
    ias="Scenario VIII",
    ssp="SSP4",
    use_pretrained=False,        # tell the function to use a user-provided model
    model_path=Path(model_path), # path to the .zip file just trained
    # country_data_path=None,    # optionally provide a custom country data file
)

df.to_excel(r"C:\Users\YourName\my_CHN_SSP4_custom_agent.xlsx", index=False)
print("Prediction results (custom agent) saved.")
```

In this mode:

- `predict_msw_scenario` **ignores** the packaged pre-trained models in `IAgents/`;  
- it uses the agent specified by `model_path` for all countries/regions;  
- all other steps (country data handling, environment construction, result aggregation)
  remain the same.

## 6. Country-level data for prediction

The package includes a default country-level dataset  
`data/country_data.xlsx`, which is used by `predict_msw_scenario`
when `country_data_path` is not specified.

### 6.1 Origin of the default dataset

The default country-level data are derived from a  
**random-forest + SSPs**–based global MSW generation projection
framework developed in:

> Cao, Q. et al.  
> *Expediting co-benefits of tailored municipal solid waste management strategies globally.*  
> Nat. Sustain. 8, 1164–1176 (2025).

In that work, country-level MSW quantities and related drivers are
projected under multiple Shared Socioeconomic Pathways (SSPs) using
a machine-learning model (random forest) calibrated on multi-source
socioeconomic, demographic and waste-generation statistics.

`country_data.xlsx` provides, for each SSP (separate Excel sheet):

- projected MSW generation and related variables at the country level;  
- a consistent **archetype assignment** for each country;  
- additional explanatory variables used in the analyses.

The package uses this dataset as a **ready-to-use example** to  
reproduce and extend the MSW intelligent-management scenarios.

### 6.2 File structure

The default `country_data.xlsx` follows these conventions:

- **Sheets**  
  - One sheet per SSP: `"SSP1"`, `"SSP2"`, `"SSP3"`, `"SSP4"`, `"SSP5"`.
  - The `ssp` argument in `predict_msw_scenario` must match one of these sheet names.

- **Required columns** (per sheet)
  - Country identifier:
    - either `country` or `iso3c` (three-letter ISO code), and  
    - internally normalised to a `country` column.
  - Archetype label:
    - either `archetype` or `baseline_mode`, and  
    - internally normalised to an `archetype` column  
      (one of `JPN`, `EU`, `US`, `CHN`, `IND`, `Permissive`).

- **Optional columns**
  - Additional variables (e.g., MSW generation projections, socioeconomic drivers)  
    may be present; they are not strictly required by the code, but are part of the
    underlying modelling described in the Nat. Sustain. paper.

The helper function inside the package will automatically:

- rename `iso3c → country` if `country` is not present;  
- rename `baseline_mode → archetype` if `archetype` is not present.

If neither `country`/`iso3c` nor `archetype`/`baseline_mode` is found,
`predict_msw_scenario` will raise a descriptive error.

### 6.3 Using your own country-level MSW projections

You are encouraged to use your **own** MSW projection data (e.g., from other
models, updated SSP trajectories, or city-level extensions). To do so:

1. Prepare an Excel file with **one sheet per scenario/pathway**  
   (e.g., `"SSP1"`, `"SSP2"`, or any other set of pathway names).
2. For each sheet, ensure that at least the following columns exist:
   - Country identifier:
     - `country` or `iso3c` (three-letter ISO codes recommended).
   - Archetype:
     - `archetype` or `baseline_mode` with values in  
       `{JPN, EU, US, CHN, IND, Permissive}` (or a subset you actually use).
3. Keep any additional columns you need for your own analysis; they will be
   carried through in the output DataFrame.

You can then point `predict_msw_scenario` to your file:

```python
from sustaincity_iagent import predict_msw_scenario

df = predict_msw_scenario(
    archetype="CHN",
    ias="Scenario VIII",
    ssp="SSP4",
    country_data_path=r"C:\path\to\my_country_data.xlsx",
)
```

As long as the **sheet name** (`ssp`) and the **column structure** match the
rules above, the framework will treat your file exactly like the packaged
`country_data.xlsx`, but using your own MSW projections.

## 7. Custom environments

Advanced users may wish to define their own RL environments, for example:

- to represent a specific city or region with customised technical options;
- to experiment with alternative intervention portfolios and constraints;
- to extend the framework to non-MSW applications within urban sustainability.

The framework is designed to accommodate such extensions with minimal changes.

### 7.1 Implementing a custom environment

To define a new environment, create a new module (e.g. `ENV_MYCITY.py`) following
the structure of the existing `ENV_*.py` files in `sustaincity_iagent.env`:

1. **Create a new file** (in your own project or as an extension of this package):

   ```text
   my_project/
     ENV_MYCITY.py
   ```

2. **Define an `MSWEnv` class** (or similarly structured class) inside `ENV_MYCITY.py`:

   - It should inherit from `gymnasium.Env` (or be API-compatible);
   - It must implement at least:
     - `__init__(self, data_file: str, reward_mode: str, ...)`;
     - `reset(self, *, seed=None, options=None) -> (obs, info)`;
     - `step(self, action) -> (obs, reward, terminated, truncated, info)`;
   - `reward_mode` is a string that determines which reward function is used  
     (e.g., `"carbon"`, `"carbon_cost"`, `"ssp4_weighted_multi"`, etc.).

3. (Optional but recommended) **Expose helper functions** for reporting:

   - `print_intervention_scheme()` – prints a human-readable description of
     the intervention portfolio (IP) structure;
   - `print_intervention_values(state: np.ndarray)` – prints the final IP values
     given a state vector.

   These functions are used by `train_msw_agent(print_details=True)` to produce
   post-training summaries.

You can inspect `ENV_CHN.py`, `ENV_US.py` etc. as concrete templates for how
to structure the observation space, action space, and reward logic.

### 7.2 Training with a custom environment

Once your custom environment module is implemented and importable, you can use
it with the high-level training API by specifying the `custom_env_module`
argument in `train_msw_agent`.

For example, suppose you have:

```text
my_project/
  ENV_MYCITY.py  # defines MSWEnv, print_intervention_scheme, print_intervention_values
  my_train_data.xlsx
```

In your training script:

```python
from sustaincity_iagent import train_msw_agent

model_path = train_msw_agent(
    archetype="CHN",                 # can still use an existing archetype tag
    ias="Scenario III",              # choose an IAS as usual
    train_data_path="my_train_data.xlsx",
    custom_env_module="ENV_MYCITY",  # dotted path to your custom module
    total_timesteps=150_000,
)
```

Under the hood:

- instead of `sustaincity_iagent.env.ENV_CHN.MSWEnv`, the framework imports
  `ENV_MYCITY.MSWEnv` as the environment class;
- all other logic (reward_mode handling, training loop, model saving) remains the same,
  as long as your `MSWEnv` constructor signature and `step`/`reset` methods match.

### 7.3 Prediction with custom environments

The default `predict_msw_scenario` function is primarily designed to work with the
packaged archetype environments (`ENV_JPN`, `ENV_EU`, …, `ENV_Permissive`) and
their corresponding pre-trained agents.

If you define a fully custom environment and train a custom agent:

- you can still use the internal utilities (e.g. how we loop over countries, how we
  structure outputs) as a reference; and
- for maximum flexibility, we recommend writing a thin wrapper script that:

  1. reads your own country-level data file;
  2. constructs your custom environment for each country/region;
  3. loads your trained agent; and
  4. runs prediction and aggregates results into a `pandas.DataFrame`.

This mirrors what `predict_msw_scenario` does internally, but avoids constraining
your environment to the exact archetype/IAS conventions used in this package.

Future versions of the framework may expose additional hooks for custom environments
within `predict_msw_scenario`. For now, the recommended pattern is:

- **Training** – use `train_msw_agent(..., custom_env_module="ENV_MYCITY")`;  
- **Prediction** – implement a short, project-specific driver script that loops over
  your custom data and uses your custom environment and agent.

## 8. Logging and reproducibility

The framework is designed to be compatible with standard RL logging and
reproducibility practices:

- **RL backend**  
  - Training uses `stable-baselines3.PPO`, which provides:
    - standard console logs (e.g., `rollout/ep_rew_mean`, `train/entropy_loss`);
    - support for integration with TensorBoard (if you enable it separately).

- **Metrics callback**  
  - `train_msw_agent` uses an internal callback (`SaveMetricsCallback`) to record
    selected metrics over time (e.g., timesteps, mean episode reward) and saves them
    to a CSV file under the chosen `output_dir`.
  - You can parse these CSV files to recreate training curves or perform
    post-hoc analysis of training dynamics.

- **Model saving**  
  - Trained agents are saved as `.zip` files using the standard Stable-Baselines3
    format (compatible with `PPO.load(...)`).
  - These model files can be loaded both via this package’s APIs and directly via
    `stable-baselines3` in your own scripts.

- **Reproducibility tips**  
  - Fix random seeds (e.g., via `numpy` and the RL backend) if you need strict
    replicability of a single run.
  - Use `python -m pip install -e .` when working from a cloned repository so that
    code and experiments remain synchronised.
  - When publishing results, we recommend:
    - archiving both code and data in a persistent repository (e.g., Zenodo);
    - documenting the exact `sustaincity-iagent` version (e.g., `v1.0.0`) and
      key hyperparameters.

---

## 9. Citation

If you use `SustainCity-IAgent-Framework`, the packaged IAgents, or derived
analyses in your work, please cite:

1. The **methodological paper1** describing the global MSW amounts:

   > Cao, Q. et al.  
   > *Expediting co-benefits of tailored municipal solid waste management strategies globally.*  
   > Nat. Sustain. 8, 1164–1176 (2025).

2. The **methodological paper2** describing the global MSW IAgents:

   > Cao, Q. et al.  
   > *To be completed*  
   > To be completed

3. The **code archive**:

   > Cao, Q.  
   > *SustainCity-IAgent-Framework: Sustainable City Intelligent-Agent Decision Framework* (v1.0.0).  
   > Zenodo. https://doi.org/10.5281/zenodo.17853494


---

## 10. License

This project is distributed under the **Apache License 2.0**.

In summary, this means you are free to:

- use the code for academic, non-commercial, and commercial purposes;
- modify and redistribute the code, including in derivative works;

Please refer to the `LICENSE` file in the repository for the full legal text.

---

## 11. Contact

For questions, suggestions, or bug reports, please:

- open an issue on the GitHub repository:  
  `https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/issues`

For enquiries specifically related to the scientific study or collaboration
opportunities, please contact: `caoql23@mails.jlu.edu.cn`
