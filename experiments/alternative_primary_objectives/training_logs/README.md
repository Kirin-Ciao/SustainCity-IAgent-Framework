# Retained formal training records — ACP S1–S3

This directory contains retained historical records for the 18 released formal PPO agents. The logs were not regenerated from figures or from new model executions. The disclosure audit on 2026-10-22 only read files, compared hashes, inspected ZIP JSON metadata and checked existing recorded numbers. No training, resume or policy inference was executed for this disclosure.

## Run and model index

All tasks used seed 1729, 200000 nominal steps, 200704 actual environment transitions, 2048 steps per rollout, 98 rollout updates, and 10 epochs per update (980 recorded `n_updates`). Each directory contains 98 rollout records, 10035 completed episodes, 934 monitoring events and one 98-row SB3 segment. Episodes end every 20 decisions; the final four transitions do not form another complete episode. They are present in the final rollout, not missing data.

| Archetype | ACP | Original source run ID | Historical records | Released model | Actual steps |
|---|---|---|---|---|---|
| JPN | ACP S1 | `20261013T024809Z_JPN_ACP_S1_seed1729_7772c909` | [JPN/ACP_S1](JPN/ACP_S1/) | [model](../models/JPN/ACP_S1.zip) | 200704 |
| JPN | ACP S2 | `20261013T025123Z_JPN_ACP_S2_seed1729_a9c355c7` | [JPN/ACP_S2](JPN/ACP_S2/) | [model](../models/JPN/ACP_S2.zip) | 200704 |
| JPN | ACP S3 | `20261013T025436Z_JPN_ACP_S3_seed1729_7a20971e` | [JPN/ACP_S3](JPN/ACP_S3/) | [model](../models/JPN/ACP_S3.zip) | 200704 |
| EU | ACP S1 | `20261013T025757Z_EU_ACP_S1_seed1729_d5f20ea3` | [EU/ACP_S1](EU/ACP_S1/) | [model](../models/EU/ACP_S1.zip) | 200704 |
| EU | ACP S2 | `20261013T030111Z_EU_ACP_S2_seed1729_7825675c` | [EU/ACP_S2](EU/ACP_S2/) | [model](../models/EU/ACP_S2.zip) | 200704 |
| EU | ACP S3 | `20261013T030423Z_EU_ACP_S3_seed1729_74186920` | [EU/ACP_S3](EU/ACP_S3/) | [model](../models/EU/ACP_S3.zip) | 200704 |
| US | ACP S1 | `20261013T030727Z_US_ACP_S1_seed1729_fd731efb` | [US/ACP_S1](US/ACP_S1/) | [model](../models/US/ACP_S1.zip) | 200704 |
| US | ACP S2 | `20261013T031029Z_US_ACP_S2_seed1729_98dd4605` | [US/ACP_S2](US/ACP_S2/) | [model](../models/US/ACP_S2.zip) | 200704 |
| US | ACP S3 | `20261013T031330Z_US_ACP_S3_seed1729_61f46271` | [US/ACP_S3](US/ACP_S3/) | [model](../models/US/ACP_S3.zip) | 200704 |
| CHN | ACP S1 | `20261013T031633Z_CHN_ACP_S1_seed1729_c666c776` | [CHN/ACP_S1](CHN/ACP_S1/) | [model](../models/CHN/ACP_S1.zip) | 200704 |
| CHN | ACP S2 | `20261013T031935Z_CHN_ACP_S2_seed1729_690e8aa0` | [CHN/ACP_S2](CHN/ACP_S2/) | [model](../models/CHN/ACP_S2.zip) | 200704 |
| CHN | ACP S3 | `20261013T032232Z_CHN_ACP_S3_seed1729_db6ac718` | [CHN/ACP_S3](CHN/ACP_S3/) | [model](../models/CHN/ACP_S3.zip) | 200704 |
| IND | ACP S1 | `20261013T032527Z_IND_ACP_S1_seed1729_9d1984fd` | [IND/ACP_S1](IND/ACP_S1/) | [model](../models/IND/ACP_S1.zip) | 200704 |
| IND | ACP S2 | `20261013T032824Z_IND_ACP_S2_seed1729_e85febf6` | [IND/ACP_S2](IND/ACP_S2/) | [model](../models/IND/ACP_S2.zip) | 200704 |
| IND | ACP S3 | `20261013T033117Z_IND_ACP_S3_seed1729_69a03c63` | [IND/ACP_S3](IND/ACP_S3/) | [model](../models/IND/ACP_S3.zip) | 200704 |
| Permissive | ACP S1 | `20261013T033413Z_Permissive_ACP_S1_seed1729_bd3676ad` | [Permissive/ACP_S1](Permissive/ACP_S1/) | [model](../models/Permissive/ACP_S1.zip) | 200704 |
| Permissive | ACP S2 | `20261013T033709Z_Permissive_ACP_S2_seed1729_badf2fdf` | [Permissive/ACP_S2](Permissive/ACP_S2/) | [model](../models/Permissive/ACP_S2.zip) | 200704 |
| Permissive | ACP S3 | `20261013T034002Z_Permissive_ACP_S3_seed1729_01b3fcbe` | [Permissive/ACP_S3](Permissive/ACP_S3/) | [model](../models/Permissive/ACP_S3.zip) | 200704 |

The [index](index.json) records model SHA256, published configuration path, frozen reward/metric versions and scales, runtime dependencies, row counts and audit results. The unchanged [released-model manifest](../models/manifest.json) is the primary source-to-model index.

## Files and interpretation

| File | Meaning |
|---|---|
| `rollouts.csv` | Custom logger **after** each PPO update; all 98 updates and all three requested diagnostics are present. The preferred analysis source. |
| `episodes.csv` | Monitor records for 10035 completed episodes; episode reward is the sum over the 20-step episode. Performance fields describe the terminal system state. |
| `sb3_segments/segment_000/progress.csv` | Original SB3 dump. `train/*` describes the **preceding** rollout update; the first row is blank because no update had yet been reported. The last update's training diagnostics are in `rollouts.csv`. Do not fill blanks or move values in this source file. |
| `progress_history.jsonl` | Append-only monitoring events: started, progress, ppo_update, checkpoint_saved, heldout_evaluation, completed. Same-timestep events are intentional. Null diagnostics before the first update are preserved. `resumed` is false throughout these 18 selected histories. |
| `progress.json` | Last monitoring snapshot; not the whole history. |
| `config.json` | Original effective formal training configuration, with only machine-specific path strings made portable. |
| `manifest.json` | Historical code/data/registry hashes, effective configuration and dependency versions. Historical Git commit metadata is unavailable (Git was not on PATH); no commit has been invented. |
| `completed.json`, `status.json` | Original completion state and historical file hashes. |
| `latest_checkpoint.json` | Original final checkpoint reference and manifest hash. This is provenance, not a distributed checkpoint bundle. |
| `validation.json` | Retained training-time finite-value/boundary/mass checks. A historical result, not a new validation or proof of optimality. |

## CSV field dictionary

All original names, rows and numeric representations are retained byte-for-byte in the CSVs.

| Fields | Definition / unit |
|---|---|
| `run_id`, `acp`, `archetype`, `seed` | Original run identity, primary-objective label, WMS archetype and random seed. |
| `metric_version`, `reward_version`, `scale` | Frozen accounting/reward contract and positive objective scale; not subsequently fitted to these curves. |
| `timesteps`, `time/total_timesteps` | Cumulative environment transitions, not optimizer epochs. X coordinates should use these recorded numbers. |
| `updates`, `train/n_updates` | Cumulative PPO epoch counter (10 per rollout), not the number of 2048-step rollouts. |
| `elapsed_seconds`, `time/time_elapsed` | Recorded elapsed wall time in seconds; the SB3 dump rounds/truncates its field. |
| `mean_episode_reward`, `rollout/ep_rew_mean` | Mean sum of rewards for the most recent 100 completed episodes. This differs from mean one-step reward. |
| `episode_reward_window` | Maximum completed-episode window size (100). |
| `length`, `rollout/ep_len_mean` | Episode length and mean episode length (decision steps). |
| `reward` | Completed-episode cumulative reward in `episodes.csv`. |
| `reward_mean`, `reward_min`, `reward_max` | Mean/minimum/maximum one-step rewards within the rollout. |
| `saturation_fraction` | Fraction of steps flagged by the frozen reward's saturation criterion. |
| `approx_kl`, `train/approx_kl` | Estimated policy KL divergence across an update; a stability diagnostic, not a performance outcome. |
| `policy_gradient_loss`, `train/policy_gradient_loss` | PPO policy objective loss; its absolute magnitude alone is not policy quality. |
| `value_loss`, `train/value_loss` | Value-function regression loss. |
| `entropy_loss`, `train/entropy_loss` | Negative entropy term reported by SB3. |
| `explained_variance`, `train/explained_variance` | Critic prediction statistic; may be negative. |
| `explained_variance_status` | Whether the explained-variance value was defined/reported. |
| `clip_fraction`, `train/clip_fraction` | Fraction of policy-ratio samples clipped by PPO. |
| `loss`, `train/loss` | Last reported total optimizer loss. |
| `fps`, `time/fps` | Recorded training transition throughput; differences between logger conventions are retained. |
| `time/iterations` | SB3 rollout iteration count. |
| `train/clip_range`, `train/learning_rate`, `train/std` | PPO clipping threshold, learning rate and policy action standard deviation. |
| `ghg`, `ghg_mean` | Net life-cycle GHG, kgCO2e/tMSW. `*_mean` averages states within the rollout. |
| `gross_cost`, `gross_cost_mean` | Gross cost, USD2020/tMSW. |
| `economic_benefits`, `economic_benefits_mean` | Recovered energy/resource revenue, USD2020/tMSW. |
| `net_cost`, `net_cost_mean` | Gross cost minus economic benefits, USD2020/tMSW. |
| `energy_input`, `energy_input_mean`, `energy_output`, `energy_output_mean` | Energy inputs/outputs, MJ/tMSW. |
| `net_energy`, `net_energy_mean` | **Net energy balance = input − output**, MJ/tMSW; negative denotes surplus. |
| `resource_output`, `resource_output_mean` | Physical resource-product output, kg/tMSW; energy products are excluded. |
| `primary_mass_relative_residual`, `primary_mass_relative_residual_mean` | Absolute primary-route mass residual divided by input mass; dimensionless. |

The JSON/JSONL monitoring fields retain their original names: `target` is the nominal transition budget, `aligned_target` the full-rollout budget, `percent_complete` the aligned progress, `eta_seconds` the recorded ETA, `pid` the original process ID, `updated_at_epoch` the original Unix timestamp, `event`/`status` the recorded state, `latest_checkpoint`/`checkpoint_event` the checkpoint reference/state, and `resumed` whether the original run resumed. Identity and diagnostic fields have the same meanings as above. `scientific_acceptance` is a historical recorded decision, not a new conclusion of this disclosure.

## Privacy and checksums

CSV files are **byte-identical** to retained originals. JSON/JSONL copies replace only local absolute path strings: repository paths become repository-relative; external workbook source references use `source_materials/`. All fields, numeric values, missing values, timestamps, events and original historical hash fields are retained.

The [publication manifest](publication_manifest.json) lists each source-relative path, original SHA256, public SHA256, size and precise changed field locations. Historical `completed.json.files` hashes refer to **original artifacts**. For a path-sanitized JSON, verify the public file with `public_sha256`; do not compare it directly with its original historical hash. The original files remain unchanged locally. The manifest/index are explicitly new disclosure documents, not reconstructed historical logs.

## Evidence and limits

Configuration, runtime/source/data hashes, original completion hashes and final ZIP/model checksums support internal consistency and provenance review. CSV/monitor histories support update-level numerical-stability analysis. The recent-100-episode reward means independently reconcile with recorded episode rewards to floating-point precision.

These retained records do not independently certify the original execution environment, prove a global optimum, establish cross-seed robustness or validate physical assumptions. Reusing the same hyperparameters does not imply identical scientific outcomes across objectives, archetypes, software/hardware or new runs.

Pilot/smoke/recovery-test runs are excluded from this formal collection. Large per-step trajectories (about 8.0 GB in total), checkpoint bundles, duplicate model archives, launcher/installation output and separate held-out evaluation files remain local. Their absence from this disclosure is not treated as missing rollout diagnostics. TensorBoard event files were not retained for these selected runs; they have not been fabricated. The final model archives remain unchanged in `models/`.
