# ACP S1–S3 历史训练日志与 GitHub 补充披露审查

审查日期：2026-10-22（Asia/Tokyo）。目标分支：`alternative-primary-objectives`。

## 1. 结论与工作边界

18/18 正式任务通过本轮历史文件一致性和来源审查，具备可公开的真实留存日志。每组保留 98 条完整更新后 rollout 记录、10035 条完整 episode、934 条监控事件和 98 条 SB3 CSV 记录；实际步数均为 200704，seed 均为 1729。正式最终模型与发布 manifest、GitHub 模型 blob 的字节哈希一致。

本轮仅递归清点、只读解析 CSV/JSON/ZIP、流式哈希和汇总既有数据；没有训练、恢复、PPO.load、反序列化模型对象、模型推理或新预测；没有从图像取数、填补缺值或补写历史事件。新生成的索引、哈希清单和本报告均明确属于 2026-10-22 披露文档，而非训练时日志。

日志披露 commit：`c1e76a8c7b5865582a81a81b80cbc238ef382263`（[提交链接](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/commit/c1e76a8c7b5865582a81a81b80cbc238ef382263)）。下列历史日志、索引和 README 均已在该提交发布；本报告在后续文档提交补充。

## 2. 18 项逐任务审查矩阵

表中 CSV 链接对应公开副本。每项原始目录均为 `experiments/alternative_primary_objectives/runs/<run_id>/`；公开目录为 `training_logs/<archetype>/ACP_Sx/`。每项 seed=1729、正式 run_kind、ACP/reward mode/version、metric version、冻结尺度、PPO 参数、数据和 registry SHA256 均与发布 manifest 和正式配置核对一致。

| Archetype | ACP | 原始 run ID | Rollout 日志/行数 | Episodes行数 | 监控事件 | Rollout步数范围 | 名义/实际步数 | 完整性 |
|---|---|---|---|---:|---:|---|---|---|
| JPN | ACP S1 | `20261013T024809Z_JPN_ACP_S1_seed1729_7772c909` | [98 CSV行](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/JPN/ACP_S1/rollouts.csv) | 10035 | 934 | 2048–200704 | 200000 / 200704 | PASS |
| JPN | ACP S2 | `20261013T025123Z_JPN_ACP_S2_seed1729_a9c355c7` | [98 CSV行](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/JPN/ACP_S2/rollouts.csv) | 10035 | 934 | 2048–200704 | 200000 / 200704 | PASS |
| JPN | ACP S3 | `20261013T025436Z_JPN_ACP_S3_seed1729_7a20971e` | [98 CSV行](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/JPN/ACP_S3/rollouts.csv) | 10035 | 934 | 2048–200704 | 200000 / 200704 | PASS |
| EU | ACP S1 | `20261013T025757Z_EU_ACP_S1_seed1729_d5f20ea3` | [98 CSV行](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/EU/ACP_S1/rollouts.csv) | 10035 | 934 | 2048–200704 | 200000 / 200704 | PASS |
| EU | ACP S2 | `20261013T030111Z_EU_ACP_S2_seed1729_7825675c` | [98 CSV行](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/EU/ACP_S2/rollouts.csv) | 10035 | 934 | 2048–200704 | 200000 / 200704 | PASS |
| EU | ACP S3 | `20261013T030423Z_EU_ACP_S3_seed1729_74186920` | [98 CSV行](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/EU/ACP_S3/rollouts.csv) | 10035 | 934 | 2048–200704 | 200000 / 200704 | PASS |
| US | ACP S1 | `20261013T030727Z_US_ACP_S1_seed1729_fd731efb` | [98 CSV行](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/US/ACP_S1/rollouts.csv) | 10035 | 934 | 2048–200704 | 200000 / 200704 | PASS |
| US | ACP S2 | `20261013T031029Z_US_ACP_S2_seed1729_98dd4605` | [98 CSV行](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/US/ACP_S2/rollouts.csv) | 10035 | 934 | 2048–200704 | 200000 / 200704 | PASS |
| US | ACP S3 | `20261013T031330Z_US_ACP_S3_seed1729_61f46271` | [98 CSV行](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/US/ACP_S3/rollouts.csv) | 10035 | 934 | 2048–200704 | 200000 / 200704 | PASS |
| CHN | ACP S1 | `20261013T031633Z_CHN_ACP_S1_seed1729_c666c776` | [98 CSV行](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/CHN/ACP_S1/rollouts.csv) | 10035 | 934 | 2048–200704 | 200000 / 200704 | PASS |
| CHN | ACP S2 | `20261013T031935Z_CHN_ACP_S2_seed1729_690e8aa0` | [98 CSV行](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/CHN/ACP_S2/rollouts.csv) | 10035 | 934 | 2048–200704 | 200000 / 200704 | PASS |
| CHN | ACP S3 | `20261013T032232Z_CHN_ACP_S3_seed1729_db6ac718` | [98 CSV行](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/CHN/ACP_S3/rollouts.csv) | 10035 | 934 | 2048–200704 | 200000 / 200704 | PASS |
| IND | ACP S1 | `20261013T032527Z_IND_ACP_S1_seed1729_9d1984fd` | [98 CSV行](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/IND/ACP_S1/rollouts.csv) | 10035 | 934 | 2048–200704 | 200000 / 200704 | PASS |
| IND | ACP S2 | `20261013T032824Z_IND_ACP_S2_seed1729_e85febf6` | [98 CSV行](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/IND/ACP_S2/rollouts.csv) | 10035 | 934 | 2048–200704 | 200000 / 200704 | PASS |
| IND | ACP S3 | `20261013T033117Z_IND_ACP_S3_seed1729_69a03c63` | [98 CSV行](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/IND/ACP_S3/rollouts.csv) | 10035 | 934 | 2048–200704 | 200000 / 200704 | PASS |
| Permissive | ACP S1 | `20261013T033413Z_Permissive_ACP_S1_seed1729_bd3676ad` | [98 CSV行](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/Permissive/ACP_S1/rollouts.csv) | 10035 | 934 | 2048–200704 | 200000 / 200704 | PASS |
| Permissive | ACP S2 | `20261013T033709Z_Permissive_ACP_S2_seed1729_badf2fdf` | [98 CSV行](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/Permissive/ACP_S2/rollouts.csv) | 10035 | 934 | 2048–200704 | 200000 / 200704 | PASS |
| Permissive | ACP S3 | `20261013T034002Z_Permissive_ACP_S3_seed1729_01b3fcbe` | [98 CSV行](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/Permissive/ACP_S3/rollouts.csv) | 10035 | 934 | 2048–200704 | 200000 / 200704 | PASS |

每组 `rollouts.csv` 时间步严格等于 2048,4096,…,200704（98条，无重复/缺口）；`updates` 是累计 optimizer epoch，10,20,…,980。`episodes.csv` 时间步为20,40,…,200700（10035条，length=20），最后4个 transition 属于未完成 episode，已包含在最终 rollout，不应虚构一个完整 episode。

监控每组包括 started×1、progress×784、ppo_update×98、checkpoint_saved×49、heldout_evaluation×1、completed×1。相同时间步的不同事件是合理记录；时间步和 epoch 时间均非递减，所有 `resumed=false`，未发现选中正式运行的失败或续训事件。其他目录的测试/恢复记录没有混入。

## 3. 模型与配置溯源

| 正式任务 | 最终模型 SHA256 | 训练 segment耗时(s) | 模型 |
|---|---|---:|---|
| JPN ACP S1 | `013e0cb3630803f196702cf52716cb832e5ab921c572045a585edbd018cf2652` | 193.547 | [已发布模型](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/models/JPN/ACP_S1.zip) |
| JPN ACP S2 | `1f1bb93a5bbc18dc4fbfb24ea4dbb1d938d153d2403718d40785e4a0937ab5b2` | 192.485 | [已发布模型](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/models/JPN/ACP_S2.zip) |
| JPN ACP S3 | `751c15e48eee27eda61660742678fb2669b5c6da1c919b90fb855726e8795487` | 200.266 | [已发布模型](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/models/JPN/ACP_S3.zip) |
| EU ACP S1 | `8430b82b51c8ab99681b078dbc04b2909b0c67d77f1a8565c225ae50a2f019aa` | 193.812 | [已发布模型](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/models/EU/ACP_S1.zip) |
| EU ACP S2 | `7bffa032c8fa11b8824480dd13c951544f34d7a31e410165145f6b63257e4238` | 191.828 | [已发布模型](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/models/EU/ACP_S2.zip) |
| EU ACP S3 | `989aaf06c480d989c52483339e10580d44091872ee4cf93c4ed8ab64ea71e85e` | 184.125 | [已发布模型](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/models/EU/ACP_S3.zip) |
| US ACP S1 | `ae01f4b76483e63f205011a9de5e616c24f004a6382d9f40d6de1629d5548efb` | 181.563 | [已发布模型](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/models/US/ACP_S1.zip) |
| US ACP S2 | `071c1f9f7a169dd112ddfeeb74f673a8727ebf61b8a95ed98810a7e1065aeeec` | 180.250 | [已发布模型](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/models/US/ACP_S2.zip) |
| US ACP S3 | `ab265a21f5d0c9b4586cd74b4010b75e68f1326d92cfefebdd859f98152378d9` | 182.422 | [已发布模型](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/models/US/ACP_S3.zip) |
| CHN ACP S1 | `25f890dcbdf1cb3288db2cf4ea2b03213a4595a2695cf97192e76d086f06d7e7` | 181.906 | [已发布模型](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/models/CHN/ACP_S1.zip) |
| CHN ACP S2 | `b22ae97294b3a1948616477230d1e558541314e456a8615bba4b1496971f092a` | 176.813 | [已发布模型](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/models/CHN/ACP_S2.zip) |
| CHN ACP S3 | `327254663e9d09e0bea37e8e1dcfef8fca5169003ab5f4a704ef061466f36df2` | 174.859 | [已发布模型](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/models/CHN/ACP_S3.zip) |
| IND ACP S1 | `b19e587606e9b41e507440bf5058139656a733ba2a92ff60f4ebf1b41b8aa5dc` | 176.219 | [已发布模型](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/models/IND/ACP_S1.zip) |
| IND ACP S2 | `03035c9824d07238f1663bedaa8de6e6366b8b1c95c5f06e27b316ad037dd064` | 173.063 | [已发布模型](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/models/IND/ACP_S2.zip) |
| IND ACP S3 | `a80e8ba30647c6c0df05a32fcece6f195508a90e8e327b5c74bc7f1fd02aca1a` | 175.579 | [已发布模型](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/models/IND/ACP_S3.zip) |
| Permissive ACP S1 | `9bd4390c12f5dc278c2ae379b87d00475ec671761027f06ff9eb356886b2f7dc` | 175.656 | [已发布模型](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/models/Permissive/ACP_S1.zip) |
| Permissive ACP S2 | `6decfe710918609d96111743958c177872f8f03e9cc2cda1d0d570b5803cbe3d` | 172.875 | [已发布模型](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/models/Permissive/ACP_S2.zip) |
| Permissive ACP S3 | `5bc0cdbecd93d73fe7c25a30d94fb7a54ca1acf9e5a4f2b67dc6f67ae348b6f8` | 174.781 | [已发布模型](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/models/Permissive/ACP_S3.zip) |

已逐项核对原始 `completed.json.files` 声明的10个文件 SHA256，18×10 项均匹配，其中包括约7,999,014,431字节的逐步 `trajectory.csv`。18个 trajectory 共3,612,672数据行，逐组1–200704连续，episode_step按20循环；扫描未发现NaN/Inf字面值。逐步其他数值列没有在本轮重新运行物理系统以独立重算；这不是新的工程性能验收。

模型 ZIP 仅解析 JSON 元数据，确认 num_timesteps=200704、_n_updates=980、seed=1729及PPO参数；没有执行模型。最后checkpoint与final_model一致，49次检查点对应每4096步保存直到200704；大体积checkpoint bundle不纳入发布。

PPO：MlpPolicy（SB3默认pi/vf [64,64]；原JSON policy_kwargs为空）、n_steps=2048、batch_size=64、n_epochs=10、learning_rate=3e-4、gamma=.99、gae_lambda=.95、clip_range=.2、ent_coef=.1、vf_coef=.5、max_grad_norm=.5，CPU、Torch单线程、episode horizon=20、无VecNormalize。既有正式配置和模型均未修改。

运行版本：Python3.12.10、stable-baselines3 2.7.0、gymnasium1.1.1、torch2.8.0、numpy2.0.2、pandas3.0.6、openpyxl3.1.5。**历史 Git commit 为空**：当时 Git 不在PATH。保留了源代码、训练数据、registry及配置SHA256，不能将本次GitHub发布commit误称为训练时记录的源码commit，也不能补造该记录。

## 4. 诊断指标及其科学意义

三项要求指标在所有18×98条更新记录中完整、有限；rollout/episode数值字段未发现空值或NaN/Inf，explained_variance均为defined。近100个已完成episode reward均值与rollout记录独立聚合复核的最大差约5.33×10⁻¹⁵。

- Mean episode reward：最近100个完整episode累计reward的均值；不同目标、冻结尺度间不能直接比较绝对reward为科学优劣。`reward_mean`是rollout内单步reward均值，不能替代它。
- Approximate KL：策略更新幅度诊断，可用于检查异常跳变，不能单独证明性能。
- Policy gradient loss：优化损失诊断；数值大小或接近零不等同于收敛到最佳政策。
- 其余保留value loss、entropy loss、explained variance、clip fraction、fps、耗时、reward范围/饱和率、每吨四维净指标和主路线质量残差。

**SB3时序区别**：原`sb3_segments/segment_000/progress.csv`先dump再更新，所以首行`train/*`为空，后97行对应前一轮更新。末次更新后的指标必须读取自定义`rollouts.csv`。未移动、插值或填补源CSV中的任何值。

配置/版本/完成哈希链支持训练设置与最终模型的内部一致性审查；曲线/事件支持数值稳定性、执行覆盖和完成状态审查。它们不能独立证明原机器执行过程的外部真实性、物理参数正确性、全局最优、跨seed统计鲁棒性、各ACP均衡收益或国家预测有效性。**相同训练参数不等于获得相同科学结果。** `validation.json`是既有训练时检查记录，本轮没有重算其科学结论。

## 5. 公开文件、隐私与哈希口径

每组公开11个文件，合计198个正式运行文件，加1个Permissive来源映射JSON；历史文件公开总量 50,835,482 字节，最大历史文件 2,018,673 字节。新生成的逐文件披露清单约3.7MB，是本次最大的文件；均低于[GitHub单文件100MiB上限](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)，无需LFS。不上传checkpoint或大体积trajectory。

| 历史文件类别 | 公开份数 |
|---|---:|
| `rollouts.csv` | 18 |
| `episodes.csv` | 18 |
| `config.json` | 18 |
| `manifest.json` | 18 |
| `completed.json` | 18 |
| `status.json` | 18 |
| `progress.json` | 18 |
| `progress_history.jsonl` | 18 |
| `latest_checkpoint.json` | 18 |
| `validation.json` | 18 |
| `progress.csv` | 18 |
| `pmm_economic_substitution.json` | 1 |

额外发布当前生成的 `training_logs/README.md`、`index.json`、`publication_manifest.json`、本报告，并更新根README。[完整逐文件清单](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/publication_manifest.json)含全部路径、字节数、原始/公开双SHA256和仅路径脱敏的位置；[逐任务机器可读索引](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/index.json)含运行、配置及模型链接。

54个CSV全部逐字节复制，不改变行、字段顺序或数字的文本表示。JSON/JSONL只将本机绝对路径替换为仓库相对路径或 `source_materials/` 来源名称；所有数值、布尔、null、时间戳、字段和事件完整保留。公开副本不含本机用户名、绝对驱动器路径或常见凭据模式；原文件不变。

`completed.json.files`、latest_checkpoint的manifest哈希等始终保留**原始历史哈希**。脱敏JSON字节与原件不同，必须使用publication_manifest的`public_sha256`检验公开副本；原始历史值应与`original_sha256`对照。没有为迎合公开副本而改写训练时哈希。

现有忽略规则屏蔽runs/logs/checkpoints，却不屏蔽新的`training_logs/`和补充manifests文件，因此没有扩大忽略规则例外，也没有改`.gitignore`。此前冻结registry内已有科研源文件路径，本轮未改该registry，以保持训练契约SHA256；新公开文件完成了单独隐私扫描。

## 6. 未公开与缺失记录

以下是本地递归清点分类（按目录、用途和manifest所选ID区分；非正式记录不能当作正式证据）：

| 本地类别 | 文件数 | 字节 |
|---|---:|---:|
| `.gitignore` | 1 | 63 |
| `README.md` | 1 | 9,192 |
| `audits` | 14 | 706,476 |
| `check_training.ps1` | 1 | 1,128 |
| `configs` | 27 | 84,512 |
| `evaluate_results.ps1` | 1 | 1,076 |
| `figures` | 240 | 34,973,675 |
| `formal_selected` | 2934 | 8,376,990,693 |
| `formal_training_review.md` | 1 | 18,347 |
| `generate_diagnostics.ps1` | 1 | 1,083 |
| `logs` | 16 | 10,617,533 |
| `manifests` | 63 | 20,037,900 |
| `other_run` | 2 | 8,673 |
| `pilot` | 189 | 186,635,967 |
| `pilot_report.md` | 1 | 8,315 |
| `predictions` | 370 | 328,250,993 |
| `run_audit.ps1` | 1 | 1,197 |
| `run_tests.ps1` | 1 | 1,224 |
| `run_training.ps1` | 1 | 2,783 |
| `scripts` | 24 | 326,134 |
| `tables` | 4 | 37,536 |
| `test_smoke_recovery_or_launch_failure` | 496 | 40,533,241 |
| `tests` | 13 | 112,102 |
| `training_plan.md` | 1 | 5,909 |
| `validate_training.ps1` | 1 | 1,736 |

- 保留本地：约8GB逐步轨迹、49×18 checkpoint bundles、重复model/initial_model、acceptance.json中单独held-out评估、launcher输出、安装/测试/恢复/启动失败日志、pilot记录及预测/图件。它们未混入正式训练文件集合。
- 正式原始`runs/`中的rollout、episode、监控和完成记录齐全；没有为了披露创建缺失历史记录。
- 未发现选中正式运行的TensorBoard event文件；当时以CSV/JSONL监控留存，不生成替代event。
- 历史Git commit缺失是实际元数据局限，保留null和原错误原因；checkpoint文件在本机有，但未公开，所以读者不能仅凭最终模型和本披露构成精确中断恢复包。

## 7. 两项额外复现问题

1. `configs/Permissive_s1.json`引用的`manifests/pmm_economic_substitution.json`本地存在，但此前发布遗漏；非.gitignore所致。本轮补公开路径脱敏副本，保留产品/成本、单位、donor、来源文件名/单元格和原始hash。原配置字段是来源说明，runtime使用已发布冻结registry，其SHA256仍为`0dca3a20fec6d2928dbc0954d38f6b0b8abf0dd32bb96685797deb2d0d2f03b1`，已含14项缺失经济价格的IND donor，故该遗漏是披露缺口，不是新增参数计算。未改reward或MEE。
2. 根README中关于公开版本不支持全球情景外推的限制句已删除；保留真实reset接口的校准质量范围，更新旧“日志未分发”表述为已公开日志索引。相关公开文本扫描未发现其他同类限制。训练和预测代码未改。

## 8. GitHub交付与变更范围

- [日志README与18任务入口](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/README.md)
- [199份历史文件及其双哈希清单](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/publication_manifest.json)
- [18任务完整审查矩阵](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/training_logs/index.json)
- [Permissive经济参数来源补充](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/manifests/pmm_economic_substitution.json)
- [本报告](https://github.com/Kirin-Ciao/SustainCity-IAgent-Framework/blob/alternative-primary-objectives/experiments/alternative_primary_objectives/report/training_log_publication_audit.md)

发布只在`alternative-primary-objectives`分支追加；日志披露基准是`62a996de1b7bf199374df8e1dc7e61dde71f046f`。已通过远端Git tree逐路径比对，确认所有已有18模型blob、训练模块、reward、MEE、正式配置、冻结尺度和其余已有文件（根README修改除外）与基准树完全相同。`MSW`对照SHA为`5d515fee0e29cf4fe0d1f8ad3a64b8a42c955dd0`，已在日志提交后复核未变；本地主工作区HEAD/index也未变。没有合并、训练、预测、模型覆盖或PyPI发布。本次共新增203个文件、仅修改根README；历史日志提交和后续报告提交均为正常快进追加。

本报告给出的是 **历史记录一致性审查通过**，不是对算法全局最优或论文全部科学结论的再认证。
