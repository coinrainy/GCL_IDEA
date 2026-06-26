# GRACE Baseline 1:1:8 Reproduction

## 状态

- 阶段：Stage 4 baseline 复现
- 当前决策：`REVISE`
- 实验标签：`development`，另有少量 `smoke`
- 结论：已完成 GRACE 官方代码拉取与本项目 1:1:8 复现脚本；Cora/CiteSeer 10 seeds 已跑通，但均低于截图目标，暂不能声称完全复现截图数值。

## 代码与环境

- GRACE 代码源：`https://github.com/CRIPAC-DIG/GRACE`
- 本地目录：`baselines/GRACE`
- 固定 commit：`b3b5ac3fcbaabbb50e8bd69a075b46cd82a50378`
- 本地 GPU：NVIDIA GeForce RTX 3060 12GB
- PyTorch：`2.5.1+cu121`
- PyG：`2.8.0`
- 复现配置：`configs/grace_1_1_8.yaml`
- 复现脚本：`scripts/run_grace_1_1_8.py`

## 协议

- 数据划分：stratified random `1:1:8`，即 train / val / test = 10% / 10% / 80%
- seeds：`0,1,2,3,4,5,6,7,8,9`
- GCL evaluator：后续主协议统一改为 frozen encoder + Logistic Regression downstream classifier
- 历史已跑结果说明：当前表中的 Cora/CiteSeer development 结果来自 PyTorch `Linear` layer + Adam + val early stopping，只作为诊断/历史记录保留，不再作为主表 GRACE 结果。
- 指标：`test@best validation epoch` accuracy / %
- split 保存位置：`splits/{dataset}/split_seed_{seed}.json`
- 原始结果保存位置：`results/raw/{dataset}/GRACE/*.json`

## 截图目标值

| Dataset | Target GRACE Accuracy / % |
| --- | ---: |
| Cora | 83.2±0.75 |
| CiteSeer | 72.1±1.51 |
| PubMed | 86.7±0.19 |
| DBLP | 84.1±0.34 |
| Amazon-Computers | 86.8±0.32 |
| Amazon-Photo | 91.8±0.15 |

## 当前 Logistic Regression 复现结果

| Dataset | Evaluator | Status | Seeds | 本地结果 / % | 截图目标 / % | Gap |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Cora | `logreg_val` | development | 10 | 82.78±1.02 | 83.2±0.75 | -0.42 |
| CiteSeer | `logreg_val` | development | 10 | 71.59±0.52 | 72.1±1.51 | -0.51 |
| PubMed | `logreg_val` | smoke | 1 | 83.33，2 epoch smoke only | 86.7±0.19 | 不可比 |
| DBLP | `logreg_val` | smoke | 1 | 77.49，2 epoch smoke only | 84.1±0.34 | 不可比 |
| Amazon-Computers | `logreg_val` | development-partial | 6 | 84.06±0.96 | 86.8±0.32 | -2.74 |
| Amazon-Photo | `logreg_val` | smoke | 1 | 92.35，2 epoch smoke only | 91.8±0.15 | 不可比 |

## 历史 PyTorch Linear 诊断结果

| Dataset | Evaluator | Status | Seeds | 本地结果 / % | 截图目标 / % | Gap |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Cora | PyTorch `Linear` + Adam | development | 10 | 81.44±0.91 | 83.2±0.75 | -1.76 |
| CiteSeer | PyTorch `Linear` + Adam | development | 10 | 70.65±0.69 | 72.1±1.51 | -1.45 |

## 已运行命令

```bash
git clone https://github.com/CRIPAC-DIG/GRACE.git baselines/GRACE
CUDA_VISIBLE_DEVICES=0 python scripts/run_grace_1_1_8.py --datasets Cora --seeds 0,1,2,3,4,5,6,7,8,9 --status development --log-every 100
CUDA_VISIBLE_DEVICES=0 python scripts/run_grace_1_1_8.py --datasets CiteSeer --seeds 0,1,2,3,4,5,6,7,8,9 --status development --log-every 100
CUDA_VISIBLE_DEVICES=0 python scripts/run_grace_1_1_8.py --datasets PubMed --seeds 0 --epochs 2 --eval-epochs 20 --eval-patience 5 --status smoke --log-every 1
CUDA_VISIBLE_DEVICES=0 python scripts/run_grace_1_1_8.py --datasets Cora,CiteSeer --seeds 0,1,2,3,4,5,6,7,8,9 --status development --evaluator logreg_val --log-every 200
CUDA_VISIBLE_DEVICES=0 python scripts/run_grace_1_1_8.py --datasets PubMed --seeds 0 --epochs 2 --status smoke --evaluator logreg_val --log-every 1
CUDA_VISIBLE_DEVICES=0 python scripts/run_grace_1_1_8.py --datasets DBLP,Computers,Photo --seeds 0 --epochs 2 --status smoke --evaluator logreg_val --log-every 1
CUDA_VISIBLE_DEVICES=0 python scripts/run_grace_1_1_8.py --datasets Computers,Photo --seeds 0,1,2,3,4,5,6,7,8,9 --status development --evaluator logreg_val --log-every 250
# 上一条命令在 Computers seed 6 期间人工中断，原因：前 6 seeds 已显示 Amazon-Computers 配置明显低于截图目标。
```

## Gap 判断

当前 Logistic Regression 结果已明显接近截图目标，但仍略低于截图均值。主要可疑差异：

- 截图目标可能沿用了 GRACE 官方 sklearn logistic regression 10% label protocol；官方代码没有显式 10% validation split。
- 当前主结果按项目协议执行 `train/val/test = 10%/10%/80%`，使用 Logistic Regression，并通过 val 选择 `C`。
- 历史已跑的 PyTorch `Linear` layer + Adam + validation early stopping 结果已降级为诊断，不作为后续主结果。
- 官方 GRACE 代码固定 config seed；本项目当前按 `model_seed = split_seed = seed` 做 10 seeds。
- Amazon-Computers/Photo 不在官方 GRACE 代码支持范围内，本项目配置只是初始适配，正式复现前需进一步核查近期论文的 GRACE hyperparameters。

`2026-06-25T19:11:25Z` 配置核查：

- 截图目标值可追溯到 UGCL IJCAI 2025 Table 2，该表列出 GRACE 在 Cora/CiteSeer/PubMed/DBLP/Computers/Photo 上的节点分类结果。
- 该论文页面可检索到节点分类表，但未给出 GRACE 在 Amazon/DBLP 上的完整复现实验配置；因此不能把当前 Amazon 配置视为已确认配置。
- 已拉取 GCA 官方仓库作为配置线索：`https://github.com/CRIPAC-DIG/GCA`，commit `0f17497f3ae89718ed610c671a5b9f56033d4d51`。GCA 官方参数中 `amazon_computers.json` 为 `lr=0.01, hidden=128, proj=128, activation=rrelu, drop_edge=(0.6,0.3), drop_feature=(0.2,0.3), tau=0.2, epochs=2000`；`amazon_photo.json` 为 `lr=0.1, hidden=256, proj=64, activation=relu, drop_edge=(0.3,0.5), drop_feature=(0.1,0.1), tau=0.3, epochs=2000`。
- 但 GCA 是自适应增强方法，其官方 `log_regression` 实现也是 PyTorch `LogReg` 线性层，不等于本项目现在要求的 sklearn Logistic Regression。因此这些参数只能作为 Amazon 配置候选，不能直接声称为 GRACE baseline 官方配置。
- 当前 Amazon-Computers 用初始 GRACE-like 配置跑到 6 seeds 后均值仅 `84.06±0.96`，低于目标 `86.8±0.32`；继续跑满 10 seeds 会浪费 GPU，因此已中断该配置。

## Downstream Classifier Decision

`2026-06-25T18:34:19Z` 项目口径更新：后续所有 GCL 方法默认使用 Logistic Regression 作为下游分类器。

- 主协议推荐：`frozen encoder + sklearn LogisticRegression`。
- `1:1:8` 主表：用 train 训练 Logistic Regression，用 val 选择 `C` / 正则项 / solver，最终只在 test 上汇报一次。
- 官方 GRACE 对照：可以额外实现 `official_grace_logreg_cv`，即 sklearn `OneVsRestClassifier(LogisticRegression) + GridSearchCV`，用于核对截图或历史论文数值；该协议若没有显式 10% val，必须单独建表。
- PyTorch `Linear` layer + Adam 不再作为主评估器，只保留为诊断或历史结果。

`2026-06-25T18:39:43Z` 已按新口径重新复现 Cora/CiteSeer。Logistic Regression 使用 `OneVsRestClassifier(LogisticRegression)`，`solver=liblinear`，`C in {2^-10,...,2^9}`，每个 seed 用 train 拟合、val 选择 `C`、test 汇报。Cora 从旧 PyTorch Linear 的 `81.44±0.91` 提升到 `82.78±1.02`，CiteSeer 从 `70.65±0.69` 提升到 `71.59±0.52`，说明下游分类器口径是此前复现 gap 的主要来源之一。

## Linear Evaluator 诊断

`2026-06-25T18:32:31Z` 对 Cora 扫描了 linear evaluator 学习率。脚本新增参数：`--eval-learning-rate`、`--eval-weight-decay`、`--no-normalize-embeddings`。

| Eval LR | Seeds | Cora test@best-val / % | 备注 |
| ---: | ---: | ---: | --- |
| 0.001 | 3 | 68.16±16.45 | 明显训练不足或不稳定 |
| 0.005 | 3 | 80.18±0.91 | 低于默认 |
| 0.01 | 10 | 81.43±0.92 | 默认设置，约等于此前 81.44±0.91 |
| 0.05 | 10 | 81.83±1.22 | 最好但仍低于截图 83.2±0.75 |
| 0.1 | 3 | 81.81±1.05 | 与 0.05 接近，未继续 10 seeds |

判断：PyTorch `Linear` evaluator 参数会影响结果，尤其 LR 过小会显著变差；但把 LR 从 `0.01` 调到 `0.05` 后只提升约 `0.4` 个点，仍不能解释 Cora 到截图目标的全部 gap。因此当前更像是 evaluator 协议差异和 seed/split 协议差异叠加，而不是单纯参数设置不当。该诊断不改变新的主口径：后续统一转向 Logistic Regression。

## 下一步

1. 增加对照协议 `official_grace_logreg_cv`，核对截图目标是否来自 GRACE 官方 10% label + sklearn CV 口径。
2. 为 Amazon-Computers/Photo 建立明确的配置候选表，至少包括当前初始配置、GCA 官方 Amazon 参数迁移版、UGCL/近年论文若公开代码中的 GRACE baseline 参数。
3. 对 PubMed/DBLP/Amazon 启动全量长跑前，先确认 evaluator 协议和 Amazon hyperparameters，否则会消耗数小时 GPU 但仍可能与目标不可比。
4. 若目标必须严格复现截图数值，则将 `strict_project_1_1_8` 和 `paper_target_protocol` 分成两张表，禁止混用。
