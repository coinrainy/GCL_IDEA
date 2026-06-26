# Benchmark Protocol

## Status

当前状态：`Draft`。进入 formal 实验前必须改为 `Frozen`。

协议冻结后，不得因为结果不理想而修改数据划分、seed 列表、evaluator、early stopping、baseline、超参数搜索空间或汇总方式。任何协议变更必须写入 `research-wiki/log.md`。

## Task

本项目任务为图对比学习用于节点分类。

- 主指标：Accuracy；类别不平衡时可补充 Macro-F1 或 Balanced Accuracy。
- 主报告方式：`test@best validation epoch`。
- 正式结果：10 seeds 的 mean±std，并保留每个 seed 原始值。
- 主表只允许使用 `formal` 结果。

## Experiment Labels

| Label | 含义 | 可否进主表 |
| --- | --- | --- |
| `smoke` | 检查代码能否运行 | 否 |
| `pilot` | 判断 idea 是否有继续价值 | 否 |
| `development` | 开发、调参、debug | 否 |
| `formal` | 协议冻结后的正式实验 | 是 |

禁止把 `smoke`、`pilot`、`development` 结果用于 SOTA、robust、comprehensive 或主 claim。

## Data Split Policy

默认主实验采用 stratified random `1:1:8` split，即 train / validation / test = `10% / 10% / 80%`。

所有方法、baseline、ablation 和 diagnostic variants 必须使用完全相同的 split 文件和相同 seed 列表。禁止每个方法单独随机划分。

正式实验默认 seeds 为 `0,1,2,3,4,5,6,7,8,9`。结果文件必须分别记录 `split_seed` 和 `model_seed`；推荐 `split_seed = model_seed = seed`。

Split 文件必须保存并版本化，优先且默认保存为 JSON，避免依赖会被 `.gitignore` 全局忽略的 `.pt` 文件。推荐路径：

```text
splits/{dataset}/split_seed_{seed}.json
```

每个 split 至少记录 `dataset_name`、`split_type`、`split_seed`、`train/val/test` 索引或 mask、样本数量和类别分布。

## Public Split Exceptions

Planetoid public split：Cora、CiteSeer、PubMed 若使用 public split，必须单独标注；Planetoid public split 不能和 random `1:1:8` split 在同一主表直接比较。

Wiki-CS：优先使用官方 splits；若额外构造 `1:1:8`，必须标注为 `custom random 1:1:8 split`，不能与官方 split 混合比较。

OGB：优先使用 official split + official evaluator，不默认改成 `1:1:8`。

Heterophily benchmarks：若已有公认 fixed splits，优先使用 fixed splits；fixed split 和 random split 不混合比较。自定义划分必须标注为 custom split。

## Evaluation Protocol

主表默认报告 `test@best validation epoch`：选择验证集表现最好的 epoch，并报告该 epoch 对应的测试集结果。

禁止报告所有 epoch 中最高 test accuracy，除非明确标注为 oracle analysis，且不能进入主表。

每次运行必须记录 `early_stopping_metric`、`patience`、`max_epochs`、`best_epoch`、`valid_at_best`、`test_at_best` 和 `final_test`。

禁止使用 test set 选择模型、超参数、epoch、增强强度或 checkpoint。

## GCL Evaluation

所有 GCL 方法必须使用相同 frozen-encoder evaluator。本项目默认且优先使用 `frozen encoder + Logistic Regression downstream classifier`，不再把 PyTorch `Linear` layer 训练器作为主评估口径。

在 random `1:1:8` 主协议下，Logistic Regression 的超参数（例如 `C`、正则项）只能由 train/val 选择，或在 train 内部做交叉验证后再用 val 做协议核查；禁止使用 test set 选择 `C`、epoch、正则强度或任何 evaluator 参数。

若为了复现 GRACE 原始代码或历史论文数值而采用 `sklearn LogisticRegression + GridSearchCV` 且没有显式 validation split，必须单独标注为 `official_grace_logreg_cv` 或等价协议，不能与本项目 `1:1:8` 主表混合。

必须记录 encoder、hidden dim、projection dim、augmentation、contrastive loss、temperature、positive/negative sampling、pretrain epochs、evaluator type、Logistic Regression solver、`C` 候选集合、`C` 选择规则、evaluator seed。

若使用 fine-tuning，必须单独建表，不能和 frozen evaluation 混合比较。

## Baseline Protocol

正式实验前必须复现关键 baseline，并记录 `paper_result`、`code_source`、`commit_hash`、`dataset`、`split`、`metric`、`our_result`、`gap_from_paper` 和可能原因。

所有方法必须使用相同 split 和相同 seed 列表。

GCL 方法必须使用相同 frozen-encoder evaluator。

GCN、GAT、GraphSAGE 等监督 baseline 不使用 GCL 的 frozen-encoder evaluator，但必须使用相同 split、相同 early stopping 规则、相同特征输入、相同 preprocessing 和可比较训练预算。

本方法不能拥有明显更大的搜索空间而不披露。若 baseline 复现明显低于原论文，不能用该弱结果声称本方法更优。

## Formal Results

Formal 结果必须满足：协议已冻结，split 文件已保存，seed 列表已固定，evaluator 已锁定，baseline 已复现，汇总脚本已实现，且不再根据 test set 调整方法。

正式报告必须包含 mean、std、per-seed results、valid_at_best、test_at_best 和 final_test。

## Result Files

每次运行必须保存原始结果，推荐路径：

```text
results/raw/{dataset}/{method}/{run_id}.json
```

每个结果文件至少包含 `run_id`、`method`、`dataset`、`split_type`、`split_seed`、`model_seed`、`config_path`、`commit_hash`、`best_epoch`、`valid_at_best`、`test_at_best`、`final_test`、`status` 和 `log_path`。

汇总结果保存到 `results/summary/`。论文表格必须由脚本从原始 JSON/CSV 自动生成，不能手工复制最佳数字。

## Ablation Requirements

每个核心模块都必须有消融，最低包括 base method、full method、remove core component、replace core component with simple alternative、`lambda=0` 或等价退化。

如果删除某模块后结果不变，则不能声称该模块是核心贡献。

## SOTA Claim Rule

只有同时满足同任务、同数据集、同 split、同 evaluator、formal 结果、多 seed 稳定、最新相关工作已检查、结果可追溯到原始文件，才能使用 SOTA、state of the art 或 new best result。

SOTA 可比性记录必须写入 `research-wiki` 对应 paper / experiment / claim 条目；涉及论文关系时写入 `research-wiki/papers/`，涉及结果证据时写入 `research-wiki/experiments/`，涉及主张状态时写入 `research-wiki/claims/`。

不满足上述条件时，只能使用 `competitive`、`strong under our protocol`、`improves over reproduced baselines` 或 `preliminary`。

## Prohibited Practices

禁止 test set 调参、混用 public split 和 random split、只报告最好 seed、删除失败 seed、不保存 split 文件、不保存原始结果、本方法和可比 GCL baseline 使用不同 evaluator、监督 baseline 使用不同 split、本方法拥有更大搜索空间但不披露、pilot/development 结果进入主表或支持 SOTA claim、手工从日志中挑最好数字、把不同协议下的结果直接比较。

## Stage Gate

| Stage | 通过条件 |
| --- | --- |
| Stage 0 | 协议文件、research-wiki、reviewer audit 准备完成 |
| Stage 1 | 文献地图和 gap matrix 完成 |
| Stage 2 | 选出 1 个主 idea 和 1 个备选 idea |
| Stage 3 | Problem Anchor 和实验计划冻结 |
| Stage 4 | baseline 复现可信 |
| Stage 5 | pilot 支持机制假设 |
| Stage 6 | formal 结果支持主 claim |

若任一阶段出现 `FAIL` 或 `BLOCKED`，不得进入下一阶段。所有阶段决策写入 `research-wiki/log.md`，并在对应 idea / experiment / claim 页面中保留状态。
