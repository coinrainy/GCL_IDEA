# GCN Baseline 1:1:8 Reproduction

## 状态

- 阶段：Stage 4 baseline 复现
- 当前决策：`REVISE`
- 实验标签：`development`，另有少量 `smoke` / diagnostic
- 结论：已完成 GCN 官方仓库拉取与项目协议版 PyG 监督 GCN 复现。Wiki-CS 与截图目标基本对齐；Cora、CiteSeer、PubMed、Computers、Photo 均高于截图目标，说明当前项目实现和截图协议仍存在训练/预处理/超参数差异，不能称为严格复现截图数值。

## 代码与环境

- GCN 代码源：`https://github.com/tkipf/gcn`
- 本地目录：`baselines/GCN`
- 固定 commit：`39a4089fe72ad9f055ed6fdb9746abdcfebc4d81`
- 可运行实现：项目内 PyTorch Geometric 监督 GCN
- 复现配置：`configs/gcn_1_1_8.yaml`
- 复现脚本：`scripts/run_gcn_1_1_8.py`
- 汇总脚本：`scripts/summarize_baseline_results.py`
- 汇总文件：`results/summary/gcn_1_1_8_development_summary.md`

## 协议

- GCN 是监督 baseline，不使用 GRACE/GCL 的 frozen encoder + Logistic Regression evaluator。
- 分类器未替换为 Logistic Regression；模型为标准两层 GCN：`GCNConv -> ReLU/dropout -> GCNConv`，第二层直接输出类别 logits，并用 `cross_entropy` 训练。
- 指标：`test@best validation epoch` accuracy / %
- Planetoid / Amazon split：项目 stratified random `1:1:8`
- Wiki-CS split：PyG 官方 split；训练使用 `train_mask`，early stopping 使用 `stopping_mask`，测试使用 `test_mask`
- seeds：`0,1,2,3,4,5,6,7,8,9`
- 原始结果保存位置：`results/raw/{dataset}/GCN/*.json`
- 汇总规则：`latest raw JSON per dataset/seed`，只筛选 `status=development`

## 截图目标值

| Dataset | Target GCN Accuracy / % |
| --- | ---: |
| Cora | 82.17±0.59 |
| CiteSeer | 71.46±0.97 |
| PubMed | 84.16±0.23 |
| Wiki-CS | 76.89±0.37 |
| Amazon-Computers | 86.34±0.48 |
| Amazon-Photo | 92.35±0.25 |

## 当前 development 复现结果

| Dataset | Seeds | 本地结果 / % | 截图目标 / % | Gap |
| --- | ---: | ---: | ---: | ---: |
| Cora | 10 | 83.52±0.94 | 82.17±0.59 | +1.35 |
| CiteSeer | 10 | 72.61±0.52 | 71.46±0.97 | +1.15 |
| PubMed | 10 | 86.49±0.27 | 84.16±0.23 | +2.33 |
| Wiki-CS | 10 | 76.78±0.56 | 76.89±0.37 | -0.11 |
| Amazon-Computers | 10 | 89.84±0.39 | 86.34±0.48 | +3.50 |
| Amazon-Photo | 10 | 93.11±0.44 | 92.35±0.25 | +0.76 |

## 关键诊断

1. GCN 分类器没有改成 Logistic Regression。当前实现是端到端监督 GCN，第二层 `GCNConv` 直接输出类别 logits。
2. 首轮 development 结果突然偏高的主要原因是 GCN 容量被设得过强：Cora/CiteSeer 使用 `hidden_dim=512`，PubMed/Amazon/Wiki-CS 使用 `hidden_dim=256`；这不是 Kipf GCN 常见的 `hidden1=16` baseline。
3. 官方 `tkipf/gcn` 默认设置为 `hidden1=16`、`epochs=200`、`early_stopping=10`，且按 validation loss 早停。项目脚本已新增 `early_stopping_metric=val_loss` 支持。
4. Planetoid 诊断表明 hidden/预算能解释 Cora/CiteSeer 的大部分偏高：`hidden=16, epochs=200, patience=10, val_loss` 下，Cora `82.52±0.97`，CiteSeer `72.22±0.69`，分别只比截图高 `+0.35` 和 `+0.76`。
5. PubMed 在同一官方式预算下仍为 `85.67±0.31`，比截图 `84.16±0.23` 高 `+1.51`，剩余差异更可能来自 split、PyG/TF 实现差异、特征预处理细节或截图论文中的 PubMed-specific 设置。
6. Wiki-CS / Amazon 最初沿用 `NormalizeFeatures()` 会显著异常：Wiki-CS seed 0 曾只有约 `24%`，Amazon-Computers seed 0 曾只有约 `67%`。关闭这些数据集的特征归一化后，Wiki-CS 回到约 `77%`，Amazon 回到 `90% / 93%` 附近。
7. Amazon hidden=64 诊断仍偏高：Computers `89.78±0.33` vs `86.34±0.48`，Photo `93.20±0.29` vs `92.35±0.25`，说明 Amazon 不能只靠降低 hidden dim 对齐，还需核查截图 split/正则/早停/预处理协议。
8. Wiki-CS 需要显式区分 `val_mask`、`stopping_mask`、`test_mask`。当前使用 `stopping_mask` 做 early stopping，`test_mask` 只用于最终汇报。

## 诊断结果

| Diagnostic | Dataset | Seeds | 结果 / % | 截图目标 / % | Gap |
| --- | --- | ---: | ---: | ---: | ---: |
| strong initial config (`hidden=512/256`) | Cora | 10 | 83.52±0.94 | 82.17±0.59 | +1.35 |
| official-like (`hidden=16, 200 epochs, val_loss`) | Cora | 10 | 82.52±0.97 | 82.17±0.59 | +0.35 |
| strong initial config (`hidden=512/256`) | CiteSeer | 10 | 72.61±0.52 | 71.46±0.97 | +1.15 |
| official-like (`hidden=16, 200 epochs, val_loss`) | CiteSeer | 10 | 72.22±0.69 | 71.46±0.97 | +0.76 |
| strong initial config (`hidden=512/256`) | PubMed | 10 | 86.49±0.27 | 84.16±0.23 | +2.33 |
| official-like (`hidden=16, 200 epochs, val_loss`) | PubMed | 10 | 85.67±0.31 | 84.16±0.23 | +1.51 |
| strong initial config (`hidden=256`) | Computers | 10 | 89.84±0.39 | 86.34±0.48 | +3.50 |
| `hidden=64` | Computers | 10 | 89.78±0.33 | 86.34±0.48 | +3.44 |
| strong initial config (`hidden=256`) | Photo | 10 | 93.11±0.44 | 92.35±0.25 | +0.76 |
| `hidden=64` | Photo | 10 | 93.20±0.29 | 92.35±0.25 | +0.85 |

## 已运行命令

```bash
git clone https://github.com/tkipf/gcn.git baselines/GCN
CUDA_VISIBLE_DEVICES=0 python scripts/run_gcn_1_1_8.py --datasets Cora --seeds 0 --epochs 5 --patience 5 --status smoke --log-every 1
CUDA_VISIBLE_DEVICES=0 python scripts/run_gcn_1_1_8.py --datasets Cora,CiteSeer,PubMed,WikiCS,Computers,Photo --seeds 0,1,2,3,4,5,6,7,8,9 --status development --log-every 200
CUDA_VISIBLE_DEVICES= python scripts/run_gcn_1_1_8.py --datasets WikiCS --seeds 1,2,3,4,5,6,7,8,9 --status development --log-every 300
CUDA_VISIBLE_DEVICES= python scripts/run_gcn_1_1_8.py --datasets WikiCS --seeds 0 --status development --log-every 300
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python scripts/run_gcn_1_1_8.py --datasets Computers,Photo --seeds 0,1,2,3,4,5,6,7,8,9 --status development --log-every 300
python scripts/summarize_baseline_results.py --method GCN --status development --output-prefix gcn_1_1_8_development
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python scripts/run_gcn_1_1_8.py --datasets Cora,CiteSeer,PubMed,Computers,Photo --seeds 0,1,2,3,4,5,6,7,8,9 --hidden-dim 16 --status diagnostic_hidden16 --log-every 300
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python scripts/run_gcn_1_1_8.py --datasets Cora,CiteSeer,PubMed --seeds 0,1,2,3,4,5,6,7,8,9 --hidden-dim 16 --epochs 200 --patience 10 --early-stopping-metric val_loss --status diagnostic_hidden16_budget200_pat10_valloss --log-every 100
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python scripts/run_gcn_1_1_8.py --datasets Computers,Photo --seeds 0,1,2,3,4,5,6,7,8,9 --hidden-dim 64 --status diagnostic_hidden64 --log-every 300
```

## 下一步

1. 查明截图目标对应的 GCN 具体实现和超参数，尤其是 Amazon 的特征预处理、hidden dim、weight decay、early stopping 与 split。
2. 将 `strict_project_1_1_8` 和 `paper_target_protocol` 分开记录，避免高于截图的 development 结果被误用为正式 claim。
3. 若后续要进入 formal，需要先冻结 GCN 监督 baseline 协议，并用同一汇总脚本重新生成正式表。
