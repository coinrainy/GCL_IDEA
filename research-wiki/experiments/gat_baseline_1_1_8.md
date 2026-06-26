# GAT Baseline 1:1:8 Reproduction

## 状态

- 阶段：Stage 4 baseline 复现
- 当前决策：`REVISE`
- 实验标签：`development`
- 结论：已按截图中的 GAT 行完成 6 个数据集、10 seeds 的项目协议版复现。Cora / CiteSeer 与截图目标基本接近；PubMed、DBLP、Computers、Photo 明显高于截图目标，说明当前 PyG 实现、特征预处理、split 或超参数协议与截图论文设置不一致，不能称为严格复现截图数值。

## 代码与环境

- GAT 代码源：`https://github.com/PetarV-/GAT`
- 本地目录：`baselines/GAT`
- 固定 commit：`5af87e7fce2b90ae1cbd621cd58059036a3c7436`
- 可运行实现：项目内 PyTorch Geometric 监督 GAT
- 复现配置：`configs/gat_1_1_8.yaml`
- 复现脚本：`scripts/run_gat_1_1_8.py`
- 汇总脚本：`scripts/summarize_baseline_results.py`
- 汇总文件：`results/summary/gat_1_1_8_development_summary.md`

## 协议

- GAT 是监督 baseline，不使用 GRACE/GCL 的 frozen encoder + Logistic Regression evaluator。
- 模型为两层 GAT：`GATConv(8 heads x 8 hidden) -> ELU/dropout -> GATConv(output head)`。
- 默认超参参考官方 GAT Cora 示例：`lr=0.005`、`weight_decay=0.0005`、`dropout=0.6`、`patience=100`。
- 指标：`test@best validation epoch` accuracy / %
- split：项目 stratified random `1:1:8`
- seeds：`0,1,2,3,4,5,6,7,8,9`
- 原始结果保存位置：`results/raw/{dataset}/GAT/*.json`
- 汇总规则：`latest raw JSON per dataset/seed`，只筛选 `status=development`

## 截图目标值

| Dataset | Target GAT Accuracy / % |
| --- | ---: |
| Cora | 83.0±0.30 |
| CiteSeer | 72.5±0.65 |
| PubMed | 79.0±0.40 |
| DBLP | 81.5±0.03 |
| Amazon-Computers | 85.2±0.09 |
| Amazon-Photo | 91.2±0.05 |

## 当前 development 复现结果

| Dataset | Seeds | 本地结果 / % | 截图目标 / % | Gap |
| --- | ---: | ---: | ---: | ---: |
| Cora | 10 | 83.44±0.83 | 83.00±0.30 | +0.44 |
| CiteSeer | 10 | 72.57±0.43 | 72.50±0.65 | +0.07 |
| PubMed | 10 | 85.05±0.29 | 79.00±0.40 | +6.05 |
| DBLP | 10 | 82.60±0.30 | 81.50±0.03 | +1.10 |
| Amazon-Computers | 10 | 89.42±0.47 | 85.20±0.09 | +4.22 |
| Amazon-Photo | 10 | 93.16±0.29 | 91.20±0.05 | +1.96 |

## 关键诊断

1. Cora / CiteSeer 基本贴近截图，说明 GAT 主体实现和 Planetoid 小图训练链路是合理的。
2. PubMed 高出约 6.05 个点，差距远超截图 std，不能视为同协议复现。
3. Computers / Photo 关闭特征归一化后结果显著高于截图；若开启归一化可能会接近另一种协议，但必须另开 diagnostic 记录，不能覆盖本次 development 结果。
4. 官方 `PetarV-/GAT` 仓库只提供 TensorFlow 1.x 的 Cora 示例。本次为项目内 PyG 统一实现，因此是“项目协议版复现”，不是逐行运行官方代码的全数据集复现。

## 已运行命令

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_gat_1_1_8.py --datasets Cora --seeds 0 --epochs 5 --patience 5 --status smoke --log-every 1
CUDA_VISIBLE_DEVICES=0 python scripts/run_gat_1_1_8.py --datasets Cora,CiteSeer,PubMed,DBLP,Computers,Photo --seeds 0,1,2,3,4,5,6,7,8,9 --status development --log-every 200
python scripts/summarize_baseline_results.py --method GAT --status development --output-prefix gat_1_1_8_development
```

## 下一步

1. 若目标是严格贴合截图，需要查明截图对应的 GAT 数据 split、Amazon 特征预处理、PubMed 设置和实现来源。
2. 建议另建 `paper_target_protocol` diagnostic：尝试 Amazon 开启 `NormalizeFeatures()`、缩小训练预算或改用截图论文代码中的 GAT 设置。
3. 进入 formal 前，需要冻结监督 baseline 协议，并重新跑 GAT formal 结果。
