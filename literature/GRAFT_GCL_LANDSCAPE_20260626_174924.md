# GRAFT-GCL Literature Boundary

- 时间：2026-06-26T17:49:24Z
- 触发原因：ORBIT-GCL `OR-M0-001` Cora seed0 negative smoke
- 新方向：Graph prior-task fitted contrastive learning
- Claim status：literature boundary only；未实现、未跑实验、无性能 claim。

## ORBIT No-Go Boundary

ORBIT full minimal 在 Cora seed0 smoke 中得到 `82.98`，低于 GRACE `84.78` 与 matched GRACE `84.96`。这说明 operator-response basis target 有一定信号，但不足以超过强 matched control。

本轮 pivot 明确禁止：

- 继续调 ORBIT operator bank、basis rank、teacher response、loss weight。
- 继续 response sibling / response certificate / operator-response target。
- 继续用 held-out response MSE、rank、boundary 或 probe margin 当作主要贡献。

## Nearby Literature

### GraphPFN / Prior-Data Fitted Graph Foundation Model

GraphPFN 提出从合成 attributed graphs 的 prior 中采样结构、属性与 target，并训练 graph foundation model 做 node-level prediction。它说明“synthetic graph/task prior”是一个正在升温的方向。边界是：GraphPFN 是跨图 foundation/in-context/finetuning 路线；GRAFT-GCL 只在当前真实图上生成一族 label-free downstream-like tasks，用 GCL encoder 学 frozen representation，不引入真实标签或跨图 foundation pretraining。

Source: https://arxiv.org/abs/2509.21489

### AutoSSL

AutoSSL 研究如何自动组合多个 graph self-supervised pretext tasks，并用 homophily 作为搜索指导。边界是：GRAFT-GCL 不搜索已有 pretext 的权重，也不把多 pretext teacher 简单蒸馏；它先定义一个显式 label-generating task prior，再对该任务族的 supervised-contrastive meta-risk 做拟合。

Source: https://arxiv.org/abs/2106.05470

### AGSSL / Multi-Teacher Automated Graph SSL

AGSSL 使用多个 pretext teachers，再把不同 inductive bias 的 teacher knowledge 集成到 student。边界是：GRAFT-GCL 不从已有 teachers 提取知识，也不做 ORBIT/teacher response 蒸馏；它的 supervision 来自无标签任务生成器产生的 synthetic class partitions。

Source: https://arxiv.org/abs/2210.02099

### Pseudo Contrastive Learning

PCL 是半监督图学习框架，使用模型预测产生 pseudo-label / negative pairs，并用 contrastive objective 训练。边界是：GRAFT-GCL 不使用真实 train labels 训练 pseudo-labeler，不用 test 或 validation 选伪标签，不生成“真实任务的伪标签”；它生成的是一族可能的 node-classification 任务作为 prior samples。

Source: https://arxiv.org/html/2302.09532v3

## Gap

现有路线大多属于以下几类：

1. node-node agreement：GRACE/MVGRL/BGRL 类方法。
2. masked input/latent reconstruction：GraphMAE/GraphMAE2/GCMAE 类方法。
3. pretext/teacher aggregation：AutoSSL/AGSSL 类方法。
4. pseudo-label 或 pair mining：PCL、CAST/BEACON/IRIS 类机制。
5. operator/response/diagnostic target：ORBIT/SPECTRA/PACER 等本项目 no-go。

缺口是：

> 在不使用真实标签的前提下，直接模拟“节点分类任务本身”的分布，并用这种任务族来训练 graph contrastive representation。

GRAFT-GCL 的核心假设是：相比预测输入、节点对、operator response 或单个 pretext，一个 encoder 若能在许多 label-free plausible node-classification tasks 上形成可线性分离的 representation，可能更贴近最终 frozen Logistic Regression evaluator。

## Hard Boundary for GRAFT

GRAFT 必须证明自己不是：

- pseudo-label self-training；
- clustering pseudo-label + SupCon；
- AutoSSL/AGSSL 多 pretext stacking；
- GraphPFN 式跨图 foundation model；
- label propagation / PPR mining；
- train-label probe routing；
- ORBIT response target 换名。

否则应立即 `KILL`。
