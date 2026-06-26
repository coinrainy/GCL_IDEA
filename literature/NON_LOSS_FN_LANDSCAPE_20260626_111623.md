# 非 Loss-only 假负样本 GCL 文献边界

生成时间：2026-06-26T11:16:23Z  
子流程：`/idea-discovery` corrective rerun  
状态：literature landscape；不包含实验结果，不产生性能 claim

## 为什么 BOND 降级

BOND-GCL 的主机制是 basin-level denominator aggregation。它比普通 pair reweighting 稍有新意，但仍主要发生在 loss / denominator 层。用户反馈其“只是在 loss 上做文章”，该判断与 fresh reviewer 对 BOND 的风险判断一致。因此 BOND 不再作为当前主线，仅保留为未来 loss-side baseline / ablation。

## 新方向的边界

新的方向必须把创新放在训练信号来源上，而不是放在 loss 参数上。近期 prior 说明：

- CGC 已经覆盖 counterfactual hard negative generation。
- BalanceGCL 已经覆盖 graph-level balanced hard negatives + semantic positives。
- RGCL/GCA/ACGA 已覆盖 learned/adversarial augmentation。
- DoG 已覆盖 graph diffusion synthetic structures for node classification and GCL。
- SPGCL 2026 提出 message passing 会导致 positive pre-alignment，说明“让 positives 重新有信息量”是重要切口。
- GraphMAE/MaskGAE 等 masked modeling 可以作为 semantic critic，但不能成为全部贡献。

## 收束结论

最有希望的非 loss-only 切口是：

```text
SIVA-GCL-positive-core:
Semantic Stability Critic constrained Semantic-Preserving Intervention Positive
```

它不把 virtual hard negative 作为主贡献，而是先解决 positive pre-alignment：对每个 anchor 搜索一个尽可能远离原视图、但由 masked-context semantic critic 判定语义稳定的 node-local ego intervention positive。

## 阶段决策

Decision: `REVISE_TO_SIVA_POSITIVE_SMOKE_PLANNING`.

当前只允许写 proposal 和 smoke plan。必须通过 Cora seed=0 smoke 的机制指标，才能进入 pilot：

- SIVA-positive 的 view distance 高于 GRACE/GCA；
- semantic stability 不崩；
- post-GNN positive similarity / positive gradient 比 GRACE 更 informative；
- GraphMAE-only、critic-only、random intervention 不能匹配 SIVA。

