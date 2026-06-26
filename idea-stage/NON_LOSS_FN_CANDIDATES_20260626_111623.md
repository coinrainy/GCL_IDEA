# 非 Loss-only 假负样本 GCL 新候选

生成时间：2026-06-26T11:16:23Z  
子流程：`/idea-discovery` corrective rerun  
触发原因：用户反馈 BOND-GCL 仍像 loss-side trick，创新性不足  
方向：图对比学习假负样本，面向普通节点分类，追求更像 2026 年论文主机制的 idea  
状态：candidate brief；未实现代码、未跑新实验、未产生性能 claim

## 本轮修正目标

BOND-GCL 的 reviewer 分数为 `6.0/10`，但其主机制仍是 loss-side denominator aggregation。用户判断“只是在 loss 上做文章，不太好”，这个判断成立。因此本轮将 BOND 降级为 future baseline / ablation，不再作为当前主线。

新的 idea 必须满足：

1. 主贡献不能是 InfoNCE 分母、pair weight、temperature、margin、采样概率等 loss-only 变化。
2. 必须改变训练信号的来源：view generation、counterfactual intervention、semantic augmentation、generative pretext、或 node-local environment construction。
3. 必须仍然服务 false-negative GCL：减少真实节点 negatives 的依赖，或让 positives 更有信息量，从源头缓解 false negatives。
4. 必须能在当前 GRACE/GCA 基线之上做 smoke，而不是需要重写整个项目。

## 新近文献边界

| Prior | 关键机制 | 对新 idea 的约束 |
|---|---|---|
| CGC / Generating Counterfactual Hard Negative Samples for GCL | 生成与原图相似但标签不同的人工 hard negatives | 不能只是“生成 hard negative”；必须做 node-level、无标签、不同生成接口 |
| BalanceGCL, AAAI 2026 | balanced hard negative graphs + fine-grained semantic-aware positive graphs | 不能只是 graph-level balanced positive/negative generation；节点级要有新定义 |
| RGCL | rationale-aware view generator 保留语义结构 | 不能只是 rationale subgraph / learned mask |
| ACGA | adversarial contrastive graph augmentation + counterfactual regularization | 不能只是 adversarial augmentation |
| DoG, 2025 | diffusion generates synthetic graph structures for node classification and GCL | 不能只是用 diffusion 造 synthetic nodes/edges |
| SPGCL, 2026 | message passing causes positive pre-alignment; use energy-aware propagation/positive sampling | 新方法应让 positive views 重新变得 informative，而不是只修 negatives |
| GraphMAE / Masked graph modeling | reconstruction provides generative self-supervision | 可作为 semantic stability critic，但不能变成 GraphMAE + GRACE 拼接 |

## Top Candidate: SIVA-GCL

**名称**：SIVA-GCL / Semantic Intervention View Augmentation for Graph Contrastive Learning  
**推荐级别**：Top 1 for fresh review  
**贡献类型**：node-level intervention view engine；不是 loss trick  

### 方法直觉

当前 GCL 把同一节点的两个随机增强视图当正样本，把其他真实节点当负样本。问题有两层：

- 正样本经 message passing 后可能已经预对齐，学习信号弱；
- 真实节点 hard negatives 很容易是假负样本。

SIVA-GCL 改变训练样本来源：不再主要从真实节点中找 hard negatives，而是为每个 anchor 的 ego-context 生成两类 **node-local intervention views**：

1. **semantic-preserving diverse positives**：尽可能改变 ego-context 中非语义因素，让正样本和原节点足够不同，但仍能被 masked-context critic 重构为同一语义代码。
2. **semantic-flipping virtual negatives**：通过最小干预把 anchor 的 masked-context semantic code 推向相邻 pseudo-code，形成虚拟 hard negative，而不是拿真实高相似节点当负样本。

### 具体构建

SIVA 由三个模块组成：

#### 1. Semantic Stability Critic

训练一个轻量 masked-context critic `C`，输入 anchor ego graph 的 masked feature / masked edge context，输出：

- reconstructed feature/context；
- EMA semantic code `q_i`；
- stability score `stab(v_i, intervention)`。

这个 critic 不使用 test label。它只判断“干预后是否还能从上下文恢复同一 anchor 的稳定语义”。

#### 2. Intervention View Search

对每个 anchor，在 feature mask、edge mask、neighbor-drop、ego-context substitution 中搜索干预 `a`：

```text
positive intervention:
  maximize view distance from original
  subject to C says semantic code stable

negative intervention:
  minimize edit distance
  subject to C says semantic code crosses to nearest non-self pseudo-code
```

第一版不需要训练复杂 generator，可用 differentiable mask + straight-through / greedy search 实现。

#### 3. Contrast with Virtual Views

训练时使用：

- anchor vs SIVA-positive：主 alignment；
- anchor vs SIVA-virtual-negative：hard negative；
- 真实节点 negatives 只保留 easy/random subset，降低 false-negative exposure。

InfoNCE 形式可以保留，但贡献不在 loss，而在 **干预视图工厂**。

### 与 prior 的差异

- vs CGC：CGC 生成 graph-level / sample-level counterfactual hard negatives；SIVA 是 node-local ego-context 干预，并同时生成 diverse positives 与 virtual negatives。
- vs BalanceGCL：BalanceGCL 依赖 graph-level class distribution/balancing；SIVA 不使用类别标签，不为每个 class 生成负图，而是用 masked-context semantic code 做无标签干预。
- vs RGCL / GCA：不是简单保留 rationale 或按重要性增强；SIVA 要求 positives 最大化非平凡变化，同时由 critic 约束语义稳定。
- vs SPGCL：SPGCL 调整 propagation/positive sampling；SIVA 直接生成 anti-prealignment positives，使正样本在 message passing 后仍有学习信号。
- vs DoG：DoG 生成 synthetic graph structures 加入原图；SIVA 生成 node-local training views，不改变原图数据集，不主打 graph diffusion。

### 为什么可能提升

- 真实 hard negatives 替换为 virtual negatives，降低 false-negative 采样风险。
- 正样本更 informative，避免 DSR/BOND 只在 negative side 修补。
- 第一版可接入 GRACE：先用现有 encoder + masked critic 生成 view masks，再沿用 frozen encoder + LogReg 评估。

### 最小 smoke

- Cora seed=0。
- 对照：GRACE、GCA-style random augmentation、GraphMAE-only、positive-only SIVA、virtual-negative-only SIVA、SIVA full、random intervention control、semantic critic shuffled control。
- 成功信号：SIVA-positive 的 view distance 大于 GRACE augmentations，但 semantic reconstruction/stability 不崩；virtual negatives 的 label-only offline false-negative rate 低于 real hard negatives；LogReg 不低于 GRACE。

### 风险

- critic 可能学到 trivial identity / degree signal。
- virtual negative 可能离数据流形太远。
- 实现复杂度高于 BOND，需要先做 non-generator greedy SIVA smoke。
- reviewer 可能认为这是 CGC + GraphMAE + GCA 的组合，需要用 node-local intervention contract 和 anti-prealignment positives 来收缩主贡献。

## Backup 1: MINT-GCL

**名称**：MINT-GCL / Masked Intervention Tokenizer for Graph Contrastive Learning  
**机制**：把每个 ego graph 压缩成若干 semantic intervention tokens，训练时 contrast node 与 tokens，而不是 node-node negatives。  
**优点**：更像 tokenizer / representation interface；false negatives 不再来自真实节点。  
**风险**：容易被 prototype / clustering / VQ-GraphMAE prior 吸收，工程较重。

## Backup 2: DoCI-GCL

**名称**：DoCI-GCL / Diffusion-on-Context Interventions for GCL  
**机制**：借鉴 DoG，但不是生成节点加入图，而是用轻量 diffusion/denoising 生成 anchor ego-context 的 counterfactual training views。  
**优点**：生成式更强，2026 观感好。  
**风险**：DoG 太近，且 diffusion 成本高，不适合首轮 smoke。

## Backup 3: ORBIT-GCL

**名称**：ORBIT-GCL / Structural-Orbit Positive Synthesis for Node-level GCL  
**机制**：用局部结构 orbit / motif role 找到可交换 ego-context，合成 positives，减少真实 negatives。  
**优点**：非 loss，解释性强。  
**风险**：可能只适合结构角色明显的数据，Cora/Amazon 上信号不稳。

## 初步排序

| Rank | Idea | 创新性预估 | 工程难度 | 提升概率 | 当前建议 |
|---:|---|---|---|---|---|
| 1 | SIVA-GCL | 中高 | 中高 | 中 | 进入 fresh reviewer |
| 2 | MINT-GCL | 中高 | 高 | 不确定 | backlog |
| 3 | DoCI-GCL | 中 | 高 | 不确定 | 不首推 |
| 4 | ORBIT-GCL | 中 | 中 | 低中 | backup |
| 5 | BOND-GCL | 中低 | 中 | 中 | 降级为 loss baseline |

## 给 reviewer 的问题

1. SIVA 是否比 BOND 更像一个可投稿的机制贡献？
2. SIVA 是否仍被 CGC / BalanceGCL / RGCL / ACGA / DoG 覆盖？
3. 如果要让它足够新，应该保留哪个最小核心：semantic critic、positive intervention、virtual negatives，还是三者必须同时存在？
4. 最小 smoke 应如何设计，避免一开始就工程过重？

