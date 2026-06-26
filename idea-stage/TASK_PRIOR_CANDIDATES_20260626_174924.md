# Task-Prior Pivot Candidates after ORBIT No-Go

- 时间：2026-06-26T17:49:24Z
- 输入证据：ORBIT-GCL `OR-M0-001` negative smoke
- 约束：禁止继续 ORBIT 小调；禁止继续 response/operator/basis 目标
- Claim status：idea generation only；未实现、未跑实验。

## Failure Synthesis

ORBIT 的结果给出一个有用信号：response-basis target 可以超过 Graph-JEPA、GCMAE-style 和 same-node response control，但不能超过 GRACE/matched GRACE。这意味着“更精细的自监督 target”并不会自动转化为 frozen Logistic Regression accuracy。

结合此前 no-go：

- pair mining / certificate：CAST、IRIS、BEACON 未稳定超过 kNN/CAST controls。
- representation diagnostics：SPECTRA rank/boundary 改善但 accuracy 下降。
- evaluator proxy：PACER margin 改善但 random route / SupCon control 更强。
- masked evidence：MEGA 明显低于 GRACE/GCMAE-style。
- operator response：ORBIT 低于 matched GRACE。

下一轮必须改变 supervision object，而不是继续调 target 细节。

## Candidate Ranking

### 1. GRAFT-GCL / Graph Prior-Task Fitted Contrastive Learning

**核心机制**：在同一个真实图上生成一族 label-free plausible node-classification tasks，每个 task 给出 synthetic class partitions。Encoder 通过 task-conditioned supervised contrastive meta-risk 学表示，真实标签只在最终 frozen Logistic Regression evaluator 中使用。

为什么是机制级 pivot：

- supervision unit 从节点对/response/输入 token 改成“下游任务样本”；
- 不是伪标签真实任务，而是任务先验采样；
- 不是多 pretext stacking，而是一个显式 label-generating prior；
- 可以形成完整论文：problem、method、task prior theory、controls、ablation、failure modes。

状态：`TOP1_REVISE_TO_PRE_REVIEW_AND_M0_SMOKE_PLANNING`。

### 2. FACTOR-GCL / Feature-Topology Factorized Task Prior

只学习 feature-only、topology-only、feature-topology interaction 三类任务先验，并用 factor consistency 约束 representation。创新性较低，容易变成 GRAFT 的 ablation。

状态：`KEEP_AS_ABLATION`。

### 3. CURVE-GCL / Curriculum over Synthetic Homophily Regimes

按 homophily/heterophily 难度逐步生成 synthetic tasks，训练 encoder 适应多种 edge-label correlation。风险是太像训练策略或 curriculum trick。

状态：`WEAK_PIVOT`。

### 4. TASK-JEPA-GCL

从 partial synthetic labels 预测 held-out synthetic task assignments。风险是回到 JEPA/GraphMAE target prediction。

状态：`KILL_AS_TOO_CLOSE_TO_JEPA`。

### 5. PRIOR-PACER

用 train-label validation 选择 task-prior mixture。风险是回到 PACER 的 train-label route，且可能引入 evaluator leakage 争议。

状态：`KILL_FOR_THIS_STAGE`。

## Selected Direction

选择 **GRAFT-GCL**，但只允许进入 pre-review + Cora seed0 smoke planning。实现前必须补一个 fresh scientific review 或至少一次严格 novelty audit；本轮不启动代码。

