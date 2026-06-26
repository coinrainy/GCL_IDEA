# IPU-GCL Novelty Brief

生成时间：2026-06-26T09:03:47Z  
阶段：fresh `/idea-discovery`，旧 BCE/CER-GCL 已归档，不作为候选继续推进。  
对象：IPU-GCL / Interval Positive-Unlabeled Graph Contrastive Learning。

## Proposed Method

IPU-GCL 针对节点级图对比学习中的假负样本与 hard negative 混淆问题。它不把非增强节点对直接压成一个点式 pseudo-positive probability，而是为每个 anchor-candidate pair 构造正样本概率区间 `[l_ij, u_ij]`：

- `l_ij` 高：把节点 `j` 作为 anchor `i` 的保守软正样本，进入 numerator。
- `u_ij` 低且当前 embedding similarity 高：把节点 `j` 作为 reliable hard true negative，进入加权 denominator。
- `l_ij` 低但 `u_ij` 高：视为 ambiguous pair，不转正，也不强推远，只保留 stop-gradient denominator 或完全 abstain。

该方法作为 GRACE/GCA loss 插件实现，不改变 encoder、数据 split、evaluator 或 baseline 预算。

## Core Claims To Check

1. **Interval semantics**：图节点 pair 的语义关系在训练早期不可精确识别，用 `[l,u]` 区间比单点 pseudo-label 更适合处理 false negative / hard negative 混淆。
2. **Conservative attraction**：只拉近 high-lower-bound candidate positives，可以缓解 positive scarcity，同时降低误转正风险。
3. **Reliable hard negatives**：只从 low-upper-bound 且 embedding-hard 的候选中选 hard negatives，可以避免普通 hard negative mining 选中 false negatives。
4. **GRACE/GCA-compatible mechanism**：该方法只改 InfoNCE target construction / pair mask，不引入 learnable negative metric network、KAN、大型 prototype 或 LLM 语义视图。

## Closest Prior Work

| Prior work | Overlap | Needed delta |
|---|---|---|
| ProGCL | 区分 hard negative 与 false negative，估计 negative 为 true negative 的概率。 | IPU-GCL 需要证明 interval target 不是 ProGCL-style probability weighting，而是同时控制 attraction / repulsion / abstention。 |
| Enhancing GCL with Node Similarity, KDD 2024 | 理想目标、扩正样本、去假负、node similarity distribution。 | IPU-GCL 需要避免只提出另一个 similarity；核心必须是 lower/upper bound 和 ambiguous pair abstention。 |
| IFL-GCL, SIGIR 2025 | PU learning 视角，用 InfoNCE similarity 提取语义信息并 corrected InfoNCE。 | IPU-GCL 的差异应是不用点估计 posterior 直接重采样，而是保留 epistemic uncertainty interval，防止 self-training overconfidence。 |
| NML-GCL, IJCAI 2025 | learnable negative metric network 区分 false negatives / true negatives，bi-level training。 | IPU-GCL 应保持轻量、非参数或半参数，无额外 metric learner；主要贡献在 interval target construction。 |
| Soft Target InfoNCE / DRO-InfoNCE | 通用 soft target、distributional robustness、ambiguity set contrastive learning。 | IPU-GCL 需要把 novelty 锚定到 graph pair semantics、positive scarcity 与 false-negative/hard-negative 分离，而不是泛泛“soft target”。 |
| Incremental False Negative Detection, ICLR 2022 | 随训练动态检测并移除 false negatives。 | IPU-GCL 不能只做 curriculum removal；必须证明 interval lower/upper bounds 比 hard removal 更稳。 |
| Khan-GCL / generated hard negatives | 构造更有区分性的 hard negatives。 | IPU-GCL 不生成样本，只选择 reliable hard true negatives；避免 KAN/生成器复杂度。 |

## Proposed Minimal Mechanism

1. Warmup：用 vanilla GRACE 预训练 `w` epochs，并记录多增强、多 checkpoint 的 pair similarity / rank stability。
2. 区间构造：
   - `s_feat`: feature cosine 或 feature-neighborhood agreement。
   - `s_struct`: PPR / shared-neighbor / k-hop proximity。
   - `s_embed`: EMA embedding similarity。
   - `s_stab`: pair rank stability across augmentations/checkpoints。
   - `l_ij = conservative_agreement(s_feat, s_struct, s_embed, s_stab)`。
   - `u_ij = liberal_union(s_feat, s_struct, s_embed, s_stab)`。
3. Loss：
   - self positive always positive。
   - high `l_ij` candidate positives get small soft-positive mass。
   - low `u_ij` + high current similarity pairs become reliable hard negatives。
   - ambiguous pairs abstain from strong gradients。
4. Evaluation：frozen encoder + Logistic Regression；同配图 `1:1:8`；异配图 official fixed split。

## Mandatory Ablations

- vanilla GRACE / GCA。
- point-estimate soft target, matching IFL-style single posterior as closely as possible。
- only negative downweight。
- only positive expansion。
- no abstention: ambiguous pairs forced into point targets。
- interval without stability。
- interval without structure。
- interval without feature。
- full IPU-GCL。

## Stop Rules

- 若 full interval 不优于 point-estimate soft target，则不能声称 interval 是核心贡献。
- 若 only positive expansion 或 only negative downweight 匹配 full method，则不能声称吸引/排斥联合约束是核心贡献。
- 若 ambiguous abstention 导致 uniformity/rank collapse，则停止或降级为 diagnostic。
- 若收益只在同配 Planetoid 数据出现，在 Amazon/heterophily fixed split 明显恶化，则不能声称通用 false-negative solution。
- 所有结果在 formal 前只能是 pilot/development，不支持 SOTA、robust、comprehensive claim。

## Desired Reviewer Output

请判断该 idea 是否值得进入 `/research-refine-pipeline`，输出：

1. Verdict：GO / REVISE / PIVOT / KILL / BLOCKED。
2. Novelty score：0-10。
3. 最接近 5 篇 prior work：overlap 与 remaining delta。
4. 对四个 core claims 分别判定 HIGH / MEDIUM / LOW novelty。
5. 若不是 KILL：给出最小可保留版本、必须改名/改叙事点、必须做 ablation、必须避免的 claim。
6. 一句话结论：是否允许进入 method refinement 和 experiment planning。
