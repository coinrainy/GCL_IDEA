<!-- ARIS-CODEX:BEGIN -->
## ARIS Codex Skill Scope
ARIS Codex packages installed in this project: skills-codex
Managed entries: 80
Manifest: `.aris/installed-skills-codex.txt`
ARIS repo root: `/root/tools/aris`
Project skill path: `.agents/skills/<skill-name>`
For ARIS Codex workflows, prefer the project-local skills under `.agents/skills/`.
When a skill needs ARIS helper scripts, resolve the repo root from the manifest or set it explicitly:
`ARIS_REPO=$(awk -F'	' '$1=="repo_root"{print $2; exit}' "/root/autodl-tmp/GCL_IDEA/.aris/installed-skills-codex.txt")`
Do not edit or delete symlinked skills in place; update upstream or rerun:
`bash /root/tools/aris/tools/install_aris_codex.sh "/root/autodl-tmp/GCL_IDEA" --reconcile`
For copied Codex installs, use:
`bash /root/tools/aris/tools/smart_update_codex.sh --project "/root/autodl-tmp/GCL_IDEA"`
<!-- ARIS-CODEX:END -->

## 项目记录

# GCL Node Classification Research Contract

## Research Objective

本项目研究图对比学习用于节点分类，目标是形成一个问题定义清晰、
机制创新明确、实验可证伪、具有 2026 年顶会/顶刊投稿潜力的方法。

SOTA 不是预设事实，只能由严格、公平、可追溯的正式实验支持。

## Primary Scope

- 主任务：节点分类。
- 主线：图对比学习。

## Experimental Status Labels

所有实验必须标记为：

- smoke：只检查代码能否运行。
- pilot：用于判断 idea 是否继续，不支持论文性能 claim。
- development：用于开发和调参，不支持最终 claim。
- formal：固定协议后的正式结果，可进入论文表格。

禁止把 smoke、pilot、development 结果称为 SOTA、robust 或 comprehensive。

## Evaluation Integrity

- 只允许使用数据集真实标签和公认官方评测脚本。
- 所有方法必须使用相同 split 和相同 seed 列表。
- GCL 方法必须使用相同 frozen-encoder evaluator。
- GCL 方法的下游分类器默认且优先统一为 Logistic Regression；不再把 PyTorch `Linear` layer 训练器作为主评估口径。
- 在 `1:1:8` 协议下，Logistic Regression 的 `C`、正则项、solver 等 evaluator 参数只能由 train/val 或 train 内部 CV 确定，禁止用 test set 选择。
- GCN、GAT、GraphSAGE 等监督 baseline 不使用 GCL 的 frozen-encoder evaluator，但必须使用相同 split、相同 early stopping 规则。
- 没有官方划分时，必须保存并版本化 split 文件，优先使用 JSON 格式。
- 每次运行保存 commit、配置、seed、命令、日志、结果文件路径。
- 汇总表必须从原始 JSON/CSV 自动生成，不能手工复制最佳结果。
- 正式小型/中型实验默认 10 seeds，报告 mean±std 和每个 seed 原始值。
- 不删除失败结果或负结果。

## Baseline Rules

- 在实现主方法前，必须先复现最强相关 baseline。
- baseline 复现明显低于原论文时，不能据此宣称优于该 baseline。
- 优先使用作者代码或可信复现实现。
- 必须记录参数量、训练时间、显存和调参预算。
- 本方法不能拥有明显更大的搜索空间而不披露。

## Reviewer Routing

当需要 reviewer 时：

- novelty 和方法审查使用 fresh gcl_scientific_reviewer。
- 实验完整性审查使用 fresh gcl_experiment_auditor。
- result-to-claim 审查使用 fresh gcl_claim_reviewer。
- 独立 verdict 之间不得复用同一个 reviewer agent。
- reviewer 只接收文件路径、审查目标和输出格式。
- 不向 reviewer 提供 executor 的总结或希望得到的结论。

## Decision Discipline

每个阶段必须给出：

- GO
- REVISE
- PIVOT
- KILL
- BLOCKED

所有阶段决策写入 `research-wiki/log.md`，并在对应的 idea / experiment / claim 页面中保留状态。
失败 idea 必须写入 `research-wiki/ideas/`，避免后续重复生成。

## Data Split Rules

- 主实验默认采用 stratified random 1:1:8 split，即 train / validation / test = 10% / 10% / 80%。
- 所有方法必须使用完全相同的 split 文件。
- split 文件必须保存为 JSON 并版本化，不能运行时临时随机生成后不保存。
- 正式实验默认使用 seeds 0-9。
- 每个 seed 对应固定 split，并用于所有 baseline 和本方法。
- Wiki-CS、OGB 和异配图 fixed-split benchmark 优先遵循官方或公认划分。
- Planetoid public split 和 random 1:1:8 split 不能混在同一主表直接比较。
- 所有结果必须显式标注 split 类型。
- 若 split 协议不一致，禁止宣称方法优于对方。

## 任务同步记录

- 2026-06-25T17:57:43Z：已启用 `research-wiki/` 项目知识库。后续论文、idea、实验、claim 与阶段决策应同步写入对应页面，并通过 `research-wiki/log.md` 保留追加式审计记录。
- 2026-06-25T18:02:35Z：准备将项目初始化为 Git 仓库并公开上传至 GitHub `coinrainy/GCL_IDEA`；新增 `README.md` 和 `.gitignore`，公开仓库排除本机 `.agents/skills/` 绝对路径符号链接。
- 2026-06-25T18:03:22Z：已创建公开 GitHub 仓库并推送 `main` 分支，地址为 `https://github.com/coinrainy/GCL_IDEA`。
- 2026-06-25T18:26:29Z：已将 GRACE 官方实现 `https://github.com/CRIPAC-DIG/GRACE` 拉取到本地 `baselines/GRACE`，固定 commit `b3b5ac3fcbaabbb50e8bd69a075b46cd82a50378`；新增 `configs/grace_1_1_8.yaml` 和 `scripts/run_grace_1_1_8.py` 支持 stratified random 1:1:8 split、PyTorch `Linear` 诊断评估器、Amazon 数据集适配、结果 JSON 保存和截图目标值 gap 对照。Cora/CiteSeer 10 seeds development 复现已完成但低于截图目标，阶段决策为 `REVISE`，详见 `research-wiki/experiments/grace_baseline_1_1_8.md`。
- 2026-06-25T18:32:31Z：针对 GRACE Cora 复现 gap 完成 linear evaluator LR 诊断。`eval_lr=0.05` 将 Cora 10 seeds 从约 `81.44±0.91` 提升到 `81.83±1.22`，仍低于截图 `83.2±0.75`；判断 evaluator 参数有影响但不是唯一主因。脚本新增 `--eval-learning-rate`、`--eval-weight-decay`、`--no-normalize-embeddings` 便于后续诊断。
- 2026-06-25T18:34:19Z：项目评估口径更新：后续 GCL 方法统一使用 frozen encoder + Logistic Regression downstream classifier；PyTorch `Linear` layer 训练器仅可作为诊断或历史结果保留，不作为主表评估器。`1:1:8` 协议下 Logistic Regression 的 `C` 等 evaluator 参数必须由 train/val 或 train 内部 CV 确定，禁止 test set 调参。
- 2026-06-25T18:39:43Z：已按 Logistic Regression downstream classifier 重新复现 GRACE：Cora 10 seeds 为 `82.78±1.02`，CiteSeer 10 seeds 为 `71.59±0.52`，均明显接近截图目标；PubMed 完成 2 epoch smoke，`83.33`，仅证明链路可跑，不作为正式结果。阶段仍为 `REVISE`，等待 PubMed/DBLP/Amazon 全量长跑与协议核查。
- 2026-06-25T19:29:56Z：已将 GCN 官方实现 `https://github.com/tkipf/gcn` 拉取到本地 `baselines/GCN`，固定 commit `39a4089fe72ad9f055ed6fdb9746abdcfebc4d81`；新增 `configs/gcn_1_1_8.yaml`、`scripts/run_gcn_1_1_8.py` 和 `scripts/summarize_baseline_results.py`。GCN 作为监督 baseline 未改成 Logistic Regression，使用两层 `GCNConv` 端到端分类。development 10 seeds 汇总：Cora `83.52±0.94`、CiteSeer `72.61±0.52`、PubMed `86.49±0.27`、Wiki-CS `76.78±0.56`、Computers `89.84±0.39`、Photo `93.11±0.44`；除 Wiki-CS 基本对齐外多数高于截图目标，阶段决策为 `REVISE`，需继续核查截图对应的 GCN 协议。
- 2026-06-25T19:47:26Z：完成 GCN 高分原因诊断。未发现 split overlap 或分类器替换；主要问题是初始 GCN 配置容量过强（`hidden_dim=512/256`）且训练预算过长（`1000` epochs / `100` patience），不符合 `tkipf/gcn` 的 `hidden1=16`、`epochs=200`、`early_stopping=10`、validation-loss early stopping。已为脚本新增 `--early-stopping-metric val_loss`，并将 `configs/gcn_1_1_8.yaml` 的 Planetoid 默认改为保守官方式设置。诊断结果：Cora `82.52±0.97`、CiteSeer `72.22±0.69` 基本接近截图；PubMed `85.67±0.31`、Amazon 仍偏高，需继续核查 split/预处理/正则协议。
- 2026-06-25T19:11:25Z：继续复现剩余数据集。DBLP、Computers、Photo 的 `logreg_val` smoke 均通过；Amazon-Computers 初始配置全量跑到 6 seeds 后为 `84.06±0.96`，显著低于截图 `86.8±0.32`，判断为配置不完整/不匹配并中断剩余 seeds。已拉取 GCA 官方仓库 `https://github.com/CRIPAC-DIG/GCA` 作为 Amazon 参数线索，固定 commit `0f17497f3ae89718ed610c671a5b9f56033d4d51`；GCA Amazon 参数只能作为候选，不能直接声称为 GRACE baseline 官方配置。
- 2026-06-26T04:57:04Z：按用户要求不再使用 GCA 参数，完成 Amazon 受控小搜索。Computers 最佳 seed0/1 候选为 `lr=0.0005,tau=0.2,epochs=1000`，结果 `87.04±0.43`；Photo 最佳 seed0/1 候选为当前配置延长到 `epochs=1000`，结果 `91.95±0.32`。已更新 `configs/grace_1_1_8.yaml` 的 Amazon 主配置；这些仍是 candidate/development 证据，正式 claim 前必须跑 10 seeds。
- 2026-06-26T06:48:01Z：已用调参后的 GRACE Amazon 配置完成 10 seeds development 复现。Amazon-Computers 为 `86.63±0.49`，接近截图 `86.8±0.32`；Amazon-Photo 为 `91.93±0.54`，接近截图 `91.8±0.15`。结果仍为 development，不支持 formal/SOTA claim；详见 `research-wiki/experiments/grace_baseline_1_1_8.md`。
- 2026-06-25T19:15:20Z：已核验 GCA 官方实现位于 `baselines/GCA`，远端为 `https://github.com/CRIPAC-DIG/GCA.git`，当前固定 commit `0f17497f3ae89718ed610c671a5b9f56033d4d51`（`origin/main`）。该仓库作为 GCA baseline 复现与 Amazon 参数参考来源。
- 2026-06-25T19:16:07Z：已将 MVGRL 官方实现 `https://github.com/kavehhassani/mvgrl` 拉取到本地 `baselines/MVGRL`，固定 commit `628ed2bdb4496f8519556f7b9e25f93b57cf6902`（`master`）。该仓库作为 MVGRL baseline 复现来源；当前仅完成源码拉取与版本记录，尚未开始协议适配或实验运行。
- 2026-06-25T19:15:40Z：已将 GAT 官方实现 `https://github.com/PetarV-/GAT` 拉取到本地 `baselines/GAT`，固定 commit `5af87e7fce2b90ae1cbd621cd58059036a3c7436`（`master`）；该仓库作为监督 GAT baseline 复现来源，后续必须使用项目统一 split 与 early stopping 规则。
- 2026-06-25T19:49:32Z：已参考截图中 GAT 行完成项目协议版复现。新增 `configs/gat_1_1_8.yaml` 与 `scripts/run_gat_1_1_8.py`，完成 Cora/CiteSeer/PubMed/DBLP/Computers/Photo 10 seeds development runs，并由 `scripts/summarize_baseline_results.py` 自动汇总。结果：Cora `83.44±0.83`、CiteSeer `72.57±0.43` 基本接近截图；PubMed `85.05±0.29`、DBLP `82.60±0.30`、Computers `89.42±0.47`、Photo `93.16±0.29` 明显高于截图。阶段决策为 `REVISE`，详见 `research-wiki/experiments/gat_baseline_1_1_8.md`。
- 2026-06-26T05:51:13Z：已按用户要求只运行 `/research-lit`，为后续 GPT idea 生成整理文献与 gap map；新增 `literature/LIT_REVIEW_GCL_CORE.md`（21 篇 GCL / Graph SSL 相关论文）、`literature/LIT_REVIEW_TRANSFERABLE_METHODS.md`（18 篇可迁移方法论文）、`literature/GAP_MAP_FOR_GPT_IDEA_CREATOR.md` 与 `idea-stage/GPT_IDEA_CREATOR_CONTEXT.md`，并同步更新 `research-wiki/gap_map.md`、`research-wiki/query_pack.md` 和 `research-wiki/log.md`。本次未运行 `/idea-creator`、`/idea-discovery`，未实现代码，未跑 pilot/GPU 实验，未产生性能 claim。当前阶段 decision：`READY_FOR_GPT_IDEA_CREATOR`。
- 2026-06-26T06:41:16Z：继续按 `/research-lit` 深入图对比学习假负样本方向；新增 `literature/LIT_REVIEW_GCL_FALSE_NEGATIVES.md`（24 篇聚焦论文，其中图/GCL 直接相关 12 篇、通用 false-negative / hard-negative / PU learning 机制 12 篇）和 `literature/FALSE_NEGATIVE_GAP_MAP_FOR_GPT_IDEA_CREATOR.md`，并同步更新 `idea-stage/GPT_IDEA_CREATOR_CONTEXT.md`、`research-wiki/gap_map.md`、`research-wiki/query_pack.md`、`research-wiki/log.md`。本次未生成 idea，未实现代码，未跑 pilot/GPU 实验，未产生性能 claim。当前阶段 decision：`READY_FOR_FALSE_NEGATIVE_IDEA_CREATOR`。
- 2026-06-26T08:23:32Z：已按默认参数运行 `/wiki-enrich missing`。预检查通过，`research-wiki/` 与 `/root/tools/aris/tools/research_wiki.py` 均存在；但当前没有 `research-wiki/papers/*.md` 论文脚手架，因此候选论文数为 0，未补全文献页面、未修改 Connections/Abstract、未运行实验或产生 claim。
- 2026-06-26T08:25:34Z：完成 GAT 复现异常诊断。未发现 split overlap；主要问题是截图 target 与项目 `1:1:8` 协议不一致，尤其 PubMed 的训练节点数从 Planetoid public split 的 60 增至 `1:1:8` 的 1971（约 `32.9x`），导致本地 GAT PubMed 从截图约 `79%` 抬升到 `85.05%`。PubMed public split 5 seeds 诊断回落到 `77.40±0.22` / 官方式 checkpoint `77.66±0.29`。Amazon 开启 PyG `NormalizeFeatures()` 会明显打崩，不是直接对齐截图的修复。GAT 阶段决策仍为 `REVISE`，需单独实现 `paper_target_protocol`。
- 2026-06-26T08:44:00Z：按 `/idea-discovery` 继续 GPT idea-creator 后续步骤，完成 BCE-GCL 深度查新与 fresh `gcl_scientific_reviewer` 审查。新增 `idea-stage/BCE_GCL_NOVELTY_BRIEF_20260626_083442.md`、`idea-stage/BCE_GCL_NOVELTY_CHECK_20260626_083442.md`、`research-wiki/ideas/bce_gcl.md` 与 `research-wiki/ideas/cer_gcl.md`，并保存 reviewer trace 至 `.aris/traces/novelty-check/2026-06-26_run01/`。结论：BCE-GCL 原始 Boundary-Conditioned 叙事被判定 `PIVOT`，novelty score `4/10`；不得原样进入实现或 `/research-refine-pipeline`。允许转为 CER-GCL（Contrastive Eligibility Routing / Audit）后继续 refinement。未实现代码、未跑 pilot/GPU 实验、未产生性能 claim。
- 2026-06-26T08:54:00Z：完成 CER-GCL 的 `/research-refine-pipeline`。Round-1 方法审查为 `REVISE`、score `6.7`；按 reviewer 要求将三路 routing 收缩为二路 objective routing（eligible anchors 使用 full InfoNCE，ineligible anchors 使用 positive-only stop-gradient consistency），删除 `s_loss` gate，仅作 diagnostic；Round-2 复评为 `READY_FOR_PILOT_PLANNING`、score `7.7`。新增并同步 `refine-logs/FINAL_PROPOSAL.md`、`REVIEW_SUMMARY.md`、`REFINEMENT_REPORT.md`、`EXPERIMENT_PLAN.md`、`EXPERIMENT_TRACKER.md`、`PIPELINE_SUMMARY.md`。当前仅允许 smoke/pilot implementation；仍未实现代码、未跑实验、未产生性能 claim。
- 2026-06-26T08:56:00Z：已将 `idea-stage/IDEA_REPORT.md` 渲染为 `idea-stage/IDEA_REPORT.html`（遵循 `/idea-discovery` 的 `--no-review` 渲染语义，HTML 仅作阅读视图），并因本轮产物超过 15 个新增 `MANIFEST.md` 作为输出索引。Markdown 文件仍是 canonical source。
- 2026-06-26T08:50:34Z：用户明确要求放弃之前的 BCE/CER-GCL idea 工作并重新开始新方向；已将 `research-wiki/ideas/cer_gcl.md` 标记为 `ARCHIVED_BY_USER` / `abandoned_by_user`。后续不得继续该 idea 的实现、pilot 或 formal 实验，除非用户明确重新激活。
- 2026-06-26T08:57:00Z：已将 `.aris/traces/` 与 `.aris/meta/` 加入 `.gitignore`，保证 reviewer trace 与本地事件记录保留在本机、不误提交到公开仓库。
- 2026-06-26T09:01:32Z：已在排除归档 BCE/CER-GCL 旧线后生成 10 个新的普通节点分类 GCL idea，并写入 `idea-stage/FRESH_GCL_IDEAS_20260626_090132.md` 与固定入口 `idea-stage/FRESH_GCL_IDEAS.md`。Top1 为 `Ambiguity-Set Soft Target InfoNCE`，阶段决策为 `GO_FOR_TOP1_REFINEMENT`。本次未实现代码、未跑 pilot/GPU/formal 实验、未产生性能 claim。
- 2026-06-26T09:08:32Z：完成 fresh `/idea-discovery` 收束与 reviewer-informed refinement。Top idea 已重命名并收缩为 `IGT-GCL / Interval-Guarded Pair Targeting for Graph Contrastive Learning`；fresh `gcl_scientific_reviewer` 给出 `REVISE`、novelty score `5.5/10`、confidence `0.78`，允许进入收缩版 smoke/pilot planning，但不允许直接实现 formal claim。已更新 `idea-stage/IDEA_REPORT.md`、`refine-logs/FINAL_PROPOSAL.md`、`refine-logs/EXPERIMENT_PLAN.md`、`refine-logs/EXPERIMENT_TRACKER.md`、`research-wiki/ideas/igt_gcl.md` 与 `MANIFEST.md`，并保存 trace 至 `.aris/traces/novelty-check/2026-06-26_run02/`。本次未实现代码、未跑 smoke/pilot/full formal 实验、未产生性能 claim。最终 decision：`REVISE`。
- 2026-06-26T09:14:00Z：已将最新 `idea-stage/IDEA_REPORT.md` 重新渲染为 `idea-stage/IDEA_REPORT.html`（offline HTML 阅读版，source sha256 前缀 `9904a7884241`）。Markdown 仍为 canonical source，HTML 不作为 claim 依据。
- 2026-06-26T09:21:55Z：按 `/idea-creator` 仅做 false-negative GCL idea 生成，将 IGT-GCL 作为低新颖性反例，显式避开 point posterior / interval / abstention / simple sample reweighting 变体。新增 `idea-stage/FN_IDEAS_BEYOND_IGT_20260626_092155.md` 与固定入口 `idea-stage/FN_IDEAS_BEYOND_IGT.md`，生成 10 个候选 idea；Top 2 推荐为 `DSR-GCL / Decoupled Semantic-Residual GCL` 与 `CNG-GCL / Conflict-Negative Graph Selection`，并新增 `research-wiki/ideas/dsr_gcl.md`、`research-wiki/ideas/cng_gcl.md`。本次未改代码、未跑实验、未产生性能 claim。当前 decision：`GO_FOR_TOP2_REFINEMENT`。
- 2026-06-26T09:24:55Z：完成 `/idea-discovery` beyond IGT-GCL 的查新、review 与 refinement 收束。Top idea 为 `DSR-GCL / Decoupled Semantic-Residual Graph Contrastive Learning`，核心机制为 `Spectral Gradient Firewall`；fresh `gcl_scientific_reviewer` 给出 `REVISE_IDEA`、novelty score `6.2/10`、confidence `0.72`，认为仅小幅强于 IGT-GCL 的 `5.5/10`，尚不能进入 `GO_TO_EXPERIMENT_BRIDGE`。已更新 `idea-stage/IDEA_REPORT.md`、`idea-stage/IDEA_CANDIDATES.md`、`refine-logs/FINAL_PROPOSAL.md`、`refine-logs/EXPERIMENT_PLAN.md`、`refine-logs/EXPERIMENT_TRACKER.md`、`research-wiki/ideas/dsr_gcl.md` 与 `MANIFEST.md`，并保存 trace 至 `.aris/traces/novelty-check/2026-06-26_run03/`。本次未实现代码、未跑 smoke/pilot/full formal 实验、未产生性能 claim。最终 decision：`REVISE_IDEA`。
- 2026-06-26T09:30:00Z：已将最新 DSR-GCL 版 `idea-stage/IDEA_REPORT.md` 重新渲染为 `idea-stage/IDEA_REPORT.html`（offline HTML 阅读版，source sha256 前缀 `0a51c0f70551`）。Markdown 仍为 canonical source。
- 2026-06-26T09:40:55Z：按用户要求继续做一轮 DSR-GCL 机制 refine。已将 `Spectral Gradient Firewall` 压实为 `parameter-isolated two-channel firewall`，明确默认不使用 shared trunk；新增数学定义 `firewall_leak_param = ||grad(theta_sem; L_res)|| / (||grad(theta_res; L_res)|| + eps)`、训练伪代码、config switches、implementation checklist 与 A0-A11 ablation matrix。新增 `refine-logs/DSR_MECHANISM_SPEC.md` 并更新 `refine-logs/FINAL_PROPOSAL.md`、`EXPERIMENT_PLAN.md`、`EXPERIMENT_TRACKER.md`、`research-wiki/ideas/dsr_gcl.md` 与 `MANIFEST.md`。本次未实现代码、未跑 smoke/pilot/full formal 实验、未产生性能 claim。decision 仍为 `REVISE_IDEA`，但机制已达到未来 smoke 实现前的规格化状态。
- 2026-06-26T10:02:00Z：完成 DSR-GCL 最小 smoke（仅 Cora seed=0，stratified random `1:1:8`，frozen encoder + Logistic Regression evaluator）。新增 `configs/dsr_smoke.yaml` 与 `scripts/run_dsr_smoke.py`，结果保存在 `results/raw/Cora/DSR_GCL_SMOKE/dsr_smoke_Cora_seed0_20260626T095525Z/`，训练日志在 `logs/dsr_smoke/dsr_smoke_Cora_seed0_20260626T095525Z/`，汇总为 `results/summary/dsr_smoke_Cora_seed0_20260626T095525Z_summary.md`。A9 DSR-full 的 firewall leakage 为 `0.0`，A5 no-firewall leakage 为 `1.2867`，说明诊断可区分负梯度是否进入 semantic channel；但 A9 concat `77.08` test@best 低于 A2 semantic-only `78.27`、A5 no-firewall `81.13`、A4 same-head `83.63` 与 A0 GRACE `84.78`。阶段结论为 `REVISE/PIVOT_REQUIRED`，不进入 formal，不做性能 claim。
- 2026-06-26T10:11:03Z：完成 DSR-GCL audit-smoke（仍仅 Cora seed=0，stratified random `1:1:8`，frozen encoder + Logistic Regression evaluator），目标是审计失败原因和修正不公平 ablation。更新 `configs/dsr_smoke.yaml` 与 `scripts/run_dsr_smoke.py`，新增 A5a semantic-InfoNCE-only、A5b budget-matched no-firewall、A5c scaled no-firewall、A4b parameter-matched single-head control，并输出 embedding-level 表、参数量表、leakage 表、raw JSON 与 log 路径。结果：A9 DSR-full `78.69`，A3 residual-only `30.21`，A4 `83.63`，A4b `80.81`，A5b `79.57`，A5c `81.00`，A0 GRACE `84.78`。多条 kill rule 触发，最终阶段结论为 `PIVOT_REQUIRED`；禁止继续当前 DSR-GCL formal 或性能 claim。
- 2026-06-26T10:26:55Z：完成 DSR-GCL fix-audit-smoke（仍仅 Cora seed=0，不跑 formal/多数据集 pilot），用于判断上一轮失败是否来自实现/公式问题。修复 `scripts/run_dsr_smoke.py`：VICReg 改为作用在 raw projected `p_sem`，InfoNCE 内部 normalize，DSR 同时输出 `h_sem/h_res/h_concat` 与 `z_sem/z_res/z_concat`，主评估改为与 GRACE 公平的 encoder-level `h_concat`；`configs/dsr_smoke.yaml` 启用 `make_undirected_after_dropout` 以缓解 edge dropout 后非对称图的 low-pass 谱解释风险；`refine-logs/DSR_MECHANISM_SPEC.md` 同步为 symmetric VICReg-style objective，删除未实现的 stop-gradient alignment 叙述。结果：A9 fixed DSR-full `h_concat=69.05`，A2 `h_sem=68.68`，A3 `h_res=29.94`，A5b `76.57`，A5c `78.09`，A4b `81.96`，A0 GRACE `84.78`。由于发现 VICReg-normalization、evaluation representation、formula-code mismatch，上一轮 `PIVOT_REQUIRED` 暂时降级为 `REVISE_IMPLEMENTATION_BEFORE_PIVOT`；修复后单 seed 仍不支持 formal 或性能 claim。
- 2026-06-26T10:42:30Z：按 `/experiment-bridge` 执行 collection-only bridge。由于当前 `EXPERIMENT_TRACKER.md` 已将 DSR 状态标为 `REVISE_IMPLEMENTATION_BEFORE_PIVOT` 且 full deployment gate blocked，本次未启动新 GPU run、未跑 Pilot-A/B、未跑 formal；仅整理已有 Cora seed=0 fix-audit-smoke 为 `refine-logs/EXPERIMENT_RESULTS_20260626_104230.md` / `refine-logs/EXPERIMENT_RESULTS.md`，并新增 local-only code review `refine-logs/EXPERIMENT_CODE_REVIEW_20260626_104230.md` / `refine-logs/EXPERIMENT_CODE_REVIEW.md`。Bridge decision：`DONE_BLOCKED`，当前不 ready for `/auto-review-loop`。
- 2026-06-26T12:23:42Z：按 `/experiment-bridge` 为 IRIS-GCL 完成计划内最小 smoke。新增 `configs/iris_smoke.yaml` 与 `scripts/run_iris_smoke.py`，只运行 Cora seed=0、stratified random `1:1:8`、frozen GRACE encoder + Logistic Regression evaluator。结果保存在 `results/raw/Cora/IRIS_GCL_SMOKE/iris_smoke_Cora_seed0_20260626T121843Z/`，汇总为 `results/summary/iris_smoke_Cora_seed0_20260626T121843Z_summary.md/json`，并同步 `refine-logs/EXPERIMENT_RESULTS.md`、`EXPERIMENT_CODE_REVIEW.md`、`EXPERIMENT_TRACKER.md` 与 `research-wiki/ideas/iris_gcl.md`。I5 IRIS full 为 `76.48`、label agreement `0.2418`，明显弱于 I7 no anti-proximity `84.59` / `0.7787` 与 I4 CAST proxy `85.65` / `0.7548`；触发 kill rules，decision：`PIVOT_REQUIRED`。本轮不支持 Pilot-A/B、formal 或任何性能提升 claim。
- 2026-06-26T12:31:30Z：继续修正 IRIS 为 `R2-IRIS / Residualized Response Invariant Signatures`。在 `configs/iris_smoke.yaml` 和 `scripts/run_iris_smoke.py` 新增 I10-I13：residualized response、raw response no residual、residual+soft proximity penalty、response+CAST hybrid；residualizer 控制 feature similarity、embedding similarity、graph proximity 与 degree gap，不使用标签。仅重跑 Cora seed=0 smoke，汇总为 `results/summary/r2_iris_smoke_Cora_seed0_20260626T123039Z_summary.md/json`。I10 为 `84.46` / label agreement `0.7783`，I13 为 `84.87` / `0.7968`，修复了硬 anti-proximity 崩坏但仍未超过 CAST/PMGCL accuracy 或 kNN label agreement。decision：`REVISE_NOT_PILOT`；仍不支持 Pilot-A/B、formal 或性能提升 claim。
- 2026-06-26T10:49:51Z：按 `/idea-discovery` 重新探索图对比学习假负样本方向，目标偏向 2026 年论文潜力与更可能提升的实验路线。DSR-GCL 因 Cora seed=0 fix-audit-smoke 弱信号被降权，不继续作为当前主线；新 Top1 为 `BOND-GCL / Basin-capped Objective for Negative Debiasing`，核心机制是 `basin-level negative mass aggregation`。fresh `gcl_scientific_reviewer` 给出 `REVISE`、novelty `6.0/10`、confidence `0.74`，要求证明其不是 ordinary pair reweighting 或 E2Neg-style small-negative sampling。已更新 `idea-stage/IDEA_REPORT.md`、`idea-stage/IDEA_CANDIDATES.md`、`idea-stage/BOND_GCL_NOVELTY_CHECK.md`、`refine-logs/FINAL_PROPOSAL.md`、`refine-logs/EXPERIMENT_PLAN.md`、`refine-logs/EXPERIMENT_TRACKER.md`、`refine-logs/PIPELINE_SUMMARY.md` 与 `research-wiki/ideas/bond_gcl.md`。本轮未实现代码、未跑新实验、未产生性能 claim。当前 decision：`REVISE_TO_SMOKE_PLANNING`。
- 2026-06-26T10:50:00Z：已将 BOND-GCL 版 `idea-stage/IDEA_REPORT.md` 渲染为 `idea-stage/IDEA_REPORT.html`（offline HTML 阅读版，source sha256 前缀 `585d3d4483fc`）。Markdown 仍为 canonical source，HTML 不作为 claim 依据。
- 2026-06-26T11:16:23Z：根据用户反馈“BOND-GCL 只是在 loss 上做文章、创新性不足”，完成非 loss-only 假负样本方向 corrective `/idea-discovery`。BOND-GCL 已降级为 archived/mainline-baseline；新 Top1 为 `SIVA-GCL-positive-core / Semantic Stability Critic constrained Semantic-Preserving Intervention Positive`，核心机制是用 masked-context semantic stability critic 约束 node-local intervention-positive search，主动生成“足够不同但语义稳定”的正视图，减少对真实节点 hard negatives 的依赖。fresh `gcl_scientific_reviewer` 给出 `REVISE`、novelty `6.7/10`、confidence `0.70`，要求删除/降级 virtual-negative、加入 GraphMAE-only、critic-shuffled、random intervention 和 positive prealignment 指标对照。已更新 `idea-stage/IDEA_REPORT.md`、`idea-stage/IDEA_CANDIDATES.md`、`idea-stage/SIVA_GCL_NOVELTY_CHECK.md`、`refine-logs/FINAL_PROPOSAL.md`、`refine-logs/EXPERIMENT_PLAN.md`、`refine-logs/EXPERIMENT_TRACKER.md`、`refine-logs/PIPELINE_SUMMARY.md` 与 `research-wiki/ideas/siva_gcl.md`。本轮未实现代码、未跑新实验、未产生性能 claim。当前 decision：`REVISE_TO_SIVA_POSITIVE_SMOKE_PLANNING`。
- 2026-06-26T11:17:00Z：已将 SIVA-GCL 版 `idea-stage/IDEA_REPORT.md` 渲染为 `idea-stage/IDEA_REPORT.html`（offline HTML 阅读版，source sha256 前缀 `f1246689f917`）。Markdown 仍为 canonical source，HTML 不作为 claim 依据。
- 2026-06-26T11:27:25Z：根据用户继续要求“idea 要足够有创新性，不要只在 loss 或普通 critic 上做文章”，完成 WILLOW-GCL 方向 corrective `/idea-discovery`。SIVA-GCL 已降为 mandatory control；新活跃候选为 `WILLOW-GCL / Latent Ego Target-Prediction Certified Views`，核心机制是 `latent ego target-prediction certificate + certified intervention positive search`，即用 JEPA-like latent context-target prediction error 认证 node-local hard positive views。fresh `gcl_scientific_reviewer` 给出 `REVISE`、novelty `7.0/10`、confidence `0.68`，要求加入 Graph-JEPA-only、SIVA reconstruction-critic、matched random positive、certificate-shuffled WILLOW 等强对照。已更新 `idea-stage/IDEA_REPORT.md`、`idea-stage/WILLOW_GCL_NOVELTY_CHECK.md`、`refine-logs/FINAL_PROPOSAL.md`、`refine-logs/EXPERIMENT_PLAN.md`、`refine-logs/EXPERIMENT_TRACKER.md` 与 `research-wiki/ideas/willow_gcl.md`。本轮未实现代码、未跑 smoke/pilot/formal 实验、未产生性能 claim。当前 decision：`REVISE_TO_WILLOW_SMOKE_PLANNING`。
- 2026-06-26T11:28:00Z：已将 WILLOW-GCL 版 `idea-stage/IDEA_REPORT.md` 渲染为 `idea-stage/IDEA_REPORT.html`（offline HTML 阅读版，source sha256 前缀 `351f7cebe5af`）。Markdown 仍为 canonical source，HTML 不作为 claim 依据。
- 2026-06-26T12:15:00Z：继续根据用户要求提升 idea 创新性，完成 CAST-GCL corrective refinement。由于 WILLOW-GCL 仍可能被解释为 Graph-JEPA/PCR + learned augmentation scorer，已将其降为 module/control；新活跃候选为 `CAST-GCL / Certificate-guided Semantic Transport for Graph Contrastive Learning`。CAST 用 `latent ego target-prediction certificate` 评分跨节点低能量 intervention path，构造 multi-positive / neutral semantic closure，直接处理 false negatives 而不是调 loss。已新增 `idea-stage/CAST_GCL_CANDIDATE_20260626_121500.md`、`literature/CAST_PRIOR_BOUNDARY.md`，并更新 `idea-stage/IDEA_REPORT.md`、`refine-logs/FINAL_PROPOSAL.md`、`EXPERIMENT_PLAN.md`、`EXPERIMENT_TRACKER.md` 与 `research-wiki/ideas/cast_gcl.md`。本轮未运行 fresh reviewer、未实现代码、未跑 smoke/pilot/formal 实验、未产生性能 claim。当前 decision：`REVISE_TO_CAST_PRE_REVIEW`。
- 2026-06-26T12:16:00Z：已将 CAST-GCL 版 `idea-stage/IDEA_REPORT.md` 渲染为 `idea-stage/IDEA_REPORT.html`（offline HTML 阅读版，source sha256 前缀 `43eb58874792`）。Markdown 仍为 canonical source，HTML 不作为 claim 依据。
- 2026-06-26T12:35:00Z：已让 fresh `gcl_scientific_reviewer` 审查 CAST-GCL。结论为 `REVISE`、novelty `6.8/10`、confidence `0.72`；reviewer 建议保留 CAST 作为当前最好候选，不回退到 WILLOW，但必须把 transport 定义压实，并证明它不是 kNN/PPR/BMM/candidate-pool positive mining。已新增 `idea-stage/CAST_GCL_NOVELTY_CHECK.md`、`refine-logs/CAST_MECHANISM_SPEC.md`，并将 `FINAL_PROPOSAL.md`、`EXPERIMENT_PLAN.md`、`EXPERIMENT_TRACKER.md`、`research-wiki/ideas/cast_gcl.md` 更新为 `REVISE_TO_CAST_REVISED_PRE_SMOKE`。本轮未实现代码、未跑 smoke/pilot/formal 实验、未产生性能 claim。
- 2026-06-26T12:36:00Z：已将 revised CAST-GCL 版 `idea-stage/IDEA_REPORT.md` 渲染为 `idea-stage/IDEA_REPORT.html`（offline HTML 阅读版，source sha256 前缀 `eb847d096f7e`）。Markdown 仍为 canonical source，HTML 不作为 claim 依据。
- 2026-06-26T13:05:00Z：为寻找“最好的、足够创新”的 idea，新增 `IRIS-GCL / Interventional Response Invariant Signatures` challenger，并让 fresh `gcl_scientific_reviewer` 直接比较 CAST 与 IRIS。reviewer verdict 为 `SWITCH_TO_IRIS`，CAST novelty `6.5/10`，IRIS novelty `7.2/10`、confidence `0.61`；判断 CAST 更成熟但太接近 positive mining，IRIS 更高风险但更像 2026 论文主线。已将 fixed `IDEA_REPORT.md`、`FINAL_PROPOSAL.md`、`EXPERIMENT_PLAN.md`、`EXPERIMENT_TRACKER.md` 切到 IRIS，新增 `idea-stage/IRIS_GCL_CHALLENGER_20260626_130500.md`、`idea-stage/IRIS_VS_CAST_REVIEW.md`、`literature/IRIS_PRIOR_BOUNDARY.md` 与 `research-wiki/ideas/iris_gcl.md`；CAST 降为 mandatory control。当前 decision：`SWITCH_TO_IRIS_REVISE_BEFORE_SMOKE`。本轮未实现代码、未跑 smoke/pilot/formal 实验、未产生性能 claim。
- 2026-06-26T13:06:00Z：已将 IRIS-GCL 版 `idea-stage/IDEA_REPORT.md` 渲染为 `idea-stage/IDEA_REPORT.html`（offline HTML 阅读版，source sha256 前缀 `bca4c8056153`）。Markdown 仍为 canonical source，HTML 不作为 claim 依据。
- 2026-06-26T13:15:00Z：新增 `idea-stage/BEST_IDEA_DECISION.md` / `BEST_IDEA_DECISION_20260626_131500.md`，正式固化 idea-selection 结论：当前 best idea 为 `IRIS-GCL / Interventional Response Invariant Signatures`。该结论只完成 idea 选择，不代表实验验证；当前仍无代码实现、无 smoke/pilot/formal 结果、无性能 claim，下一步仅允许 Cora seed=0 smoke。
