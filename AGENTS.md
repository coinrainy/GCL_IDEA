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
- 2026-06-25T19:15:20Z：已核验 GCA 官方实现位于 `baselines/GCA`，远端为 `https://github.com/CRIPAC-DIG/GCA.git`，当前固定 commit `0f17497f3ae89718ed610c671a5b9f56033d4d51`（`origin/main`）。该仓库作为 GCA baseline 复现与 Amazon 参数参考来源。
- 2026-06-25T19:16:07Z：已将 MVGRL 官方实现 `https://github.com/kavehhassani/mvgrl` 拉取到本地 `baselines/MVGRL`，固定 commit `628ed2bdb4496f8519556f7b9e25f93b57cf6902`（`master`）。该仓库作为 MVGRL baseline 复现来源；当前仅完成源码拉取与版本记录，尚未开始协议适配或实验运行。
- 2026-06-25T19:15:40Z：已将 GAT 官方实现 `https://github.com/PetarV-/GAT` 拉取到本地 `baselines/GAT`，固定 commit `5af87e7fce2b90ae1cbd621cd58059036a3c7436`（`master`）；该仓库作为监督 GAT baseline 复现来源，后续必须使用项目统一 split 与 early stopping 规则。
- 2026-06-25T19:49:32Z：已参考截图中 GAT 行完成项目协议版复现。新增 `configs/gat_1_1_8.yaml` 与 `scripts/run_gat_1_1_8.py`，完成 Cora/CiteSeer/PubMed/DBLP/Computers/Photo 10 seeds development runs，并由 `scripts/summarize_baseline_results.py` 自动汇总。结果：Cora `83.44±0.83`、CiteSeer `72.57±0.43` 基本接近截图；PubMed `85.05±0.29`、DBLP `82.60±0.30`、Computers `89.42±0.47`、Photo `93.16±0.29` 明显高于截图。阶段决策为 `REVISE`，详见 `research-wiki/experiments/gat_baseline_1_1_8.md`。
