# GCL Node Classification Research

本仓库用于记录图对比学习在节点分类任务上的研究计划、实验协议和研究 wiki。

当前阶段以问题定义、benchmark 协议、文献与实验审计规则为主。仓库中的内容不构成 SOTA 或最终性能 claim；任何正式结论必须来自冻结协议后的 formal 实验，并保留可追溯的 split、seed、配置、日志和原始结果。

## 核心文件

- `AGENTS.md`：项目研究约束、评测纪律和任务同步记录。
- `BENCHMARK_PROTOCOL.md`：节点分类 benchmark 协议草案。
- `research-wiki/`：研究日志、gap map、query pack 与知识库索引。
- `.codex/agents/`：只读 reviewer / auditor agent 配置。

## 实验原则

- 主任务：节点分类。
- 主线：图对比学习。
- 默认 split：stratified random `1:1:8`，即 train / validation / test = `10% / 10% / 80%`。
- formal 实验默认 10 seeds，报告 mean±std 和每个 seed 原始值。
- pilot、development、smoke 结果不能用于论文主 claim。
