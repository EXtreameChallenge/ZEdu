# ZEdu 版本演进史

> ZEdu 是一个持续演进的大项目。以下版本均属于同一项目体系，主仓库为当前 `ZEdu-Duo`，历史过渡版本按时间线归档记录。本文档由 2026-10-09 整理。

## 版本时间线

| 阶段 | 版本 | 时间 | 定位 | 技术栈 | 状态 |
|------|------|------|------|--------|------|
| ① 早期 | **ZEdu v6.x** | 2026-07-03 | 第一代 AI 私教（含华数杯竞赛实践） | Java 17 · Spring Boot 3.3.5 · React 19 · Electron 33 · LangChain4j | 历史归档 |
| ② 过渡 | **ZEdu Lite** | 2026-08-04 | 轻量重装版：80% 教育 AI 能力、20% 部署成本、5 分钟开箱即用 | Python 3.11 · FastAPI · SQLite · ChromaDB · React 19 · Electron 31 | 历史归档 |
| ③ 变体 | **ZEdu Liquid Glass** | 2026-09 | 设计变体：Liquid Glass 液态玻璃拟态 + Golden Time / Parchment Dark 双主题 | 与 ZEdu-Duo 同构 | 实验/归档 |
| ④ 变体 | **ZEdu Experiment** | 2026-09 | 功能实验版本 | 与 ZEdu-Duo 同构 | 实验/归档 |
| ⑤ 当前 | **ZEdu Duo**（本仓库） | 2026-09 至今 | 双端双角色多智能体 AI 教育平台，当前 v4.0.41 | Python 3.11 · FastAPI · LangGraph · SQLite · ChromaDB · React 19 · Electron | **主版本** |

## 各版本说明

### ① ZEdu v6.x（2026-07-03）

- 目录：`D:\Project\OpenSourcePlan\ZEdu20260703`（独立 git 仓库）
- 第一代产品，Java 后端 + LangChain4j 智能体，含华数杯竞赛题实践与论文材料
- 版本标识：v6.2.0

### ② ZEdu Lite（2026-08-04）

- 目录：`D:\Project\OpenSourcePlan\ZEduLite`（独立 git 仓库，另有 zip/tar 归档与汇报 PPTX）
- 定位：ZEdu 的轻量重装版——砍掉重运维依赖（MySQL/Redis/Kafka），保留全部教育 AI 核心：
  - 苏格拉底 6 阶段引导、8 维错题诊断、BKT 知识追踪、FSRS 间隔重复、GraphRAG 三路检索、自适应出题、作文四维批改、自进化闭环
- 新增极简体验：命令面板、5 分钟启动、Electron 双击（端口 8765）
- 技术栈：FastAPI (Python 3.11) · SQLite + Alembic · ChromaDB · GLM-4-Flash + 降级链 · React 19 + Vite 6 + Tailwind CSS 4 + shadcn/ui · Electron 31
- 设计语言：Golden Time / Parchment Dark 双主题、Liquid Glass 液态玻璃、Fraunces 衬线字体、20-48px 大圆角
- 文档体系：01-PRD 至 12-Roadmap 共 12 篇设计文档

### ③ ZEdu Liquid Glass（2026-09）

- 目录：`D:\Project\OpenSourcePlan\ZEdu-Liquid-Glass`（独立 git 仓库）
- 与 ZEdu-Duo 同构的完整项目（backend/frontend/tests/docs/packaging 全量），侧重 Liquid Glass 视觉设计验证

### ④ ZEdu Experiment（2026-09）

- 目录：`D:\Project\OpenSourcePlan\ZEdu-Experiment`（独立 git 仓库）
- 与 ZEdu-Duo 同构的完整项目，用于功能实验

### ⑤ ZEdu Duo（当前主版本）

- 本仓库，当前版本 v4.0.41，详见 [README](../README.md)
- 关键演进：Lite 阶段验证的教育 AI 核心（苏格拉底/BKT/8维诊断/GraphRAG/间隔重复）全部保留并升级为 LangGraph 四阶段 Agent 流水线（Profiler → Planner → Tutor → Verifier），新增双端（Web + Electron）、双角色（教师端 + 学生端）、双闭环（学习闭环 + 错题闭环）
- 新增能力：个性化资源中心（六类资源多智能体流水线）、自建高校课程知识库（《人工智能导论》13 章）、AI 输出审批中心、四层测试金字塔（525 个测试）

## 版本间关系

```
ZEdu v6.x (2026-07, Java)
      │  轻量重装
      ▼
ZEdu Lite (2026-08, Python/FastAPI) ── 教育AI核心验证
      │  双端/双角色/多智能体升级
      ▼
ZEdu Duo (2026-09 → 至今, 主版本 v4.0.41)
      ├── ZEdu Liquid Glass (设计变体)
      └── ZEdu Experiment (实验变体)
```

## 归档说明

- 各过渡版本保留在项目同级目录（`D:\Project\OpenSourcePlan\`），均为独立 git 仓库，未并入主仓库以保持主仓库整洁
- `ZEduLite.zip`（`D:\Project\OpenSourcePlan\ZEduLite.zip`）为 Lite 版打包归档，便于分发
- 主仓库历史中曾误提交的 203 MiB 比赛提交包（`submission/软件开发+ZEdu-Duo.zip`）已于 2026-10-09 通过 `filter-branch` 移除（本地备份 tag：`backup-before-filter`）
