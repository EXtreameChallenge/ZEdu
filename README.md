---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: 'f7e2edf6-aedb-4d2f-b4e8-3ee78c421756'
  PropagateID: 'f7e2edf6-aedb-4d2f-b4e8-3ee78c421756'
  ReservedCode1: 'e82677f0-1651-4af2-8e17-98f70e9540a3'
  ReservedCode2: 'e82677f0-1651-4af2-8e17-98f70e9540a3'
---

# ZEdu Duo

> 双端双角色多智能体AI教育平台 — 让学生真正学会，而不是直接给答案

**当前版本：v4.0.41** | [查看更新日志](CHANGELOG.md)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19+-61dafb.svg)](https://react.dev/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-4.0.41-gold.svg)](CHANGELOG.md)

## 项目简介

ZEdu Duo 是一个基于多智能体协作的AI教育平台。"Duo"有三重含义：
- **双端**：Web端 + 桌面端（Electron）
- **双角色**：教师端（建课/班级/统计）+ 学生端（学习/练习/复习）
- **双闭环**：学习闭环（画像→诊断→路径→学习→评估）+ 错题闭环（8维诊断→溯源→变式→掌握）

### 产品定位：不给答案的 AI 陪练

当通用 AI 比谁给答案更快时，我们反着做：四阶段 Agent 流水线（画像→规划→苏格拉底引导→分题型验证）让学生自己推出答案。每一次作答实时更新 BKT 掌握度、自动沉淀 8 维错因诊断、按遗忘曲线安排复习——聊天记录变成学情数据，学情数据反过来决定下一题怎么教。全流程本地 SQLite 存储加教师端 AI 输出审批中心，学生练得放心、老师管得住。

## 版本演进

> ZEdu 是一个持续演进的大项目。本仓库（ZEdu Duo）为当前主版本，历史过渡版本（ZEdu v6.x、ZEdu Lite、Liquid Glass、Experiment）均属于同一项目体系。完整谱系见 [docs/version-history.md](docs/version-history.md)。

| 版本 | 时间 | 定位 |
|------|------|------|
| ZEdu v6.x | 2026-07 | 第一代 AI 私教（Java / Spring Boot / LangChain4j） |
| ZEdu Lite | 2026-08 | 轻量重装版（Python / FastAPI，5 分钟开箱即用，教育 AI 核心验证） |
| **ZEdu Duo（本仓库）** | 2026-09 至今 | 双端双角色多智能体平台，当前 v4.0.41 |
| ZEdu Liquid Glass / Experiment | 2026-09 | 设计 / 实验变体（与 Duo 同构） |

### 核心创新：四阶段Agent流水线

不同于传统AI助教直接给答案，ZEdu Duo通过四个专业Agent协作，引导学生自主思考：

```
Profiler（画像分析）→ Planner（路径规划）→ Tutor（苏格拉底引导）→ Verifier（答案验证）
```

- **Profiler**：分析学生历史答题数据，识别薄弱知识点和认知水平
- **Planner**：基于BKT知识追踪和最近发展区（ZPD）理论，规划最优学习路径
- **Tutor**：苏格拉底6阶段引导，通过提问启发学生，不直接给答案
- **Verifier**：数学题做数值容差比对，编程题用AST+测试用例，文科做质量评估（低置信度不判对错）

## 功能特性

### 学生端
- 多模式对话：普通问答 / 苏格拉底辅导 / 自适应出题 / 作文批改
- 个性化资源中心：规划→知识库接地→生成→审校→（按意见）修订 的多智能体流水线产出六类资源——讲解文档、思维导图（ReactFlow交互渲染）、练习题库、拓展阅读、代码实操、多模态教学动画（分镜驱动网页播放器）；阶段进度落库可轮询、可取消、刷新可续，引用溯源与审校结论随资源展示
- 学习仪表盘可视化：掌握度分布柱状图、学习状态构成、知识图谱×BKT掌握度着色视图
- 8维错题诊断：概念/策略/计算/迁移/审题/负荷/遗忘/前置缺失
- BKT知识追踪：80个预定义知识点实时掌握度更新
- GraphRAG知识引擎：三路检索 + 混合Rerank（语义×词法） + 引用溯源
- 自建高校课程知识库《人工智能导论》：13 章团队自撰讲义（308 个知识块、约 12.2 万字），标题感知分块、章节路径入向量、引用可回溯到具体小节；`scripts/build_course_kb.py` 一条命令重建并跑 13 道课内题命中自检（当前 13/13）
- 自研间隔重复调度：按遗忘曲线稳定性乘子安排复习时间（四档评分，非 fsrs 库）
- 知识图谱可视化：知识点层级关系和前置依赖一目了然

### 教师端
- 班级管理：创建班级、添加学生、查看班级概况
- 学生画像：每个学生的掌握度雷达图和薄弱点分析
- 班级统计：整体掌握度分布、常见错题、学习进度

### 工程特性
- 内容安全：敏感词过滤 + AI免责声明 + 输入限制
- 合规设计：年龄确认（14+）+ 隐私政策 + 数据可删除
- API限流：防止滥用和恶意调用
- 多模型容错：多Key轮询 + 降级链 + 响应缓存

## 技术栈

### 后端
- **语言**：Python 3.11+
- **框架**：FastAPI 0.115+
- **Agent编排**：LangGraph 0.2+（StateGraph + 条件边）
- **数据库**：SQLite（SQLAlchemy 2.0异步 + aiosqlite + NullPool + WAL）
- **向量库**：ChromaDB
- **知识追踪**：BKT 贝叶斯知识追踪（自实现4参数HMM，参考 pyBKT 算法）
- **间隔重复**：自研乘子式调度（借 FSRS 的遗忘曲线思想，未引入 fsrs 库）
- **认证**：JWT（python-jose）
- **文档解析**：python-docx（含表格）/ pypdf（逐页带真实页码）

### 前端
- **框架**：React 19 + TypeScript
- **构建**：Vite 6
- **样式**：Tailwind CSS 4
- **状态管理**：Zustand
- **图表**：Recharts
- **图谱可视化**：ReactFlow
- **Markdown渲染**：react-markdown + remark-gfm

### AI模型
- **主模型**：智谱GLM-4-Flash
- **降级模型**：通义千问 Qwen-Turbo / 讯飞星火 generalv3.5（可选通道，配置 XINGHUO_API_KEY 启用）
- **嵌入模型**：智谱 embedding-3（API 调用，1024 维）

## 快速开始

### 环境要求
- Python 3.11+
- Node.js 18+
- npm 或 pnpm

### 1. 克隆项目
```bash
git clone https://github.com/EXtreameChallenge/ZEdu.git
cd ZEdu
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env，填入你的API密钥
```

### 3. 启动后端
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```
后端将在 http://localhost:8000 启动，API文档在 http://localhost:8000/docs

### 4. 启动前端
```bash
cd frontend
npm install
npm run dev
```
前端将在 http://localhost:5173 启动

### 5. 演示账号
- 教师：teacher@demo.com / demo123
- 学生：student1@demo.com / demo123（另有 student2/student3@demo.com）

## 项目结构

```
ZEdu-Duo/
├── backend/              # Python FastAPI后端
│   ├── agents/           # LangGraph多智能体（核心）
│   ├── rag/              # GraphRAG知识引擎
│   ├── tutor/            # 教育核心算法（BKT/间隔重复调度/错题诊断）
│   ├── teacher/          # 教师端
│   ├── auth/             # 认证
│   ├── security/         # 安全中间件
│   ├── content_safety/   # 内容安全
│   ├── profile/          # 学生画像（6维）
│   ├── plan/             # 学习计划生成
│   ├── personas/         # 人格系统（4种教学人格）
│   ├── evolution/        # 画像自进化（辩证式）
│   ├── memory/           # 双维度记忆（总结/检索）
│   ├── resources/        # 个性化资源中心（六类资源流水线）
│   ├── skills/           # 技能系统
│   ├── settings/         # 系统设置
│   ├── approval/         # 审批系统
│   ├── tools/            # 工具系统（计算器/代码执行/知识检索）
│   ├── desktop/          # 桌面端适配（静态托管/环境）
│   └── compression/      # 上下文压缩
├── frontend/             # React前端
│   └── src/
│       ├── pages/        # 页面组件（10页）
│       ├── components/   # 通用组件
│       ├── hooks/        # 自定义Hooks（缓存/语音/动效偏好）
│       ├── i18n/         # 中英国际化
│       ├── stores/       # Zustand状态
│       └── api/          # API客户端
├── docs/                 # 设计文档（13篇 + 3篇比赛文档）
├── tests/                # 后端 pytest 用例（525 个：515 通过 / 10 跳过）
├── scripts/              # 工具脚本
├── knowledge-base/       # 预定义知识图谱 + course/ 自建课程讲义（13章）
└── .loop/                # Loop Engineering状态文件
```

## 测试

项目采用「后端 pytest + 前端 Vitest + E2E Playwright + 性能 Locust」四层测试金字塔。所有测试默认跑在隔离的临时 SQLite 库上，**不会触碰演示库 `data/zedu_duo.db`**（`tests/conftest.py` 在会话结束时会校验演示库未被改动）。

### 后端测试（pytest）

```bash
# 仓库根目录，使用项目自带 venv
.\venv\Scripts\python.exe -m pytest tests/ -v --tb=short

# 只跑某一类
.\venv\Scripts\python.exe -m pytest tests/test_data_privacy_compliance.py -v
.\venv\Scripts\python.exe -m pytest tests/test_security.py -v
```

`pytest.ini` 已配置 `asyncio_mode=auto`，异步用例无需手动装饰；`addopts` 默认带 `--cov=backend`，跑完整套件会自动输出覆盖率。

### 前端测试（Vitest + node:test）

```bash
cd frontend
npm test            # vitest run（src/**/*.test.ts(x)）
npm run test:unit   # node:test 跑 i18n / electron-main / markdown 纯逻辑用例
npm run test:all    # vitest + node:test 全量
```

### E2E 测试（Playwright）

```bash
cd frontend
npx playwright install --with-deps   # 首次安装浏览器
npx playwright test                  # 跑 e2e/*.spec.ts
```

E2E 需要后端（:8000）与前端（:5173）同时启动；用例会为每个场景注册独立用户，互不污染。

### 性能测试（Locust）

```bash
# 先启动后端，再在仓库根目录压测
.\venv\Scripts\locust.exe -f locustfile.py --headless `
  -u 10 -r 2 -t 60s --csv=.temp/perf_results --host http://127.0.0.1:8000
```

### 覆盖率报告

```bash
# 后端：终端 + XML（coverage.xml）
.\venv\Scripts\python.exe -m pytest tests/ --cov=backend --cov-report=term --cov-report=xml

# 前端：v8 覆盖率
cd frontend && npm run test:coverage
```

## 贡献指南

### 代码规范

- **Python**：[Ruff](https://docs.astral.sh/ruff/) 同时做 lint 与 format（配置见根目录 `ruff.toml`，line-length=120，启用 E/W/F/I/B/UP/C4）。
- **TypeScript/TSX**：ESLint v9（flat config）+ Prettier 格式化。

### Commit 规范（Conventional Commits）

提交信息统一遵循 [Conventional Commits](https://www.conventionalcommits.org/)，由 commitlint 在 `commit-msg` 钩子强制校验：

```
<type>(<scope>): <subject>
```

常用 type：`feat` / `fix` / `test` / `docs` / `refactor` / `perf` / `style` / `chore` / `ci`。

### 提交前钩子（pre-commit + husky）

```bash
pip install pre-commit
pre-commit install

cd frontend && npm install   # husky 钩子自动安装
```

### 提 PR 前自检清单

1. `ruff check .` 与 `cd frontend && npx tsc --noEmit`、`npx eslint src/` 全部通过。
2. 新增/修改后端行为时补充对应 pytest 用例；`pytest tests/` 全绿。
3. 前端改动跑 `npm run test:all`。
4. 涉及认证/隐私/安全的改动，确认 `tests/test_data_privacy*.py`、`tests/test_security*.py` 通过。
5. Commit message 符合 Conventional Commits。

## CI/CD

GitHub Actions（`.github/workflows/ci.yml`）在 push 到 `main/master/develop` 及向 `main/master` 提 PR 时触发，分两个并行 Job：

- **backend**（ubuntu-latest, Python 3.11）：装依赖 → `ruff check .` → `pytest tests/ --cov=backend` → `pip-audit` 依赖漏洞扫描。
- **frontend**（ubuntu-latest, Node 20）：`npm ci` → `tsc --noEmit` → `eslint src/` → 跑单测 → `npm run build` → `npm audit`。

## 开发路线图

- [x] 项目骨架和基础架构
- [x] 四阶段Agent流水线核心（LangGraph 7节点图 + SSE过程可视化）
- [x] 学生端完整闭环（画像→规划→引导→验证→复习）
- [x] 教师端MVP（班级/统计/画像）
- [x] 测试和文档（后端 pytest 525 用例 + 前端 Vitest 81 用例 + Playwright E2E 10 用例）
- [x] 个性化资源中心（5类多模态学习资源生成）
- [x] Docker 部署（docker-compose + Nginx + systemd 脚本）
- [x] Electron 桌面端壳（打包配置就绪）

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 致谢

感谢以下开源项目的启发：
- [MITS](https://github.com/Siesher/MITS) — 四阶段流水线架构
- [TraceBack](https://github.com/zxh123456-source/Trace-Back) — 苏格拉底错题系统
- [pyBKT](https://github.com/CAHLR/pyBKT) — 贝叶斯知识追踪
- [LangGraph](https://github.com/langchain-ai/langgraph) — 多智能体编排框架
- [fsrs4anki](https://github.com/open-spaced-repetition/fsrs4anki) — 间隔重复遗忘曲线的调度思想参考（代码自研）
- [Mr. Ranedeer](https://github.com/JushBJJ/Mr.-Ranedeer-AI-Tutor) — 六维个性化配置设计
- 辩证式自进化画像（Hermes 思路）— 正题/反题/合题的画像演进机制
- 多智能体协作可视化（Marvis 风格）— 智能体执行链实时呈现

主要开源依赖：
- [lucide-react](https://github.com/lucide-icons/lucide) — 图标库
- [framer-motion](https://github.com/motiondivision/motion) — 动效
- [recharts](https://github.com/recharts/recharts) — 数据可视化
- [reactflow](https://github.com/xyflow/xyflow) — 知识图谱渲染
- [react-markdown](https://github.com/remarkjs/react-markdown) + remark-gfm — Markdown 渲染
