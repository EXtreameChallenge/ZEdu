# ZEdu Duo 快速运行指南

## 系统要求
- Python 3.11+
- Node.js 18+
- npm 或 pnpm

## 快速开始

### 1. 配置环境变量
```bash
cp .env.example .env

# 编辑 .env，填入你的 AI 模型 API 密钥
# 至少需要配置一个模型通道（如智谱AI）才能使用AI功能
```

### 2. 安装后端依赖
```bash
cd backend
pip install -r requirements.txt
```

### 3. 安装前端依赖
```bash
cd frontend
npm install
```

### 4. 启动后端
```bash
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. 启动前端（开发模式）
```bash
cd frontend
npm run dev
```
浏览器访问 http://localhost:5173

### 6. 启动 Electron 桌面端（可选）
```bash
cd frontend
npm run electron:dev
```

## 演示账号（已内置在数据库中）
- 学生账号：student1@demo.com / demo123
- 教师账号：teacher@demo.com / demo123

## 数据库说明
- 主数据库：data/zedu_duo.db（SQLite，已包含演示数据）
- 向量数据库：data/chroma_db/（ChromaDB，已包含知识库索引）
- 定时任务数据库：data/scheduler.db
- Agent运行记录：data/agent_runs.db

## 项目结构
```
backend/          # FastAPI 后端 + LangGraph 多智能体引擎
frontend/         # React 19 + TypeScript + Vite + Electron
data/             # 数据库文件（已包含演示数据）
tests/            # 自动化测试（525个用例）
scripts/          # 工具脚本
docs/             # 文档
```

## 常见问题
- **AI功能不可用**：检查 .env 中是否配置了有效的 API 密钥
- **端口被占用**：修改 backend/config.py 中的端口配置
- **数据库报错**：删除 data/*.db 后重启，系统会自动初始化
- **前端白屏**：确认后端已启动，检查 Vite 代理配置
