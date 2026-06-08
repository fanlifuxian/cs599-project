# 🧑‍⚕️ 个性化健康规划多智能体平台

> **Personalized Health Planning Multi-Agent Platform**

一个基于 LangGraph 的多智能体协作系统，整合饮食规划、运动指导、睡眠管理和健康咨询四大专业 Agent，为用户提供个性化的健康管理建议。

---

## 项目简介

本项目是 **CS599 企业级应用软件设计与开发** 课程大作业，选择 **方向一：Agentic AI 原生开发**。

通过 **Supervisor-Worker 多智能体协作模式**，将用户健康需求自动路由到饮食、运动、睡眠三个专业 Agent，最后由咨询总控 Agent 整合输出，形成完整的健康规划闭环。

---

## 方向

🎯 **方向一：Agentic AI 原生开发** — 多智能体协作系统

---

## 技术栈

| 类别 | 技术 |
|------|------|
| **AI IDE** | Trae CN |
| **LLM** | DeepSeek API (也支持 OpenAI 兼容 API) |
| **Agent 框架** | LangGraph + LangChain |
| **协议** | Function Calling (OpenAI 兼容) |
| **语言** | Python 3.11+ |
| **UI** | Streamlit (Web Demo) |
| **容器** | Docker |
| **基础设施** | GitHub |

### 核心技术要素（6/6 全覆盖）

| 要素 | 实现方式 |
|------|----------|
| ✅ SDD 规格驱动开发 | Pydantic 数据模型作为可执行规格 + docs/ 规格文档 |
| ✅ 工具使用 / Function Calling | 每个 Agent 配备 3-4 个领域工具函数，OpenAI 兼容 Tool Schema |
| ✅ 记忆机制 | 短时记忆 (LangChain ConversationBuffer) + 长时记忆 (JSON 文件持久化) |
| ✅ 状态管理与多步骤推理 | LangGraph StateGraph，Supervisor-Worker 路由状态机 |
| ✅ 多智能体协作 | 4 Agent 协作：Consultation → Diet/Exercise/Sleep → Synthesize |
| ✅ 可观测性与评估 | 结构化日志、LangGraph 追踪、Agent 置信度评分 |

---

## 架构概览

```
User Input → Supervisor (Consultation Agent)
                ↓
         ┌──────┼──────┐
         ↓      ↓      ↓
      Diet   Exercise  Sleep
      Agent   Agent    Agent
         ↓      ↓      ↓
         └──────┼──────┘
                ↓
         Supervisor Synthesizes → User Response
```

### 四个 Agent 的职责

| Agent | 角色 | 工具 |
|-------|------|------|
| 🥗 **Diet Agent** | 饮食营养专家 | `calculate_bmr`, `calculate_tdee`, `generate_meal_plan`, `analyze_nutrition` |
| 🏃 **Exercise Agent** | 运动健身专家 | `generate_workout_plan`, `recommend_exercise`, `calculate_calorie_burn` |
| 😴 **Sleep Agent** | 睡眠健康专家 | `analyze_sleep_quality`, `generate_sleep_plan`, `get_sleep_hygiene_tips` |
| 🧠 **Consultation Agent** | 健康咨询总控 | `get_user_profile_summary`, `set_health_goals`, `track_progress`, `health_risk_assessment` |

---

## 目录结构

```
cs599-project/
├── docs/                          # 项目文档（大作业报告 PDF）
├── src/
│   ├── agents/                    # Agent 实现
│   │   ├── base_agent.py          # 基类：LLM + 工具调用循环
│   │   ├── diet_agent.py          # 饮食 Agent
│   │   ├── exercise_agent.py      # 运动 Agent
│   │   ├── sleep_agent.py         # 睡眠 Agent
│   │   └── consultation_agent.py  # 咨询总控 Agent（Supervisor）
│   ├── tools/                     # Function Calling 工具
│   │   ├── diet_tools.py          # 饮食相关工具
│   │   ├── exercise_tools.py      # 运动相关工具
│   │   ├── sleep_tools.py         # 睡眠相关工具
│   │   └── common_tools.py        # 通用工具（档案、目标、追踪）
│   ├── graph/                     # LangGraph 编排
│   │   ├── state.py               # 图状态定义
│   │   └── health_graph.py        # Supervisor-Worker 图构建
│   ├── memory/                    # 记忆管理
│   │   └── memory_manager.py      # 会话缓冲 + 文件持久化
│   ├── models/                    # Pydantic 数据模型（SDD 可执行规格）
│   │   └── schemas.py
│   ├── config/                    # 配置管理
│   │   └── settings.py            # 环境变量配置（无硬编码 Key）
│   ├── ui/                        # Web 界面
│   │   └── streamlit_app.py       # Streamlit Demo UI
│   └── main.py                    # CLI 入口
├── tests/                         # 测试
├── .env.example                   # 环境变量模板
├── .gitignore
├── Dockerfile
├── LICENSE                        # MIT
├── requirements.txt
└── README.md
```

---

## 环境搭建

### 1. 克隆仓库

```bash
git clone <your-repo-url>
cd cs599-project
```

### 2. 安装依赖

```bash
# 推荐使用虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key
# 必填：DEEPSEEK_API_KEY=sk-xxxxxxxx
```

⚠️ **严禁在代码中硬编码 API Key！** 所有密钥通过 `.env` 文件管理。

### 4. 启动应用

**CLI 模式：**
```bash
python src/main.py
```

**Web UI 模式（推荐 Demo 使用）：**
```bash
streamlit run src/ui/streamlit_app.py
```

### 5. Docker 部署

```bash
# 构建镜像
docker build -t health-agents .

# 运行容器
docker run -p 8501:8501 --env-file .env health-agents

# 访问 http://localhost:8501
```

---

## 使用示例

### 示例对话 1：减重咨询
```
用户: 我想科学减重，身高170cm，体重80kg，平时久坐
系统: [Consultation Agent 路由到 DietAgent + ExerciseAgent]
🥗 Diet Agent: BMR=1705kcal, TDEE=2046kcal, 建议每日摄入1546kcal...
🏃 Exercise Agent: 推荐初级减重计划：快走+深蹲+开合跳...
🧠 Consultation: [整合建议] 饮食热量缺口500kcal + 每周3次运动...
```

### 示例对话 2：睡眠改善
```
用户: 我最近总是睡不着，每天只睡5小时
系统: [Consultation Agent 路由到 SleepAgent]
😴 Sleep Agent: 睡眠质量评分45(D)，建议：固定22:30上床，睡前1小时禁用手机...
```

---

## 项目状态

- [x] Proposal — 架构设计与 Spec 初稿
- [x] MVP — 核心闭环 Demo (v0.1)
- [x] Final — 完整代码 + 文档

---

## 学术声明

本项目为 CS599 课程大作业，遵循学术诚信原则：
- 代码基于 LangGraph/LangChain 开源框架构建
- LLM API 调用使用 DeepSeek 官方 API
- 工具函数中的营养/运动数据部分基于公开科学公式（Mifflin-St Jeor、MET 等）
- 无抄袭，无代码雷同

---

## License

MIT License — 详见 [LICENSE](LICENSE) 文件
