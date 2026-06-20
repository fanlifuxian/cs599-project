# 🧑‍⚕️ 个性化健康规划多智能体平台 v2.0

> **Personalized Health Planning Multi-Agent Platform**  
> CS599 企业级应用软件设计与开发 · 方向一：Agentic AI 原生开发  
> 多智能体协作 · MCP · RAG · ChromaDB · LangGraph

一个基于 LangGraph 构建的**企业级**多智能体健康规划系统，整合饮食规划、运动指导、睡眠管理和健康咨询四大专业 Agent，通过 Supervisor-Worker + Reflection 模式协作，为用户提供个性化、循证的健康管理建议。

---

## 项目简介

现代人的健康管理面临碎片化挑战：饮食、运动、睡眠相互关联但往往被独立处理。本项目通过 **多智能体协作** 实现跨领域健康规划的有机整合：

- 🥗 **Diet Agent** — 注册营养师，精准计算 BMR/TDEE，设计个性化饮食方案
- 🏃 **Exercise Agent** — 运动生理学家，基于 ACSM FITT-VP 原则开具运动处方
- 😴 **Sleep Agent** — 睡眠医学专家，运用 CBT-I 改善睡眠质量
- 🧠 **Consultation Agent** — 健康顾问总控，智能路由、质量审查、结果融合

**v2.0 企业级升级亮点**：MCP 协议集成 · Agentic RAG 循证知识库 · ChromaDB 向量记忆 · Reflection 质量门禁 · Langfuse 可观测性 · FastAPI 生产级 API · 流式响应 · 熔断重试

---

## 方向

🎯 **方向一：Agentic AI 原生开发** — 多智能体协作系统 (Supervisor-Worker + Reflection)

---

## 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| **AI IDE** | Trae CN | 课程指定 AI 原生 IDE |
| **LLM** | DeepSeek API / OpenAI / Anthropic / Ollama | 多 Provider 支持，环境变量切换 |
| **Agent 框架** | LangGraph 0.2+ | StateGraph 编排 + Checkpointer 持久化 |
| **协议** | Function Calling + **MCP** | OpenAI 兼容 Tool Schema + MCP 健康知识服务 |
| **记忆** | **ChromaDB** (向量) + ConversationBuffer + JSON | 三层记忆架构 |
| **RAG** | **Agentic RAG** (自建知识库) | 20+ 循证健康文档，6 大领域 |
| **可观测性** | **Langfuse** + 结构化日志 | Tracing、Token 追踪、成本估算 |
| **API** | **FastAPI** + SSE | RESTful API + 流式响应 |
| **UI** | Streamlit 1.40+ | 增强 Web 界面 + 仪表盘 |
| **容器** | Docker (Multi-Stage) | 生产级镜像 |
| **语言** | Python 3.11+ | 全异步支持 |
| **测试** | Pytest + pytest-cov | 单元测试 + 集成测试 + 评估基准 |
| **基础设施** | GitHub | 版本控制 |

### 核心技术要素（6/6 全覆盖 + MCP 加分项）

| # | 要素 | 实现方式 | 亮点 |
|---|------|----------|------|
| 1 | ✅ **SDD 规格驱动开发** | Pydantic v2 完整数据模型 (~400行) | Product Spec + API Spec 一体化 |
| 2 | ✅ **工具使用/Function Calling/MCP** | 6组工具 15+ 函数 + MCP 3工具 | 健康指南/药物互作/医学参考 |
| 3 | ✅ **记忆机制** | ChromaDB 向量 + 短期缓冲 + 文件持久化 | 语义检索 + 记忆巩固 + 偏好学习 |
| 4 | ✅ **状态管理与多步推理** | LangGraph StateGraph + ReAct + Reflection | Supervisor-Worker + 质量门禁 |
| 5 | ✅ **多智能体协作** | 4 Agent 并行/串行协作 | 意图分析 → 并行路由 → 反思 → 融合 |
| 6 | ✅ **可观测性与评估** | Langfuse Tracing + 10+ 评估用例 | Token 追踪 + 成本估算 + Benchmark |
| ➕ | 🔬 **MCP 协议** | 健康知识 MCP Server | WHO/CDC/AASM 权威指南查询 |
| ➕ | 📚 **Agentic RAG** | 20+ 循证文档知识库 | 6 大健康领域的证据检索 |
| ➕ | 🌊 **流式响应** | FastAPI SSE + Streamlit | 实时流式输出 |

---

## 架构概览

```
User Input → Supervisor (Consultation Agent)
                ├── 意图分析 + 安全筛查
                ├── 智能路由决策
                ↓
         ┌──────┼──────┐  (并行执行)
         ↓      ↓      ↓
      Diet   Exercise  Sleep
      Agent   Agent    Agent
      (+MCP   (+MCP    (+MCP
       +RAG)   +RAG)    +RAG)
         ↓      ↓      ↓
         └──────┼──────┘
                ↓
         Reflection Node  (质量门禁)
                ↓
         Synthesis Node   (跨领域融合)
                ↓
         User Response
```

### Agent 工具矩阵

| Agent | 核心工具 | MCP 工具 | RAG 工具 |
|-------|---------|---------|----------|
| 🥗 **Diet** | `calculate_bmr`, `calculate_tdee`, `generate_meal_plan`, `analyze_nutrition` | `query_health_guidelines` (nutrition), `check_drug_interactions` | `search_health_knowledge` (nutrition) |
| 🏃 **Exercise** | `generate_workout_plan`, `recommend_exercise`, `calculate_calorie_burn` | `query_health_guidelines` (exercise), `get_medical_reference` | `search_health_knowledge` (exercise) |
| 😴 **Sleep** | `analyze_sleep_quality`, `generate_sleep_plan`, `get_sleep_hygiene_tips` | `query_health_guidelines` (sleep), `get_medical_reference` | `search_health_knowledge` (sleep) |
| 🧠 **Consultation** | `get_user_profile_summary`, `set_health_goals`, `track_progress`, `health_risk_assessment` | 全部 MCP 工具 | 全部 RAG 工具 |

---

## 目录结构

```
cs599-project/
├── docs/                          # 📄 项目文档
│   ├── CS599_大作业报告.pdf         # 最终提交报告
│   └── architecture.md             # 详细架构说明
├── src/
│   ├── agents/                     # 🤖 Agent 实现
│   │   ├── base_agent.py           # 基类（重试/熔断/流式/观测）
│   │   ├── consultation_agent.py   # Supervisor（路由/反思/合成）
│   │   ├── diet_agent.py           # 饮食专家（MCP+RAG）
│   │   ├── exercise_agent.py       # 运动专家（MCP+RAG）
│   │   ├── sleep_agent.py          # 睡眠专家（MCP+RAG）
│   │   └── response_parser.py      # 回复解析（置信度/来源/注意事项）
│   ├── tools/                      # 🔧 Function Calling 工具
│   │   ├── diet_tools.py           # 营养计算/饮食计划
│   │   ├── exercise_tools.py       # 运动处方/热量估算
│   │   ├── sleep_tools.py          # 睡眠分析/CBT-I
│   │   ├── common_tools.py         # 档案/目标/追踪
│   │   ├── mcp_tools.py            # MCP：健康指南/药物互作
│   │   └── rag_tools.py            # RAG：循证知识库检索
│   ├── graph/                      # 🔀 LangGraph 编排
│   │   ├── state.py                # 增强状态定义
│   │   └── health_graph.py         # Supervisor-Worker + Reflection
│   ├── memory/                     # 🧠 三层记忆架构
│   │   ├── memory_manager.py       # 记忆管理器
│   │   └── vector_store.py         # ChromaDB 向量存储
│   ├── models/                     # 📐 SDD 可执行规格
│   │   └── schemas.py              # Pydantic v2 完整数据模型
│   ├── prompts/                    # 💬 提示词（独立于代码）
│   │   ├── manager.py              # PromptManager 加载/缓存/格式化
│   │   ├── diet.yaml               # Diet Agent 提示词
│   │   ├── exercise.yaml           # Exercise Agent 提示词
│   │   ├── sleep.yaml              # Sleep Agent 提示词
│   │   └── consultation.yaml       # Supervisor 提示词
│   ├── routing/                    # 🧭 意图路由
│   │   └── router.py               # 关键词+协同+安全筛查
│   ├── middleware/                  # 🛡️ 容错中间件
│   │   └── resilience.py           # 熔断器/指数退避重试/优雅降级
│   ├── observability/              # 📊 可观测性
│   │   ├── tracer.py               # Langfuse 追踪 + 成本估算
│   │   └── evaluator.py            # 评估框架 + 基准测试
│   ├── api/                        # 🌐 API 服务
│   │   └── server.py               # FastAPI + SSE 流式
│   ├── ui/                         # 🖥️ Web 界面
│   │   └── streamlit_app.py        # 增强 Streamlit UI
│   ├── config/                     # ⚙️ 配置管理
│   │   └── settings.py             # 环境变量（无硬编码 Key）
│   └── main.py                     # 💻 CLI 入口
├── tests/
│   ├── test_agents.py              # 工具函数单元测试
│   └── test_integration.py         # 集成测试 + 评估测试
├── .env.example                    # 🔑 环境变量模板
├── .gitignore
├── Dockerfile                      # 🐳 多阶段构建
├── LICENSE                         # MIT
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
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，至少填入:
#   DEEPSEEK_API_KEY=sk-xxxxxxxx
#   或 OPENAI_API_KEY=sk-xxxxxxxx
```

⚠️ **严禁在代码中硬编码 API Key！** 所有密钥通过 `.env` 文件管理。

### 4. 启动应用

**CLI 模式：**
```bash
python src/main.py
```

**Web UI 模式（Demo 推荐）：**
```bash
streamlit run src/ui/streamlit_app.py
```

**API 服务器模式（生产）：**
```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 4
# 访问 http://localhost:8000/docs 查看交互式 API 文档
```

### 5. Docker 部署

```bash
# 构建镜像
docker build -t health-agents .

# Streamlit UI
docker run -p 8501:8501 --env-file .env health-agents

# FastAPI Server
docker run -p 8000:8000 --env-file .env health-agents \
  uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 4

# 访问
# Streamlit: http://localhost:8501
# FastAPI:   http://localhost:8000/docs
```

### 6. 运行测试

```bash
# 全部测试
pytest tests/ -v

# 含覆盖率
pytest tests/ -v --cov=src --cov-report=html
```

---

## 使用示例

### 示例 1：科学减重咨询
```
用户: 我想科学减重，身高170cm，体重80kg，平时久坐

系统: [Consultation Agent → 意图分析: 饮食+运动 → 并行调用]
🥗 Diet: BMR=1648.75kcal, TDEE=1978.5kcal, 建议每日摄入1478.5kcal
🏃 Exercise: 初级减重计划，每周3次HIIT+2次快走

🧠 健康顾问:
根据您的数据（BMI 27.7，偏胖），我为您整合了饮食和运动专家的建议：
1. 饮食：每日摄入~1500kcal，蛋白质90g+，参考地中海饮食模式
2. 运动：从初级HIIT+快走开始，每周消耗~1000kcal
3. 协同：运动日适当增加蛋白质，防止肌肉流失
📋 下一步: 本周完成3次训练 + 记录3天饮食日记
```

### 示例 2：睡眠改善
```
用户: 我最近失眠，每天12点躺下2点才睡着，只睡5小时

系统: [Consultation → 路由到 SleepAgent]
😴 Sleep: 睡眠质量评分45(D级)，入睡延迟过长(120min)

🧠 健康顾问:
您的入睡延迟严重超标，我基于CBT-I原则为您制定方案：
1. 刺激控制：躺下20分钟未入睡就起床，等有困意再回床
2. 睡眠限制：暂时限制在床时间为6小时
3. 睡前程序：21:00减少屏幕→22:00泡脚→22:30阅读→23:00熄灯
4. 环境优化：卧室温度18-22°C，使用遮光窗帘
📋 预计1-2周入睡时间从120min缩短至30min内
```

### 示例 3：API 调用
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "我身高165体重75想减重，怎么办？", "stream": false}'
```

---

## 项目状态

- [x] Proposal — 架构设计与 Spec 初稿
- [x] MVP — 核心闭环 Demo (v0.1)
- [x] v2.0 企业级升级 — MCP + RAG + ChromaDB + Reflection + FastAPI
- [x] Final — 完整代码 + 架构文档 + 测试套件

---

## 关键技术亮点

### 1. MCP 协议集成
实现了 MCP 兼容的健康知识工具：`query_health_guidelines`、`check_drug_interactions`、`get_medical_reference`，使 Agent 能够查询 WHO/CDC/AASM 等权威机构的官方指南。

### 2. Agentic RAG
自建 20+ 篇循证健康知识文档，覆盖营养学、运动科学、睡眠医学、心理健康、预防医学六大领域，每个文档标注来源和可靠性评分。

### 3. Reflection 质量门禁
在 Agent 并行输出后，Reflection 节点自动进行置信度检查、一致性检测和完整性验证，确保最终回复质量。

### 4. 企业级容错
- 指数退避重试 + 随机抖动
- 熔断器模式（5次失败后自动熔断30秒）
- 优雅降级（API 失败时返回通用健康建议）
- 速率限制保护

### 5. 全链路可观测性
- Langfuse 分布式追踪
- Token 用量实时统计
- API 成本估算
- 请求级 Trace ID 贯穿全链路

---

## 学术声明

本项目为 CS599 企业级应用软件设计与开发课程大作业，遵循学术诚信原则：
- Agent 框架基于 LangGraph/LangChain 开源生态构建
- LLM API 调用使用 DeepSeek/OpenAI 官方 API
- 工具函数中的营养/运动数据基于公开科学公式（Mifflin-St Jeor、MET、ACSM 指南等）
- 健康知识库内容基于 WHO/CDC/AASM/ACSM 等权威机构的公开指南
- 无抄袭，无代码雷同

**引用声明**：使用了 `langgraph`、`langchain`、`chromadb`、`sentence-transformers`、`fastapi`、`streamlit` 等开源项目。

---

## License

MIT License — 详见 [LICENSE](LICENSE) 文件
