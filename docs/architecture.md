# 🧑‍⚕️ 个性化健康规划多智能体平台 — 系统架构文档

> **Architecture Document v2.0**  
> CS599 企业级应用软件设计与开发 · 方向一：Agentic AI 原生开发  
> 多智能体协作系统

---

## 一、架构概览

本系统采用 **Supervisor-Worker + Reflection** 多智能体架构，基于 LangGraph 构建。

### 核心架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          👤 User Interface                          │
│                   CLI (main.py)  │  Web UI (Streamlit)              │
│                                  │  API (FastAPI + SSE)             │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   🧠 Consultation Agent (Supervisor)                 │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Intent      │  │ Route        │  │ Safety       │               │
│  │ Analysis    │──│ Decision     │──│ Check        │               │
│  └─────────────┘  └──────────────┘  └──────────────┘               │
└───────┬──────────────────┬──────────────────┬──────────────────────┘
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ 🥗 Diet      │  │ 🏃 Exercise  │  │ 😴 Sleep     │
│    Agent     │  │    Agent     │  │    Agent     │
│              │  │              │  │              │
│ • BMR/TDEE   │  │ • FITT-VP    │  │ • CBT-I      │
│ • Meal Plan  │  │ • Workout    │  │ • Sleep Plan │
│ • Nutrition  │  │ • Calories   │  │ • Hygiene    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    🔍 Reflection Node (Quality Gate)                 │
│  • Confidence check  • Consistency check  • Completeness check     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   📝 Synthesis Node (Final Response)                 │
│  • Cross-domain integration  • Next steps  • Health alerts          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、技术架构

### 2.1 技术栈总览

| 层级 | 技术 | 说明 |
|------|------|------|
| **Agent 框架** | LangGraph 0.2+ | 状态图编排，checkpointer 持久化 |
| **LLM** | DeepSeek/OpenAI/Anthropic/Ollama | 多 Provider 支持，通过环境变量切换 |
| **协议** | Function Calling + MCP | OpenAI 兼容 Tool Schema + MCP 工具 |
| **记忆** | ChromaDB (向量) + ConversationBuffer (短期) + JSON (持久化) | 三层记忆架构 |
| **RAG** | 自建健康知识库 (20+ 循证文档) | 语义检索 + 关键词混合匹配 |
| **可观测性** | Langfuse + 控制台日志 | Tracing、Token 用量、成本估算 |
| **API** | FastAPI + SSE | RESTful + 流式响应 |
| **UI** | Streamlit 1.40+ | Web Demo 界面 |
| **容器** | Docker Multi-Stage | 生产级镜像 |
| **测试** | Pytest + pytest-cov | 单元测试 + 集成测试 + 评估基准 |

### 2.2 六大核心技术要素覆盖

| # | 要素 | 实现方式 | 完成度 |
|---|------|----------|--------|
| 1 | **SDD 规格驱动** | Pydantic v2 完整数据模型 (schemas.py ~400行) | ✅ 100% |
| 2 | **工具使用/Function Calling/MCP** | 6 组工具 (Diet/Exercise/Sleep/Common/MCP/RAG)，15+ 函数 | ✅ 100% |
| 3 | **记忆机制** | 三层架构：短期缓冲 + 向量语义记忆 (ChromaDB) + 文件持久化 | ✅ 100% |
| 4 | **状态管理与多步骤推理** | LangGraph StateGraph + Supervisor-Worker + Reflection | ✅ 100% |
| 5 | **多智能体协作** | 4 Agent (Diet/Exercise/Sleep/Consultation) 并行/串行协作 | ✅ 100% |
| 6 | **可观测性与评估** | Langfuse Tracing + 评估基准 (10+ case) + Token 追踪 | ✅ 100% |

---

## 三、Agent 交互流程

### 3.1 标准请求流程

```
User: "我想科学减重10公斤"
  │
  ├─[1] Supervisor: 意图分析
  │     ├─ 检测关键词: "减重" → Diet + Exercise
  │     ├─ 安全筛查: 无危急症状
  │     └─ 决定: route to [diet, exercise]
  │
  ├─[2] Parallel Workers:
  │     ├─ DietAgent:
  │     │   ├─ calculate_bmr(70kg, 170cm, 30, male) → BMR=1648.75
  │     │   ├─ calculate_tdee(1648.75, sedentary) → TDEE=1978.5
  │     │   ├─ search_health_knowledge("减重饮食") → 循证依据
  │     │   └─ generate_meal_plan(1478.5) → 一日三餐计划
  │     │
  │     └─ ExerciseAgent:
  │         ├─ query_health_guidelines("exercise", "WHO") → 运动指南
  │         ├─ generate_workout_plan("lose_weight", "beginner")
  │         └─ recommend_exercise("lose_weight", 30, "home")
  │
  ├─[3] Reflection: 质量审查
  │     ├─ Diet 置信度 0.85 ✅
  │     ├─ Exercise 置信度 0.82 ✅
  │     ├─ 一致性检查: 饮食热量缺口 + 运动消耗匹配 ✅
  │     └─ Quality Gate: PASSED
  │
  └─[4] Synthesis: 整合输出
        ├─ 体重现状分析 (BMI 24.2, 偏胖)
        ├─ 热量目标: 每日摄入 1500 kcal (缺口 500 kcal)
        ├─ 运动计划: 每周3次 HIIT + 2次力量训练
        ├─ 协同策略: 饮食+运动的周计划表
        └─ 下一步行动清单
```

### 3.2 安全协议流程

```
User: "我胸口疼了三天了"
  │
  └─[1] Supervisor: Safety Check
        ├─ 检测危急关键词: "胸痛"
        └─ Priority: URGENT
        └─ Response: "⚠️ 建议立即就医...不提供自行处理建议"
           (No agent routing)
```

---

## 四、数据流设计

### 4.1 LangGraph State 结构

```python
HealthGraphState:
  ├── messages: list              # 对话历史
  ├── user_input: str             # 当前用户输入
  ├── user_profile: dict|null     # 用户健康档案
  ├── health_goals: list[dict]    # 健康目标列表
  ├── supervisor_decision: dict   # 路由决策
  ├── agent_outputs: dict[str,dict]  # 各Agent输出
  ├── reflection_needed: bool     # 是否需要反思
  ├── reflection_notes: str       # 反思结果
  ├── quality_gate_passed: bool   # 质量门禁
  ├── final_response: dict        # 最终回复
  ├── iteration_count: int        # 迭代计数
  ├── errors: list[str]           # 错误列表
  ├── trace_id: str               # 追踪ID
  └── token_usage: dict[str,int]  # Token统计
```

### 4.2 记忆架构

```
┌─────────────────────────────────────────┐
│           Memory Architecture            │
├─────────────────────────────────────────┤
│  Tier 1: Short-term Memory              │
│  ┌───────────────────────────────────┐  │
│  │ LangChain ConversationBuffer      │  │
│  │ • Current session messages        │  │
│  │ • Max 20 exchanges                │  │
│  └───────────────────────────────────┘  │
├─────────────────────────────────────────┤
│  Tier 2: Long-term Semantic Memory      │
│  ┌───────────────────────────────────┐  │
│  │ ChromaDB Vector Store             │  │
│  │ • Cross-session retrieval         │  │
│  │ • Semantic similarity search       │  │
│  │ • Memory consolidation             │  │
│  │ • Embedding: multilingual-MiniLM  │  │
│  └───────────────────────────────────┘  │
├─────────────────────────────────────────┤
│  Tier 3: Persistent Storage             │
│  ┌───────────────────────────────────┐  │
│  │ JSON Files                        │  │
│  │ • User profiles                   │  │
│  │ • Health goals                    │  │
│  │ • Metrics history                 │  │
│  │ • Response archive                │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

---

## 五、目录结构

```
cs599-project/
├── docs/
│   ├── CS599_大作业报告.pdf      # 最终报告
│   └── architecture.md            # 本架构文档
├── src/
│   ├── agents/                    # Agent 实现
│   │   ├── base_agent.py          # 基类（重试/熔断/流式/观测）
│   │   ├── consultation_agent.py  # Supervisor（路由/反思/合成）
│   │   ├── diet_agent.py          # 饮食专家（MCP+RAG）
│   │   ├── exercise_agent.py      # 运动专家（MCP+RAG）
│   │   ├── sleep_agent.py         # 睡眠专家（MCP+RAG）
│   │   └── response_parser.py      # 回复解析（置信度/来源/注意事项）
│   ├── tools/                     # Function Calling 工具
│   │   ├── diet_tools.py          # BMR/TDEE/饮食计划/营养分析
│   │   ├── exercise_tools.py      # 训练计划/运动推荐/热量估算
│   │   ├── sleep_tools.py         # 睡眠分析/睡眠计划/卫生建议
│   │   ├── common_tools.py        # 档案/目标/追踪/风险评估
│   │   ├── mcp_tools.py           # MCP：健康指南/药物互作/医学参考
│   │   └── rag_tools.py           # RAG：循证知识库检索
│   ├── graph/                     # LangGraph 编排
│   │   ├── state.py               # 增强状态定义
│   │   └── health_graph.py        # Supervisor-Worker + Reflection
│   ├── memory/                    # 记忆管理
│   │   ├── memory_manager.py      # 三层记忆管理器
│   │   └── vector_store.py        # ChromaDB 向量存储
│   ├── models/                    # SDD 可执行规格
│   │   └── schemas.py             # Pydantic v2 完整数据模型
│   ├── prompts/                   # 提示词（独立于代码）
│   │   ├── manager.py             # PromptManager 加载/缓存/格式化
│   │   ├── diet.yaml              # Diet Agent 提示词
│   │   ├── exercise.yaml          # Exercise Agent 提示词
│   │   ├── sleep.yaml             # Sleep Agent 提示词
│   │   └── consultation.yaml      # Supervisor 提示词
│   ├── routing/                   # 意图路由
│   │   └── router.py              # 关键词+协同+安全筛查
│   ├── middleware/                 # 容错中间件
│   │   └── resilience.py          # 熔断器/指数退避重试/优雅降级
│   ├── observability/             # 可观测性
│   │   ├── tracer.py              # Langfuse 追踪 + 成本估算
│   │   └── evaluator.py           # 评估框架 + 基准测试
│   ├── api/                       # API 服务
│   │   └── server.py              # FastAPI + SSE 流式
│   ├── ui/                        # Web 界面
│   │   └── streamlit_app.py       # Streamlit v2 增强 UI
│   ├── config/                    # 配置管理
│   │   └── settings.py            # 环境变量 + 多Provider
│   └── main.py                    # CLI 入口
├── tests/
│   ├── test_agents.py             # 工具函数单元测试
│   └── test_integration.py        # 集成测试 + 评估测试
├── .env.example                   # 环境变量模板（含所有配置）
├── .gitignore
├── Dockerfile                     # 多阶段构建
├── LICENSE                        # MIT
├── requirements.txt
└── README.md
```

---

## 六、关键设计决策

### 6.1 为什么选择 Supervisor-Worker 模式
- **动态路由**：根据用户意图灵活决定调用哪些专家
- **并行执行**：饮食+运动可同时分析，减少延迟
- **易于扩展**：新增 Agent 只需注册路由规则
- **Quality Gate**：Reflection 节点统一把关质量

### 6.2 为什么三层记忆架构
- **短期**：满足当前对话的上下文需求（低延迟）
- **向量长期**：跨会话的语义检索（用户偏好学习）
- **文件持久化**：结构化数据的可靠存储（档案/目标/指标）

### 6.3 为什么集成 MCP 协议
- 医疗健康领域需要权威指南支撑（WHO/CDC/AASM）
- 药物相互作用检查是真实场景的刚需
- MCP 标准化的工具协议便于未来扩展

---

## 七、扩展路线图

- [ ] 本地 Ollama 模型支持的完整测试
- [ ] LangSmith 替代方案深度集成
- [ ] 移动端适配 (PWA)
- [ ] 可穿戴设备数据接入 (Apple Health / Google Fit)
- [ ] 多语言国际化 (English, 日本語)
- [ ] HIPAA 合规改造（医疗数据隐私保护）
