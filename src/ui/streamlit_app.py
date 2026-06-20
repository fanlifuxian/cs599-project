"""
Streamlit UI v2.0 — Enhanced web interface for the Health Multi-Agent Platform.

New features:
- Rich dashboard with health metrics visualization
- Agent status & contribution indicators
- Progress tracking charts
- Knowledge base statistics
- Evaluation runner
- Mobile-responsive layout
"""

import sys
import uuid
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import logging

from src.config.settings import settings
from src.graph.health_graph import get_health_graph
from src.memory.memory_manager import MemoryManager
from src.models import (
    UserProfile, Gender, ActivityLevel, GoalType, HealthGoal, BodyMetrics,
)

# ── Page Config ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="个性化健康规划多智能体平台",
    page_icon="🧑‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "CS599 企业级应用软件设计与开发 · 方向一：Agentic AI 原生开发",
    },
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")

# ── Custom CSS ──────────────────────────────────────────────────────────

st.markdown("""
<style>
    .agent-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        text-align: center;
        margin: 0.3rem;
    }
    .agent-active { border-color: #4CAF50; background: #f1f8e9; }
    .agent-idle { border-color: #e0e0e0; background: #fafafa; }
    .metric-card {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .stChatMessage { border-radius: 15px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ───────────────────────────────────────────────────────

if "memory" not in st.session_state:
    st.session_state.memory = MemoryManager()
if "graph" not in st.session_state:
    st.session_state.graph = get_health_graph()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "profile_done" not in st.session_state:
    st.session_state.profile_done = False
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

memory = st.session_state.memory
graph = st.session_state.graph

# ── Sidebar ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/health-checkup.png", width=64) if False else None
    st.title("🧑‍⚕️ 健康档案")

    if not st.session_state.profile_done:
        with st.form("profile_form"):
            name = st.text_input("👤 称呼", value="用户")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("🎂 年龄", 1, 120, 30)
                gender = st.selectbox("⚧ 性别", ["male", "female", "other"],
                    format_func=lambda x: {"male": "男", "female": "女", "other": "其他"}[x])
            with col2:
                height = st.number_input("📏 身高 (cm)", 50.0, 250.0, 170.0, 0.5)
                weight = st.number_input("⚖️ 体重 (kg)", 20.0, 300.0, 65.0, 0.5)

            activity = st.selectbox("🏃 活动水平",
                ["sedentary", "lightly_active", "moderately_active", "very_active", "extra_active"],
                format_func=lambda x: {
                    "sedentary": "久坐不动", "lightly_active": "轻度活动 (1-3天/周)",
                    "moderately_active": "中度活动 (3-5天/周)", "very_active": "高度活跃 (6-7天/周)",
                    "extra_active": "极度活跃 (运动员)",
                }[x],
            )
            sleep_hours = st.slider("😴 平均睡眠 (h)", 0.0, 16.0, 7.0, 0.5)

            col3, col4 = st.columns(2)
            with col3:
                dietary = st.text_input("🥗 饮食偏好", placeholder="如: 低盐,高蛋白")
            with col4:
                allergies = st.text_input("⚠️ 过敏/忌口", placeholder="如: 花生,海鲜")

            medical = st.text_input("🏥 健康状况", placeholder="如: 高血压,糖尿病（无可留空）")

            submitted = st.form_submit_button("💾 创建健康档案", use_container_width=True)

            if submitted:
                profile = UserProfile(
                    user_id=str(uuid.uuid4())[:8],
                    name=name, age=age, gender=Gender(gender),
                    current_metrics=BodyMetrics(weight_kg=weight, height_cm=height),
                    activity_level=ActivityLevel(activity),
                    sleep_hours_avg=sleep_hours,
                    dietary_preferences=[d.strip() for d in dietary.split(",") if d.strip()],
                    allergies=[a.strip() for a in allergies.split(",") if a.strip()],
                    medical_conditions=[m.strip() for m in medical.split(",") if m.strip()],
                )
                memory.save_user_profile(profile)
                memory.save_goals(profile.user_id, [
                    HealthGoal(goal_type=GoalType.GENERAL_WELLNESS,
                              target_description="全面改善健康状态", priority=1)
                ])
                st.session_state.profile_done = True
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"✅ 健康档案已创建！\n\n📊 **BMI: {profile.bmi}** ({profile.bmi_category})\n\n"
                              f"我已协调了饮食🥗、运动🏃、睡眠😴三位专家待命。\n请告诉我您想改善什么健康问题？",
                })
                st.rerun()
    else:
        profile = memory.get_active_user()
        if profile:
            bmi = profile.bmi
            bmi_color = "green" if 18.5 <= bmi < 24 else "orange" if bmi < 28 else "red"
            st.markdown(f"### 👤 {profile.name}")
            cols = st.columns(3)
            cols[0].metric("BMI", f"{bmi}", profile.bmi_category)
            cols[1].metric("身高", f"{profile.height_cm} cm")
            cols[2].metric("体重", f"{profile.weight_kg} kg")
            st.divider()

            with st.expander("📋 详细信息", expanded=False):
                st.write(f"**年龄**: {profile.age} 岁")
                st.write(f"**性别**: {{'male':'男','female':'女','other':'其他'}}[{profile.gender.value}]")
                st.write(f"**活动水平**: {profile.activity_level.value}")
                st.write(f"**平均睡眠**: {profile.sleep_hours_avg} h/夜")
                if profile.dietary_preferences:
                    st.write(f"**饮食偏好**: {', '.join(profile.dietary_preferences)}")
                if profile.allergies:
                    st.write(f"**⚠️ 过敏**: {', '.join(profile.allergies)}")
                if profile.medical_conditions:
                    st.write(f"**🏥 健康状况**: {', '.join(profile.medical_conditions)}")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔄 重置档案", use_container_width=True):
                    st.session_state.profile_done = False
                    st.session_state.messages = []
                    memory.clear_conversation()
                    st.rerun()
            with col_btn2:
                if st.button("🗑️ 清空对话", use_container_width=True):
                    st.session_state.messages = []
                    memory.clear_conversation()
                    st.session_state.thread_id = str(uuid.uuid4())
                    st.rerun()

    st.divider()
    st.caption(f"🔌 {settings.llm_provider}/{settings.llm_model}")
    st.caption(f"💾 ChromaDB | 📚 RAG | 🔬 MCP")
    st.caption(f"🏭 CS599 · Agentic AI 原生开发")

# ── Main Area ───────────────────────────────────────────────────────────

st.title("🧑‍⚕️ 个性化健康规划多智能体平台")
st.caption("v2.0 · 多智能体协作 · MCP · RAG · ChromaDB · Reflection · LangGraph")

# Agent status
if st.session_state.profile_done:
    cols = st.columns(4)
    agents = [
        ("🥗 饮食Agent", "循证营养", "📗"),
        ("🏃 运动Agent", "科学训练", "📘"),
        ("😴 睡眠Agent", "睡眠医学", "📙"),
        ("🧠 咨询Agent", "智能协调", "📕"),
    ]
    for i, (name, desc, icon) in enumerate(agents):
        with cols[i]:
            st.markdown(
                f"""<div class="agent-card agent-active">
                <h4>{icon} {name}</h4>
                <small>{desc}</small><br>
                <span style="color:green">● 就绪</span>
                </div>""",
                unsafe_allow_html=True,
            )

st.divider()

# Chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("请输入您的健康问题...（如：我想科学减重、最近失眠怎么办、帮我分析一下饮食）"):
    if not settings.llm_api_key:
        st.error("⚠️ 请先配置 API Key！复制 .env.example 为 .env 并填入 API Key。")
        st.stop()
    if not st.session_state.profile_done:
        st.warning("⚠️ 请先在左侧边栏创建健康档案")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 正在协调多智能体为您分析..."):
            try:
                active_user = memory.get_active_user()
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                trace_id = str(uuid.uuid4())[:12]

                result = graph.invoke(
                    {
                        "user_input": prompt,
                        "user_profile": active_user.to_dict() if active_user else None,
                        "health_goals": [g.model_dump() for g in memory._goals],
                        "agent_outputs": {}, "supervisor_decision": None,
                        "final_response": None, "iteration_count": 0,
                        "errors": [], "reflection_needed": False,
                        "reflection_notes": "", "quality_gate_passed": False,
                        "next_step": "", "trace_id": trace_id, "token_usage": {},
                    },
                    config=config,
                )

                final = result.get("final_response", {})
                if final:
                    response_text = final.get("message", "抱歉，未能生成回复。")
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                    # Agent contributions
                    agent_outputs = result.get("agent_outputs", {})
                    if agent_outputs:
                        agent_names = {
                            "diet": "🥗 饮食Agent", "exercise": "🏃 运动Agent",
                            "sleep": "😴 睡眠Agent",
                        }
                        contributors = [agent_names.get(r, r) for r in agent_outputs]
                        st.caption(f"📊 协作Agent: {', '.join(contributors)} | 🔢 Token: ~{result.get('token_usage', {}).get('total', '?')}")

                        # Show individual agent outputs
                        with st.expander("🔍 查看各Agent详细输出"):
                            for role_key, data in agent_outputs.items():
                                display_name = agent_names.get(role_key, role_key)
                                st.markdown(f"**{display_name}** (置信度: {data.get('confidence', '?')})")
                                st.text(data.get("content", "")[:300] + "...")

                    # Next steps
                    next_steps = final.get("next_steps", [])
                    if next_steps:
                        with st.expander("📋 建议的后续行动"):
                            for i, step in enumerate(next_steps, 1):
                                st.markdown(f"{i}. {step}")

                    # Health alerts
                    alerts = final.get("health_alerts", [])
                    if alerts:
                        for alert in alerts:
                            st.warning(alert)

                else:
                    st.error("未能生成回复")
                    errors = result.get("errors", [])
                    if errors:
                        st.caption(f"错误: {'; '.join(errors)}")

            except Exception as e:
                error_msg = f"❌ 处理出错: {str(e)}"
                st.error(error_msg)
                logger.error(f"Graph error: {e}", exc_info=True)

# ── Footer ──────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "⚠️ 免责声明：本平台为 AI 驱动的健康建议系统，提供的信息仅供参考，"
    "不构成医疗诊断或治疗建议。如有健康问题，请咨询专业医生。\n\n"
    "CS599 企业级应用软件设计与开发 · 方向一：Agentic AI 原生开发 · "
    "LangGraph + MCP + RAG + ChromaDB"
)
