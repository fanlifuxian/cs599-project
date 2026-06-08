"""
Streamlit UI for the 个性化健康规划多智能体平台.
Provides a web-based chat interface for demo and presentation.
"""

import sys
import uuid
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import logging

from src.config.settings import settings
from src.graph.health_graph import get_health_graph
from src.memory.memory_manager import MemoryManager
from src.models.schemas import UserProfile, Gender, ActivityLevel, GoalType, HealthGoal

# ── Page Config ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="个性化健康规划多智能体平台",
    page_icon="🧑‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logging ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")

# ── Session State Initialization ────────────────────────────────────────

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


# ── Sidebar — User Profile ──────────────────────────────────────────────

with st.sidebar:
    st.title("🧑‍⚕️ 健康档案")

    if not st.session_state.profile_done:
        with st.form("profile_form"):
            name = st.text_input("称呼", value="用户")
            age = st.number_input("年龄", min_value=1, max_value=120, value=30)
            gender = st.selectbox("性别", ["male", "female", "other"], format_func=lambda x: {"male": "男", "female": "女", "other": "其他"}[x])
            height = st.number_input("身高 (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.1)
            weight = st.number_input("体重 (kg)", min_value=20.0, max_value=300.0, value=65.0, step=0.1)
            activity = st.selectbox(
                "活动水平",
                ["sedentary", "lightly_active", "moderately_active", "very_active", "extra_active"],
                format_func=lambda x: {
                    "sedentary": "久坐不动",
                    "lightly_active": "轻度活动",
                    "moderately_active": "中度活动",
                    "very_active": "高度活跃",
                    "extra_active": "极度活跃",
                }[x],
            )
            sleep_hours = st.slider("平均睡眠时长 (小时)", 0.0, 16.0, 7.0, 0.5)

            dietary = st.text_input("饮食偏好 (逗号分隔)", placeholder="如: 低盐, 高蛋白")
            allergies = st.text_input("过敏/忌口", placeholder="如: 花生, 海鲜")

            submitted = st.form_submit_button("💾 创建健康档案", use_container_width=True)

            if submitted:
                profile = UserProfile(
                    user_id=str(uuid.uuid4())[:8],
                    name=name,
                    age=age,
                    gender=Gender(gender),
                    height_cm=height,
                    weight_kg=weight,
                    activity_level=ActivityLevel(activity),
                    sleep_hours_avg=sleep_hours,
                    dietary_preferences=[d.strip() for d in dietary.split(",") if d.strip()],
                    allergies=[a.strip() for a in allergies.split(",") if a.strip()],
                )
                memory.save_user_profile(profile)

                goals = [
                    HealthGoal(
                        goal_type=GoalType.GENERAL_WELLNESS,
                        target_description="全面改善健康状态",
                        priority=1,
                    )
                ]
                memory.save_goals(profile.user_id, goals)

                st.session_state.profile_done = True
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"✅ 健康档案已创建！\n\n📊 BMI: {profile.bmi} ({profile.bmi_category})\n\n您可以开始向我咨询任何健康问题，我会协调饮食、运动、睡眠三个专业Agent为您服务！",
                })
                st.rerun()
    else:
        profile = memory.get_active_user()
        if profile:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("BMI", f"{profile.bmi}", profile.bmi_category)
            with col2:
                st.metric("身高", f"{profile.height_cm} cm")
            with col3:
                st.metric("体重", f"{profile.weight_kg} kg")

            st.divider()
            st.caption(f"年龄: {profile.age} | 性别: {profile.gender.value}")
            st.caption(f"活动水平: {profile.activity_level.value}")
            if profile.dietary_preferences:
                st.caption(f"饮食偏好: {', '.join(profile.dietary_preferences)}")
            if profile.allergies:
                st.caption(f"⚠️ 过敏: {', '.join(profile.allergies)}")

            if st.button("🔄 重置档案", use_container_width=True):
                st.session_state.profile_done = False
                st.session_state.messages = []
                memory.clear_conversation()
                st.rerun()

    st.divider()
    st.caption("---")
    st.caption(f"🔌 LLM: {settings.llm_provider} / {settings.llm_model}")

# ── Main Chat Area ──────────────────────────────────────────────────────

st.title("🧑‍⚕️ 个性化健康规划多智能体平台")
st.caption("多智能体协作 · 饮食 🥗 + 运动 🏃 + 睡眠 😴 + 咨询 🧠")

# Agent status indicators
if st.session_state.profile_done:
    cols = st.columns(4)
    agents_status = [
        ("🥗 饮食Agent", "就绪", "green"),
        ("🏃 运动Agent", "就绪", "green"),
        ("😴 睡眠Agent", "就绪", "green"),
        ("🧠 咨询Agent", "协调中", "blue"),
    ]
    for i, (name, status, color) in enumerate(agents_status):
        with cols[i]:
            st.markdown(f"**{name}**\n:{color}[{status}]")

st.divider()

# Chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("请输入您的健康问题..."):
    if not settings.llm_api_key:
        st.error("⚠️ 请先配置 API Key！复制 .env.example 为 .env 并填入您的 DeepSeek API Key。")
        st.stop()

    if not st.session_state.profile_done:
        st.warning("请先在左侧边栏创建健康档案")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke graph
    with st.chat_message("assistant"):
        with st.spinner("🤔 正在协调多智能体为您分析..."):
            try:
                active_user = memory.get_active_user()
                config = {"configurable": {"thread_id": st.session_state.thread_id}}

                result = graph.invoke(
                    {
                        "user_input": prompt,
                        "user_profile": active_user.model_dump() if active_user else None,
                        "health_goals": [g.model_dump() for g in memory._goals],
                        "agent_outputs": {},
                        "supervisor_decision": None,
                        "final_response": None,
                        "iteration_count": 0,
                        "errors": [],
                    },
                    config=config,
                )

                final = result.get("final_response", {})
                if final:
                    response_text = final.get("message", "抱歉，未能生成回复。")
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                    # Show agent contributions
                    agent_outputs = result.get("agent_outputs", {})
                    if agent_outputs:
                        agent_names = {
                            "diet": "🥗 饮食Agent",
                            "exercise": "🏃 运动Agent",
                            "sleep": "😴 睡眠Agent",
                        }
                        contributors = [agent_names.get(r, r) for r in agent_outputs]
                        st.caption(f"📊 本次协作Agent: {', '.join(contributors)}")

                    # Next steps
                    next_steps = final.get("next_steps", [])
                    if next_steps:
                        with st.expander("📋 建议的后续行动"):
                            for step in next_steps:
                                st.markdown(f"- {step}")
            except Exception as e:
                error_msg = f"❌ 处理出错: {str(e)}"
                st.error(error_msg)
                logger.error(f"Graph error: {e}", exc_info=True)

# ── Footer ──────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "⚠️ 免责声明：本平台为 AI 驱动的健康建议系统，提供的信息仅供参考，"
    "不构成医疗诊断或治疗建议。如有健康问题，请咨询专业医生。"
    " | CS599 企业级应用软件设计与开发 · 方向一：Agentic AI 原生开发"
)
