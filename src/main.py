"""
CLI entry point for the 个性化健康规划多智能体平台.
Runs the health multi-agent graph in interactive mode.
"""

import logging
import sys
import uuid
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import settings
from src.graph.health_graph import get_health_graph
from src.memory.memory_manager import MemoryManager
from src.models.schemas import UserProfile, Gender, ActivityLevel, GoalType, HealthGoal


def setup_logging():
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║        个性化健康规划多智能体平台                        ║
║        Personalized Health Multi-Agent Platform          ║
║                                                          ║
║  🥗 Diet Agent     — 饮食营养专家                       ║
║  🏃 Exercise Agent — 运动健身专家                       ║
║  😴 Sleep Agent    — 睡眠健康专家                       ║
║  🧠 Consultation   — 健康咨询总控                       ║
║                                                          ║
║  Powered by LangGraph + DeepSeek API                     ║
╚══════════════════════════════════════════════════════════╝
    """)


def quick_setup(memory: MemoryManager) -> UserProfile:
    """Quick user profile setup wizard."""
    print("\n📋 让我们先快速建立您的健康档案（输入 'skip' 跳过）\n")

    profile_data = {
        "user_id": str(uuid.uuid4())[:8],
        "name": input("您的称呼: ").strip() or "用户",
    }

    if profile_data["name"].lower() == "skip":
        return None

    try:
        profile_data["age"] = int(input("年龄: "))
        if profile_data["age"] < 1 or profile_data["age"] > 120:
            print("年龄不合法，使用默认值 30")
            profile_data["age"] = 30
    except ValueError:
        profile_data["age"] = 30

    gender_input = input("性别 (m/f/o): ").strip().lower()
    gender_map = {"m": Gender.MALE, "f": Gender.FEMALE, "o": Gender.OTHER}
    profile_data["gender"] = gender_map.get(gender_input, Gender.OTHER)

    try:
        profile_data["height_cm"] = float(input("身高 (cm): "))
    except ValueError:
        profile_data["height_cm"] = 170.0

    try:
        profile_data["weight_kg"] = float(input("体重 (kg): "))
    except ValueError:
        profile_data["weight_kg"] = 65.0

    print("\n活动水平:")
    print("  1. 久坐不动  2. 轻度活动  3. 中度活动  4. 高度活跃  5. 极度活跃")
    act_input = input("选择 (1-5, 默认 2): ").strip()
    act_map = {
        "1": ActivityLevel.SEDENTARY,
        "2": ActivityLevel.LIGHTLY_ACTIVE,
        "3": ActivityLevel.MODERATELY_ACTIVE,
        "4": ActivityLevel.VERY_ACTIVE,
        "5": ActivityLevel.EXTRA_ACTIVE,
    }
    profile_data["activity_level"] = act_map.get(act_input, ActivityLevel.LIGHTLY_ACTIVE)

    profile = UserProfile(**profile_data)
    memory.save_user_profile(profile)

    bmi = profile.bmi
    print(f"\n✅ 档案已创建！BMI: {bmi} ({profile.bmi_category})")
    return profile


def main():
    """Main interactive loop."""
    setup_logging()
    logger = logging.getLogger("main")
    print_banner()

    # Check API key
    if not settings.llm_api_key:
        print("⚠️  未配置 API Key！")
        print("   请复制 .env.example 为 .env 并填入您的 API Key")
        print(f"   当前 LLM Provider: {settings.llm_provider}")
        return

    print(f"🔌 LLM Provider: {settings.llm_provider}")
    print(f"📦 Model: {settings.llm_model}")

    memory = MemoryManager()

    # Quick setup
    profile = quick_setup(memory)

    # Set default goals
    if profile:
        goals = [
            HealthGoal(
                goal_type=GoalType.GENERAL_WELLNESS,
                target_description="全面改善健康状态，获得个性化健康指导",
                priority=1,
            )
        ]
        memory.save_goals(profile.user_id, goals)

    # Initialize graph
    print("\n🚀 正在初始化多智能体协作引擎...")
    graph = get_health_graph()
    print("✅ 就绪！开始对话吧（输入 'quit' 退出，'reset' 重置对话）\n")

    # Config for LangGraph (thread ID for conversation persistence)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    while True:
        try:
            user_input = input("💬 您: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！祝您健康！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！祝您健康！")
            break

        if user_input.lower() == "reset":
            memory.clear_conversation()
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            print("🔄 对话已重置\n")
            continue

        # Invoke the graph
        print("\n🤔 正在协调多智能体为您分析...")
        try:
            result = graph.invoke(
                {
                    "user_input": user_input,
                    "user_profile": memory.get_active_user().model_dump() if memory.get_active_user() else None,
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
                print(f"\n🧑‍⚕️ 健康顾问:\n{final.get('message', '')}")

                next_steps = final.get("next_steps", [])
                if next_steps:
                    print("\n📋 建议的后续行动:")
                    for step in next_steps:
                        print(f"  {step}")
            else:
                print("\n⚠️ 未能生成回复，请稍后重试。")

            # Show which agents contributed
            agent_outputs = result.get("agent_outputs", {})
            if agent_outputs:
                agent_names = []
                for role in agent_outputs:
                    names = {"diet": "🥗 饮食Agent", "exercise": "🏃 运动Agent", "sleep": "😴 睡眠Agent"}
                    agent_names.append(names.get(role, role))
                print(f"\n📊 本次参与协作的Agent: {', '.join(agent_names)}")

        except Exception as e:
            logger.error(f"Graph invocation error: {e}", exc_info=True)
            print(f"\n❌ 处理出错: {e}")

        print()


if __name__ == "__main__":
    main()
