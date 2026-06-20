"""
CLI entry point for the 个性化健康规划多智能体平台 v2.0.

Runs the enterprise health multi-agent graph in interactive mode.
Supports:
- Interactive chat with multi-agent coordination
- Profile setup wizard
- Rich console output
- Command shortcuts (reset, stats, eval)
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
from src.models import (
    UserProfile, Gender, ActivityLevel, GoalType, HealthGoal, BodyMetrics,
)


def setup_logging():
    """Configure structured logging."""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    if settings.log_format == "json":
        log_format = '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'

    handlers = [logging.StreamHandler()]
    if settings.log_file:
        handlers.append(logging.FileHandler(settings.log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format=log_format,
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def print_banner():
    """Display the platform banner."""
    print(r"""
╔══════════════════════════════════════════════════════════════════╗
║           个性化健康规划多智能体平台 v2.0                        ║
║           Personalized Health Multi-Agent Platform               ║
║                                                                  ║
║   🥗 Diet Agent      — 循证营养 · BMR/TDEE · 饮食设计          ║
║   🏃 Exercise Agent  — 科学训练 · FITT-VP · 运动处方          ║
║   😴 Sleep Agent     — 睡眠医学 · CBT-I · 昼夜节律            ║
║   🧠 Consultation    — 智能路由 · 质量审查 · 多方协同          ║
║                                                                  ║
║   🔬 MCP · RAG · ChromaDB · Langfuse · Reflection               ║
║   Powered by LangGraph + DeepSeek API                            ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def quick_setup(memory: MemoryManager) -> UserProfile | None:
    """Interactive user profile setup wizard."""
    print("\n📋 让我们先快速建立您的健康档案（输入 'skip' 跳过）\n")

    name = input("您的称呼: ").strip() or "用户"
    if name.lower() == "skip":
        return None

    try:
        age = int(input("年龄: ") or "30")
        age = max(1, min(120, age))
    except ValueError:
        age = 30

    gender_map = {"m": Gender.MALE, "f": Gender.FEMALE, "": Gender.OTHER}
    gender_input = input("性别 (m/f/o，默认 o): ").strip().lower()
    gender = gender_map.get(gender_input, Gender.OTHER)

    try:
        height = float(input("身高 (cm，默认 170): ") or "170")
        height = max(50, min(250, height))
    except ValueError:
        height = 170.0

    try:
        weight = float(input("体重 (kg，默认 65): ") or "65")
        weight = max(20, min(300, weight))
    except ValueError:
        weight = 65.0

    print("\n活动水平:")
    for i, (key, label) in enumerate([
        ("sedentary", "久坐不动"),
        ("lightly_active", "轻度活动 (1-3天/周)"),
        ("moderately_active", "中度活动 (3-5天/周)"),
        ("very_active", "高度活跃 (6-7天/周)"),
        ("extra_active", "极度活跃 (运动员)"),
    ], 1):
        print(f"  {i}. {label}")
    act_input = input("选择 (1-5，默认 1): ").strip()
    act_map = {
        "1": ActivityLevel.SEDENTARY, "2": ActivityLevel.LIGHTLY_ACTIVE,
        "3": ActivityLevel.MODERATELY_ACTIVE, "4": ActivityLevel.VERY_ACTIVE,
        "5": ActivityLevel.EXTRA_ACTIVE,
    }
    activity = act_map.get(act_input, ActivityLevel.SEDENTARY)

    try:
        sleep_hours = float(input("平均睡眠时长 (小时/夜，默认 7): ") or "7")
        sleep_hours = max(0, min(16, sleep_hours))
    except ValueError:
        sleep_hours = 7.0

    diet_input = input("饮食偏好 (逗号分隔，如: 低盐,高蛋白): ").strip()
    allergies_input = input("过敏/忌口 (逗号分隔，如: 花生,海鲜): ").strip()

    profile = UserProfile(
        user_id=str(uuid.uuid4())[:8],
        name=name,
        age=age,
        gender=gender,
        current_metrics=BodyMetrics(weight_kg=weight, height_cm=height),
        activity_level=activity,
        sleep_hours_avg=sleep_hours,
        dietary_preferences=[d.strip() for d in diet_input.split(",") if d.strip()],
        allergies=[a.strip() for a in allergies_input.split(",") if a.strip()],
    )

    memory.save_user_profile(profile)
    print(f"\n✅ 档案已创建！BMI: {profile.bmi} ({profile.bmi_category})")
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
        print(f"   支持的 Provider: deepseek, openai, anthropic, ollama")
        return

    print(f"🔌 LLM Provider: {settings.llm_provider}")
    print(f"📦 Model: {settings.llm_model}")
    print(f"💾 Vector Store: ChromaDB ({settings.chroma_persist_dir})")
    print(f"📚 RAG Knowledge Base: enabled" if settings.feature_rag else "📚 RAG: disabled")
    print(f"🔬 MCP Protocol: enabled" if settings.feature_mcp else "🔬 MCP: disabled")

    memory = MemoryManager()

    # Setup profile
    profile = quick_setup(memory)
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
    print("✅ 就绪！开始对话吧")
    print("   💡 命令: 'quit' 退出 | 'reset' 重置对话 | 'stats' 系统状态 | 'eval' 运行评估\n")

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    while True:
        try:
            user_input = input("💬 您: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！祝您健康！")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("quit", "exit", "q"):
            print("👋 再见！祝您健康！")
            break

        if cmd == "reset":
            memory.clear_conversation()
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            print("🔄 对话已重置\n")
            continue

        if cmd == "stats":
            print(f"🔌 LLM: {settings.llm_provider}/{settings.llm_model}")
            print(f"💾 DataDir: {settings.data_dir}")
            print(f"👤 ActiveUser: {memory.get_active_user().name if memory.get_active_user() else 'None'}")
            try:
                from src.memory.vector_store import get_vector_store
                vs_stats = get_vector_store().get_collection_stats()
                print(f"🧠 VectorStore: {vs_stats}")
            except Exception as e:
                print(f"🧠 VectorStore: unavailable ({e})")
            continue

        if cmd == "eval":
            print("🔬 运行评估套件...")
            try:
                from src.observability.evaluator import get_evaluator

                def eval_run(user_input: str):
                    mm = MemoryManager()
                    g = get_health_graph()
                    if memory.get_active_user():
                        mm.save_user_profile(memory.get_active_user())
                    result = g.invoke(
                        {
                            "user_input": user_input,
                            "user_profile": mm.get_active_user().to_dict() if mm.get_active_user() else None,
                            "health_goals": [g.model_dump() for g in mm._goals],
                            "agent_outputs": {}, "supervisor_decision": None,
                            "final_response": None, "iteration_count": 0,
                            "errors": [], "reflection_needed": False,
                            "reflection_notes": "", "quality_gate_passed": False,
                            "next_step": "", "trace_id": str(uuid.uuid4())[:12],
                            "token_usage": {},
                        },
                        config={"configurable": {"thread_id": str(uuid.uuid4())}},
                    )
                    agent_raw = result.get("agent_outputs", {})
                    final = result.get("final_response", {})
                    agent_msgs = {}
                    from src.models import AgentMessage, AgentRole
                    for rk, data in agent_raw.items():
                        try:
                            agent_msgs[rk] = AgentMessage(**data)
                        except Exception:
                            agent_msgs[rk] = AgentMessage(role=AgentRole.CONSULTATION, content=str(data))
                    return agent_msgs, final.get("message", "")

                evaluator = get_evaluator()
                suite = evaluator.run_benchmark(eval_run)
                print(f"\n📊 评估结果: pass_rate={suite.pass_rate:.1%}, "
                      f"avg_quality={suite.avg_quality:.3f}, "
                      f"avg_latency={suite.avg_latency_ms:.0f}ms")
            except Exception as e:
                print(f"❌ 评估失败: {e}")
            continue

        # Invoke graph
        print("\n🤔 正在协调多智能体为您分析...")
        try:
            trace_id = str(uuid.uuid4())[:12]
            result = graph.invoke(
                {
                    "user_input": user_input,
                    "user_profile": memory.get_active_user().to_dict() if memory.get_active_user() else None,
                    "health_goals": [g.model_dump() for g in memory._goals],
                    "agent_outputs": {},
                    "supervisor_decision": None,
                    "final_response": None,
                    "iteration_count": 0,
                    "errors": [],
                    "reflection_needed": False,
                    "reflection_notes": "",
                    "quality_gate_passed": False,
                    "next_step": "",
                    "trace_id": trace_id,
                    "token_usage": {},
                },
                config=config,
            )

            final = result.get("final_response", {})
            if final:
                print(f"\n🧑‍⚕️ 健康顾问:\n{final.get('message', '')}")

                next_steps = final.get("next_steps", [])
                if next_steps:
                    print("\n📋 建议的后续行动:")
                    for i, step in enumerate(next_steps, 1):
                        print(f"  {i}. {step}")
            else:
                print("\n⚠️ 未能生成回复，请稍后重试。")

            # Show contributing agents
            agent_outputs = result.get("agent_outputs", {})
            if agent_outputs:
                names = {"diet": "🥗 饮食Agent", "exercise": "🏃 运动Agent",
                         "sleep": "😴 睡眠Agent", "consultation": "🧠 咨询Agent"}
                agents_list = [names.get(r, r) for r in agent_outputs]
                print(f"\n📊 协作Agent: {', '.join(agents_list)}")

            # Quality gate info
            if result.get("reflection_notes"):
                print(f"🔍 质量审查: {'✅ 通过' if result.get('quality_gate_passed') else '⚠️ 请注意'}")

            # Token usage
            token_usage = result.get("token_usage", {})
            if token_usage:
                total = token_usage.get("total", sum(token_usage.values()))
                print(f"🔢 Token 用量: ~{total}")

        except Exception as e:
            logger.error(f"Graph invocation error: {e}", exc_info=True)
            print(f"\n❌ 处理出错: {e}")

        print()


if __name__ == "__main__":
    main()
