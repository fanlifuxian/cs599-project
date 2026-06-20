"""
Enterprise-grade Health Multi-Agent LangGraph — Supervisor-Worker + Reflection.

Enhanced graph flow:
    START → supervisor_node
                ↓
         ┌──────┼──────┐       (conditional parallel routing)
         ↓      ↓      ↓
      diet   exercise  sleep    (run in parallel when possible)
         ↓      ↓      ↓
         └──────┼──────┘
                ↓
         reflect_node            (quality gate — critique & improve)
                ↓
         synthesize_node → END

Key enhancements:
- Parallel agent execution via LangGraph's Send API
- Reflection/quality-gate node before final synthesis
- Token usage aggregation across agents
- Structured error handling with graceful degradation
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send

from src.graph.state import HealthGraphState
from src.agents.consultation_agent import ConsultationAgent
from src.agents.diet_agent import DietAgent
from src.agents.exercise_agent import ExerciseAgent
from src.agents.sleep_agent import SleepAgent
from src.memory.memory_manager import MemoryManager
from src.models import AgentRole, AgentMessage, SupervisorDecision

logger = logging.getLogger(__name__)


# ── Agent singletons (lazy init) ────────────────────────────────────────

_diet_agent: DietAgent | None = None
_exercise_agent: ExerciseAgent | None = None
_sleep_agent: SleepAgent | None = None
_consultation_agent: ConsultationAgent | None = None
_memory_manager: MemoryManager | None = None


def _get_memory() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def _get_consultation() -> ConsultationAgent:
    global _consultation_agent
    if _consultation_agent is None:
        _consultation_agent = ConsultationAgent()
    return _consultation_agent


def _get_diet() -> DietAgent:
    global _diet_agent
    if _diet_agent is None:
        _diet_agent = DietAgent()
    return _diet_agent


def _get_exercise() -> ExerciseAgent:
    global _exercise_agent
    if _exercise_agent is None:
        _exercise_agent = ExerciseAgent()
    return _exercise_agent


def _get_sleep() -> SleepAgent:
    global _sleep_agent
    if _sleep_agent is None:
        _sleep_agent = SleepAgent()
    return _sleep_agent


# ── Node Functions ──────────────────────────────────────────────────────

def supervisor_node(state: HealthGraphState) -> dict:
    """
    Supervisor node (Consultation Agent):
    1. Stores user message in memory
    2. Performs deep intent analysis
    3. Decides routing to specialist agents
    4. Handles non-routed queries directly
    """
    user_input = state.get("user_input", "")
    if not user_input and state.get("messages"):
        last_msg = state["messages"][-1]
        user_input = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

    trace_id = state.get("trace_id", str(uuid.uuid4())[:8])
    logger.info(f"[{trace_id}] Supervisor processing: {user_input[:100]}...")

    memory = _get_memory()
    consultation = _get_consultation()

    # Store user message
    memory.add_user_message(user_input)

    # Build context
    context = memory.build_agent_context(user_query=user_input)

    # Get routing decision
    decision = consultation.detect_route(user_input)
    logger.info(f"[{trace_id}] Routing decision: {decision.target_agents}, "
                f"route={decision.should_route}, reflect={decision.requires_reflection}")

    # If no routing needed, handle with consultation tools directly
    if not decision.should_route or not decision.target_agents:
        response = consultation.invoke(user_input, context)
        memory.add_ai_message(response.content)

        return {
            "user_input": user_input,
            "user_profile": context.get("user_profile"),
            "health_goals": context.get("health_goals", []),
            "supervisor_decision": decision.model_dump(),
            "agent_outputs": {},
            "reflection_needed": False,
            "reflection_notes": "",
            "quality_gate_passed": True,
            "next_step": "end",
            "final_response": {
                "message": response.content,
                "agent_contributions": {"consultation": response.content},
                "plans": {},
                "next_steps": [],
                "confidence_level": "medium",
                "timestamp": datetime.now().isoformat(),
            },
            "iteration_count": state.get("iteration_count", 0) + 1,
            "errors": state.get("errors", []),
            "trace_id": trace_id,
            "token_usage": state.get("token_usage", {}),
        }

    return {
        "user_input": user_input,
        "user_profile": context.get("user_profile"),
        "health_goals": context.get("health_goals", []),
        "supervisor_decision": decision.model_dump(),
        "agent_outputs": state.get("agent_outputs", {}),
        "reflection_needed": decision.requires_reflection,
        "reflection_notes": "",
        "quality_gate_passed": False,
        "next_step": "route",
        "iteration_count": state.get("iteration_count", 0) + 1,
        "errors": state.get("errors", []),
        "trace_id": trace_id,
        "token_usage": state.get("token_usage", {}),
    }


def diet_node(state: HealthGraphState) -> dict:
    """Diet specialist agent node with full context."""
    trace_id = state.get("trace_id", "unknown")
    logger.info(f"[{trace_id}] 🥗 DietAgent invoking...")

    memory = _get_memory()
    agent = _get_diet()
    user_input = state.get("user_input", "")
    context = memory.build_agent_context(user_query=user_input)

    result = agent.invoke(user_input, context)

    agent_outputs = dict(state.get("agent_outputs", {}))
    agent_outputs[AgentRole.DIET.value] = result.model_dump()

    # Track token usage
    token_usage = dict(state.get("token_usage", {}))
    token_usage["diet"] = result.token_usage.get("total_tokens", 0)

    logger.info(f"[{trace_id}] DietAgent done — confidence: {result.confidence:.2f}, "
                f"tools: {result.tool_calls_made}")

    return {
        "agent_outputs": agent_outputs,
        "token_usage": token_usage,
    }


def exercise_node(state: HealthGraphState) -> dict:
    """Exercise specialist agent node with full context."""
    trace_id = state.get("trace_id", "unknown")
    logger.info(f"[{trace_id}] 🏃 ExerciseAgent invoking...")

    memory = _get_memory()
    agent = _get_exercise()
    user_input = state.get("user_input", "")
    context = memory.build_agent_context(user_query=user_input)

    result = agent.invoke(user_input, context)

    agent_outputs = dict(state.get("agent_outputs", {}))
    agent_outputs[AgentRole.EXERCISE.value] = result.model_dump()

    token_usage = dict(state.get("token_usage", {}))
    token_usage["exercise"] = result.token_usage.get("total_tokens", 0)

    logger.info(f"[{trace_id}] ExerciseAgent done — confidence: {result.confidence:.2f}, "
                f"tools: {result.tool_calls_made}")

    return {
        "agent_outputs": agent_outputs,
        "token_usage": token_usage,
    }


def sleep_node(state: HealthGraphState) -> dict:
    """Sleep specialist agent node with full context."""
    trace_id = state.get("trace_id", "unknown")
    logger.info(f"[{trace_id}] 😴 SleepAgent invoking...")

    memory = _get_memory()
    agent = _get_sleep()
    user_input = state.get("user_input", "")
    context = memory.build_agent_context(user_query=user_input)

    result = agent.invoke(user_input, context)

    agent_outputs = dict(state.get("agent_outputs", {}))
    agent_outputs[AgentRole.SLEEP.value] = result.model_dump()

    token_usage = dict(state.get("token_usage", {}))
    token_usage["sleep"] = result.token_usage.get("total_tokens", 0)

    logger.info(f"[{trace_id}] SleepAgent done — confidence: {result.confidence:.2f}, "
                f"tools: {result.tool_calls_made}")

    return {
        "agent_outputs": agent_outputs,
        "token_usage": token_usage,
    }


def reflect_node(state: HealthGraphState) -> dict:
    """
    Reflection node — quality gate before synthesis.

    Checks:
    1. Are all expected agent outputs present?
    2. Do any agent outputs have low confidence?
    3. Are there contradictions between agents?
    """
    trace_id = state.get("trace_id", "unknown")
    logger.info(f"[{trace_id}] 🔍 Reflection node — quality check...")

    agent_outputs = state.get("agent_outputs", {})
    errors = list(state.get("errors", []))
    reflection_notes = []

    # Check 1: All routed agents produced output
    decision_data = state.get("supervisor_decision", {})
    expected = decision_data.get("target_agents", [])
    actual = list(agent_outputs.keys())

    missing = [a for a in expected if a not in actual]
    if missing:
        note = f"⚠️ 以下 Agent 未产出结果: {missing}"
        reflection_notes.append(note)
        errors.append(note)
        logger.warning(f"[{trace_id}] {note}")

    # Check 2: Confidence check
    for role_key, data in agent_outputs.items():
        confidence = data.get("confidence", 0.0)
        if confidence < 0.4:
            note = f"⚠️ {role_key} Agent 置信度偏低 ({confidence:.2f})"
            reflection_notes.append(note)
        elif confidence >= 0.85:
            reflection_notes.append(f"✅ {role_key} Agent 置信度良好 ({confidence:.2f})")

    # Check 3: Quick contradiction detection (keyword-based)
    all_content = " ".join(data.get("content", "") for data in agent_outputs.values())
    contradiction_pairs = [
        (["低碳水", "低脂"], "低碳水与低脂同时出现，可能互相矛盾"),
        (["高碳水", "低碳水"], "碳水建议存在矛盾"),
    ]
    for keywords, note in contradiction_pairs:
        if all(kw in all_content for kw in keywords):
            reflection_notes.append(f"⚠️ 可能存在矛盾: {note}")

    quality_passed = len([n for n in reflection_notes if "⚠️" in n]) == 0

    logger.info(f"[{trace_id}] Reflection done — passed={quality_passed}, "
                f"notes: {len(reflection_notes)}")

    return {
        "reflection_needed": False,
        "reflection_notes": "\n".join(reflection_notes),
        "quality_gate_passed": quality_passed,
        "errors": errors,
    }


def synthesize_node(state: HealthGraphState) -> dict:
    """
    Synthesis node: combines specialist outputs into a cohesive final response.
    Includes optional reflection-based improvement.
    """
    trace_id = state.get("trace_id", "unknown")
    logger.info(f"[{trace_id}] 📝 Synthesizing final response...")

    memory = _get_memory()
    consultation = _get_consultation()

    agent_outputs_raw = state.get("agent_outputs", {})

    # Deserialize AgentMessages
    agent_messages: dict[str, AgentMessage] = {}
    for role_key, data in agent_outputs_raw.items():
        try:
            agent_messages[role_key] = AgentMessage(**data)
        except Exception as e:
            logger.error(f"[{trace_id}] Failed to parse {role_key} output: {e}")

    decision = SupervisorDecision(**state.get("supervisor_decision", {}))
    user_input = state.get("user_input", "")
    user_profile = state.get("user_profile")

    # Synthesize
    response = consultation.synthesize(
        user_message=user_input,
        agent_outputs=agent_messages,
        decision=decision,
        user_profile=user_profile,
    )

    # Run reflection if needed
    if state.get("reflection_notes"):
        response.message = (
            f"> 🔍 质量审查通过\n\n{response.message}"
            if state.get("quality_gate_passed")
            else f"> ⚠️ 部分建议置信度偏低，请酌情参考\n\n{response.message}"
        )

    # Save to memory
    memory.add_ai_message(response.message)
    if user_profile and "user_id" in user_profile:
        memory.save_response(user_profile.get("user_id", "unknown"), response)

    # Track total tokens
    token_usage = state.get("token_usage", {})
    total_tokens = sum(token_usage.values())

    logger.info(f"[{trace_id}] Synthesis complete — "
                f"agents: {list(agent_messages.keys())}, "
                f"total_tokens: {total_tokens}")

    return {
        "final_response": response.model_dump(),
        "next_step": "end",
        "token_usage": {**token_usage, "total": total_tokens},
    }


# ── Routing Functions ───────────────────────────────────────────────────

def route_after_supervisor(state: HealthGraphState) -> list[str]:
    """
    Conditional edge from supervisor — decide which agents to invoke.
    Returns node names for parallel execution.
    """
    decision_data = state.get("supervisor_decision", {})
    target_agents = decision_data.get("target_agents", [])
    should_route = decision_data.get("should_route", False)

    if not should_route or not target_agents:
        logger.info("[Router] No routing needed → END")
        return ["__end__"]

    # Map AgentRole to node name
    role_to_node = {
        "diet": "diet_agent",
        "exercise": "exercise_agent",
        "sleep": "sleep_agent",
    }

    nodes = [role_to_node[role] for role in target_agents if role in role_to_node]
    logger.info(f"[Router] Parallel routing to: {nodes}")
    return nodes if nodes else ["__end__"]


def route_after_specialist(state: HealthGraphState) -> str:
    """
    After specialists complete, check if reflection is needed.
    If yes → reflect, if no → synthesize.
    """
    if state.get("reflection_needed", False) and state.get("agent_outputs"):
        has_multiple = len(state.get("agent_outputs", {})) >= 2
        if has_multiple:
            logger.info("[Router] Multiple agents → reflect before synthesize")
            return "reflect"

    return "synthesize"


def route_after_reflect(state: HealthGraphState) -> str:
    """After reflection, always go to synthesize."""
    return "synthesize"


# ── Graph Builder ───────────────────────────────────────────────────────

def build_health_graph() -> StateGraph:
    """
    Build and compile the enterprise health multi-agent LangGraph.

    Architecture:
        supervisor → [diet | exercise | sleep] (parallel) → reflect → synthesize → END
    """
    graph = StateGraph(HealthGraphState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("diet_agent", diet_node)
    graph.add_node("exercise_agent", exercise_node)
    graph.add_node("sleep_agent", sleep_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("synthesize", synthesize_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor → conditional parallel routing
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "diet_agent": "diet_agent",
            "exercise_agent": "exercise_agent",
            "sleep_agent": "sleep_agent",
            "__end__": END,
        },
    )

    # Each specialist → reflection check (or direct to synthesize)
    graph.add_conditional_edges(
        "diet_agent",
        route_after_specialist,
        {"reflect": "reflect", "synthesize": "synthesize"},
    )
    graph.add_conditional_edges(
        "exercise_agent",
        route_after_specialist,
        {"reflect": "reflect", "synthesize": "synthesize"},
    )
    graph.add_conditional_edges(
        "sleep_agent",
        route_after_specialist,
        {"reflect": "reflect", "synthesize": "synthesize"},
    )

    # Reflect → synthesize
    graph.add_edge("reflect", "synthesize")

    # Synthesize → END
    graph.add_edge("synthesize", END)

    # Compile with memory checkpointer
    memory_saver = MemorySaver()
    compiled = graph.compile(checkpointer=memory_saver)

    logger.info("[Graph] Enterprise health multi-agent graph compiled successfully.")
    logger.info("[Graph] Nodes: supervisor, diet_agent, exercise_agent, sleep_agent, reflect, synthesize")
    return compiled


# ── Module-level singleton ──────────────────────────────────────────────

_health_graph: StateGraph | None = None


def get_health_graph() -> StateGraph:
    """Get or create the compiled health graph singleton."""
    global _health_graph
    if _health_graph is None:
        _health_graph = build_health_graph()
    return _health_graph
