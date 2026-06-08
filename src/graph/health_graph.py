"""
Health Multi-Agent LangGraph — Supervisor-Worker orchestration.

Graph flow:
    START → supervisor_node
                ↓
         ┌──────┼──────┐
         ↓      ↓      ↓       (conditional routing)
      diet   exercise  sleep
         ↓      ↓      ↓
         └──────┼──────┘
                ↓
         synthesize_node → END

The Supervisor (Consultation Agent) analyzes user intent and decides
which specialist agents to invoke. Specialists run in parallel, then
the supervisor synthesizes a final response.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import HealthGraphState
from src.agents.consultation_agent import ConsultationAgent
from src.agents.diet_agent import DietAgent
from src.agents.exercise_agent import ExerciseAgent
from src.agents.sleep_agent import SleepAgent
from src.memory.memory_manager import MemoryManager
from src.models.schemas import AgentRole

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
    2. Detects intent and decides routing
    3. If no routing needed, handles directly
    """
    user_input = state.get("user_input", "")
    if not user_input and state.get("messages"):
        user_input = state["messages"][-1].content if hasattr(state["messages"][-1], 'content') else str(state["messages"][-1])

    logger.info(f"[Supervisor] Processing: {user_input[:100]}...")

    memory = _get_memory()
    consultation = _get_consultation()

    # Add user message to memory
    memory.add_user_message(user_input)

    # Build context
    context = memory.build_agent_context()

    # Get routing decision
    decision = consultation.detect_route(user_input)

    # If profile-related, use consultation tools
    if not decision.should_route:
        # Let consultation handle directly
        response = consultation.invoke(user_input, context)
        memory.add_ai_message(response.content)

        return {
            "user_input": user_input,
            "user_profile": context.get("user_profile"),
            "health_goals": context.get("health_goals", []),
            "supervisor_decision": decision.model_dump(),
            "agent_outputs": {},
            "next_step": "end",
            "final_response": {
                "message": response.content,
                "agent_contributions": {"consultation": response.content},
                "plans": {},
                "next_steps": [],
                "timestamp": response.content[:50] if response.content else "",
            },
            "iteration_count": state.get("iteration_count", 0) + 1,
            "errors": state.get("errors", []),
        }

    return {
        "user_input": user_input,
        "user_profile": context.get("user_profile"),
        "health_goals": context.get("health_goals", []),
        "supervisor_decision": decision.model_dump(),
        "agent_outputs": state.get("agent_outputs", {}),
        "next_step": "route",
        "iteration_count": state.get("iteration_count", 0) + 1,
        "errors": state.get("errors", []),
    }


def diet_node(state: HealthGraphState) -> dict:
    """Diet specialist agent node."""
    logger.info("[DietAgent] Invoking...")
    memory = _get_memory()
    agent = _get_diet()

    context = memory.build_agent_context()
    user_input = state.get("user_input", "")

    result = agent.invoke(user_input, context)

    agent_outputs = dict(state.get("agent_outputs", {}))
    agent_outputs[AgentRole.DIET.value] = result.model_dump()

    return {"agent_outputs": agent_outputs}


def exercise_node(state: HealthGraphState) -> dict:
    """Exercise specialist agent node."""
    logger.info("[ExerciseAgent] Invoking...")
    memory = _get_memory()
    agent = _get_exercise()

    context = memory.build_agent_context()
    user_input = state.get("user_input", "")

    result = agent.invoke(user_input, context)

    agent_outputs = dict(state.get("agent_outputs", {}))
    agent_outputs[AgentRole.EXERCISE.value] = result.model_dump()

    return {"agent_outputs": agent_outputs}


def sleep_node(state: HealthGraphState) -> dict:
    """Sleep specialist agent node."""
    logger.info("[SleepAgent] Invoking...")
    memory = _get_memory()
    agent = _get_sleep()

    context = memory.build_agent_context()
    user_input = state.get("user_input", "")

    result = agent.invoke(user_input, context)

    agent_outputs = dict(state.get("agent_outputs", {}))
    agent_outputs[AgentRole.SLEEP.value] = result.model_dump()

    return {"agent_outputs": agent_outputs}


def synthesize_node(state: HealthGraphState) -> dict:
    """
    Synthesis node: combines specialist outputs into final response.
    """
    logger.info("[Synthesize] Combining agent outputs...")
    memory = _get_memory()
    consultation = _get_consultation()

    agent_outputs_raw = state.get("agent_outputs", {})

    # Deserialize AgentMessages
    from src.models.schemas import AgentMessage, SupervisorDecision
    agent_messages: dict[str, AgentMessage] = {}
    for role_key, data in agent_outputs_raw.items():
        agent_messages[role_key] = AgentMessage(**data)

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

    # Save to memory
    memory.add_ai_message(response.message)
    if user_profile and "user_id" in user_profile:
        memory.save_response(user_profile["user_id"], response)

    return {
        "final_response": response.model_dump(),
        "next_step": "end",
    }


# ── Routing Functions ───────────────────────────────────────────────────

def route_after_supervisor(state: HealthGraphState) -> list[str]:
    """
    Conditional edge: decide which agents to invoke.
    If supervisor decides to route, return list of target agent node names.
    Otherwise, go directly to END.
    """
    decision_data = state.get("supervisor_decision", {})
    target_agents = decision_data.get("target_agents", [])
    should_route = decision_data.get("should_route", False)

    if not should_route or not target_agents:
        return ["__end__"]

    # Map AgentRole to node name
    role_to_node = {
        "diet": "diet_agent",
        "exercise": "exercise_agent",
        "sleep": "sleep_agent",
    }

    nodes = [role_to_node[role] for role in target_agents if role in role_to_node]
    logger.info(f"[Router] Routing to: {nodes}")
    return nodes if nodes else ["__end__"]


def route_after_specialist(state: HealthGraphState) -> str:
    """
    After all specialist agents have run, go to synthesize.
    """
    return "synthesize"


# ── Graph Builder ───────────────────────────────────────────────────────

def build_health_graph() -> StateGraph:
    """
    Build and compile the health multi-agent LangGraph.

    Returns a compiled graph ready for invocation.
    """
    graph = StateGraph(HealthGraphState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("diet_agent", diet_node)
    graph.add_node("exercise_agent", exercise_node)
    graph.add_node("sleep_agent", sleep_node)
    graph.add_node("synthesize", synthesize_node)

    # Set entry point
    graph.set_entry_point("supervisor")

    # Supervisor → conditional routing to specialists (or END)
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

    # Each specialist → synthesize
    graph.add_edge("diet_agent", "synthesize")
    graph.add_edge("exercise_agent", "synthesize")
    graph.add_edge("sleep_agent", "synthesize")

    # Synthesize → END
    graph.add_edge("synthesize", END)

    # Compile with memory checkpointer for conversation persistence
    memory_saver = MemorySaver()
    compiled = graph.compile(checkpointer=memory_saver)

    logger.info("[Graph] Health multi-agent graph compiled successfully.")
    return compiled


# ── Module-level singleton ──────────────────────────────────────────────

_health_graph = None


def get_health_graph() -> StateGraph:
    """Get or create the compiled health graph singleton."""
    global _health_graph
    if _health_graph is None:
        _health_graph = build_health_graph()
    return _health_graph
