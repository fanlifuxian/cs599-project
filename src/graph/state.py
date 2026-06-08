"""
LangGraph State definition — the shared state that flows through the multi-agent graph.
Uses TypedDict for LangGraph compatibility.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict
from langgraph.graph.message import add_messages


class HealthGraphState(TypedDict):
    """
    The shared state for the health multi-agent LangGraph.

    LangGraph merges node return values into this state.
    `add_messages` reducer appends messages to the list.
    """

    # Conversation
    messages: Annotated[list, add_messages]
    user_input: str

    # User context (persisted across turns)
    user_profile: Optional[dict]
    health_goals: list[dict]

    # Orchestration
    supervisor_decision: Optional[dict]  # SupervisorDecision serialized
    agent_outputs: dict[str, dict]  # {role: AgentMessage serialized}

    # Final output
    final_response: Optional[dict]  # AgentResponse serialized

    # Control
    iteration_count: int
    errors: list[str]
    next_step: str  # "route", "synthesize", "end"
