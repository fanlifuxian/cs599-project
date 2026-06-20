"""
LangGraph State definition — enhanced shared state for enterprise multi-agent orchestration.

Supports:
- Reflection loop (critique → improve)
- Quality gate (pass/fail before responding)
- Parallel agent execution with merge
- Token usage tracking per agent
- Error accumulation for debugging
- Trace ID for observability
"""

from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict
from langgraph.graph.message import add_messages


class HealthGraphState(TypedDict):
    """
    Enhanced shared state for the health multi-agent LangGraph.

    Flow:
        START → supervisor → [diet | exercise | sleep] → reflect → synthesize → END
                                                                    ↑___________|
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

    # Reflection & Quality
    reflection_needed: bool  # Whether to run reflection
    reflection_notes: str  # Reflection output
    quality_gate_passed: bool  # Did synthesis pass quality check?

    # Final output
    final_response: Optional[dict]  # AgentResponse serialized

    # Control flow
    iteration_count: int
    errors: list[str]
    next_step: str  # "route" | "synthesize" | "reflect" | "end"

    # Observability
    trace_id: str
    token_usage: dict[str, int]  # {agent_role: total_tokens}
