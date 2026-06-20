"""
Exercise Agent — 运动健身专家。

职责单一：定义角色身份和工具集，提示词委托给 PromptManager。
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.models import AgentRole
from src.prompts.manager import get_prompt_manager
from src.tools.exercise_tools import EXERCISE_TOOLS_SCHEMA, EXERCISE_TOOLS_MAP
from src.tools.mcp_tools import MCP_TOOLS_SCHEMA, MCP_TOOLS_MAP
from src.tools.rag_tools import RAG_TOOLS_SCHEMA, RAG_TOOLS_MAP


class ExerciseAgent(BaseAgent):
    """运动健身专家 Agent — 运动处方、训练计划、热量估算、安全指导。"""

    role = AgentRole.EXERCISE
    tools_schema = EXERCISE_TOOLS_SCHEMA + MCP_TOOLS_SCHEMA + RAG_TOOLS_SCHEMA
    tools_map = {**EXERCISE_TOOLS_MAP, **MCP_TOOLS_MAP, **RAG_TOOLS_MAP}

    def _build_system_prompt(self, context: dict[str, Any]) -> str:
        return get_prompt_manager().build_exercise_prompt(context)
