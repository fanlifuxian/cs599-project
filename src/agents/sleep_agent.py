"""
Sleep Agent — 睡眠健康专家。

职责单一：定义角色身份和工具集，提示词委托给 PromptManager。
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.models import AgentRole
from src.prompts.manager import get_prompt_manager
from src.tools.sleep_tools import SLEEP_TOOLS_SCHEMA, SLEEP_TOOLS_MAP
from src.tools.mcp_tools import MCP_TOOLS_SCHEMA, MCP_TOOLS_MAP
from src.tools.rag_tools import RAG_TOOLS_SCHEMA, RAG_TOOLS_MAP


class SleepAgent(BaseAgent):
    """睡眠健康专家 Agent — 睡眠分析、改善计划、CBT-I、卫生教育。"""

    role = AgentRole.SLEEP
    tools_schema = SLEEP_TOOLS_SCHEMA + MCP_TOOLS_SCHEMA + RAG_TOOLS_SCHEMA
    tools_map = {**SLEEP_TOOLS_MAP, **MCP_TOOLS_MAP, **RAG_TOOLS_MAP}

    def _build_system_prompt(self, context: dict[str, Any]) -> str:
        return get_prompt_manager().build_sleep_prompt(context)
