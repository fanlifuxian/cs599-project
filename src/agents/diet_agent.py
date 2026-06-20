"""
Diet Agent — 饮食营养专家。

职责单一：定义角色身份和工具集，提示词委托给 PromptManager。
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.models import AgentRole
from src.prompts.manager import get_prompt_manager
from src.tools.diet_tools import DIET_TOOLS_SCHEMA, DIET_TOOLS_MAP
from src.tools.mcp_tools import MCP_TOOLS_SCHEMA, MCP_TOOLS_MAP
from src.tools.rag_tools import RAG_TOOLS_SCHEMA, RAG_TOOLS_MAP


class DietAgent(BaseAgent):
    """饮食健康专家 Agent — 营养计算、饮食计划、食物分析、循证建议。"""

    role = AgentRole.DIET
    tools_schema = DIET_TOOLS_SCHEMA + MCP_TOOLS_SCHEMA + RAG_TOOLS_SCHEMA
    tools_map = {**DIET_TOOLS_MAP, **MCP_TOOLS_MAP, **RAG_TOOLS_MAP}

    def _build_system_prompt(self, context: dict[str, Any]) -> str:
        return get_prompt_manager().build_diet_prompt(context)
