"""
Tools module — enterprise-grade function calling tools for health agents.

Tool categories:
- diet_tools: Nutrition calculation, meal planning, food analysis
- exercise_tools: Workout planning, exercise recommendation, calorie estimation
- sleep_tools: Sleep quality analysis, sleep planning, hygiene tips
- common_tools: User profile, goal setting, progress tracking, risk assessment
- mcp_tools: MCP protocol — health guidelines, drug interactions, medical reference
- rag_tools: Agentic RAG — evidence-based health knowledge retrieval
"""

from src.tools.diet_tools import DIET_TOOLS_SCHEMA, DIET_TOOLS_MAP
from src.tools.exercise_tools import EXERCISE_TOOLS_SCHEMA, EXERCISE_TOOLS_MAP
from src.tools.sleep_tools import SLEEP_TOOLS_SCHEMA, SLEEP_TOOLS_MAP
from src.tools.common_tools import COMMON_TOOLS_SCHEMA, COMMON_TOOLS_MAP
from src.tools.mcp_tools import MCP_TOOLS_SCHEMA, MCP_TOOLS_MAP
from src.tools.rag_tools import RAG_TOOLS_SCHEMA, RAG_TOOLS_MAP

__all__ = [
    "DIET_TOOLS_SCHEMA", "DIET_TOOLS_MAP",
    "EXERCISE_TOOLS_SCHEMA", "EXERCISE_TOOLS_MAP",
    "SLEEP_TOOLS_SCHEMA", "SLEEP_TOOLS_MAP",
    "COMMON_TOOLS_SCHEMA", "COMMON_TOOLS_MAP",
    "MCP_TOOLS_SCHEMA", "MCP_TOOLS_MAP",
    "RAG_TOOLS_SCHEMA", "RAG_TOOLS_MAP",
]
