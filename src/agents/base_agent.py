"""
Base Agent — abstract class wrapping LLM + tool binding + invocation loop.
Uses OpenAI-compatible API for tool calling (Function Calling).
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

from openai import OpenAI
from src.config.settings import settings
from src.models.schemas import AgentRole, AgentMessage

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all specialist agents.
    Handles LLM initialization, tool calling loop, and structured output.
    """

    role: AgentRole
    system_prompt: str
    tools_schema: list[dict] = []
    tools_map: dict[str, Callable] = {}

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.model = settings.llm_model
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens

        if not settings.llm_api_key:
            logger.warning(
                f"[{self.role.value}] No API key configured. "
                "Set DEEPSEEK_API_KEY or OPENAI_API_KEY in .env"
            )

    @abstractmethod
    def _build_system_prompt(self, context: dict[str, Any]) -> str:
        """Build the system prompt with user context injected."""
        ...

    def invoke(self, user_message: str, context: dict[str, Any] | None = None) -> AgentMessage:
        """
        Invoke the agent with a user message.
        Runs the tool-calling loop and returns a structured AgentMessage.
        """
        context = context or {}
        system_prompt = self._build_system_prompt(context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        tool_calls_made: list[str] = []
        logger.info(f"[{self.role.value}] Invoking with message: {user_message[:100]}...")

        try:
            # First LLM call — may request tool calls
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools_schema if self.tools_schema else None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            msg = response.choices[0].message

            # Tool calling loop
            max_tool_rounds = 5
            round_count = 0

            while msg.tool_calls and round_count < max_tool_rounds:
                round_count += 1
                # Append assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

                # Execute each tool call
                for tc in msg.tool_calls:
                    func_name = tc.function.name
                    tool_calls_made.append(func_name)

                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    logger.info(f"[{self.role.value}] Tool call: {func_name}({args})")

                    if func_name in self.tools_map:
                        try:
                            result = self.tools_map[func_name](**args)
                        except Exception as e:
                            result = {"error": str(e)}
                    else:
                        result = {"error": f"Unknown tool: {func_name}"}

                    # Append tool result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })

                # Continue the conversation
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools_schema if self.tools_schema else None,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                msg = response.choices[0].message

            # Final response
            content = msg.content or ""

            # Parse confidence from content if present
            confidence = 0.85
            if "confidence:" in content.lower():
                try:
                    conf_line = [l for l in content.split("\n") if "confidence:" in l.lower()][0]
                    confidence = float(conf_line.split(":")[-1].strip().rstrip("%")) / 100
                    if confidence > 1:
                        confidence /= 100
                except (ValueError, IndexError):
                    pass

            logger.info(f"[{self.role.value}] Response generated ({len(content)} chars)")

            return AgentMessage(
                role=self.role,
                content=content,
                confidence=confidence,
                tool_calls_made=tool_calls_made,
            )

        except Exception as e:
            logger.error(f"[{self.role.value}] Error: {e}")
            return AgentMessage(
                role=self.role,
                content=f"抱歉，我在处理您的请求时遇到了问题：{str(e)}。请稍后重试或换个方式提问。",
                confidence=0.0,
                tool_calls_made=tool_calls_made,
            )
