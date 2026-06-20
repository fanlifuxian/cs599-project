"""
Base Agent — 纯 LLM + 工具调用循环。

职责单一：
- 管理 LLM 客户端（多 Provider）
- 执行 ReAct 工具调用循环
- 委托熔断/重试给 infrastructure/resilience.py
- 委托回复解析给 agents/parser.py
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Generator

from openai import OpenAI

from src.config.settings import settings
from src.models import AgentRole, AgentMessage
from src.middleware.resilience import CircuitBreaker, retry_with_backoff, get_fallback_message
from src.agents.response_parser import (
    extract_confidence, extract_sources, extract_caveats, extract_reasoning_trace,
)

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    所有健康专家 Agent 的抽象基类。

    子类只需实现：
    - role: AgentRole — Agent 身份
    - _build_system_prompt(context) — 构建系统提示词
    - tools_schema / tools_map — 工具定义
    """

    role: AgentRole
    system_prompt: str = ""
    tools_schema: list[dict] = []
    tools_map: dict[str, Callable] = {}

    def __init__(self):
        self.model = settings.llm_model
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        self._client: OpenAI | None = None

        # 委托给独立模块
        self.circuit = CircuitBreaker(name=self.role.value)

        if not settings.llm_api_key:
            logger.warning(f"[{self.role.value}] No API key configured")

    # ── 懒加载 LLM 客户端 ───────────────────────────────────────────────

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            api_key = settings.llm_api_key or "sk-placeholder"
            self._client = OpenAI(
                api_key=api_key,
                base_url=settings.llm_base_url,
                timeout=settings.request_timeout,
                max_retries=0,
            )
        return self._client

    # ── 子类必须实现 ────────────────────────────────────────────────────

    @abstractmethod
    def _build_system_prompt(self, context: dict[str, Any]) -> str:
        ...

    # ── 核心调用（同步）─────────────────────────────────────────────────

    def invoke(
        self, user_message: str, context: dict[str, Any] | None = None
    ) -> AgentMessage:
        """同步调用 Agent，执行完整的工具调用循环。"""
        context = context or {}

        if self.circuit.is_open:
            return self._fallback("熔断器开启")

        system_prompt = self._build_system_prompt(context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        tool_calls_made: list[str] = []
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        logger.info(f"[{self.role.value}] Invoking — {user_message[:100]}...")

        try:
            msg = self._call_llm(messages, token_usage)
            self.circuit.record_success()

            # ── ReAct 工具调用循环 ──────────────────────────────────
            for _ in range(settings.agent_max_tool_rounds):
                if not msg.tool_calls:
                    break
                messages = self._handle_tool_calls(messages, msg, tool_calls_made)
                msg = self._call_llm(messages, token_usage)
                self.circuit.record_success()

            content = msg.content or ""

            return AgentMessage(
                role=self.role,
                content=content,
                confidence=extract_confidence(content, len(tool_calls_made)),
                tool_calls_made=tool_calls_made,
                reasoning_trace=extract_reasoning_trace(tool_calls_made),
                sources=extract_sources(content),
                caveats=extract_caveats(content),
                token_usage=token_usage,
            )

        except Exception as e:
            self.circuit.record_failure()
            logger.error(f"[{self.role.value}] Failed: {e}")
            return self._fallback(str(e))

    # ── 流式调用 ────────────────────────────────────────────────────────

    def invoke_stream(
        self, user_message: str, context: dict[str, Any] | None = None
    ) -> Generator[str, None, AgentMessage]:
        """流式调用 Agent，yield 文本块。"""
        context = context or {}

        if self.circuit.is_open:
            yield "服务暂时不可用，请稍后重试。\n"
            return AgentMessage(role=self.role, content="", confidence=0.0, tool_calls_made=[])

        system_prompt = self._build_system_prompt(context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        tool_calls_made: list[str] = []
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            msg = self._call_llm(messages, token_usage)
            self.circuit.record_success()

            for _ in range(settings.agent_max_tool_rounds):
                if not msg.tool_calls:
                    break
                messages = self._handle_tool_calls(messages, msg, tool_calls_made)
                msg = self._call_llm(messages, token_usage)
                self.circuit.record_success()

            full_content = ""
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    full_content += delta
                    yield delta

            return AgentMessage(
                role=self.role,
                content=full_content,
                confidence=extract_confidence(full_content, len(tool_calls_made)),
                tool_calls_made=tool_calls_made,
                token_usage=token_usage,
            )

        except Exception as e:
            self.circuit.record_failure()
            logger.error(f"[{self.role.value}] Stream failed: {e}")
            yield f"\n\n（处理出错：{str(e)[:100]}）"
            return AgentMessage(role=self.role, content=f"Error: {e}", confidence=0.0, tool_calls_made=[])

    # ── LLM 调用（带重试）───────────────────────────────────────────────

    def _call_llm(self, messages: list[dict], token_usage: dict[str, int]):
        """调用 LLM API，自动重试。"""
        @retry_with_backoff()
        def _do_call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools_schema or None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            if response.usage:
                token_usage["prompt_tokens"] += response.usage.prompt_tokens or 0
                token_usage["completion_tokens"] += response.usage.completion_tokens or 0
                token_usage["total_tokens"] += response.usage.total_tokens or 0
            return response.choices[0].message

        return _do_call()

    # ── 工具调用处理 ────────────────────────────────────────────────────

    def _handle_tool_calls(
        self, messages: list[dict], msg, tool_calls_made: list[str]
    ) -> list[dict]:
        """执行所有工具调用并追加结果到消息列表。"""
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            func_name = tc.function.name
            tool_calls_made.append(func_name)
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            logger.info(f"[{self.role.value}] 🔧 {func_name}({str(args)[:100]})")
            try:
                result = self.tools_map[func_name](**args) if func_name in self.tools_map else {"error": f"Unknown: {func_name}"}
            except Exception as e:
                result = {"error": str(e)}
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

        return messages

    # ── 降级 ────────────────────────────────────────────────────────────

    def _fallback(self, error: str) -> AgentMessage:
        """生成优雅降级回复。"""
        return AgentMessage(
            role=self.role,
            content=get_fallback_message(self.role.value, error),
            confidence=0.0,
            tool_calls_made=[],
            caveats=["服务暂时不可用，以下为通用建议"],
        )
