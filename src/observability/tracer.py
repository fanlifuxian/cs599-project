"""
Observability Tracer — Langfuse + console tracing for agent execution.

Provides:
- Trace context management across agent invocations
- Span creation for each agent call, tool call, and synthesis
- Token usage and cost estimation
- Latency tracking
- Graceful fallback to console logging when Langfuse is unavailable
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional

from src.config.settings import settings
from src.models import Trace, TraceSpan

logger = logging.getLogger(__name__)


class HealthTracer:
    """
    Tracing manager for health multi-agent platform.

    Supports:
    - Langfuse (cloud) when LANGFUSE_ENABLED=true
    - Console logging (always available)
    - OpenTelemetry (future)
    """

    def __init__(self):
        self._langfuse_client = None
        self._active_traces: dict[str, Trace] = {}
        self._active_spans: dict[str, list[TraceSpan]] = {}
        self._enabled = settings.langfuse_enabled

        if self._enabled:
            self._init_langfuse()

    def _init_langfuse(self):
        """Initialize Langfuse client if configured."""
        try:
            if settings.langfuse_public_key and settings.langfuse_secret_key:
                import langfuse
                self._langfuse_client = langfuse.Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
                logger.info("Langfuse tracing enabled")
            else:
                logger.info("Langfuse keys not configured — using console tracing")
                self._enabled = False
        except ImportError:
            logger.warning("langfuse package not installed — using console tracing")
            self._enabled = False
        except Exception as e:
            logger.warning(f"Langfuse init failed: {e} — using console tracing")
            self._enabled = False

    # ── Trace Management ──────────────────────────────────────────────────

    def start_trace(self, user_input: str, session_id: str = "") -> str:
        """Start a new trace for a user request."""
        trace_id = str(uuid.uuid4())[:12]
        trace = Trace(
            trace_id=trace_id,
            session_id=session_id,
            user_input=user_input[:200],
            created_at=datetime.now().isoformat(),
        )
        self._active_traces[trace_id] = trace
        self._active_spans[trace_id] = []

        logger.info(f"[Trace:{trace_id}] Started — input: {user_input[:100]}...")
        return trace_id

    def start_span(
        self,
        trace_id: str,
        name: str,
        agent_role: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ) -> str:
        """Start a new span within a trace."""
        span = TraceSpan(
            span_id=str(uuid.uuid4())[:8],
            parent_span_id=parent_span_id,
            name=name,
            agent_role=agent_role,
            status="running",
            metadata={},
        )

        if trace_id in self._active_spans:
            self._active_spans[trace_id].append(span)

        return span.span_id

    def end_span(
        self,
        trace_id: str,
        span_id: str,
        status: str = "success",
        tokens: dict[str, int] | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Complete a span."""
        spans = self._active_spans.get(trace_id, [])
        for span in spans:
            if span.span_id == span_id:
                span.status = status
                span.end_time = datetime.now().isoformat()
                span.error_message = error
                if metadata:
                    span.metadata.update(metadata)
                if tokens:
                    span.input_tokens = tokens.get("prompt_tokens", 0)
                    span.output_tokens = tokens.get("completion_tokens", 0)

                # Calculate duration
                if span.start_time:
                    try:
                        start_dt = datetime.fromisoformat(span.start_time)
                        end_dt = datetime.fromisoformat(span.end_time)
                        span.duration_ms = (end_dt - start_dt).total_seconds() * 1000
                    except (ValueError, TypeError):
                        pass
                break

    def end_trace(
        self,
        trace_id: str,
        final_response: str | None = None,
        agent_count: int = 0,
    ) -> Optional[Trace]:
        """Complete a trace and optionally send to Langfuse."""
        trace = self._active_traces.pop(trace_id, None)
        if not trace:
            return None

        trace.spans = self._active_spans.pop(trace_id, [])
        trace.final_response = (final_response or "")[:200]
        trace.agent_count = agent_count

        # Aggregate metrics
        total_tokens = sum(
            s.input_tokens + s.output_tokens for s in trace.spans
        )
        trace.total_tokens = total_tokens
        trace.total_duration_ms = sum(
            s.duration_ms or 0 for s in trace.spans
        )
        trace.total_cost_est = self._estimate_cost(total_tokens)

        # Send to Langfuse if enabled
        if self._enabled and self._langfuse_client:
            self._send_to_langfuse(trace)

        logger.info(
            f"[Trace:{trace_id}] Completed — spans: {len(trace.spans)}, "
            f"tokens: {total_tokens}, duration: {trace.total_duration_ms:.0f}ms, "
            f"cost: ${trace.total_cost_est:.4f}"
        )

        return trace

    # ── Cost Estimation ───────────────────────────────────────────────────

    def _estimate_cost(self, total_tokens: int) -> float:
        """Estimate API cost based on model pricing (per 1M tokens)."""
        model = settings.llm_model.lower()
        provider = settings.llm_provider

        # Approximate pricing per 1M tokens (input+output blended)
        pricing = {
            "deepseek": {"deepseek-chat": 0.20, "deepseek-reasoner": 0.55},
            "openai": {"gpt-4o": 5.00, "gpt-4o-mini": 0.30, "gpt-3.5-turbo": 0.50},
            "anthropic": {"claude-sonnet-4-6": 3.00, "claude-haiku-4-5": 1.00},
            "ollama": {"default": 0.0},
        }

        provider_pricing = pricing.get(provider, {})
        per_million = provider_pricing.get(model, provider_pricing.get("default", 1.0))

        return (total_tokens / 1_000_000) * per_million

    def _send_to_langfuse(self, trace: Trace):
        """Send trace data to Langfuse."""
        try:
            langfuse_trace = self._langfuse_client.trace(
                id=trace.trace_id,
                name="health-agent-request",
                metadata={
                    "agent_count": trace.agent_count,
                    "total_tokens": trace.total_tokens,
                    "cost": trace.total_cost_est,
                },
            )

            for span in trace.spans:
                langfuse_trace.span(
                    id=span.span_id,
                    name=span.name,
                    metadata={
                        "agent_role": span.agent_role,
                        "tool_calls": span.tool_calls,
                        "duration_ms": span.duration_ms,
                        "status": span.status,
                    },
                )

            self._langfuse_client.flush()
            logger.debug(f"[Trace:{trace.trace_id}] Sent to Langfuse")
        except Exception as e:
            logger.warning(f"Langfuse send failed: {e}")

    # ── Context Manager ───────────────────────────────────────────────────

    @contextmanager
    def trace_context(self, user_input: str, session_id: str = ""):
        """Context manager for automatic trace lifecycle."""
        trace_id = self.start_trace(user_input, session_id)
        try:
            yield trace_id
        finally:
            self.end_trace(trace_id)

    def get_stats(self) -> dict:
        """Get tracer statistics."""
        return {
            "enabled": self._enabled,
            "langfuse_connected": self._langfuse_client is not None,
            "active_traces": len(self._active_traces),
        }


# Module-level singleton
_tracer: Optional[HealthTracer] = None


def get_tracer() -> HealthTracer:
    """Get or create the tracer singleton."""
    global _tracer
    if _tracer is None:
        _tracer = HealthTracer()
    return _tracer
