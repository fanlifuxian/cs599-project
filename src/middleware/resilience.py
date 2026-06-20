"""
Resilience Middleware — 熔断器 + 指数退避重试 + 优雅降级。

职责单一：仅负责 API 调用的容错逻辑，与 Agent 业务逻辑完全解耦。

Patterns:
- CircuitBreaker: 连续失败 N 次后自动熔断，30s 后半开尝试
- RetryWithBackoff: 指数退避 + 随机抖动，避免惊群效应
- GracefulFallback: 各 Agent 角色的降级回复模板
"""

from __future__ import annotations

import logging
import random
import time
from functools import wraps
from typing import Callable, TypeVar

from src.config.settings import settings

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


# ═══════════════════════════════════════════════════════════════════════════════
# Circuit Breaker
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """
    熔断器 — 防止级联故障。

    状态机：CLOSED → (failures >= threshold) → OPEN → (timeout elapsed) → HALF_OPEN
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        name: str = "default",
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.name = name
        self._failure_count = 0
        self._open_until = 0.0

    @property
    def is_open(self) -> bool:
        if self._open_until > time.time():
            return True
        if self._failure_count >= self.failure_threshold:
            self._open_until = time.time() + self.reset_timeout
            logger.warning(f"[CircuitBreaker:{self.name}] OPEN — {self._failure_count} failures")
            return True
        return False

    def record_success(self):
        self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self):
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            logger.error(f"[CircuitBreaker:{self.name}] Threshold reached — opening circuit")

    def reset(self):
        self._failure_count = 0
        self._open_until = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Retry with Exponential Backoff + Jitter
# ═══════════════════════════════════════════════════════════════════════════════

def retry_with_backoff(
    max_retries: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
):
    """
    装饰器：为函数添加指数退避重试。

    每次重试前等待：min(base_delay ** attempt + random(0, 1), max_delay)
    """
    max_retries = max_retries if max_retries is not None else settings.max_retries
    base_delay = base_delay if base_delay is not None else settings.retry_backoff
    max_delay = max_delay if max_delay is not None else settings.retry_max_delay

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        delay = min(
                            base_delay ** attempt + random.uniform(0, 1),
                            max_delay,
                        )
                        logger.warning(
                            f"[Retry] {func.__name__} attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
            raise last_error  # type: ignore
        return wrapper  # type: ignore
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# Graceful Fallback Messages
# ═══════════════════════════════════════════════════════════════════════════════

FALLBACK_MESSAGES: dict[str, str] = {
    "diet": (
        "抱歉，饮食健康助手暂时无法完成详细分析。\n\n"
        "以下是一些通用建议：\n"
        "• 保持均衡饮食，多吃蔬菜水果\n"
        "• 控制盐和糖的摄入\n"
        "• 每天饮水 1.5-2 升\n\n"
        "请稍后重试或换个方式描述您的问题。"
    ),
    "exercise": (
        "抱歉，运动健身助手暂时无法完成详细分析。\n\n"
        "以下是一些通用建议：\n"
        "• 每周至少 150 分钟中等强度运动\n"
        "• 结合有氧和力量训练\n"
        "• 运动前充分热身\n\n"
        "请稍后重试或换个方式描述您的问题。"
    ),
    "sleep": (
        "抱歉，睡眠健康助手暂时无法完成详细分析。\n\n"
        "以下是一些通用建议：\n"
        "• 保持规律作息，固定起床时间\n"
        "• 睡前 1 小时远离屏幕\n"
        "• 卧室保持凉爽黑暗\n\n"
        "请稍后重试或换个方式描述您的问题。"
    ),
    "consultation": (
        "抱歉，健康咨询助手暂时无法完成分析。\n\n"
        "以下是一些通用建议：\n"
        "• 定期体检，关注健康指标\n"
        "• 保持积极心态\n"
        "• 如有不适及时就医\n\n"
        "请稍后重试。"
    ),
}


def get_fallback_message(role: str, error: str = "") -> str:
    """获取指定角色的优雅降级回复。"""
    base = FALLBACK_MESSAGES.get(role, FALLBACK_MESSAGES["consultation"])
    if error:
        base += f"\n\n（技术细节：{error[:100]}）"
    return base
