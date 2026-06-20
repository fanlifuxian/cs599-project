"""
Response Parser — 从 Agent 回复中提取结构化信息。

职责单一：仅负责解析 LLM 回复文本，提取置信度、引用来源、注意事项。
不涉及 LLM 调用、工具执行或业务逻辑。
"""

from __future__ import annotations

import re
from typing import Any


def extract_confidence(content: str, tool_calls_count: int = 0) -> float:
    """
    从回复文本中提取置信度评分。

    策略：
    1. 显式标记：confidence: 0.85 / 置信度: 85%
    2. 启发式：使用了工具的回复默认 0.75，未使用工具默认 0.65
    """
    markers = [
        r"confidence\s*:\s*([\d.]+)\s*%?",
        r"置信度\s*[：:]\s*([\d.]+)\s*%?",
        r"confidence score\s*:\s*([\d.]+)",
    ]

    for marker in markers:
        match = re.search(marker, content, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                if val > 1:
                    val /= 100
                return max(0.0, min(1.0, val))
            except ValueError:
                pass

    # 启发式回退
    return 0.75 if tool_calls_count > 0 else 0.65


def extract_sources(content: str) -> list[str]:
    """从回复中提取引用的信息来源。"""
    sources = []
    for line in content.split("\n"):
        stripped = line.strip()
        if any(kw in stripped.lower() for kw in ["来源", "参考", "source", "reference", "根据"]):
            # 只保留有实质内容的行
            if len(stripped) > 8:
                sources.append(stripped)
    return sources


def extract_caveats(content: str) -> list[str]:
    """从回复中提取注意事项和免责声明。"""
    caveats = []
    for line in content.split("\n"):
        stripped = line.strip()
        if any(kw in stripped for kw in ["注意", "警告", "免责", "disclaimer", "⚠️", "风险"]):
            if len(stripped) > 8:
                caveats.append(stripped)
    return caveats


def extract_reasoning_trace(tool_calls_made: list[str]) -> str:
    """根据工具调用列表生成推理追踪。"""
    if not tool_calls_made:
        return ""
    return f"Tools used: {', '.join(tool_calls_made)}"
