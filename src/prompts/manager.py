"""
Prompt Manager — 集中管理所有 Agent 提示词。

职责单一：加载、缓存、格式化 YAML 提示词模板。
修改提示词只需编辑 YAML 文件，无需触及 agent 代码。

Usage:
    from src.prompts.manager import PromptManager
    pm = PromptManager()
    prompt = pm.build("diet", user_profile={...}, goals=[...])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent


class PromptManager:
    """
    提示词管理器 — 加载 YAML 提示词模板并根据上下文格式化。

    设计原则：
    - 提示词与代码完全解耦
    - 支持模板变量替换
    - 内置缓存，避免重复 I/O
    """

    _cache: dict[str, dict] = {}

    # ── Agent prompt builders (public API) ───────────────────────────────

    def build_diet_prompt(self, context: dict[str, Any]) -> str:
        return self._build("diet_agent", context)

    def build_exercise_prompt(self, context: dict[str, Any]) -> str:
        return self._build("exercise_agent", context)

    def build_sleep_prompt(self, context: dict[str, Any]) -> str:
        return self._build("sleep_agent", context)

    def build_consultation_prompt(self, context: dict[str, Any]) -> str:
        return self._build("consultation_agent", context)

    # ── Core builder ─────────────────────────────────────────────────────

    def _build(self, name: str, context: dict[str, Any]) -> str:
        """Load YAML template and format it with context."""
        template = self._load(name)
        if not template:
            return f"Error: prompt '{name}' not found"

        parts: list[str] = []

        # ── Identity section ──────────────────────────────────────────
        parts.append(template.get("identity", ""))

        # ── Expertise ──────────────────────────────────────────────────
        expertise = template.get("expertise", [])
        if expertise:
            parts.append("## 你的专业领域")
            for i, exp in enumerate(expertise, 1):
                parts.append(f"{i}. **{exp['name']}**：{exp['description']}")

        # ── Team intro (consultation only) ───────────────────────────────
        if team_intro := template.get("team_intro"):
            parts.append(team_intro)

        # ── Responsibilities ─────────────────────────────────────────────
        responsibilities = template.get("responsibilities", [])
        if responsibilities:
            parts.append("## 你的核心职责")
            for i, resp in enumerate(responsibilities, 1):
                parts.append(f"{i}. {resp}")

        # ── Workflow ─────────────────────────────────────────────────────
        workflow = template.get("workflow", [])
        if workflow:
            parts.append("## 工作流程")
            for wf in workflow:
                tool_ref = f" — 使用 `{wf['tool']}`" if "tool" in wf else ""
                parts.append(f"{wf['step']}. {wf['action']}{tool_ref}：{wf['description']}")

        # ── Tools ────────────────────────────────────────────────────────
        tools = template.get("tools", [])
        if tools:
            parts.append("## 你的工具")
            for tool in tools:
                parts.append(f"- `{tool['name']}` — {tool['description']}")

        # ── Knowledge ────────────────────────────────────────────────────
        knowledge = template.get("knowledge", [])
        if knowledge:
            parts.append("## 核心知识")
            for k in knowledge:
                parts.append(f"- {k}")

        # ── User context (formatted from context vars) ───────────────────
        user_profile = context.get("user_profile")
        if user_profile:
            profile_tpl = template.get("profile_template", "")
            parts.append(self._format_profile(profile_tpl, user_profile))
        else:
            parts.append(template.get("profile_empty", ""))

        # ── Goals ────────────────────────────────────────────────────────
        goals = context.get("health_goals", [])
        if goals:
            goals_tpl = template.get("goals_template", "## 用户健康目标\n{goals_list}")
            goals_list = "\n".join(
                f"- [{g.get('goal_type', '')}] {g.get('target_description', '')} (优先级: {g.get('priority', 3)})"
                for g in goals
            )
            parts.append(goals_tpl.replace("{goals_list}", goals_list))

        # ── Progress ─────────────────────────────────────────────────────
        progress = context.get("progress", {})
        if progress:
            progress_tpl = template.get("progress_template", "")
            parts.append(
                progress_tpl.format(
                    adherence=progress.get("adherence_score", 0),
                    weight_trend=progress.get("weight_trend", "N/A"),
                )
                if progress_tpl else ""
            )

        # ── Routing rules (consultation only) ────────────────────────────
        routing_rules = template.get("routing_rules", {})
        if routing_rules:
            parts.append("## 路由决策规则")
            for scenario, desc in routing_rules.items():
                parts.append(f"- **{scenario.replace('_', ' ')}** → {desc}")

        # ── Safety protocol ──────────────────────────────────────────────
        safety = template.get("safety_protocol")
        if safety:
            parts.append(safety)

        # ── Response rules ───────────────────────────────────────────────
        response_rules = template.get("response_rules", [])
        if response_rules:
            parts.append("## 回复规范")
            for i, rule in enumerate(response_rules, 1):
                parts.append(f"{i}. {rule}")

        return "\n\n".join(parts)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _load(self, name: str) -> dict:
        """Load a YAML prompt template (with caching)."""
        if name in self._cache:
            return self._cache[name]

        filepath = _PROMPTS_DIR / f"{name}.yaml"
        if not filepath.exists():
            logger.error(f"Prompt file not found: {filepath}")
            return {}

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._cache[name] = data or {}
            logger.debug(f"Prompt loaded: {name}")
            return self._cache[name]
        except Exception as e:
            logger.error(f"Failed to load prompt {name}: {e}")
            return {}

    @staticmethod
    def _format_profile(template: str, profile: dict) -> str:
        """Format user profile into the template."""
        bmi = round(profile["weight_kg"] / ((profile["height_cm"] / 100) ** 2), 1)
        if bmi < 18.5:
            bmi_cat = "偏瘦"
        elif bmi < 24:
            bmi_cat = "正常"
        elif bmi < 28:
            bmi_cat = "偏胖"
        else:
            bmi_cat = "肥胖"

        activity_labels = {
            "sedentary": "久坐不动", "lightly_active": "轻度活动",
            "moderately_active": "中度活动", "very_active": "高度活跃",
            "extra_active": "极度活跃",
        }

        return template.format(
            age=profile.get("age", "?"),
            gender=profile.get("gender", "未知"),
            height_cm=profile.get("height_cm", "?"),
            weight_kg=profile.get("weight_kg", "?"),
            bmi=bmi,
            bmi_category=bmi_cat,
            activity_level=activity_labels.get(profile.get("activity_level", "sedentary"), "未知"),
            dietary_preferences=", ".join(profile.get("dietary_preferences", [])) or "无特殊",
            allergies=", ".join(profile.get("allergies", [])) or "无",
            medical_conditions=", ".join(profile.get("medical_conditions", [])) or "无特殊",
            sleep_hours_avg=profile.get("sleep_hours_avg", "?"),
            exercise_preferences=", ".join(profile.get("exercise_preferences", [])) or "未指定",
            available_equipment=", ".join(profile.get("available_equipment", [])) or "未指定（默认徒手训练）",
        )

    def clear_cache(self):
        """Clear the prompt cache (useful for hot-reloading during development)."""
        self._cache.clear()
        logger.info("Prompt cache cleared")


# Module-level singleton
_prompt_manager: PromptManager | None = None


def get_prompt_manager() -> PromptManager:
    """Get or create the prompt manager singleton."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
