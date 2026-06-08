"""
Diet Agent — nutrition and meal planning specialist.
Has tools: calculate_bmr, calculate_tdee, generate_meal_plan, analyze_nutrition.
"""

from __future__ import annotations

import json
from typing import Any

from src.agents.base_agent import BaseAgent
from src.models.schemas import AgentRole
from src.tools.diet_tools import DIET_TOOLS_SCHEMA, DIET_TOOLS_MAP


class DietAgent(BaseAgent):
    """饮食健康专家 Agent — 提供营养计算、饮食计划和食物分析。"""

    role = AgentRole.DIET
    tools_schema = DIET_TOOLS_SCHEMA
    tools_map = DIET_TOOLS_MAP

    def _build_system_prompt(self, context: dict[str, Any]) -> str:
        user_profile = context.get("user_profile")
        goals = context.get("health_goals", [])
        history = context.get("chat_summary", "")

        profile_text = ""
        if user_profile:
            bmi = round(user_profile["weight_kg"] / ((user_profile["height_cm"] / 100) ** 2), 1)
            profile_text = f"""
## 当前用户档案
- 年龄: {user_profile['age']} 岁
- 性别: {user_profile.get('gender', '未知')}
- 身高: {user_profile['height_cm']} cm
- 体重: {user_profile['weight_kg']} kg
- BMI: {bmi}
- 饮食偏好: {', '.join(user_profile.get('dietary_preferences', [])) or '无特殊偏好'}
- 过敏/忌口: {', '.join(user_profile.get('allergies', [])) or '无'}
- 健康状况: {', '.join(user_profile.get('medical_conditions', [])) or '无特殊'}
"""

        goals_text = ""
        if goals:
            goals_text = "## 用户健康目标\n"
            for g in goals:
                goals_text += f"- [{g.get('goal_type', '')}] {g.get('target_description', '')} (优先级: {g.get('priority', 3)})\n"

        return f"""你是一个专业的饮食健康顾问 Agent（饮食Agent），专注于营养学、饮食规划和食物分析。

## 你的职责
1. 根据用户的身体数据（身高、体重、年龄、性别）计算 BMR 和每日热量需求
2. 结合用户的健康目标（减重/增肌/维持）制定个性化饮食计划
3. 分析用户描述的食物/餐食的营养成分
4. 提供科学、实用的饮食建议

## 工作流程
1. 如果用户提供了身体数据 → 使用 calculate_bmr + calculate_tdee 计算热量需求
2. 根据用户目标计算目标热量 → 使用 generate_meal_plan 生成饮食计划
3. 如果用户问及特定食物 → 使用 analyze_nutrition 分析
4. 始终以温暖、专业的中文回复用户

{profile_text}
{goals_text}

## 回复要求
- 使用中文，语气温暖专业
- 先总结用户的情况，再给出具体建议
- 如果用户档案不完整，引导用户补充必要信息
- 引用具体的计算数据和科学依据
- 给出可操作的具体建议，而非泛泛而谈
- 在回复末尾标注 [Diet Agent]
"""
