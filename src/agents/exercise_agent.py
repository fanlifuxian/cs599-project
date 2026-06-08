"""
Exercise Agent — workout and fitness specialist.
Has tools: generate_workout_plan, recommend_exercise, calculate_calorie_burn.
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.models.schemas import AgentRole
from src.tools.exercise_tools import EXERCISE_TOOLS_SCHEMA, EXERCISE_TOOLS_MAP


class ExerciseAgent(BaseAgent):
    """运动健身专家 Agent — 提供运动计划、运动推荐和热量消耗估算。"""

    role = AgentRole.EXERCISE
    tools_schema = EXERCISE_TOOLS_SCHEMA
    tools_map = EXERCISE_TOOLS_MAP

    def _build_system_prompt(self, context: dict[str, Any]) -> str:
        user_profile = context.get("user_profile")
        goals = context.get("health_goals", [])

        profile_text = ""
        if user_profile:
            bmi = round(user_profile["weight_kg"] / ((user_profile["height_cm"] / 100) ** 2), 1)
            activity_labels = {
                "sedentary": "久坐不动",
                "lightly_active": "轻度活动",
                "moderately_active": "中度活动",
                "very_active": "高度活跃",
                "extra_active": "极度活跃",
            }
            profile_text = f"""
## 当前用户档案
- 年龄: {user_profile['age']} 岁
- 性别: {user_profile.get('gender', '未知')}
- 身高: {user_profile['height_cm']} cm
- 体重: {user_profile['weight_kg']} kg
- BMI: {bmi}
- 当前活动水平: {activity_labels.get(user_profile.get('activity_level', 'sedentary'), '未知')}
- 健康状况: {', '.join(user_profile.get('medical_conditions', [])) or '无特殊'}
"""

        goals_text = ""
        if goals:
            goals_text = "## 用户健康目标\n"
            for g in goals:
                goals_text += f"- [{g.get('goal_type', '')}] {g.get('target_description', '')} (优先级: {g.get('priority', 3)})\n"

        return f"""你是一个专业的运动健身顾问 Agent（运动Agent），专注于运动科学、训练计划和体能提升。

## 你的职责
1. 根据用户的身体数据和目标制定个性化运动计划
2. 考虑用户的体能水平和可用器材，推荐合适的运动
3. 估算运动消耗的热量
4. 提供运动安全指导和渐进式训练建议

## 工作流程
1. 了解用户的健身目标（减重/增肌/保持/改善睡眠/减压）
2. 评估用户的体能水平（初级/中级/高级）
3. 使用 generate_workout_plan 生成运动计划
4. 使用 recommend_exercise 推荐具体运动
5. 使用 calculate_calorie_burn 估算热量消耗

{profile_text}
{goals_text}

## 回复要求
- 使用中文，语气激励人心
- 根据用户体能水平推荐合适的运动强度
- 强调运动安全和正确姿势
- 考虑用户的伤病和身体限制
- 给出渐进式的训练建议
- 提醒热身和拉伸的重要性
- 在回复末尾标注 [Exercise Agent]
"""
