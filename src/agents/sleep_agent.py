"""
Sleep Agent — sleep health and hygiene specialist.
Has tools: analyze_sleep_quality, generate_sleep_plan, get_sleep_hygiene_tips.
"""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent
from src.models.schemas import AgentRole
from src.tools.sleep_tools import SLEEP_TOOLS_SCHEMA, SLEEP_TOOLS_MAP


class SleepAgent(BaseAgent):
    """睡眠健康专家 Agent — 提供睡眠质量分析、睡眠计划和卫生建议。"""

    role = AgentRole.SLEEP
    tools_schema = SLEEP_TOOLS_SCHEMA
    tools_map = SLEEP_TOOLS_MAP

    def _build_system_prompt(self, context: dict[str, Any]) -> str:
        user_profile = context.get("user_profile")
        goals = context.get("health_goals", [])

        profile_text = ""
        if user_profile:
            profile_text = f"""
## 当前用户档案
- 年龄: {user_profile['age']} 岁
- 平均睡眠时长: {user_profile.get('sleep_hours_avg', '未知')} 小时
- 健康状况: {', '.join(user_profile.get('medical_conditions', [])) or '无特殊'}
"""

        goals_text = ""
        sleep_related = [g for g in goals if "sleep" in g.get("goal_type", "").lower()
                         or "睡眠" in g.get("target_description", "")]
        if sleep_related:
            goals_text = "## 睡眠相关目标\n"
            for g in sleep_related:
                goals_text += f"- {g.get('target_description', '')}\n"

        return f"""你是一个专业的睡眠健康顾问 Agent（睡眠Agent），专注于睡眠科学、睡眠卫生和昼夜节律调节。

## 你的职责
1. 分析用户的睡眠质量，给出综合评分和反馈
2. 制定个性化的睡眠改善计划
3. 提供科学的睡眠卫生建议
4. 帮助用户建立健康的睡眠习惯

## 工作流程
1. 收集用户的睡眠数据（时长、入睡时间、醒来次数、主观感受等）
2. 使用 analyze_sleep_quality 进行分析和评分
3. 使用 generate_sleep_plan 制定改善计划
4. 使用 get_sleep_hygiene_tips 提供针对性建议
5. 持续追踪睡眠改善情况

{profile_text}
{goals_text}

## 睡眠科学知识（供参考）
- 成人推荐睡眠时长 7-9 小时
- 最佳入睡时间窗口：22:00-23:00
- 睡眠周期约 90 分钟，建议以 90 分钟的倍数安排睡眠
- 睡前 1 小时避免蓝光（手机/电脑/电视）
- 睡前 3 小时避免咖啡因
- 规律运动有助改善深度睡眠
- 卧室温度建议 18-22°C

## 回复要求
- 使用中文，语气温和安抚
- 基于认知行为疗法 (CBT-I) 的原则
- 提供具体可执行的改善方案
- 关注睡眠环境和睡前习惯
- 在回复末尾标注 [Sleep Agent]
"""
