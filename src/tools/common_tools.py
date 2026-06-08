"""
Common Tools — shared utilities for user profile management and progress tracking.
Used primarily by the Consultation Agent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Implementations
# ═══════════════════════════════════════════════════════════════════════════════

def get_user_profile_summary(user_data: dict | None) -> dict:
    """
    Get a human-readable summary of the user's health profile.

    Args:
        user_data: User profile dict (from memory manager)
    """
    if not user_data:
        return {
            "status": "no_profile",
            "message": "尚未创建健康档案，请先提供基本信息（年龄、身高、体重、性别、活动水平）",
            "suggested_action": "请告诉我您的年龄、身高、体重、性别和日常活动水平，我来帮您建立健康档案",
        }

    bmi = round(user_data["weight_kg"] / ((user_data["height_cm"] / 100) ** 2), 1)
    if bmi < 18.5:
        bmi_cat = "偏瘦"
    elif bmi < 24.0:
        bmi_cat = "正常范围"
    elif bmi < 28.0:
        bmi_cat = "偏胖"
    else:
        bmi_cat = "肥胖"

    gender_labels = {"male": "男", "female": "女", "other": "其他"}
    activity_labels = {
        "sedentary": "久坐不动",
        "lightly_active": "轻度活动",
        "moderately_active": "中度活动",
        "very_active": "高度活跃",
        "extra_active": "极度活跃",
    }

    return {
        "status": "ok",
        "profile": {
            "name": user_data.get("name", "用户"),
            "age": user_data["age"],
            "gender": gender_labels.get(user_data.get("gender"), "未知"),
            "height_cm": user_data["height_cm"],
            "weight_kg": user_data["weight_kg"],
            "bmi": bmi,
            "bmi_category": bmi_cat,
            "activity_level": activity_labels.get(user_data.get("activity_level"), "未知"),
            "dietary_preferences": user_data.get("dietary_preferences", []),
            "allergies": user_data.get("allergies", []),
            "medical_conditions": user_data.get("medical_conditions", []),
            "avg_sleep_hours": user_data.get("sleep_hours_avg", 7.0),
        },
    }


def set_health_goals(goals: list[dict]) -> dict:
    """
    Set or update user's health goals.

    Args:
        goals: List of goal objects, each with goal_type, description, priority
    """
    valid_types = ["lose_weight", "gain_muscle", "maintain", "improve_sleep", "general_wellness"]

    validated_goals = []
    for g in goals:
        goal_type = g.get("goal_type", "general_wellness")
        if goal_type not in valid_types:
            goal_type = "general_wellness"
        validated_goals.append({
            "goal_type": goal_type,
            "target_description": g.get("description", ""),
            "target_value": g.get("target_value"),
            "deadline": g.get("deadline"),
            "priority": g.get("priority", 3),
            "created_at": datetime.now().isoformat(),
        })

    return {
        "status": "ok",
        "goals_set": len(validated_goals),
        "goals": validated_goals,
        "message": f"已设置 {len(validated_goals)} 个健康目标",
    }


def track_progress(metric: str, value: float, date: str | None = None) -> dict:
    """
    Record a progress metric for tracking over time.

    Args:
        metric: "weight", "sleep_hours", "exercise_minutes", "calories", "mood"
        value: The measured value
        date: ISO date string (defaults to today)
    """
    date = date or datetime.now().strftime("%Y-%m-%d")

    metric_labels = {
        "weight": "体重 (kg)",
        "sleep_hours": "睡眠时长 (小时)",
        "exercise_minutes": "运动时长 (分钟)",
        "calories": "摄入热量 (kcal)",
        "mood": "心情评分 (1-5)",
    }

    return {
        "metric": metric,
        "label": metric_labels.get(metric, metric),
        "value": value,
        "date": date,
        "recorded_at": datetime.now().isoformat(),
        "message": f"已记录 {metric_labels.get(metric, metric)}: {value} ({date})",
    }


def health_risk_assessment(user_data: dict | None) -> dict:
    """
    Perform a basic health risk assessment based on user profile.

    Args:
        user_data: User profile dict
    """
    if not user_data:
        return {"status": "no_data", "message": "需要用户健康档案才能进行风险评估"}

    risks = []
    bmi = round(user_data["weight_kg"] / ((user_data["height_cm"] / 100) ** 2), 1)

    if bmi >= 28:
        risks.append({"level": "high", "area": "体重管理", "detail": f"BMI {bmi} 属于肥胖范围，建议制定减重计划"})
    elif bmi >= 24:
        risks.append({"level": "medium", "area": "体重管理", "detail": f"BMI {bmi} 属于偏胖范围，建议关注体重"})
    elif bmi < 18.5:
        risks.append({"level": "medium", "area": "营养", "detail": f"BMI {bmi} 偏瘦，建议增加营养摄入"})

    if user_data.get("sleep_hours_avg", 7) < 6:
        risks.append({"level": "high", "area": "睡眠", "detail": "平均睡眠不足 6 小时，长期睡眠不足增加多种疾病风险"})
    elif user_data.get("sleep_hours_avg", 7) < 7:
        risks.append({"level": "medium", "area": "睡眠", "detail": "睡眠略不足推荐量，建议增加 30-60 分钟"})

    if user_data.get("activity_level") == "sedentary":
        risks.append({"level": "medium", "area": "运动", "detail": "久坐不动的生活方式增加心血管疾病风险"})

    if not risks:
        risks.append({"level": "low", "area": "综合", "detail": "未发现明显健康风险因素，保持当前良好习惯"})

    return {
        "status": "ok",
        "bmi": bmi,
        "risks": risks,
        "total_risks": len([r for r in risks if r["level"] != "low"]),
        "recommendation": "建议定期关注以上风险因素，如有不适及时就医",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Function Calling Schema Definitions
# ═══════════════════════════════════════════════════════════════════════════════

COMMON_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_user_profile_summary",
            "description": "获取用户健康档案的摘要信息，包括 BMI、基本信息等",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_data": {"type": "object", "description": "用户档案数据（从 memory manager 获取）"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_health_goals",
            "description": "设置或更新用户的健康目标",
            "parameters": {
                "type": "object",
                "properties": {
                    "goals": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "goal_type": {"type": "string", "description": "目标类型"},
                                "description": {"type": "string", "description": "目标描述"},
                                "priority": {"type": "integer", "description": "优先级 1(最高)-5(最低)"},
                            },
                        },
                        "description": "健康目标列表",
                    },
                },
                "required": ["goals"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "track_progress",
            "description": "记录健康指标用于长期追踪",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["weight", "sleep_hours", "exercise_minutes", "calories", "mood"],
                        "description": "健康指标类型",
                    },
                    "value": {"type": "number", "description": "指标数值"},
                    "date": {"type": "string", "description": "日期 (YYYY-MM-DD)"},
                },
                "required": ["metric", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "health_risk_assessment",
            "description": "基于用户档案进行基础健康风险评估",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_data": {"type": "object", "description": "用户档案数据"},
                },
                "required": [],
            },
        },
    },
]

COMMON_TOOLS_MAP = {
    "get_user_profile_summary": get_user_profile_summary,
    "set_health_goals": set_health_goals,
    "track_progress": track_progress,
    "health_risk_assessment": health_risk_assessment,
}
