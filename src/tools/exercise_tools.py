"""
Exercise Agent Tools — workout planning, exercise recommendations, calorie burn estimation.
All tools expose OpenAI-compatible function calling schemas.
"""

from __future__ import annotations

from typing import Optional
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Implementations
# ═══════════════════════════════════════════════════════════════════════════════

def generate_workout_plan(
    goal: str,
    fitness_level: str = "beginner",
    duration_minutes: int = 45,
    equipment: list[str] | None = None,
    injuries: list[str] | None = None,
) -> dict:
    """
    Generate a personalized workout plan based on user goals and fitness level.

    Args:
        goal: "lose_weight", "gain_muscle", "maintain", "general_wellness"
        fitness_level: "beginner", "intermediate", "advanced"
        duration_minutes: Target workout duration (15-90)
        equipment: Available equipment e.g. ["dumbbells", "treadmill"]
        injuries: Any injuries or physical limitations
    """
    equipment = equipment or []
    injuries = injuries or []

    # Workout template by goal + level
    templates = {
        "lose_weight": {
            "beginner": {
                "warm_up": "原地踏步 3 分钟 + 关节活动 2 分钟",
                "exercises": [
                    {"name": "快走/慢跑交替", "type": "cardio", "duration_minutes": 15, "sets": None, "reps": None,
                     "calories_burned_est": 120, "notes": "快走 2min + 慢跑 1min 交替"},
                    {"name": "深蹲", "type": "strength", "duration_minutes": 5, "sets": 3, "reps": "12-15",
                     "calories_burned_est": 40, "notes": "徒手深蹲，注意膝盖不超过脚尖"},
                    {"name": "俯卧撑（跪姿）", "type": "strength", "duration_minutes": 5, "sets": 3, "reps": "8-12",
                     "calories_burned_est": 35, "notes": "核心收紧，身体成一条线"},
                    {"name": "开合跳", "type": "cardio", "duration_minutes": 5, "sets": 3, "reps": "30秒/组",
                     "calories_burned_est": 50, "notes": "组间休息 30 秒"},
                    {"name": "平板支撑", "type": "strength", "duration_minutes": 3, "sets": 3, "reps": "20-30秒",
                     "calories_burned_est": 20, "notes": "保持核心收紧"},
                ],
                "cool_down": "全身拉伸 5 分钟",
                "total_calories_burned": 265,
                "tips": ["运动心率控制在最大心率的 60-70%", "每周至少 3 次", "运动前后注意补水"],
            },
            "intermediate": {
                "warm_up": "跳绳 3 分钟 + 动态拉伸 3 分钟",
                "exercises": [
                    {"name": "跑步机/HIIT 跑步", "type": "cardio", "duration_minutes": 20, "sets": None, "reps": None,
                     "calories_burned_est": 200, "notes": "心率维持在最大心率的 70-80%"},
                    {"name": "负重深蹲", "type": "strength", "duration_minutes": 6, "sets": 4, "reps": "10-12",
                     "calories_burned_est": 60, "notes": "可用哑铃负重，核心收紧"},
                    {"name": "波比跳", "type": "hiit", "duration_minutes": 5, "sets": 4, "reps": "10",
                     "calories_burned_est": 70, "notes": "全力完成，组间休息 40 秒"},
                    {"name": "哑铃划船", "type": "strength", "duration_minutes": 5, "sets": 3, "reps": "12/侧",
                     "calories_burned_est": 40, "notes": "保持腰背挺直"},
                ],
                "cool_down": "泡沫轴放松 + 静态拉伸 7 分钟",
                "total_calories_burned": 370,
                "tips": ["控制组间休息 30-60 秒", "每周 4-5 次", "搭配蛋白质摄入促进恢复"],
            },
        },
        "gain_muscle": {
            "beginner": {
                "warm_up": "动态拉伸 5 分钟 + 轻重量热身 3 分钟",
                "exercises": [
                    {"name": "哑铃卧推", "type": "strength", "duration_minutes": 8, "sets": 3, "reps": "10-12",
                     "calories_burned_est": 45, "notes": "选择能做 10-12 次的重量"},
                    {"name": "坐姿划船", "type": "strength", "duration_minutes": 8, "sets": 3, "reps": "10-12",
                     "calories_burned_est": 40, "notes": "感受背部发力"},
                    {"name": "腿部推举", "type": "strength", "duration_minutes": 7, "sets": 3, "reps": "12-15",
                     "calories_burned_est": 55, "notes": "全幅度动作"},
                    {"name": "哑铃侧平举", "type": "strength", "duration_minutes": 5, "sets": 3, "reps": "12-15",
                     "calories_burned_est": 25, "notes": "轻重量，控制发力"},
                    {"name": "卷腹", "type": "strength", "duration_minutes": 5, "sets": 3, "reps": "15-20",
                     "calories_burned_est": 25, "notes": "慢起慢落，感受腹肌收缩"},
                ],
                "cool_down": "全身拉伸 5 分钟",
                "total_calories_burned": 190,
                "tips": ["增肌需要热量盈余（+300kcal/天）", "每天摄入 1.6-2.0g 蛋白质/每公斤体重", "保证至少 7 小时睡眠帮助肌肉恢复"],
            },
        },
        "general_wellness": {
            "beginner": {
                "warm_up": "关节活动 3 分钟 + 原地踏步 2 分钟",
                "exercises": [
                    {"name": "快走", "type": "cardio", "duration_minutes": 20, "sets": None, "reps": None,
                     "calories_burned_est": 120, "notes": "保持轻快步伐，能说话但不能唱歌的强度"},
                    {"name": "太极拳/八段锦", "type": "flexibility", "duration_minutes": 10, "sets": 1, "reps": None,
                     "calories_burned_est": 50, "notes": "动作缓慢，配合呼吸"},
                    {"name": "靠墙静蹲", "type": "strength", "duration_minutes": 3, "sets": 3, "reps": "30秒",
                     "calories_burned_est": 20, "notes": "膝角约 90 度"},
                    {"name": "猫牛式 + 鸟狗式", "type": "flexibility", "duration_minutes": 5, "sets": 2, "reps": "10",
                     "calories_burned_est": 15, "notes": "脊柱灵活性训练"},
                ],
                "cool_down": "深呼吸放松 3 分钟",
                "total_calories_burned": 205,
                "tips": ["每周保持 3-5 次运动", "循序渐进，不要急于求成", "找到你喜欢的运动方式才能坚持"],
            },
        },
    }

    # Fallback chain: goal+level → goal+beginner → general_wellness+beginner
    goal_templates = templates.get(goal, templates["general_wellness"])
    plan = goal_templates.get(fitness_level) or list(goal_templates.values())[0]

    # Calculate total duration: warm_up(~5min) + exercises + cool_down(~5min)
    warm_up_minutes = 5
    cool_down_minutes = 5
    try:
        if "warm_up" in plan:
            warm_str = plan["warm_up"]
            for part in warm_str.replace("分钟", "").split():
                try:
                    warm_up_minutes = int(part)
                except ValueError:
                    pass
    except Exception:
        warm_up_minutes = 5
    try:
        if "cool_down" in plan:
            cool_str = plan["cool_down"]
            for part in cool_str.replace("分钟", "").split():
                try:
                    cool_down_minutes = int(part)
                except ValueError:
                    pass
    except Exception:
        cool_down_minutes = 5
    total_minutes = warm_up_minutes + sum(e.get("duration_minutes", 5) for e in plan["exercises"]) + cool_down_minutes

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "goal": goal,
        "fitness_level": fitness_level,
        "warm_up": plan["warm_up"],
        "exercises": plan["exercises"],
        "cool_down": plan["cool_down"],
        "total_duration_minutes": total_minutes,
        "total_calories_burned": plan["total_calories_burned"],
        "equipment_used": equipment,
        "injuries_considered": injuries,
        "tips": plan.get("tips", []),
    }


def recommend_exercise(
    goal: str,
    available_time_minutes: int = 30,
    location: str = "home",
    equipment: list[str] | None = None,
) -> dict:
    """
    Recommend a specific exercise or short workout based on constraints.

    Args:
        goal: "lose_weight", "gain_muscle", "maintain", "stress_relief", "improve_sleep"
        location: "home", "gym", "outdoor", "office"
        available_time_minutes: How much time the user has
        equipment: Available equipment
    """
    equipment = equipment or []

    recommendations = {
        "home": {
            "lose_weight": {
                "name": "家庭 HIIT 燃脂训练",
                "description": "无需器械，高效燃脂",
                "exercises": [
                    {"name": "开合跳", "duration": "40秒", "rest": "20秒"},
                    {"name": "高抬腿", "duration": "40秒", "rest": "20秒"},
                    {"name": "深蹲跳", "duration": "40秒", "rest": "20秒"},
                    {"name": "登山者", "duration": "40秒", "rest": "20秒"},
                    {"name": "波比跳（简化版）", "duration": "30秒", "rest": "30秒"},
                ],
                "rounds": max(1, available_time_minutes // 6),
                "calories_est": max(1, available_time_minutes // 6) * 45,
            },
            "improve_sleep": {
                "name": "睡前放松瑜伽",
                "description": "帮助身心放松，促进睡眠",
                "exercises": [
                    {"name": "婴儿式", "duration": "2分钟", "rest": ""},
                    {"name": "猫牛式", "duration": "2分钟", "rest": ""},
                    {"name": "仰卧脊柱扭转", "duration": "2分钟/侧", "rest": ""},
                    {"name": "腿靠墙上举", "duration": "3分钟", "rest": ""},
                    {"name": "深呼吸放松", "duration": "3分钟", "rest": ""},
                ],
                "rounds": 1,
                "calories_est": 50,
            },
        },
        "gym": {
            "lose_weight": {"name": "健身房循环训练", "description": "结合有氧和力量的高效训练", "rounds": max(1, available_time_minutes // 10)},
            "gain_muscle": {"name": "力量训练（推拉腿分化）", "description": "今天推荐「推」日: 卧推+肩推+三头下压", "rounds": 1},
        },
        "outdoor": {
            "lose_weight": {"name": "户外跑步", "description": f"建议慢跑 {available_time_minutes} 分钟，配速 6-7 min/km"},
            "general_wellness": {"name": "户外快走", "description": f"快走 {available_time_minutes} 分钟，保持心率在 120-140 bpm"},
        },
    }

    # Get recommendation with fallback
    loc_recs = recommendations.get(location, recommendations["home"])
    rec = loc_recs.get(goal)
    if not rec:
        # Try to find any matching goal
        for loc_key in ["home", "gym", "outdoor"]:
            for goal_key, val in recommendations.get(loc_key, {}).items():
                if goal_key == goal:
                    rec = val
                    break
            if rec:
                break
    if not rec:
        rec = {"name": "基础全身运动", "description": f"{available_time_minutes} 分钟轻中度运动",
               "exercises": [{"name": "快走/慢跑", "duration": f"{available_time_minutes}分钟"}], "rounds": 1, "calories_est": available_time_minutes * 5}

    rec["goal"] = goal
    rec["location"] = location
    rec["available_time_minutes"] = available_time_minutes

    return rec


def calculate_calorie_burn(
    exercise_name: str, duration_minutes: int, weight_kg: float, intensity: str = "moderate"
) -> dict:
    """
    Estimate calories burned for a given exercise.

    Args:
        exercise_name: Name of the exercise
        duration_minutes: Duration in minutes
        weight_kg: User's weight in kilograms
        intensity: "light", "moderate", "vigorous"
    """
    # MET (Metabolic Equivalent of Task) values
    met_db = {
        "跑步": {"light": 6.0, "moderate": 8.3, "vigorous": 11.0},
        "快走": {"light": 3.5, "moderate": 4.5, "vigorous": 5.5},
        "游泳": {"light": 5.0, "moderate": 7.0, "vigorous": 10.0},
        "骑车": {"light": 4.0, "moderate": 6.8, "vigorous": 10.0},
        "跳绳": {"light": 8.0, "moderate": 10.0, "vigorous": 12.0},
        "深蹲": {"light": 3.5, "moderate": 5.0, "vigorous": 7.0},
        "俯卧撑": {"light": 3.8, "moderate": 5.5, "vigorous": 7.5},
        "开合跳": {"light": 6.0, "moderate": 8.0, "vigorous": 10.0},
        "瑜伽": {"light": 2.5, "moderate": 3.5, "vigorous": 5.0},
        "举重": {"light": 3.0, "moderate": 5.0, "vigorous": 7.0},
    }

    # Try to find matching exercise
    met = None
    matched_name = exercise_name
    for key, values in met_db.items():
        if key in exercise_name or exercise_name in key:
            met = values.get(intensity, values["moderate"])
            matched_name = key
            break

    if met is None:
        met = 5.0  # Default moderate MET
        matched_name = exercise_name

    # Calories = MET × weight(kg) × hours
    calories = round(met * weight_kg * (duration_minutes / 60), 1)

    return {
        "exercise": matched_name,
        "duration_minutes": duration_minutes,
        "weight_kg": weight_kg,
        "intensity": intensity,
        "met_value": met,
        "calories_burned": calories,
        "formula": f"{met:.1f} MET × {weight_kg}kg × {duration_minutes / 60:.2f}h = {calories:.1f} kcal",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Function Calling Schema Definitions
# ═══════════════════════════════════════════════════════════════════════════════

EXERCISE_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "generate_workout_plan",
            "description": "根据用户目标和体能水平生成个性化运动计划",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "enum": ["lose_weight", "gain_muscle", "maintain", "general_wellness"],
                        "description": "健身目标",
                    },
                    "fitness_level": {
                        "type": "string",
                        "enum": ["beginner", "intermediate", "advanced"],
                        "description": "当前体能水平",
                    },
                    "duration_minutes": {"type": "integer", "description": "目标运动时长（分钟）", "minimum": 10, "maximum": 120},
                    "equipment": {"type": "array", "items": {"type": "string"}, "description": "可用器械"},
                    "injuries": {"type": "array", "items": {"type": "string"}, "description": "伤病/身体限制"},
                },
                "required": ["goal", "fitness_level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_exercise",
            "description": "根据时间、场地等约束推荐合适的运动",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "运动目标"},
                    "available_time_minutes": {"type": "integer", "description": "可用时间（分钟）"},
                    "location": {
                        "type": "string",
                        "enum": ["home", "gym", "outdoor", "office"],
                        "description": "运动场地",
                    },
                    "equipment": {"type": "array", "items": {"type": "string"}, "description": "可用器材"},
                },
                "required": ["goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_calorie_burn",
            "description": "估算某项运动消耗的热量",
            "parameters": {
                "type": "object",
                "properties": {
                    "exercise_name": {"type": "string", "description": "运动名称"},
                    "duration_minutes": {"type": "integer", "description": "运动时长（分钟）"},
                    "weight_kg": {"type": "number", "description": "体重（千克）"},
                    "intensity": {
                        "type": "string",
                        "enum": ["light", "moderate", "vigorous"],
                        "description": "运动强度",
                    },
                },
                "required": ["exercise_name", "duration_minutes", "weight_kg"],
            },
        },
    },
]

EXERCISE_TOOLS_MAP = {
    "generate_workout_plan": generate_workout_plan,
    "recommend_exercise": recommend_exercise,
    "calculate_calorie_burn": calculate_calorie_burn,
}
