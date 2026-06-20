"""
Diet Agent Tools — nutrition calculation, meal planning, food analysis.
All tools expose OpenAI-compatible function calling schemas.
"""

from __future__ import annotations

import json
import math
from typing import Optional
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Implementations
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> dict:
    """
    Calculate Basal Metabolic Rate (BMR) using the Mifflin-St Jeor equation.

    Args:
        weight_kg: Weight in kilograms
        height_cm: Height in centimeters
        age: Age in years
        gender: "male", "female", or "other"
    """
    if gender == "male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    elif gender == "female":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 78  # average

    bmr = round(bmr, 1)

    return {
        "bmr": bmr,
        "formula": "Mifflin-St Jeor",
        "parameters": {
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "age": age,
            "gender": gender,
        },
        "note": "BMR 是维持基本生命活动所需的热量（静息代谢）",
    }


def calculate_tdee(bmr: float, activity_level: str) -> dict:
    """
    Calculate Total Daily Energy Expenditure (TDEE) from BMR and activity level.

    Args:
        bmr: Basal Metabolic Rate
        activity_level: One of "sedentary", "lightly_active", "moderately_active", "very_active", "extra_active"
    """
    multipliers = {
        "sedentary": 1.2,
        "lightly_active": 1.375,
        "moderately_active": 1.55,
        "very_active": 1.725,
        "extra_active": 1.9,
    }

    multiplier = multipliers.get(activity_level, 1.2)
    tdee = round(bmr * multiplier, 1)

    activity_labels = {
        "sedentary": "久坐不动（几乎不运动）",
        "lightly_active": "轻度活动（每周运动 1-3 天）",
        "moderately_active": "中度活动（每周运动 3-5 天）",
        "very_active": "高度活跃（每周运动 6-7 天）",
        "extra_active": "极度活跃（高强度体力劳动/运动员）",
    }

    return {
        "tdee": tdee,
        "bmr": bmr,
        "activity_multiplier": multiplier,
        "activity_level": activity_level,
        "activity_label": activity_labels.get(activity_level, "未知"),
        "note": f"TDEE = BMR × {multiplier}，是维持当前体重每日所需摄入的热量",
    }


def calculate_target_calories(tdee: float, goal: str) -> dict:
    """
    Calculate daily calorie target based on goal.

    Args:
        tdee: Total Daily Energy Expenditure
        goal: "lose_weight", "gain_muscle", "maintain", or "general_wellness"
    """
    adjustments = {
        "lose_weight": {"adjustment": -500, "label": "减重（每日热量缺口 500 kcal）"},
        "gain_muscle": {"adjustment": 300, "label": "增肌（每日热量盈余 300 kcal）"},
        "maintain": {"adjustment": 0, "label": "维持体重"},
        "improve_sleep": {"adjustment": 0, "label": "维持体重（关注睡眠质量）"},
        "general_wellness": {"adjustment": 0, "label": "维持体重（日常健康）"},
    }

    config = adjustments.get(goal, adjustments["general_wellness"])
    target = round(tdee + config["adjustment"], 1)

    return {
        "target_calories": target,
        "tdee": tdee,
        "adjustment": config["adjustment"],
        "goal": goal,
        "goal_label": config["label"],
    }


def generate_meal_plan(
    target_calories: float,
    dietary_preferences: list[str] | None = None,
    allergies: list[str] | None = None,
    meals_per_day: int = 3,
) -> dict:
    """
    Generate a daily meal plan with calorie and macro targets.

    Args:
        target_calories: Daily calorie target
        dietary_preferences: e.g. ["vegetarian", "low_carb"]
        allergies: e.g. ["peanuts", "seafood"]
        meals_per_day: Number of meals (2-5)
    """
    dietary_preferences = dietary_preferences or []
    allergies = allergies or []

    # Macro split: 30% protein, 45% carbs, 25% fat
    protein_cal = target_calories * 0.30
    carbs_cal = target_calories * 0.45
    fat_cal = target_calories * 0.25

    # Meal distribution
    meal_calories = target_calories / meals_per_day

    # Meal template library
    breakfast_options = [
        {
            "name": "燕麦牛奶早餐",
            "foods": [
                {"name": "燕麦片", "portion": "50g", "calories": 190, "protein_g": 6.5, "carbs_g": 33.0, "fat_g": 3.5},
                {"name": "全脂牛奶", "portion": "250ml", "calories": 150, "protein_g": 8.0, "carbs_g": 12.0, "fat_g": 8.0},
                {"name": "水煮蛋", "portion": "1个", "calories": 70, "protein_g": 6.3, "carbs_g": 0.6, "fat_g": 5.0},
                {"name": "香蕉", "portion": "1根", "calories": 105, "protein_g": 1.3, "carbs_g": 27.0, "fat_g": 0.4},
            ],
            "total": 515,
        },
        {
            "name": "中式杂粮粥早餐",
            "foods": [
                {"name": "小米粥", "portion": "1碗(300ml)", "calories": 140, "protein_g": 4.0, "carbs_g": 28.0, "fat_g": 1.5},
                {"name": "蒸红薯", "portion": "100g", "calories": 86, "protein_g": 1.6, "carbs_g": 20.0, "fat_g": 0.1},
                {"name": "茶叶蛋", "portion": "1个", "calories": 73, "protein_g": 6.5, "carbs_g": 0.8, "fat_g": 5.0},
                {"name": "凉拌黄瓜", "portion": "100g", "calories": 15, "protein_g": 0.8, "carbs_g": 2.9, "fat_g": 0.1},
            ],
            "total": 314,
        },
    ]

    lunch_options = [
        {
            "name": "鸡胸肉蔬菜饭",
            "foods": [
                {"name": "糙米饭", "portion": "150g", "calories": 168, "protein_g": 3.5, "carbs_g": 35.0, "fat_g": 1.5},
                {"name": "鸡胸肉", "portion": "150g", "calories": 195, "protein_g": 35.0, "carbs_g": 0, "fat_g": 4.0},
                {"name": "西兰花", "portion": "100g", "calories": 34, "protein_g": 2.8, "carbs_g": 7.0, "fat_g": 0.4},
                {"name": "橄榄油", "portion": "5ml", "calories": 45, "protein_g": 0, "carbs_g": 0, "fat_g": 5.0},
            ],
            "total": 442,
        },
        {
            "name": "三文鱼藜麦沙拉",
            "foods": [
                {"name": "三文鱼", "portion": "120g", "calories": 250, "protein_g": 24.0, "carbs_g": 0, "fat_g": 16.0},
                {"name": "藜麦", "portion": "80g(熟)", "calories": 96, "protein_g": 3.5, "carbs_g": 17.0, "fat_g": 1.5},
                {"name": "混合生菜", "portion": "150g", "calories": 25, "protein_g": 2.0, "carbs_g": 4.0, "fat_g": 0.3},
                {"name": "橄榄油醋汁", "portion": "10ml", "calories": 70, "protein_g": 0, "carbs_g": 1.0, "fat_g": 7.0},
            ],
            "total": 441,
        },
    ]

    dinner_options = [
        {
            "name": "清蒸鱼配时蔬",
            "foods": [
                {"name": "鲈鱼(清蒸)", "portion": "150g", "calories": 158, "protein_g": 28.0, "carbs_g": 0, "fat_g": 5.0},
                {"name": "白米饭", "portion": "100g", "calories": 116, "protein_g": 2.6, "carbs_g": 25.9, "fat_g": 0.3},
                {"name": "炒青菜", "portion": "150g", "calories": 60, "protein_g": 2.5, "carbs_g": 5.0, "fat_g": 3.0},
                {"name": "紫菜蛋花汤", "portion": "1碗", "calories": 40, "protein_g": 3.0, "carbs_g": 2.0, "fat_g": 1.5},
            ],
            "total": 374,
        },
    ]

    snack_options = [
        {"name": "希腊酸奶+蓝莓", "calories": 150, "protein_g": 12.0, "carbs_g": 15.0, "fat_g": 4.0},
        {"name": "混合坚果(无盐)", "calories": 170, "protein_g": 5.0, "carbs_g": 6.0, "fat_g": 15.0},
        {"name": "苹果", "calories": 95, "protein_g": 0.5, "carbs_g": 25.0, "fat_g": 0.3},
    ]

    # Build meal plan
    meals = []

    # Breakfast
    bf = breakfast_options[0]
    meals.append({
        "meal_type": "breakfast",
        "time": "07:30",
        "name": bf["name"],
        "foods": bf["foods"],
        "total_calories": bf["total"],
        "notes": "早餐应包含优质蛋白和复合碳水",
    })

    # Lunch
    ln = lunch_options[0]
    meals.append({
        "meal_type": "lunch",
        "time": "12:00",
        "name": ln["name"],
        "foods": ln["foods"],
        "total_calories": ln["total"],
        "notes": "午餐建议摄入足量蛋白质",
    })

    # Dinner
    dn = dinner_options[0]
    meals.append({
        "meal_type": "dinner",
        "time": "18:30",
        "name": dn["name"],
        "foods": dn["foods"],
        "total_calories": dn["total"],
        "notes": "晚餐建议清淡，避免高脂",
    })

    # Snack (if needed)
    if meals_per_day >= 4:
        snacks = snack_options[0]
        meals.append({
            "meal_type": "snack",
            "time": "15:30",
            "name": snacks["name"],
            "foods": [snacks],
            "total_calories": snacks["calories"],
            "notes": "下午加餐帮助维持能量",
        })

    total_actual = sum(m["total_calories"] for m in meals)
    total_protein = sum(f.get("protein_g", 0) for m in meals for f in m["foods"])
    total_carbs = sum(f.get("carbs_g", 0) for m in meals for f in m["foods"])
    total_fat = sum(f.get("fat_g", 0) for m in meals for f in m["foods"])

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "target_calories": target_calories,
        "target_protein_g": round(target_calories * 0.30 / 4, 1),
        "target_carbs_g": round(target_calories * 0.45 / 4, 1),
        "target_fat_g": round(target_calories * 0.25 / 9, 1),
        "meals": meals,
        "daily_calories_total": round(total_actual, 1),
        "daily_protein_g": round(total_protein, 1),
        "daily_carbs_g": round(total_carbs, 1),
        "daily_fat_g": round(total_fat, 1),
        "dietary_preferences": dietary_preferences,
        "allergies_excluded": allergies,
        "tips": [
            "每天饮水 1.5-2 升，运动时适当增加",
            "尽量选择全谷物替代精制碳水",
            "控制盐摄入量 < 6g/天",
            "蔬菜每天至少 300-500g",
        ],
    }


def analyze_nutrition(food_description: str) -> dict:
    """
    Provide nutritional analysis for a described food or meal.

    Args:
        food_description: Description of food/meal (e.g. "一碗牛肉面", "chicken salad with dressing")
    """
    # This is a simulated analysis — real implementation would use a nutrition API
    foods_db = {
        "牛肉面": {"calories": 450, "protein_g": 20, "carbs_g": 55, "fat_g": 15, "fiber_g": 3},
        "鸡胸肉沙拉": {"calories": 320, "protein_g": 35, "carbs_g": 12, "fat_g": 14, "fiber_g": 5},
        "汉堡薯条": {"calories": 850, "protein_g": 30, "carbs_g": 85, "fat_g": 42, "fiber_g": 4},
        "寿司拼盘": {"calories": 400, "protein_g": 18, "carbs_g": 65, "fat_g": 6, "fiber_g": 2},
        "火锅(一人份)": {"calories": 800, "protein_g": 45, "carbs_g": 35, "fat_g": 50, "fiber_g": 4},
        "蛋炒饭": {"calories": 380, "protein_g": 12, "carbs_g": 48, "fat_g": 15, "fiber_g": 2},
        "三明治": {"calories": 350, "protein_g": 15, "carbs_g": 40, "fat_g": 14, "fiber_g": 3},
    }

    # Fuzzy lookup
    matched = None
    for key, data in foods_db.items():
        if key in food_description:
            matched = data
            matched["food_name"] = key
            break

    if not matched:
        # Generic estimate
        matched = {
            "food_name": food_description,
            "calories": 400,
            "protein_g": 15,
            "carbs_g": 45,
            "fat_g": 18,
            "fiber_g": 3,
        }

    matched["analysis"] = _nutrition_feedback(matched)

    return matched


def _nutrition_feedback(data: dict) -> str:
    """Generate brief nutrition feedback."""
    feedbacks = []
    if data["calories"] > 700:
        feedbacks.append("⚠️ 此餐热量较高，建议搭配低热量食物平衡")
    if data["protein_g"] < 15:
        feedbacks.append("💡 蛋白质不足，建议增加瘦肉/豆制品/蛋奶")
    if data["fat_g"] > 30:
        feedbacks.append("⚠️ 脂肪含量偏高")
    if data["fiber_g"] < 3:
        feedbacks.append("💡 膳食纤维不足，建议增加蔬菜或全谷物")
    if not feedbacks:
        feedbacks.append("✅ 此餐营养较为均衡")
    return "；".join(feedbacks)


# ═══════════════════════════════════════════════════════════════════════════════
# Function Calling Schema Definitions (OpenAI-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

DIET_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculate_bmr",
            "description": "计算用户的基础代谢率（BMR），使用 Mifflin-St Jeor 公式",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg": {"type": "number", "description": "体重（千克）"},
                    "height_cm": {"type": "number", "description": "身高（厘米）"},
                    "age": {"type": "integer", "description": "年龄"},
                    "gender": {"type": "string", "enum": ["male", "female", "other"], "description": "性别"},
                },
                "required": ["weight_kg", "height_cm", "age", "gender"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_tdee",
            "description": "根据 BMR 和活动水平计算每日总热量消耗（TDEE）",
            "parameters": {
                "type": "object",
                "properties": {
                    "bmr": {"type": "number", "description": "基础代谢率（BMR）"},
                    "activity_level": {
                        "type": "string",
                        "enum": ["sedentary", "lightly_active", "moderately_active", "very_active", "extra_active"],
                        "description": "活动水平",
                    },
                },
                "required": ["bmr", "activity_level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_meal_plan",
            "description": "生成一日饮食计划，包含每餐的食物、热量和宏量营养素",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_calories": {"type": "number", "description": "每日目标热量（kcal）"},
                    "dietary_preferences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "饮食偏好（如 vegetarian, low_carb）",
                    },
                    "allergies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "过敏/忌口食物",
                    },
                    "meals_per_day": {"type": "integer", "description": "每日餐次（2-5）", "minimum": 2, "maximum": 5},
                },
                "required": ["target_calories"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_nutrition",
            "description": "分析某种食物或一餐的营养成分",
            "parameters": {
                "type": "object",
                "properties": {
                    "food_description": {"type": "string", "description": "食物或餐食的描述"},
                },
                "required": ["food_description"],
            },
        },
    },
]

DIET_TOOLS_MAP = {
    "calculate_bmr": calculate_bmr,
    "calculate_tdee": calculate_tdee,
    "generate_meal_plan": generate_meal_plan,
    "analyze_nutrition": analyze_nutrition,
}
