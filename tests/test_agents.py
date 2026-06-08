"""
Tests for the health multi-agent platform.
Tests tool functions directly (no API key needed).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestDietTools:
    """Test diet-related tool functions."""

    def test_calculate_bmr(self):
        from src.tools.diet_tools import calculate_bmr
        result = calculate_bmr(weight_kg=70, height_cm=175, age=30, gender="male")
        assert "bmr" in result
        assert result["bmr"] > 0
        # Mifflin-St Jeor for male: 10*70 + 6.25*175 - 5*30 + 5 = 1648.75
        assert abs(result["bmr"] - 1648.75) < 0.5

    def test_calculate_bmr_female(self):
        from src.tools.diet_tools import calculate_bmr
        result = calculate_bmr(weight_kg=60, height_cm=165, age=25, gender="female")
        # 10*60 + 6.25*165 - 5*25 - 161 = 1345.25
        assert abs(result["bmr"] - 1345.25) < 0.5

    def test_calculate_tdee(self):
        from src.tools.diet_tools import calculate_tdee
        result = calculate_tdee(bmr=1648.75, activity_level="moderately_active")
        assert "tdee" in result
        assert abs(result["tdee"] - 1648.75 * 1.55) < 0.5
        assert result["activity_multiplier"] == 1.55

    def test_generate_meal_plan(self):
        from src.tools.diet_tools import generate_meal_plan
        result = generate_meal_plan(target_calories=1800, meals_per_day=3)
        assert "meals" in result
        assert len(result["meals"]) == 3
        assert "breakfast" in result["meals"][0]["meal_type"]
        assert result["daily_calories_total"] > 0

    def test_analyze_nutrition(self):
        from src.tools.diet_tools import analyze_nutrition
        result = analyze_nutrition("牛肉面")
        assert "calories" in result
        assert result["calories"] > 0
        assert "analysis" in result


class TestExerciseTools:
    """Test exercise-related tool functions."""

    def test_generate_workout_plan(self):
        from src.tools.exercise_tools import generate_workout_plan
        result = generate_workout_plan(goal="lose_weight", fitness_level="beginner", duration_minutes=45)
        assert "warm_up" in result
        assert "exercises" in result
        assert len(result["exercises"]) > 0
        assert result["total_calories_burned"] > 0

    def test_recommend_exercise(self):
        from src.tools.exercise_tools import recommend_exercise
        result = recommend_exercise(goal="lose_weight", available_time_minutes=20, location="home")
        assert "name" in result
        assert result["location"] == "home"

    def test_calculate_calorie_burn(self):
        from src.tools.exercise_tools import calculate_calorie_burn
        result = calculate_calorie_burn(exercise_name="跑步", duration_minutes=30, weight_kg=70, intensity="moderate")
        assert "calories_burned" in result
        assert result["calories_burned"] > 0
        # MET 8.3 * 70 * 0.5 = 290.5
        assert abs(result["calories_burned"] - 290.5) < 1.0


class TestSleepTools:
    """Test sleep-related tool functions."""

    def test_analyze_sleep_quality(self):
        from src.tools.sleep_tools import analyze_sleep_quality
        result = analyze_sleep_quality(
            sleep_hours=7.5, fall_asleep_minutes=15, wake_ups=0,
            screen_time_before_bed_minutes=15, caffeine_after_4pm=False, exercise_today=True
        )
        assert "score" in result
        assert result["score"] >= 80  # Should be good quality
        assert "grade" in result

    def test_analyze_sleep_quality_poor(self):
        from src.tools.sleep_tools import analyze_sleep_quality
        result = analyze_sleep_quality(
            sleep_hours=4.5, fall_asleep_minutes=60, wake_ups=3,
            screen_time_before_bed_minutes=90, caffeine_after_4pm=True, exercise_today=False
        )
        assert result["score"] < 60  # Should be poor

    def test_generate_sleep_plan(self):
        from src.tools.sleep_tools import generate_sleep_plan
        result = generate_sleep_plan(target_sleep_hours=8.0)
        assert "recommended_bedtime" in result
        assert "recommended_wake_time" in result
        assert "pre_sleep_routine" in result
        assert len(result["pre_sleep_routine"]) > 0

    def test_get_sleep_hygiene_tips(self):
        from src.tools.sleep_tools import get_sleep_hygiene_tips
        result = get_sleep_hygiene_tips(topic="insomnia")
        assert "tips" in result
        assert len(result["tips"]) > 0


class TestCommonTools:
    """Test common/shared tool functions."""

    def test_get_user_profile_summary_no_profile(self):
        from src.tools.common_tools import get_user_profile_summary
        result = get_user_profile_summary(None)
        assert result["status"] == "no_profile"

    def test_get_user_profile_summary_with_profile(self):
        from src.tools.common_tools import get_user_profile_summary
        user_data = {
            "name": "测试用户", "age": 30, "gender": "male",
            "height_cm": 175, "weight_kg": 70,
            "activity_level": "moderately_active",
            "dietary_preferences": [], "allergies": [],
            "medical_conditions": [], "sleep_hours_avg": 7.5,
        }
        result = get_user_profile_summary(user_data)
        assert result["status"] == "ok"
        assert result["profile"]["bmi"] == 22.9

    def test_set_health_goals(self):
        from src.tools.common_tools import set_health_goals
        goals = [
            {"goal_type": "lose_weight", "description": "减重5kg", "priority": 1},
            {"goal_type": "improve_sleep", "description": "改善睡眠质量", "priority": 2},
        ]
        result = set_health_goals(goals)
        assert result["goals_set"] == 2
        assert len(result["goals"]) == 2

    def test_track_progress(self):
        from src.tools.common_tools import track_progress
        result = track_progress(metric="weight", value=70.5)
        assert result["metric"] == "weight"
        assert result["value"] == 70.5

    def test_health_risk_assessment(self):
        from src.tools.common_tools import health_risk_assessment
        user_data = {
            "weight_kg": 90, "height_cm": 170, "age": 35,
            "sleep_hours_avg": 5.5, "activity_level": "sedentary",
        }
        result = health_risk_assessment(user_data)
        assert result["status"] == "ok"
        assert result["total_risks"] > 0


class TestModels:
    """Test Pydantic data models."""

    def test_user_profile_creation(self):
        from src.models.schemas import UserProfile, Gender, ActivityLevel
        profile = UserProfile(
            user_id="test001", name="测试", age=30,
            gender=Gender.MALE, height_cm=175, weight_kg=70,
            activity_level=ActivityLevel.MODERATELY_ACTIVE,
        )
        assert profile.bmi == 22.9
        assert profile.bmi_category == "正常"

    def test_user_profile_bmi_obesity(self):
        from src.models.schemas import UserProfile, Gender, ActivityLevel
        profile = UserProfile(
            user_id="test002", name="测试", age=40,
            gender=Gender.FEMALE, height_cm=160, weight_kg=80,
            activity_level=ActivityLevel.SEDENTARY,
        )
        assert profile.bmi == 31.2
        assert profile.bmi_category == "肥胖"

    def test_graph_state_default(self):
        from src.models.schemas import GraphState
        state = GraphState(user_input="test")
        assert state.user_input == "test"
        assert state.iteration_count == 0
        assert state.agent_outputs == {}
        assert state.final_response is None
