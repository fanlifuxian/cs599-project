"""
Pydantic data models — executable SDD specs.
These define the contract for all agent inputs/outputs and system state.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class ActivityLevel(str, Enum):
    SEDENTARY = "sedentary"           # 几乎不运动
    LIGHTLY_ACTIVE = "lightly_active"  # 每周运动 1-3 天
    MODERATELY_ACTIVE = "moderately_active"  # 每周运动 3-5 天
    VERY_ACTIVE = "very_active"        # 每周运动 6-7 天
    EXTRA_ACTIVE = "extra_active"      # 高强度体力劳动/运动员


class GoalType(str, Enum):
    LOSE_WEIGHT = "lose_weight"
    GAIN_MUSCLE = "gain_muscle"
    MAINTAIN = "maintain"
    IMPROVE_SLEEP = "improve_sleep"
    GENERAL_WELLNESS = "general_wellness"


class MealType(str, Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


class ExerciseType(str, Enum):
    CARDIO = "cardio"
    STRENGTH = "strength"
    FLEXIBILITY = "flexibility"
    HIIT = "hiit"
    YOGA = "yoga"
    WALKING = "walking"


class AgentRole(str, Enum):
    DIET = "diet"
    EXERCISE = "exercise"
    SLEEP = "sleep"
    CONSULTATION = "consultation"


# ═══════════════════════════════════════════════════════════════════════════════
# User Profile
# ═══════════════════════════════════════════════════════════════════════════════

class UserProfile(BaseModel):
    """User health profile — the core data entity."""
    user_id: str = Field(description="Unique user identifier")
    name: str = Field(default="用户", description="User's display name")
    age: int = Field(ge=1, le=120, description="Age in years")
    gender: Gender = Field(description="Biological gender")
    height_cm: float = Field(ge=50.0, le=250.0, description="Height in centimeters")
    weight_kg: float = Field(ge=20.0, le=300.0, description="Weight in kilograms")
    activity_level: ActivityLevel = Field(default=ActivityLevel.SEDENTARY)
    goals: list[GoalType] = Field(default_factory=lambda: [GoalType.GENERAL_WELLNESS])

    # Optional health data
    medical_conditions: list[str] = Field(default_factory=list, description="Known medical conditions")
    allergies: list[str] = Field(default_factory=list, description="Food/drug allergies")
    dietary_preferences: list[str] = Field(default_factory=list, description="e.g. vegetarian, halal")
    sleep_hours_avg: float = Field(default=7.0, ge=0, le=16, description="Average sleep hours per night")

    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def bmi(self) -> float:
        """BMI = weight(kg) / height(m)^2"""
        return round(self.weight_kg / ((self.height_cm / 100) ** 2), 1)

    @property
    def bmi_category(self) -> str:
        bmi = self.bmi
        if bmi < 18.5:
            return "偏瘦"
        elif bmi < 24.0:
            return "正常"
        elif bmi < 28.0:
            return "偏胖"
        else:
            return "肥胖"


# ═══════════════════════════════════════════════════════════════════════════════
# Health Goals
# ═══════════════════════════════════════════════════════════════════════════════

class HealthGoal(BaseModel):
    """A specific health goal set by the user."""
    goal_type: GoalType
    target_description: str = Field(description="Human-readable goal description")
    target_value: Optional[float] = Field(default=None, description="Numeric target if applicable")
    deadline: Optional[str] = Field(default=None, description="Target date in ISO format")
    priority: int = Field(default=1, ge=1, le=5, description="Priority 1 (highest) to 5 (lowest)")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Plans (Domain Outputs)
# ═══════════════════════════════════════════════════════════════════════════════

class FoodItem(BaseModel):
    """A single food item in a meal."""
    name: str
    portion: str = Field(description="e.g. '100g', '1碗'")
    calories: float
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    notes: str = ""


class Meal(BaseModel):
    """A single meal composed of food items."""
    meal_type: MealType
    time: str = Field(description="Suggested meal time, e.g. '08:00'")
    foods: list[FoodItem] = Field(default_factory=list)
    total_calories: float = 0.0
    notes: str = ""


class MealPlan(BaseModel):
    """A full-day meal plan."""
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    meals: list[Meal] = Field(default_factory=list)
    daily_calories_total: float = 0.0
    daily_protein_g: float = 0.0
    daily_carbs_g: float = 0.0
    daily_fat_g: float = 0.0
    tips: list[str] = Field(default_factory=list)


class ExerciseItem(BaseModel):
    """A single exercise in a workout."""
    name: str
    exercise_type: ExerciseType
    duration_minutes: int = Field(ge=1, le=180)
    sets: Optional[int] = None
    reps: Optional[str] = None  # e.g. "12-15" or "until failure"
    calories_burned_est: float = 0.0
    notes: str = ""


class WorkoutPlan(BaseModel):
    """A full workout plan."""
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    exercises: list[ExerciseItem] = Field(default_factory=list)
    total_duration_minutes: int = 0
    total_calories_burned: float = 0.0
    warm_up: str = ""
    cool_down: str = ""
    tips: list[str] = Field(default_factory=list)


class SleepPlan(BaseModel):
    """A sleep hygiene and schedule plan."""
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    recommended_bedtime: str = Field(description="e.g. '22:30'")
    recommended_wake_time: str = Field(description="e.g. '06:30'")
    target_hours: float = Field(ge=4, le=12)
    pre_sleep_routine: list[str] = Field(default_factory=list, description="睡前例行事项")
    environment_tips: list[str] = Field(default_factory=list, description="睡眠环境优化建议")
    avoid_items: list[str] = Field(default_factory=list, description="睡前应避免的事项")
    overall_tips: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Communication
# ═══════════════════════════════════════════════════════════════════════════════

class AgentMessage(BaseModel):
    """A message from one agent to the supervisor or user."""
    role: AgentRole
    content: str = Field(description="Agent's natural-language response")
    plan: Optional[MealPlan | WorkoutPlan | SleepPlan | dict] = Field(
        default=None, description="Structured plan output, if applicable"
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Agent's confidence in this response")
    tool_calls_made: list[str] = Field(default_factory=list, description="Tools invoked to produce this response")


class SupervisorDecision(BaseModel):
    """The supervisor's routing decision."""
    should_route: bool = Field(description="Whether to route to specialist agents")
    target_agents: list[AgentRole] = Field(default_factory=list, description="Which agents to invoke")
    reasoning: str = Field(default="", description="Why this routing decision was made")
    user_message: str = Field(default="", description="Message to show the user while agents work")


class AgentResponse(BaseModel):
    """Final synthesized response from the supervisor to the user."""
    message: str = Field(description="The main response text")
    agent_contributions: dict[str, str] = Field(
        default_factory=dict, description="Contributions from each agent, keyed by role"
    )
    plans: dict[str, Any] = Field(
        default_factory=dict, description="Structured plans, keyed by agent role"
    )
    next_steps: list[str] = Field(default_factory=list, description="Suggested next actions for the user")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ═══════════════════════════════════════════════════════════════════════════════
# LangGraph State
# ═══════════════════════════════════════════════════════════════════════════════

class GraphState(BaseModel):
    """The shared state that flows through the LangGraph."""
    # Conversation
    messages: list[dict] = Field(default_factory=list, description="Chat history (role/content dicts)")
    user_input: str = Field(default="", description="Latest user message")

    # User context
    user_profile: Optional[UserProfile] = Field(default=None)
    health_goals: list[HealthGoal] = Field(default_factory=list)

    # Agent orchestration
    supervisor_decision: Optional[SupervisorDecision] = Field(default=None)
    agent_outputs: dict[str, AgentMessage] = Field(
        default_factory=dict, description="Outputs from specialist agents, keyed by role"
    )

    # Final output
    final_response: Optional[AgentResponse] = Field(default=None)

    # Metadata
    iteration_count: int = Field(default=0, description="Number of agent routing iterations")
    errors: list[str] = Field(default_factory=list)
