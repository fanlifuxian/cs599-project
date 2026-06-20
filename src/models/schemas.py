"""
Pydantic data models — executable SDD (Spec-Driven Development) specs.
All models use Pydantic v2 with strict validation — they ARE the executable spec.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from enum import Enum


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class ActivityLevel(str, Enum):
    SEDENTARY = "sedentary"
    LIGHTLY_ACTIVE = "lightly_active"
    MODERATELY_ACTIVE = "moderately_active"
    VERY_ACTIVE = "very_active"
    EXTRA_ACTIVE = "extra_active"


class GoalType(str, Enum):
    LOSE_WEIGHT = "lose_weight"
    GAIN_MUSCLE = "gain_muscle"
    MAINTAIN = "maintain"
    IMPROVE_SLEEP = "improve_sleep"
    IMPROVE_CARDIO = "improve_cardio"
    STRESS_RELIEF = "stress_relief"
    GENERAL_WELLNESS = "general_wellness"
    CUSTOM = "custom"


class MealType(str, Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"
    PRE_WORKOUT = "pre_workout"
    POST_WORKOUT = "post_workout"


class ExerciseType(str, Enum):
    CARDIO = "cardio"
    STRENGTH = "strength"
    FLEXIBILITY = "flexibility"
    HIIT = "hiit"
    YOGA = "yoga"
    WALKING = "walking"
    SWIMMING = "swimming"
    CYCLING = "cycling"


class AgentRole(str, Enum):
    DIET = "diet"
    EXERCISE = "exercise"
    SLEEP = "sleep"
    CONSULTATION = "consultation"
    REFLECTION = "reflection"


class MCPToolType(str, Enum):
    QUERY = "query"
    ACTION = "action"
    STREAM = "stream"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# ═══ User Profile ═══
class BodyMetrics(BaseModel):
    """Body measurement snapshot."""
    weight_kg: float = Field(ge=20.0, le=300.0)
    height_cm: float = Field(ge=50.0, le=250.0)
    body_fat_pct: Optional[float] = Field(default=None, ge=2.0, le=60.0)
    waist_cm: Optional[float] = Field(default=None, ge=30.0, le=200.0)
    recorded_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def bmi(self) -> float:
        return round(self.weight_kg / ((self.height_cm / 100) ** 2), 1)

    @property
    def bmi_category(self) -> str:
        bmi = self.bmi
        if bmi < 18.5: return "偏瘦"
        elif bmi < 24.0: return "正常"
        elif bmi < 28.0: return "偏胖"
        else: return "肥胖"


class UserProfile(BaseModel):
    """User health profile — the central data entity (SDD Product Spec)."""
    user_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    name: str = Field(default="用户", min_length=1, max_length=50)
    age: int = Field(ge=1, le=120)
    gender: Gender = Field(description="Biological gender")
    current_metrics: BodyMetrics = Field(description="Latest body measurements")

    activity_level: ActivityLevel = Field(default=ActivityLevel.SEDENTARY)
    sleep_hours_avg: float = Field(default=7.0, ge=0.0, le=16.0)
    water_intake_liters: float = Field(default=1.5, ge=0.0, le=10.0)
    stress_level: int = Field(default=3, ge=1, le=10)
    smoking: bool = Field(default=False)
    alcohol_per_week: int = Field(default=0, ge=0, le=50)

    dietary_preferences: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    medical_conditions: list[str] = Field(default_factory=list)
    exercise_preferences: list[str] = Field(default_factory=list)
    available_equipment: list[str] = Field(default_factory=list)

    metrics_history: list[BodyMetrics] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def height_cm(self) -> float:
        return self.current_metrics.height_cm

    @property
    def weight_kg(self) -> float:
        return self.current_metrics.weight_kg

    @property
    def bmi(self) -> float:
        return self.current_metrics.bmi

    @property
    def bmi_category(self) -> str:
        return self.current_metrics.bmi_category

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id, "name": self.name, "age": self.age,
            "gender": self.gender.value, "height_cm": self.height_cm,
            "weight_kg": self.weight_kg, "bmi": self.bmi,
            "bmi_category": self.bmi_category,
            "activity_level": self.activity_level.value,
            "sleep_hours_avg": self.sleep_hours_avg,
            "dietary_preferences": self.dietary_preferences,
            "allergies": self.allergies,
            "medical_conditions": self.medical_conditions,
            "exercise_preferences": self.exercise_preferences,
            "available_equipment": self.available_equipment,
        }


class HealthGoal(BaseModel):
    """A specific, measurable health goal."""
    goal_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    goal_type: GoalType
    target_description: str = Field(min_length=1, max_length=500)
    target_value: Optional[float] = None
    current_value: Optional[float] = None
    unit: Optional[str] = None
    deadline: Optional[str] = None
    priority: int = Field(default=3, ge=1, le=5)
    status: Literal["active", "achieved", "abandoned"] = "active"
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""


class HealthMetric(BaseModel):
    """A single health measurement for tracking."""
    metric_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    user_id: str
    metric_type: Literal["weight", "sleep_hours", "exercise_minutes",
                         "calories_in", "calories_out", "mood", "blood_pressure",
                         "heart_rate", "steps", "water_liters", "stress_level"]
    value: float
    unit: str = ""
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""


class ProgressSummary(BaseModel):
    """Aggregated user progress over a period."""
    user_id: str
    period_start: str
    period_end: str
    metrics_recorded: int = 0
    goals_achieved: int = 0
    goals_total: int = 1
    weight_trend: Literal["down", "stable", "up"] = "stable"
    avg_sleep_hours: float = 0.0
    avg_exercise_minutes: float = 0.0
    adherence_score: float = Field(default=0.0, ge=0.0, le=100.0)
    insights: list[str] = Field(default_factory=list)

# ═══ Domain Plans ═══
class FoodItem(BaseModel):
    name: str
    portion: str
    calories: float
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    fiber_g: float = 0.0
    sugar_g: float = 0.0
    sodium_mg: float = 0.0
    notes: str = ""


class Meal(BaseModel):
    meal_type: MealType
    time: str
    foods: list[FoodItem] = Field(default_factory=list)
    total_calories: float = 0.0
    total_protein_g: float = 0.0
    notes: str = ""


class MealPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    meals: list[Meal] = Field(default_factory=list)
    daily_calories_target: float = 0.0
    daily_calories_actual: float = 0.0
    macro_split: dict[str, float] = Field(default_factory=dict)
    hydration_liters: float = 2.0
    tips: list[str] = Field(default_factory=list)
    dietary_restrictions_applied: list[str] = Field(default_factory=list)


class ExerciseItem(BaseModel):
    name: str
    exercise_type: ExerciseType
    duration_minutes: int = Field(ge=1, le=180)
    sets: Optional[int] = None
    reps: Optional[str] = None
    rest_seconds: int = 60
    intensity: Literal["light", "moderate", "vigorous"] = "moderate"
    calories_burned_est: float = 0.0
    target_heart_rate: Optional[str] = None
    notes: str = ""


class WorkoutPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    goal: GoalType = GoalType.GENERAL_WELLNESS
    fitness_level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    exercises: list[ExerciseItem] = Field(default_factory=list)
    warm_up: str = ""
    cool_down: str = ""
    total_duration_minutes: int = 0
    total_calories_burned: float = 0.0
    safety_notes: list[str] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)
    progression_plan: str = ""


class SleepPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    recommended_bedtime: str
    recommended_wake_time: str
    target_hours: float = Field(ge=4.0, le=12.0)
    pre_sleep_routine: list[str] = Field(default_factory=list)
    environment_optimizations: list[str] = Field(default_factory=list)
    avoid_items: list[str] = Field(default_factory=list)
    wind_down_start: str = ""
    morning_routine: list[str] = Field(default_factory=list)
    cbti_techniques: list[str] = Field(default_factory=list)
    overall_tips: list[str] = Field(default_factory=list)

# ═══ Agent Communication ═══
class AgentMessage(BaseModel):
    """Message from a specialist agent to the supervisor."""
    role: AgentRole
    content: str = Field(min_length=1)
    plan: Optional[Union[MealPlan, WorkoutPlan, SleepPlan, dict]] = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    tool_calls_made: list[str] = Field(default_factory=list)
    reasoning_trace: str = ""
    sources: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)


class SupervisorDecision(BaseModel):
    """The supervisor's routing and orchestration decision."""
    should_route: bool = False
    target_agents: list[AgentRole] = Field(default_factory=list)
    reasoning: str = ""
    user_message: str = ""
    priority: Literal["routine", "urgent"] = "routine"
    requires_reflection: bool = False
    expected_tools: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Final synthesized response to the user."""
    message: str = Field(min_length=1)
    agent_contributions: dict[str, str] = Field(default_factory=dict)
    plans: dict[str, Any] = Field(default_factory=dict)
    next_steps: list[str] = Field(default_factory=list)
    health_alerts: list[str] = Field(default_factory=list)
    disclaimer: str = "⚠️ 本建议由AI生成，仅供参考，不构成医疗诊断。如有健康问题请咨询专业医生。"
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    session_id: str = Field(default_factory=lambda: str(uuid4())[:8])

# ═══ Evaluation & Infra ═══
# ── Memory & RAG ─────────────────────────────────────────────────────────

class MemoryEntry(BaseModel):
    entry_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    user_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    memory_type: Literal["conversation", "profile", "goal", "preference", "insight"] = "conversation"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    embedding: Optional[list[float]] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: Optional[str] = None


class RAGDocument(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    title: str
    content: str
    source: str
    category: Literal["nutrition", "exercise", "sleep", "mental_health",
                       "chronic_disease", "preventive_care", "general"] = "general"
    tags: list[str] = Field(default_factory=list)
    reliability_score: float = Field(default=0.8, ge=0.0, le=1.0)
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())
    chunk_index: int = 0


class RAGRetrievalResult(BaseModel):
    query: str
    documents: list[RAGDocument] = Field(default_factory=list)
    relevance_scores: list[float] = Field(default_factory=list)
    synthesized_answer: str = ""
    retrieval_time_ms: float = 0.0


# ── MCP ──────────────────────────────────────────────────────────────────

class MCPToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    tool_type: MCPToolType = MCPToolType.QUERY


class MCPToolCall(BaseModel):
    call_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    server_name: str = "health-knowledge"


class MCPToolResult(BaseModel):
    call_id: str
    tool_name: str
    content: Any
    is_error: bool = False
    execution_time_ms: float = 0.0


# ── Tracing ──────────────────────────────────────────────────────────────

class TraceSpan(BaseModel):
    span_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    parent_span_id: Optional[str] = None
    name: str
    agent_role: Optional[AgentRole] = None
    start_time: str = Field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[str] = Field(default_factory=list)
    status: Literal["success", "error", "running"] = "running"
    error_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    session_id: str = ""
    user_input: str = ""
    spans: list[TraceSpan] = Field(default_factory=list)
    total_tokens: int = 0
    total_cost_est: float = 0.0
    total_duration_ms: float = 0.0
    final_response: Optional[str] = None
    agent_count: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ── Evaluation ───────────────────────────────────────────────────────────

class EvalCase(BaseModel):
    case_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    category: str
    user_input: str
    user_profile: Optional[dict] = None
    expected_topics: list[str] = Field(default_factory=list)
    expected_agents: list[AgentRole] = Field(default_factory=list)
    min_confidence: float = 0.6
    forbidden_content: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    case_id: str
    passed: bool
    actual_topics_covered: list[str] = Field(default_factory=list)
    actual_agents_invoked: list[str] = Field(default_factory=list)
    confidence_met: bool = False
    response_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_usage_correct: bool = False
    latency_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)


class EvalSuite(BaseModel):
    suite_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    cases: list[EvalCase] = Field(default_factory=list)
    results: list[EvalResult] = Field(default_factory=list)
    pass_rate: float = 0.0
    avg_quality: float = 0.0
    avg_latency_ms: float = 0.0
    executed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ── LangGraph State (Pydantic model) ─────────────────────────────────────

class GraphState(BaseModel):
    """The shared state flowing through the LangGraph (Pydantic model)."""
    messages: list[dict] = Field(default_factory=list)
    user_input: str = ""
    user_profile: Optional[dict] = None
    health_goals: list[dict] = Field(default_factory=list)
    supervisor_decision: Optional[dict] = None
    agent_outputs: dict[str, dict] = Field(default_factory=dict)
    reflection_needed: bool = False
    reflection_notes: str = ""
    quality_gate_passed: bool = False
    final_response: Optional[dict] = None
    iteration_count: int = 0
    errors: list[str] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)
    trace_id: str = Field(default_factory=lambda: str(uuid4())[:8])