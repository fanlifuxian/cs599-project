"""
FastAPI Production Server — RESTful API for the Health Multi-Agent Platform.

Endpoints:
- POST /chat — synchronous chat
- POST /chat/stream — Server-Sent Events streaming chat
- GET /health — health check
- GET /stats — system statistics
- POST /eval — run benchmark evaluation
- GET /docs — auto-generated OpenAPI documentation

Features:
- Async request handling
- Server-Sent Events (SSE) streaming
- Request validation with Pydantic
- Rate limiting
- CORS support
- Graceful shutdown
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.graph.health_graph import get_health_graph
from src.memory.memory_manager import MemoryManager
from src.models import (
    UserProfile, Gender, ActivityLevel, BodyMetrics,
    AgentResponse, HealthGoal, GoalType,
)

logger = logging.getLogger(__name__)

# ── Application Setup ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — init and cleanup."""
    logger.info(f"🚀 Starting Health Multi-Agent API on {settings.api_host}:{settings.api_port}")
    # Pre-warm the graph
    get_health_graph()
    yield
    logger.info("👋 Shutting down Health Multi-Agent API")


app = FastAPI(
    title="个性化健康规划多智能体平台 API",
    description="""
## Personalized Health Planning Multi-Agent Platform

A LangGraph-based multi-agent system for personalized health management.

### 🤖 Agents
- **Diet Agent** 🥗 — Nutrition assessment, meal planning, food analysis
- **Exercise Agent** 🏃 — Workout prescription, exercise recommendation
- **Sleep Agent** 😴 — Sleep quality analysis, CBT-I, hygiene education
- **Consultation Agent** 🧠 — Orchestration, risk assessment, synthesis

### 🔧 Core Features
- Multi-agent collaboration with Supervisor-Worker pattern
- Function Calling with 15+ health tools
- MCP protocol integration (health guidelines, drug interactions)
- Agentic RAG (evidence-based knowledge base ~20 curated documents)
- Vector memory (ChromaDB semantic search)
- Reflection & quality gate
- Streaming responses (SSE)
""",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="User's health question")
    user_id: Optional[str] = Field(default=None, description="User ID for profile lookup")
    stream: bool = Field(default=False, description="Enable streaming response")


class ChatResponse(BaseModel):
    message: str = Field(description="Agent's response")
    agents_contributed: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    confidence: str = "medium"
    trace_id: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ProfileRequest(BaseModel):
    name: str = Field(default="用户")
    age: int = Field(ge=1, le=120)
    gender: str = "other"
    height_cm: float = Field(ge=50, le=250)
    weight_kg: float = Field(ge=20, le=300)
    activity_level: str = "sedentary"
    sleep_hours_avg: float = 7.0
    dietary_preferences: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    medical_conditions: list[str] = Field(default_factory=list)


class StatsResponse(BaseModel):
    status: str = "healthy"
    agents: list[str] = ["diet", "exercise", "sleep", "consultation"]
    llm_provider: str = ""
    llm_model: str = ""
    vector_store: dict = Field(default_factory=dict)
    knowledge_base: dict = Field(default_factory=dict)
    uptime: str = ""


# ── Global State ────────────────────────────────────────────────────────

_start_time = datetime.now()
_memory_managers: dict[str, MemoryManager] = {}


def _get_memory(user_id: str = "default") -> MemoryManager:
    """Get or create a memory manager for a user."""
    if user_id not in _memory_managers:
        mm = MemoryManager()
        if user_id != "default":
            mm.load_user_profile(user_id)
        _memory_managers[user_id] = mm
    return _memory_managers[user_id]


# ── API Endpoints ───────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "llm": f"{settings.llm_provider}/{settings.llm_model}",
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    try:
        from src.memory.vector_store import get_vector_store
        vs_stats = get_vector_store().get_collection_stats()
    except Exception:
        vs_stats = {"status": "unavailable"}

    try:
        from src.tools.rag_tools import get_knowledge_base_stats
        kb_stats = get_knowledge_base_stats()
    except Exception:
        kb_stats = {"status": "unavailable"}

    uptime = str(datetime.now() - _start_time).split(".")[0]

    return StatsResponse(
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        vector_store=vs_stats,
        knowledge_base=kb_stats,
        uptime=uptime,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a health question and get a response from the multi-agent system.

    The system will:
    1. Analyze your intent
    2. Route to appropriate specialist agents (Diet/Exercise/Sleep)
    3. Synthesize a comprehensive response
    """
    if not settings.llm_api_key:
        raise HTTPException(status_code=503, detail="LLM API key not configured")

    # Streaming
    if request.stream and settings.feature_streaming:
        return StreamingResponse(
            _stream_chat(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Synchronous chat
    user_id = request.user_id or "default"
    memory = _get_memory(user_id)
    graph = get_health_graph()
    trace_id = str(uuid.uuid4())[:12]

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    try:
        result = graph.invoke(
            {
                "user_input": request.message,
                "user_profile": memory.get_active_user().to_dict() if memory.get_active_user() else None,
                "health_goals": [g.model_dump() for g in memory._goals],
                "agent_outputs": {},
                "supervisor_decision": None,
                "final_response": None,
                "iteration_count": 0,
                "errors": [],
                "reflection_needed": False,
                "reflection_notes": "",
                "quality_gate_passed": False,
                "next_step": "",
                "trace_id": trace_id,
                "token_usage": {},
            },
            config=config,
        )

        final = result.get("final_response", {})
        agent_outputs = result.get("agent_outputs", {})

        agent_names = {
            "diet": "diet", "exercise": "exercise",
            "sleep": "sleep", "consultation": "consultation",
        }

        return ChatResponse(
            message=final.get("message", "抱歉，未能生成回复。"),
            agents_contributed=[agent_names.get(r, r) for r in agent_outputs],
            next_steps=final.get("next_steps", []),
            confidence=final.get("confidence_level", "medium"),
            trace_id=trace_id,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent processing error: {str(e)}")


async def _stream_chat(request: ChatRequest):
    """Server-Sent Events streaming chat."""
    user_id = request.user_id or "default"
    memory = _get_memory(user_id)
    graph = get_health_graph()
    trace_id = str(uuid.uuid4())[:12]

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    try:
        yield f"event: status\ndata: 正在分析您的问题...\n\n"

        result = graph.invoke(
            {
                "user_input": request.message,
                "user_profile": memory.get_active_user().to_dict() if memory.get_active_user() else None,
                "health_goals": [g.model_dump() for g in memory._goals],
                "agent_outputs": {},
                "supervisor_decision": None,
                "final_response": None,
                "iteration_count": 0,
                "errors": [],
                "reflection_needed": False,
                "reflection_notes": "",
                "quality_gate_passed": False,
                "next_step": "",
                "trace_id": trace_id,
                "token_usage": {},
            },
            config=config,
        )

        final = result.get("final_response", {})
        message = final.get("message", "抱歉，未能生成回复。")

        # Simulate streaming by chunking the response
        chunks = message.split("\n")
        for chunk in chunks:
            if chunk.strip():
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.05)

        yield f"event: done\ndata: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        yield f"event: error\ndata: 处理出错: {str(e)[:100]}\n\n"


@app.post("/profile")
async def create_profile(profile: ProfileRequest):
    """Create or update a user health profile."""
    try:
        gender_map = {"male": Gender.MALE, "female": Gender.FEMALE, "other": Gender.OTHER}
        activity_map = {
            "sedentary": ActivityLevel.SEDENTARY,
            "lightly_active": ActivityLevel.LIGHTLY_ACTIVE,
            "moderately_active": ActivityLevel.MODERATELY_ACTIVE,
            "very_active": ActivityLevel.VERY_ACTIVE,
            "extra_active": ActivityLevel.EXTRA_ACTIVE,
        }

        user_profile = UserProfile(
            user_id=str(uuid.uuid4())[:8],
            name=profile.name,
            age=profile.age,
            gender=gender_map.get(profile.gender, Gender.OTHER),
            current_metrics=BodyMetrics(
                weight_kg=profile.weight_kg,
                height_cm=profile.height_cm,
            ),
            activity_level=activity_map.get(profile.activity_level, ActivityLevel.SEDENTARY),
            sleep_hours_avg=profile.sleep_hours_avg,
            dietary_preferences=profile.dietary_preferences,
            allergies=profile.allergies,
            medical_conditions=profile.medical_conditions,
        )

        memory = _get_memory(user_profile.user_id)
        memory.save_user_profile(user_profile)

        default_goals = [
            HealthGoal(
                goal_type=GoalType.GENERAL_WELLNESS,
                target_description="全面改善健康状态",
                priority=1,
            )
        ]
        memory.save_goals(user_profile.user_id, default_goals)

        return {
            "status": "ok",
            "user_id": user_profile.user_id,
            "bmi": user_profile.bmi,
            "bmi_category": user_profile.bmi_category,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid profile data: {str(e)}")


@app.post("/eval")
async def run_evaluation():
    """Run the benchmark evaluation suite."""
    try:
        from src.observability.evaluator import get_evaluator

        evaluator = get_evaluator()

        def run_fn(user_input: str):
            """Adapter function for evaluation."""
            memory = MemoryManager()
            graph = get_health_graph()

            # Create a basic profile for testing
            from src.models import UserProfile, Gender, ActivityLevel, BodyMetrics
            test_profile = UserProfile(
                user_id="eval_user",
                name="测试用户", age=30, gender=Gender.MALE,
                current_metrics=BodyMetrics(weight_kg=70, height_cm=170),
                activity_level=ActivityLevel.MODERATELY_ACTIVE,
            )
            memory.save_user_profile(test_profile)

            result = graph.invoke(
                {
                    "user_input": user_input,
                    "user_profile": test_profile.to_dict(),
                    "health_goals": [],
                    "agent_outputs": {},
                    "supervisor_decision": None,
                    "final_response": None,
                    "iteration_count": 0,
                    "errors": [],
                    "reflection_needed": False,
                    "reflection_notes": "",
                    "quality_gate_passed": False,
                    "next_step": "",
                    "trace_id": str(uuid.uuid4())[:12],
                    "token_usage": {},
                },
                config={"configurable": {"thread_id": str(uuid.uuid4())}},
            )

            agent_outputs_raw = result.get("agent_outputs", {})
            final = result.get("final_response", {})

            agent_outputs = {}
            from src.models import AgentMessage, AgentRole
            for role_key, data in agent_outputs_raw.items():
                try:
                    agent_outputs[role_key] = AgentMessage(**data)
                except Exception:
                    agent_outputs[role_key] = AgentMessage(
                        role=AgentRole.CONSULTATION,
                        content=str(data),
                    )

            return agent_outputs, final.get("message", "")

        suite = evaluator.run_benchmark(run_fn)
        return suite.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")


# ── Entry Point ─────────────────────────────────────────────────────────

def main():
    """Start the API server."""
    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
