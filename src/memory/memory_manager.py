"""
Enterprise-grade Memory Manager — hybrid memory architecture.

Three-tier memory system:
1. Short-term: LangChain ConversationBuffer (in-memory, current session)
2. Long-term: ChromaDB vector store (semantic, cross-session)
3. Persistent: JSON file storage (user profiles, goals, metrics history)

Features:
- Semantic conversation retrieval across sessions
- Automatic memory consolidation (insight extraction)
- User preference learning over time
- Memory importance scoring & decay
- Hybrid search (semantic + keyword)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.config.settings import settings


# ── Simple Conversation Buffer (replaces langchain.memory) ──────────────

class _ChatMessage:
    def __init__(self, msg_type: str, content: str):
        self.type = msg_type
        self.content = content


class _SimpleChatMemory:
    """Minimal chat message history — avoids langchain.memory dependency."""
    def __init__(self):
        self.messages: list[_ChatMessage] = []

    def add_user_message(self, content: str):
        self.messages.append(_ChatMessage("human", content))

    def add_ai_message(self, content: str):
        self.messages.append(_ChatMessage("ai", content))


class SimpleConversationBuffer:
    """Standalone conversation buffer with LangChain-compatible API."""
    def __init__(self, memory_key: str = "chat_history", return_messages: bool = True, input_key: str = "input"):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.input_key = input_key
        self.chat_memory = _SimpleChatMemory()

    def add_user_message(self, content: str):
        self.chat_memory.add_user_message(content)

    def add_ai_message(self, content: str):
        self.chat_memory.add_ai_message(content)

    def clear(self):
        self.chat_memory.messages = []
from src.models import (
    UserProfile, HealthGoal, HealthMetric, AgentResponse,
    MemoryEntry, ProgressSummary, BodyMetrics,
)

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Hybrid memory manager combining:
    - Conversation buffer (short-term, LangChain)
    - Vector store (long-term semantic, ChromaDB)
    - File persistence (profiles, goals, metrics)
    """

    def __init__(self, data_dir: str | None = None):
        self.data_dir = Path(data_dir or settings.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Tier 1: Short-term conversation memory
        self.conversation_memory: SimpleConversationBuffer = SimpleConversationBuffer(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
        )

        # Tier 2: Long-term vector memory (lazy init)
        self._vector_store = None

        # Tier 3: In-memory cache
        self._active_user: Optional[UserProfile] = None
        self._goals: list[HealthGoal] = []
        self._history: list[dict] = []
        self._metrics: list[HealthMetric] = []
        self._session_id: str = datetime.now().strftime("%Y%m%d%H%M%S")

        logger.info(f"MemoryManager initialized. Data dir: {self.data_dir}")

    @property
    def vector_store(self):
        """Lazy-init the vector store."""
        if self._vector_store is None:
            from src.memory.vector_store import get_vector_store
            self._vector_store = get_vector_store()
        return self._vector_store

    # ═══════════════════════════════════════════════════════════════════════════
    # User Profile Persistence
    # ═══════════════════════════════════════════════════════════════════════════

    def _user_file_path(self, user_id: str) -> Path:
        return self.data_dir / f"user_{user_id}.json"

    def save_user_profile(self, profile: UserProfile) -> None:
        """Persist user profile and index it in vector store."""
        profile.updated_at = datetime.now().isoformat()
        filepath = self._user_file_path(profile.user_id)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(profile.model_dump(), f, ensure_ascii=False, indent=2)

        self._active_user = profile

        # Index profile in vector memory
        self._index_profile(profile)

        logger.info(f"User profile saved: {profile.user_id} (BMI: {profile.bmi})")

    def load_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load user profile from disk."""
        filepath = self._user_file_path(user_id)
        if not filepath.exists():
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle legacy format (flat profile) vs new format (with current_metrics)
        if "current_metrics" not in data:
            from src.models import BodyMetrics
            data["current_metrics"] = BodyMetrics(
                weight_kg=data.get("weight_kg", 70),
                height_cm=data.get("height_cm", 170),
            ).model_dump()

        profile = UserProfile(**data)
        self._active_user = profile
        return profile

    def get_active_user(self) -> Optional[UserProfile]:
        return self._active_user

    def update_metrics(self, user_id: str, metrics: BodyMetrics) -> None:
        """Record new body metrics and track history."""
        profile = self._active_user
        if not profile or profile.user_id != user_id:
            profile = self.load_user_profile(user_id)

        if profile:
            profile.metrics_history.append(profile.current_metrics)
            profile.current_metrics = metrics
            profile.updated_at = datetime.now().isoformat()
            self.save_user_profile(profile)

    # ═══════════════════════════════════════════════════════════════════════════
    # Health Goals
    # ═══════════════════════════════════════════════════════════════════════════

    def _goals_file_path(self, user_id: str) -> Path:
        return self.data_dir / f"goals_{user_id}.json"

    def save_goals(self, user_id: str, goals: list[HealthGoal]) -> None:
        filepath = self._goals_file_path(user_id)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([g.model_dump() for g in goals], f, ensure_ascii=False, indent=2)
        self._goals = goals
        logger.info(f"Goals saved: {len(goals)} goals for user {user_id}")

    def load_goals(self, user_id: str) -> list[HealthGoal]:
        filepath = self._goals_file_path(user_id)
        if not filepath.exists():
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._goals = [HealthGoal(**g) for g in data]
        return self._goals

    def update_goal_progress(self, goal_id: str, current_value: float) -> None:
        """Update progress toward a specific goal."""
        for goal in self._goals:
            if goal.goal_id == goal_id:
                goal.current_value = current_value
                if goal.target_value and goal.target_value != 0:
                    goal.progress_pct = min(100, (current_value / goal.target_value) * 100)
                break
        if self._active_user:
            self.save_goals(self._active_user.user_id, self._goals)

    # ═══════════════════════════════════════════════════════════════════════════
    # Health Metrics Tracking
    # ═══════════════════════════════════════════════════════════════════════════

    def _metrics_file_path(self, user_id: str) -> Path:
        return self.data_dir / f"metrics_{user_id}.json"

    def record_metric(self, metric: HealthMetric) -> None:
        """Record a health metric."""
        self._metrics.append(metric)
        # Persist
        filepath = self._metrics_file_path(metric.user_id)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([m.model_dump() for m in self._metrics], f, ensure_ascii=False, indent=2)

        logger.debug(f"Metric recorded: {metric.metric_type}={metric.value} for {metric.user_id}")

    def get_metrics(
        self, user_id: str, metric_type: str | None = None, days: int = 30
    ) -> list[HealthMetric]:
        """Get metrics, optionally filtered by type and time window."""
        cutoff = datetime.now().replace(hour=0, minute=0, second=0)
        from datetime import timedelta
        cutoff = (cutoff - timedelta(days=days)).isoformat()

        filtered = [
            m for m in self._metrics
            if m.user_id == user_id
            and m.timestamp >= cutoff
            and (metric_type is None or m.metric_type == metric_type)
        ]
        return filtered

    def get_progress_summary(self, user_id: str, days: int = 30) -> ProgressSummary:
        """Generate a progress summary for the user."""
        metrics = self.get_metrics(user_id, days=days)
        goals = self._goals

        if not metrics:
            return ProgressSummary(
                user_id=user_id,
                period_start="",
                period_end=datetime.now().isoformat(),
            )

        # Calculate trends
        weights = [m for m in metrics if m.metric_type == "weight"]
        weight_trend = "stable"
        if len(weights) >= 2:
            diff = weights[-1].value - weights[0].value
            if diff < -0.5: weight_trend = "down"
            elif diff > 0.5: weight_trend = "up"

        sleep_metrics = [m for m in metrics if m.metric_type == "sleep_hours"]
        ex_metrics = [m for m in metrics if m.metric_type == "exercise_minutes"]

        # Count metrics by date for adherence
        dates_with_data = len(set(m.date for m in metrics))
        adherence = min(100, (dates_with_data / days) * 100)

        insights = []
        if weight_trend == "down" and any(g.goal_type.value == "lose_weight" for g in goals):
            insights.append("✅ 体重呈下降趋势，减重计划有成效")
        if sleep_metrics:
            avg_sleep = sum(m.value for m in sleep_metrics) / len(sleep_metrics)
            if avg_sleep < 7:
                insights.append(f"⚠️ 平均睡眠 {avg_sleep:.1f}h，仍低于推荐的 7-9 小时")

        return ProgressSummary(
            user_id=user_id,
            period_start=metrics[0].date if metrics else "",
            period_end=datetime.now().isoformat(),
            metrics_recorded=len(metrics),
            goals_achieved=sum(1 for g in goals if g.status == "achieved"),
            goals_total=len(goals),
            weight_trend=weight_trend,
            avg_sleep_hours=sum(m.value for m in sleep_metrics) / len(sleep_metrics) if sleep_metrics else 0,
            avg_exercise_minutes=sum(m.value for m in ex_metrics) / len(ex_metrics) if ex_metrics else 0,
            adherence_score=adherence,
            insights=insights,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Conversation Memory (Short-term + Vector Long-term)
    # ═══════════════════════════════════════════════════════════════════════════

    def add_user_message(self, content: str) -> None:
        """Record a user message in both short-term and vector memory."""
        self.conversation_memory.chat_memory.add_user_message(content)

        # Index in vector memory for cross-session retrieval
        if self._active_user:
            self._index_message(content, "human", self._active_user.user_id)

    def add_ai_message(self, content: str) -> None:
        """Record an AI response in both short-term and vector memory."""
        self.conversation_memory.chat_memory.add_ai_message(content)

        if self._active_user:
            self._index_message(content, "ai", self._active_user.user_id)

    def get_chat_history(self) -> list[dict]:
        """Retrieve full chat history."""
        messages = self.conversation_memory.chat_memory.messages
        return [{"role": m.type, "content": m.content} for m in messages]

    def get_chat_history_text(self) -> str:
        """Retrieve chat history as a formatted string."""
        messages = self.conversation_memory.chat_memory.messages
        lines = []
        for m in messages:
            role = "用户" if m.type == "human" else "助手"
            lines.append(f"{role}: {m.content}")
        return "\n".join(lines)

    def get_relevant_history(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search for relevant past conversations."""
        if not self._active_user:
            return []

        results = self.vector_store.search(
            query=query,
            user_id=self._active_user.user_id,
            memory_type="conversation",
            top_k=top_k,
        )
        return results

    def clear_conversation(self) -> None:
        """Reset conversation buffer (keep user profile and vector memory)."""
        self.conversation_memory.clear()
        logger.info("Conversation buffer cleared")

    # ═══════════════════════════════════════════════════════════════════════════
    # Response History
    # ═══════════════════════════════════════════════════════════════════════════

    def _history_file_path(self, user_id: str) -> Path:
        return self.data_dir / f"history_{user_id}.json"

    def save_response(self, user_id: str, response: AgentResponse) -> None:
        """Archive a full agent response and index it."""
        self._history.append(response.model_dump())

        # Persist
        filepath = self._history_file_path(user_id)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._history[-50:], f, ensure_ascii=False, indent=2)

        # Index in vector memory for future retrieval
        entry = MemoryEntry(
            user_id=user_id,
            content=f"Q: {self.get_chat_history()[-2]['content'] if len(self.get_chat_history()) >= 2 else ''}\nA: {response.message[:500]}",
            memory_type="conversation",
            importance=0.7,
            metadata={
                "agents": list(response.agent_contributions.keys()),
                "confidence": response.confidence_level.value,
                "plans": list(response.plans.keys()) if response.plans else [],
                "timestamp": response.timestamp,
            },
        )
        self.vector_store.add_memory(entry)

    def load_history(self, user_id: str) -> list[dict]:
        """Load archived response history."""
        filepath = self._history_file_path(user_id)
        if not filepath.exists():
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            self._history = json.load(f)
        return self._history

    # ═══════════════════════════════════════════════════════════════════════════
    # Agent Context Builder
    # ═══════════════════════════════════════════════════════════════════════════

    def build_agent_context(self, user_query: str = "") -> dict[str, Any]:
        """
        Build a comprehensive context dict for agents containing:
        - User profile
        - Health goals
        - Recent conversation history
        - Semantically relevant past conversations
        - Learned user preferences
        - Progress summary
        """
        context: dict[str, Any] = {
            "user_profile": self._active_user.to_dict() if self._active_user else None,
            "health_goals": [g.model_dump() for g in self._goals],
            "recent_history": self._history[-5:] if self._history else [],
            "chat_summary": self.get_chat_history_text()[-3000:],
        }

        # Add vector memory context if available
        if user_query and self._active_user:
            try:
                relevant = self.get_relevant_history(user_query, top_k=3)
                if relevant:
                    context["relevant_past_conversations"] = [
                        {"content": r["content"][:500], "score": r.get("relevance_score", 0)}
                        for r in relevant
                    ]

                preferences = self.vector_store.get_user_preferences(
                    self._active_user.user_id
                )
                if preferences:
                    context["learned_preferences"] = preferences
            except Exception as e:
                logger.debug(f"Vector memory context skipped: {e}")

        # Add progress data
        if self._active_user:
            try:
                progress = self.get_progress_summary(self._active_user.user_id)
                context["progress"] = progress.model_dump()
            except Exception as e:
                logger.debug(f"Progress summary skipped: {e}")

        return context

    # ═══════════════════════════════════════════════════════════════════════════
    # Private Helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _index_profile(self, profile: UserProfile) -> None:
        """Index user profile information in vector memory."""
        profile_text = (
            f"用户 {profile.name}，{profile.age}岁，{profile.gender.value}，"
            f"身高{profile.height_cm}cm，体重{profile.weight_kg}kg，BMI {profile.bmi}，"
            f"活动水平：{profile.activity_level.value}，"
            f"平均睡眠：{profile.sleep_hours_avg}h，"
            f"饮食偏好：{', '.join(profile.dietary_preferences) or '无'}，"
            f"过敏：{', '.join(profile.allergies) or '无'}，"
            f"健康状况：{', '.join(profile.medical_conditions) or '无'}，"
            f"运动偏好：{', '.join(profile.exercise_preferences) or '无'}"
        )

        entry = MemoryEntry(
            user_id=profile.user_id,
            content=profile_text,
            memory_type="profile",
            importance=1.0,
            metadata={"category": "user_profile", "bmi": profile.bmi},
        )
        self.vector_store.add_memory(entry)

    def _index_message(self, content: str, role: str, user_id: str) -> None:
        """Index a conversation message in vector memory."""
        # Only index substantive messages (> 20 chars)
        if len(content) < 20:
            return

        entry = MemoryEntry(
            user_id=user_id,
            content=content[:1000],
            memory_type="conversation",
            importance=0.3,
            metadata={
                "role": role,
                "session_id": self._session_id,
                "timestamp": datetime.now().isoformat(),
            },
        )
        try:
            self.vector_store.add_memory(entry)
        except Exception as e:
            logger.debug(f"Message indexing skipped: {e}")

    def run_consolidation(self) -> list[dict]:
        """Run memory consolidation to extract insights."""
        if not self._active_user:
            return []

        try:
            insights = self.vector_store.consolidate_memories(
                self._active_user.user_id
            )
            return [i.model_dump() for i in insights]
        except Exception as e:
            logger.debug(f"Consolidation skipped: {e}")
            return []
