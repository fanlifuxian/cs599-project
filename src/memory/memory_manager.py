"""
Memory Manager — short-term conversation buffer + long-term JSON file persistence.
Implements the "记忆机制" core technical element.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

from langchain.memory import ConversationBufferMemory
from src.models.schemas import UserProfile, HealthGoal, AgentResponse
from src.config.settings import settings

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages both short-term (in-memory conversation) and long-term
    (file-persisted user profiles and health history) memory.
    """

    def __init__(self, data_dir: str | None = None):
        self.data_dir = Path(data_dir or settings.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Short-term conversation memory (LangChain buffer)
        self.conversation_memory: ConversationBufferMemory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
        )

        # In-memory cache for active user
        self._active_user: Optional[UserProfile] = None
        self._goals: list[HealthGoal] = []
        self._history: list[dict] = []  # Past agent responses summary

        logger.info(f"MemoryManager initialized. Data dir: {self.data_dir}")

    # ── User Profile Persistence ──────────────────────────────────────────

    def _user_file_path(self, user_id: str) -> Path:
        return self.data_dir / f"user_{user_id}.json"

    def save_user_profile(self, profile: UserProfile) -> None:
        """Persist user profile to disk."""
        profile.updated_at = datetime.now().isoformat()
        filepath = self._user_file_path(profile.user_id)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(profile.model_dump(), f, ensure_ascii=False, indent=2)
        self._active_user = profile
        logger.info(f"User profile saved: {profile.user_id}")

    def load_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load user profile from disk if exists."""
        filepath = self._user_file_path(user_id)
        if not filepath.exists():
            logger.info(f"No saved profile for user: {user_id}")
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        profile = UserProfile(**data)
        self._active_user = profile
        logger.info(f"User profile loaded: {user_id}")
        return profile

    def get_active_user(self) -> Optional[UserProfile]:
        return self._active_user

    # ── Health Goals ──────────────────────────────────────────────────────

    def _goals_file_path(self, user_id: str) -> Path:
        return self.data_dir / f"goals_{user_id}.json"

    def save_goals(self, user_id: str, goals: list[HealthGoal]) -> None:
        filepath = self._goals_file_path(user_id)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([g.model_dump() for g in goals], f, ensure_ascii=False, indent=2)
        self._goals = goals

    def load_goals(self, user_id: str) -> list[HealthGoal]:
        filepath = self._goals_file_path(user_id)
        if not filepath.exists():
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._goals = [HealthGoal(**g) for g in data]
        return self._goals

    # ── Conversation Memory ───────────────────────────────────────────────

    def add_user_message(self, content: str) -> None:
        """Record a user message in conversation memory."""
        self.conversation_memory.chat_memory.add_user_message(content)

    def add_ai_message(self, content: str) -> None:
        """Record an AI response in conversation memory."""
        self.conversation_memory.chat_memory.add_ai_message(content)

    def get_chat_history(self) -> list[dict]:
        """Retrieve full chat history as list of role/content dicts."""
        messages = self.conversation_memory.chat_memory.messages
        return [{"role": m.type, "content": m.content} for m in messages]

    def get_chat_history_text(self) -> str:
        """Retrieve chat history as a formatted string for prompts."""
        messages = self.conversation_memory.chat_memory.messages
        lines = []
        for m in messages:
            role = "用户" if m.type == "human" else "助手"
            lines.append(f"{role}: {m.content}")
        return "\n".join(lines)

    def clear_conversation(self) -> None:
        """Reset conversation buffer (keep user profile)."""
        self.conversation_memory.clear()

    # ── Response History ──────────────────────────────────────────────────

    def _history_file_path(self, user_id: str) -> Path:
        return self.data_dir / f"history_{user_id}.json"

    def save_response(self, user_id: str, response: AgentResponse) -> None:
        """Append an agent response to history."""
        self._history.append(response.model_dump())
        filepath = self._history_file_path(user_id)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._history, f, ensure_ascii=False, indent=2)

    def load_history(self, user_id: str) -> list[dict]:
        """Load response history."""
        filepath = self._history_file_path(user_id)
        if not filepath.exists():
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            self._history = json.load(f)
        return self._history

    # ── Summary / Context for Agents ──────────────────────────────────────

    def build_agent_context(self) -> dict[str, Any]:
        """Build a context dict for agents containing user profile, goals, and history."""
        return {
            "user_profile": self._active_user.model_dump() if self._active_user else None,
            "health_goals": [g.model_dump() for g in self._goals],
            "recent_history": self._history[-5:] if self._history else [],  # last 5 responses
            "chat_summary": self.get_chat_history_text()[-2000:],  # last 2000 chars
        }
