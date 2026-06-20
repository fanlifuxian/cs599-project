"""
ChromaDB Vector Store — semantic long-term memory for the health multi-agent platform.

Provides:
- Semantic search over conversation history
- User preference learning & retrieval
- Health knowledge embedding & retrieval
- Memory importance scoring
- Automatic memory consolidation
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.config.settings import settings
from src.models import MemoryEntry

logger = logging.getLogger(__name__)


class HealthVectorStore:
    """
    ChromaDB-backed vector store for semantic health memory.

    Stores and retrieves:
    - Conversation history (semantic search)
    - User preferences (learned over time)
    - Health insights (extracted patterns)
    - Knowledge base references
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_dir: str | None = None,
    ):
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_dir = persist_dir or settings.chroma_persist_dir

        self._client = None
        self._collection = None
        self._embedding_fn = None
        self._initialized = False

        try:
            self._initialize()
        except Exception as e:
            logger.warning(f"ChromaDB initialization deferred: {e}")
            logger.warning("Vector memory will use fallback (in-memory) mode")

    def _initialize(self):
        """Lazy-initialize ChromaDB with embedding function."""
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Try to get or create the collection
        try:
            self._collection = self._client.get_collection(
                name=self.collection_name,
                embedding_function=self._get_embedding_fn(),
            )
        except Exception:
            self._collection = self._client.create_collection(
                name=self.collection_name,
                embedding_function=self._get_embedding_fn(),
                metadata={"description": "Health multi-agent memory store"},
            )

        self._initialized = True
        logger.info(
            f"Vector store initialized: collection='{self.collection_name}', "
            f"count={self._collection.count()}"
        )

    def _get_embedding_fn(self):
        """Get or create the embedding function."""
        if self._embedding_fn is not None:
            return self._embedding_fn

        try:
            from chromadb.utils import embedding_functions
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model,
            )
            logger.info(f"Embedding model loaded: {settings.embedding_model}")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer: {e}, using default")
            from chromadb.utils import embedding_functions
            self._embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        return self._embedding_fn

    @property
    def is_available(self) -> bool:
        return self._initialized and self._collection is not None

    # ── CRUD Operations ────────────────────────────────────────────────────

    def add_memory(self, entry: MemoryEntry) -> str:
        """Add a memory entry to the vector store."""
        if not self.is_available:
            self._add_fallback(entry)
            return entry.entry_id

        metadata = {
            "user_id": entry.user_id,
            "memory_type": entry.memory_type,
            "importance": entry.importance,
            "created_at": entry.created_at,
            **entry.metadata,
        }

        try:
            self._collection.add(
                ids=[entry.entry_id],
                documents=[entry.content],
                metadatas=[metadata],
            )
            logger.debug(f"Memory added: {entry.entry_id} ({entry.memory_type})")
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            self._add_fallback(entry)

        return entry.entry_id

    def add_memories(self, entries: list[MemoryEntry]) -> list[str]:
        """Batch-add memory entries."""
        if not self.is_available:
            for e in entries:
                self._add_fallback(e)
            return [e.entry_id for e in entries]

        if not entries:
            return []

        ids = [e.entry_id for e in entries]
        documents = [e.content for e in entries]
        metadatas = [
            {
                "user_id": e.user_id,
                "memory_type": e.memory_type,
                "importance": e.importance,
                "created_at": e.created_at,
                **e.metadata,
            }
            for e in entries
        ]

        try:
            self._collection.add(ids=ids, documents=documents, metadatas=metadatas)
        except Exception as e:
            logger.error(f"Batch add failed: {e}")
            for entry in entries:
                self._add_fallback(entry)

        return ids

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        top_k: int | None = None,
        min_importance: float = 0.0,
    ) -> list[dict]:
        """Semantic search over memory entries."""
        top_k = top_k or settings.vector_memory_top_k

        if not self.is_available:
            return self._search_fallback(query, user_id, top_k)

        # Build filter
        where_filter = None
        conditions = []
        if user_id:
            conditions.append({"user_id": user_id})
        if memory_type:
            conditions.append({"memory_type": memory_type})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
            )

            formatted = []
            if results and results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    doc = results["documents"][0][i] if results["documents"] else ""
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results.get("distances") else 1.0

                    # Filter by importance
                    if meta.get("importance", 0.5) < min_importance:
                        continue

                    formatted.append({
                        "entry_id": doc_id,
                        "content": doc,
                        "metadata": meta,
                        "relevance_score": round(1.0 - min(distance, 1.0), 4),
                    })

            return formatted

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return self._search_fallback(query, user_id, top_k)

    def search_similar_memories(
        self,
        content: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        """Search and return typed MemoryEntry objects."""
        results = self.search(content, user_id=user_id, top_k=top_k)
        entries = []
        for r in results:
            entries.append(MemoryEntry(
                entry_id=r["entry_id"],
                user_id=r["metadata"].get("user_id", ""),
                content=r["content"],
                metadata=r["metadata"],
                memory_type=r["metadata"].get("memory_type", "conversation"),
                importance=r["metadata"].get("importance", 0.5),
                relevance_score=r["relevance_score"],
                created_at=r["metadata"].get("created_at", ""),
            ))
        return entries

    def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user."""
        if not self.is_available:
            return self._delete_fallback(user_id)

        try:
            results = self._collection.get(where={"user_id": user_id})
            if results and results["ids"]:
                self._collection.delete(ids=results["ids"])
                count = len(results["ids"])
                logger.info(f"Deleted {count} memories for user {user_id}")
                return count
        except Exception as e:
            logger.error(f"Delete failed: {e}")
        return 0

    def get_collection_stats(self) -> dict:
        """Get statistics about the vector collection."""
        if not self.is_available:
            return {"status": "unavailable", "count": len(self._fallback_store)}

        try:
            return {
                "status": "available",
                "collection": self.collection_name,
                "count": self._collection.count(),
                "persist_dir": self.persist_dir,
            }
        except Exception:
            return {"status": "error", "count": 0}

    # ── Smart Memory Operations ─────────────────────────────────────────────

    def consolidate_memories(self, user_id: str) -> list[MemoryEntry]:
        """
        Consolidate related memories into insights.
        Finds patterns across conversations and generates summary insights.
        """
        # Get recent conversation memories
        recent = self.search(
            query="health goals progress preferences",
            user_id=user_id,
            memory_type="conversation",
            top_k=20,
        )

        if len(recent) < 3:
            return []

        # Extract key themes (simple TF-based approach)
        insights: list[MemoryEntry] = []
        all_text = " ".join(r["content"] for r in recent)

        # Check for recurring topics
        topic_keywords = {
            "weight": ["减重", "减肥", "体重", "瘦", "weight", "kg", "公斤"],
            "sleep": ["睡眠", "失眠", "入睡", "熬夜", "sleep"],
            "exercise": ["运动", "锻炼", "健身", "跑步", "exercise", "workout"],
            "diet": ["饮食", "吃饭", "食物", "热量", "diet", "calorie", "卡路里"],
            "stress": ["压力", "焦虑", "紧张", "stress", "anxiety"],
        }

        for topic, keywords in topic_keywords.items():
            count = sum(1 for kw in keywords if kw in all_text)
            if count >= 3:
                insights.append(MemoryEntry(
                    user_id=user_id,
                    content=f"用户频繁讨论{topic}相关话题（出现{count}次关键词匹配）",
                    memory_type="insight",
                    importance=min(0.9, 0.5 + count * 0.1),
                    metadata={"topic": topic, "keyword_count": count},
                ))

        # Store insights back
        if insights:
            self.add_memories(insights)

        return insights

    def get_user_preferences(self, user_id: str) -> dict[str, Any]:
        """Extract learned user preferences from memory."""
        preferences = self.search(
            query="user preferences dietary exercise sleep habits likes dislikes",
            user_id=user_id,
            memory_type="preference",
            top_k=10,
        )

        if not preferences:
            return {}

        prefs: dict[str, list[str]] = {}
        for p in preferences:
            meta = p.get("metadata", {})
            category = meta.get("category", "general")
            if category not in prefs:
                prefs[category] = []
            prefs[category].append(p["content"])

        return prefs

    # ── Fallback (in-memory) Store ──────────────────────────────────────────

    _fallback_store: list[dict] = []

    def _add_fallback(self, entry: MemoryEntry):
        self._fallback_store.append({
            "entry_id": entry.entry_id,
            "user_id": entry.user_id,
            "content": entry.content,
            "metadata": {
                "memory_type": entry.memory_type,
                "importance": entry.importance,
                "created_at": entry.created_at,
                **entry.metadata,
            },
        })
        # Keep fallback store bounded
        if len(self._fallback_store) > 1000:
            self._fallback_store = self._fallback_store[-500:]

    def _search_fallback(
        self, query: str, user_id: str | None, top_k: int
    ) -> list[dict]:
        """Simple keyword-based fallback search."""
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for entry in self._fallback_store:
            if user_id and entry.get("user_id") != user_id:
                continue

            content = entry.get("content", "").lower()
            # Simple TF scoring
            score = sum(1 for w in query_words if w in content)
            score += 2 * (1 if any(w in content for w in query_words) else 0)

            if score > 0:
                results.append({
                    "entry_id": entry["entry_id"],
                    "content": entry["content"],
                    "metadata": entry["metadata"],
                    "relevance_score": min(score / (len(query_words) + 2), 1.0),
                })

        results.sort(key=lambda r: r["relevance_score"], reverse=True)
        return results[:top_k]

    def _delete_fallback(self, user_id: str) -> int:
        before = len(self._fallback_store)
        self._fallback_store = [
            e for e in self._fallback_store
            if e.get("user_id") != user_id
        ]
        return before - len(self._fallback_store)


# Module-level singleton
_vector_store: Optional[HealthVectorStore] = None


def get_vector_store() -> HealthVectorStore:
    """Get or create the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = HealthVectorStore()
    return _vector_store
