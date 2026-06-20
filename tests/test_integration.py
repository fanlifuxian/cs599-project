"""
Integration tests for the health multi-agent platform.
Tests agent interactions, graph execution, and memory persistence.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestGraphState:
    """Test LangGraph state management."""

    def test_state_initialization(self):
        from src.graph.state import HealthGraphState
        state = HealthGraphState(
            messages=[],
            user_input="test query",
            user_profile=None,
            health_goals=[],
            supervisor_decision=None,
            agent_outputs={},
            final_response=None,
            iteration_count=0,
            errors=[],
            reflection_needed=False,
            reflection_notes="",
            quality_gate_passed=False,
            next_step="",
            trace_id="test123",
            token_usage={},
        )
        assert state["user_input"] == "test query"
        assert state["iteration_count"] == 0
        assert state["reflection_needed"] is False

    def test_state_with_profile(self):
        from src.graph.state import HealthGraphState
        profile = {"user_id": "u1", "name": "测试", "age": 30,
                   "weight_kg": 70, "height_cm": 175, "gender": "male"}
        state = HealthGraphState(
            messages=[],
            user_input="test",
            user_profile=profile,
            health_goals=[],
            supervisor_decision=None,
            agent_outputs={},
            final_response=None,
            iteration_count=0,
            errors=[],
            reflection_needed=False,
            reflection_notes="",
            quality_gate_passed=False,
            next_step="",
            trace_id="test123",
            token_usage={},
        )
        assert state["user_profile"]["name"] == "测试"


class TestConsultationRouting:
    """Test consultation agent routing logic."""

    @pytest.fixture
    def consultation(self):
        from src.agents.consultation_agent import ConsultationAgent
        return ConsultationAgent()

    def test_route_diet_only(self, consultation):
        decision = consultation.detect_route("我想知道每天应该摄入多少热量")
        assert decision.should_route is True
        target_values = [a.value for a in decision.target_agents]
        assert "diet" in target_values

    def test_route_exercise_only(self, consultation):
        decision = consultation.detect_route("怎么制定运动计划？")
        assert decision.should_route is True
        target_values = [a.value for a in decision.target_agents]
        assert "exercise" in target_values

    def test_route_sleep_only(self, consultation):
        decision = consultation.detect_route("我失眠了怎么办")
        assert decision.should_route is True
        target_values = [a.value for a in decision.target_agents]
        assert "sleep" in target_values

    def test_route_weight_loss_multi_agent(self, consultation):
        decision = consultation.detect_route("我想科学减重10公斤")
        assert decision.should_route is True
        target_values = [a.value for a in decision.target_agents]
        # Should route to at least diet + exercise for weight loss
        assert "diet" in target_values
        assert "exercise" in target_values

    def test_route_general_consultation(self, consultation):
        decision = consultation.detect_route("你好，介绍一下你自己")
        assert decision.should_route is False
        assert len(decision.target_agents) == 0

    def test_route_critical_safety(self, consultation):
        decision = consultation.detect_route("我最近胸痛呼吸困难")
        assert decision.priority == "urgent"
        # Should NOT route to agents for critical symptoms
        assert decision.should_route is False

    def test_route_complex_health_improvement(self, consultation):
        decision = consultation.detect_route("我想全面改善我的健康状况")
        assert decision.should_route is True
        target_values = [a.value for a in decision.target_agents]
        assert len(target_values) >= 2  # At least 2 agents


class TestMemoryManager:
    """Test the memory manager with file persistence."""

    @pytest.fixture
    def memory(self, tmp_path):
        from src.memory.memory_manager import MemoryManager
        return MemoryManager(data_dir=str(tmp_path))

    def test_save_and_load_profile(self, memory):
        from src.models import UserProfile, Gender, ActivityLevel, BodyMetrics
        profile = UserProfile(
            user_id="test123",
            name="测试用户",
            age=28,
            gender=Gender.MALE,
            current_metrics=BodyMetrics(weight_kg=75, height_cm=178),
            activity_level=ActivityLevel.MODERATELY_ACTIVE,
        )
        memory.save_user_profile(profile)

        loaded = memory.load_user_profile("test123")
        assert loaded is not None
        assert loaded.name == "测试用户"
        assert loaded.bmi == 23.7  # 75 / (1.78^2) = 23.67

    def test_save_and_load_goals(self, memory):
        from src.models import HealthGoal, GoalType
        goals = [
            HealthGoal(goal_type=GoalType.LOSE_WEIGHT, target_description="减重5kg", priority=1),
            HealthGoal(goal_type=GoalType.IMPROVE_SLEEP, target_description="改善睡眠", priority=2),
        ]
        memory.save_goals("test123", goals)

        loaded = memory.load_goals("test123")
        assert len(loaded) == 2
        assert loaded[0].goal_type == GoalType.LOSE_WEIGHT

    def test_conversation_memory(self, memory):
        memory.add_user_message("用户问题")
        memory.add_ai_message("AI回复")

        history = memory.get_chat_history()
        assert len(history) == 2
        assert history[0]["role"] == "human"
        assert history[1]["role"] == "ai"

    def test_clear_conversation(self, memory):
        memory.add_user_message("test1")
        memory.add_ai_message("test2")
        memory.clear_conversation()
        history = memory.get_chat_history()
        assert len(history) == 0

    def test_build_agent_context(self, memory):
        from src.models import UserProfile, Gender, ActivityLevel, BodyMetrics
        profile = UserProfile(
            user_id="ctx_test",
            name="上下文测试",
            age=30,
            gender=Gender.FEMALE,
            current_metrics=BodyMetrics(weight_kg=60, height_cm=165),
        )
        memory.save_user_profile(profile)

        context = memory.build_agent_context(user_query="减重饮食")
        assert context["user_profile"] is not None
        assert context["user_profile"]["name"] == "上下文测试"


class TestRAGKnowledgeBase:
    """Test RAG knowledge retrieval."""

    def test_search_nutrition(self):
        from src.tools.rag_tools import search_health_knowledge
        result = search_health_knowledge("蛋白质摄入量", category="nutrition", top_k=3)
        assert len(result.documents) > 0
        assert any("蛋白质" in doc.title for doc in result.documents)

    def test_search_exercise(self):
        from src.tools.rag_tools import search_health_knowledge
        result = search_health_knowledge("如何开始健身", category="exercise", top_k=2)
        assert len(result.documents) > 0

    def test_search_sleep(self):
        from src.tools.rag_tools import search_health_knowledge
        result = search_health_knowledge("失眠 CBT-I", category="sleep", top_k=2)
        assert len(result.documents) > 0

    def test_search_no_results(self):
        from src.tools.rag_tools import search_health_knowledge
        result = search_health_knowledge("zzzquerythatshouldnotmatchanything", top_k=3)
        # With a truly non-matching query, should return 0 or very few docs
        assert len(result.documents) <= 1  # Allow at most 1 low-relevance match

    def test_knowledge_base_stats(self):
        from src.tools.rag_tools import get_knowledge_base_stats
        stats = get_knowledge_base_stats()
        assert stats["total_documents"] > 10
        assert "nutrition" in stats["categories"]
        assert "exercise" in stats["categories"]
        assert "sleep" in stats["categories"]


class TestMCPTools:
    """Test MCP protocol tools."""

    def test_query_nutrition_guidelines(self):
        from src.tools.mcp_tools import query_health_guidelines
        result = query_health_guidelines("nutrition", "WHO")
        assert result["status"] == "ok"
        assert len(result["key_points"]) > 0

    def test_query_exercise_guidelines(self):
        from src.tools.mcp_tools import query_health_guidelines
        result = query_health_guidelines("exercise", "ACSM")
        assert result["status"] == "ok"

    def test_query_sleep_guidelines(self):
        from src.tools.mcp_tools import query_health_guidelines
        result = query_health_guidelines("sleep", "AASM")
        assert result["status"] == "ok"

    def test_drug_interaction_check(self):
        from src.tools.mcp_tools import check_drug_interactions
        result = check_drug_interactions(
            medications=["华法林", "他汀类"],
            foods=["菠菜", "西柚"],
        )
        assert result["status"] == "warning"
        assert result["interactions_found"] > 0

    def test_no_drug_interaction(self):
        from src.tools.mcp_tools import check_drug_interactions
        result = check_drug_interactions(
            medications=["维生素C"],
            foods=["苹果"],
        )
        assert result["interactions_found"] == 0

    def test_medical_reference_bmi(self):
        from src.tools.mcp_tools import get_medical_reference
        result = get_medical_reference("bmi", 22.0)
        assert result["status"] == "ok"
        assert result["category"] == "正常"


class TestEvaluationFramework:
    """Test agent evaluation framework."""

    def test_eval_case_creation(self):
        from src.models import EvalCase, AgentRole
        case = EvalCase(
            case_id="test_001",
            category="diet",
            user_input="减重怎么吃？",
            expected_topics=["减重", "饮食", "热量"],
            expected_agents=[AgentRole.DIET],
        )
        assert case.case_id == "test_001"
        assert len(case.expected_topics) == 3

    def test_evaluator_has_cases(self):
        from src.observability.evaluator import get_evaluator
        evaluator = get_evaluator()
        assert len(evaluator.cases) >= 10  # At least 10 benchmark cases

    def test_eval_result_scoring(self):
        from src.models import EvalResult
        result = EvalResult(
            case_id="test",
            passed=True,
            actual_topics_covered=["减重", "饮食"],
            response_quality_score=0.85,
        )
        assert result.passed is True
        assert result.response_quality_score == 0.85


class TestSettings:
    """Test configuration management."""

    def test_settings_loaded(self):
        from src.config.settings import settings
        assert settings.llm_provider in ("deepseek", "openai", "anthropic", "ollama")
        assert settings.temperature >= 0.0
        assert settings.max_tokens > 0

    def test_no_hardcoded_api_keys(self):
        from src.config.settings import settings
        # API key should come from env, not hardcoded
        import inspect
        source = inspect.getsource(settings.__class__)
        assert "sk-" not in source  # No hardcoded keys in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
