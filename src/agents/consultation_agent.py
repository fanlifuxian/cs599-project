"""
Consultation Agent — 健康咨询总控 (Supervisor)。

职责单一：
- 路由委托给 routing/router.py (IntentRouter)
- 提示词委托给 prompts/manager.py (PromptManager)
- 仅保留核心职责：合成回复 + 质量反思
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.agents.base_agent import BaseAgent
from src.models import (
    AgentRole, AgentMessage, AgentResponse, SupervisorDecision, ConfidenceLevel,
)
from src.prompts.manager import get_prompt_manager
from src.routing.router import get_router
from src.tools.common_tools import COMMON_TOOLS_SCHEMA, COMMON_TOOLS_MAP
from src.tools.mcp_tools import MCP_TOOLS_SCHEMA, MCP_TOOLS_MAP
from src.tools.rag_tools import RAG_TOOLS_SCHEMA, RAG_TOOLS_MAP

logger = logging.getLogger(__name__)


class ConsultationAgent(BaseAgent):
    """
    健康咨询总控 Agent — 意图分析 → 路由分发 → 结果合成 → 质量反思。

    核心职责：合成与质量管理。路由逻辑在 IntentRouter，提示词在 PromptManager。
    """

    role = AgentRole.CONSULTATION
    tools_schema = COMMON_TOOLS_SCHEMA + MCP_TOOLS_SCHEMA + RAG_TOOLS_SCHEMA
    tools_map = {**COMMON_TOOLS_MAP, **MCP_TOOLS_MAP, **RAG_TOOLS_MAP}

    def _build_system_prompt(self, context: dict[str, Any]) -> str:
        return get_prompt_manager().build_consultation_prompt(context)

    # ═══════════════════════════════════════════════════════════════════════
    # 路由（委托给 IntentRouter）
    # ═══════════════════════════════════════════════════════════════════════

    def detect_route(self, user_message: str) -> SupervisorDecision:
        """分析用户意图并决定路由 — 委托给 IntentRouter。"""
        return get_router().route(user_message)

    # ═══════════════════════════════════════════════════════════════════════
    # 合成回复（核心职责）
    # ═══════════════════════════════════════════════════════════════════════

    def synthesize(
        self,
        user_message: str,
        agent_outputs: dict[str, AgentMessage],
        decision: SupervisorDecision,
        user_profile: dict | None = None,
    ) -> AgentResponse:
        """融合多 Agent 输出为连贯的最终回复。"""
        contributions = {k: v.content for k, v in agent_outputs.items()}
        plans = {k: v.plan for k, v in agent_outputs.items() if v.plan}
        max_confidence = max((v.confidence for v in agent_outputs.values()), default=0.7)

        agent_text = self._format_agent_contributions(contributions)
        profile_ctx = self._format_profile_context(user_profile)

        synthesis_prompt = f"""你是一位资深健康顾问总控，请将以下多位专业健康专家的建议整合成一个连贯、完整、个性化的回复。

## 用户原始问题
{user_message}

## 用户概况
{profile_ctx}

## 各专业专家的建议
{agent_text}

## 整合要求
1. **温暖开场**：共情用户的问题，简述你将提供的帮助
2. **有机整合**：将饮食、运动、睡眠建议融为一体，说明它们之间的协同关系
3. **优先级排序**：按对用户最重要的先讲，用清晰的小标题引导
4. **具体可执行**：每个建议都要具体到「做什么、做多少、什么时候做」
5. **科学依据**：适当引用专家的计算数据和循证依据
6. **协同效应**：指出饮食+运动如何配合、运动如何改善睡眠等跨领域联系
7. **行动清单**：给出 3-5 条本周可以开始的行动
8. **适度使用 emoji**：增强可读性但不过度
9. **免责声明**：如有健康问题请咨询专业医生
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是私人健康顾问总控。用中文回复，温暖专业。回复应是有机整体而非分块罗列。每段体现跨领域协同。"},
                    {"role": "user", "content": synthesis_prompt},
                ],
                temperature=0.7,
                max_tokens=self.max_tokens,
            )
            synthesized = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            synthesized = self._manual_synthesize(contributions)

        next_steps = self._generate_next_steps(decision.target_agents)
        health_alerts = self._check_alerts(user_profile)

        confidence = ConfidenceLevel.HIGH if max_confidence >= 0.85 else (
            ConfidenceLevel.MEDIUM if max_confidence >= 0.6 else ConfidenceLevel.LOW
        )

        return AgentResponse(
            message=synthesized,
            agent_contributions=contributions,
            plans=plans,
            next_steps=next_steps,
            health_alerts=health_alerts,
            confidence_level=confidence,
            timestamp=datetime.now().isoformat(),
        )

    # ═══════════════════════════════════════════════════════════════════════
    # 反思（质量门禁）
    # ═══════════════════════════════════════════════════════════════════════

    def reflect(
        self,
        response: AgentResponse,
        user_message: str,
        agent_outputs: dict[str, AgentMessage],
    ) -> tuple[AgentResponse, bool]:
        """审查回复质量 — 置信度、完整性、一致性。"""
        if response.confidence_level == ConfidenceLevel.HIGH:
            return response, True

        reflection_prompt = f"""请审查以下健康建议回复的质量。

## 用户问题
{user_message}

## 生成的回复
{response.message}

## 审查标准
1. 是否完整回应用户的所有问题？
2. 建议之间是否存在矛盾？
3. 是否有具体可执行的行动步骤？
4. 是否遗漏了重要的安全提醒？

请用中文回复，格式为：
PASS: <yes/no>
IMPROVEMENTS: <如需改进，简要说明>
"""

        try:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是质量审查专家。严格但简洁地审查回复质量。"},
                    {"role": "user", "content": reflection_prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            text = result.choices[0].message.content or ""
            if "PASS: yes" in text or "PASS: Yes" in text:
                return response, True
            improvements = text.split("IMPROVEMENTS:")[-1].strip() if "IMPROVEMENTS:" in text else ""
            if improvements:
                response.message += f"\n\n---\n💡 **补充提示**：{improvements}"
            return response, True
        except Exception as e:
            logger.warning(f"Reflection failed: {e}")
            return response, True

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _format_agent_contributions(contributions: dict[str, str]) -> str:
        roles = {"diet": "🥗 饮食专家", "exercise": "🏃 运动专家", "sleep": "😴 睡眠专家"}
        parts = []
        for key, content in contributions.items():
            parts.append(f"\n### {roles.get(key, key)}的建议：\n{content}")
        return "\n".join(parts)

    @staticmethod
    def _format_profile_context(profile: dict | None) -> str:
        if not profile:
            return "无用户档案"
        bmi = round(profile["weight_kg"] / ((profile["height_cm"] / 100) ** 2), 1)
        return f"用户 {profile.get('age', '?')}岁, BMI {bmi}, 目标: {profile.get('goals', [])}"

    @staticmethod
    def _check_alerts(profile: dict | None) -> list[str]:
        alerts = []
        if not profile:
            return alerts
        bmi = round(profile["weight_kg"] / ((profile["height_cm"] / 100) ** 2), 1)
        if bmi >= 28:
            alerts.append(f"⚠️ BMI 为 {bmi}，属于肥胖范围，建议制定系统减重计划")
        if profile.get("sleep_hours_avg", 7) < 6:
            alerts.append("⚠️ 平均睡眠不足6小时，长期会增加健康风险")
        return alerts

    @staticmethod
    def _generate_next_steps(target_agents: list[AgentRole]) -> list[str]:
        steps = []
        if AgentRole.DIET in target_agents:
            steps.append("🥗 用3天时间记录饮食日记，熟悉你当前的饮食模式")
        if AgentRole.EXERCISE in target_agents:
            steps.append("🏃 本周完成至少2次训练，从低强度开始，关注身体感受")
        if AgentRole.SLEEP in target_agents:
            steps.append("🌙 今晚开始固定起床时间，连续记录3天睡眠日志")
        if len(target_agents) >= 2:
            steps.append("📊 一周后回来复诊，对比数据看改善效果")
        return steps if steps else ["💬 有任何健康问题随时回来咨询"]

    @staticmethod
    def _manual_synthesize(contributions: dict[str, str]) -> str:
        """LLM 合成失败时的手动回退。"""
        roles = {"diet": "🥗 饮食专家", "exercise": "🏃 运动专家", "sleep": "😴 睡眠专家"}
        parts = [f"### {roles.get(k, k)}\n{v}" for k, v in contributions.items()]
        return "## 您的个性化健康建议\n\n" + "\n\n---\n\n".join(parts)
