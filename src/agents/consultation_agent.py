"""
Consultation Agent — supervisor/orchestrator that coordinates specialist agents.
Has access to common_tools + routing logic to delegate to Diet/Exercise/Sleep agents.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.models.schemas import (
    AgentRole,
    AgentMessage,
    AgentResponse,
    SupervisorDecision,
)
from src.tools.common_tools import COMMON_TOOLS_SCHEMA, COMMON_TOOLS_MAP

logger = logging.getLogger(__name__)


class ConsultationAgent(BaseAgent):
    """
    健康咨询总控 Agent (Supervisor) — 负责：
    1. 与用户直接对话，了解需求
    2. 分析用户意图，决定路由到哪些专业 Agent
    3. 汇总专业 Agent 的输出，合成最终回复
    4. 管理用户档案和健康目标
    """

    role = AgentRole.CONSULTATION
    tools_schema = COMMON_TOOLS_SCHEMA
    tools_map = COMMON_TOOLS_MAP

    # Routing keywords for intent detection
    ROUTING_KEYWORDS = {
        AgentRole.DIET: [
            "饮食", "吃饭", "食物", "热量", "卡路里", "营养", "减重", "减肥", "增肌",
            "蛋白质", "碳水", "脂肪", "食谱", "meal", "diet", "food", "calorie",
            "nutrition", "weight", "lose weight", "protein", "meal plan", "吃什么",
            "早餐", "午餐", "晚餐", "零食", "食材", "烹饪", "素食", "忌口", "过敏",
        ],
        AgentRole.EXERCISE: [
            "运动", "锻炼", "健身", "跑步", "游泳", "瑜伽", "力量", "有氧",
            "深蹲", "俯卧撑", "哑铃", "workout", "exercise", "fitness", "gym",
            "cardio", "strength", "训练", "HIIT", "拉伸", "热量消耗", "燃脂",
            "瘦身", "塑形", "马甲线", "腹肌", "增肌训练",
        ],
        AgentRole.SLEEP: [
            "睡眠", "失眠", "入睡", "熬夜", "早醒", "打鼾", "sleep", "insomnia",
            "睡不着", "多梦", "易醒", "困", "疲惫", "精力", "作息", "生物钟",
            "午睡", "打盹", "睡前", "起床", "闹钟", "褪黑素",
        ],
    }

    def _build_system_prompt(self, context: dict[str, Any]) -> str:
        user_profile = context.get("user_profile")
        goals = context.get("health_goals", [])

        profile_text = ""
        if user_profile:
            profile_text = f"""
## 当前用户档案
{json.dumps(user_profile, ensure_ascii=False, indent=2)}
"""

        goals_text = ""
        if goals:
            goals_text = "## 用户健康目标\n"
            for g in goals:
                goals_text += f"- [{g.get('goal_type', '')}] {g.get('target_description', '')}\n"

        return f"""你是一个健康咨询总控 Agent（咨询Agent），负责协调饮食、运动、睡眠三个专业 Agent。

## 你的职责
1. **接待用户**：友好地了解用户的健康需求和问题
2. **档案管理**：收集和管理用户的健康档案信息
3. **意图识别**：分析用户的问题属于哪个领域（饮食/运动/睡眠/综合）
4. **任务分发**：将用户问题路由给相应的专业 Agent
5. **结果汇总**：整合各专业 Agent 的输出，给用户一个完整的回复

## 你的工具
- get_user_profile_summary: 查看用户健康档案
- set_health_goals: 设置健康目标
- health_risk_assessment: 进行健康风险评估
- track_progress: 记录健康指标

{profile_text}
{goals_text}

## 路由规则
根据用户问题的关键词判断应调用哪些专业 Agent：
- **饮食相关** → 调用 DietAgent（饮食/营养/热量/食谱/减重饮食）
- **运动相关** → 调用 ExerciseAgent（运动/健身/训练/燃脂）
- **睡眠相关** → 调用 SleepAgent（睡眠/失眠/熬夜/入睡困难）
- **综合问题** → 调用多个 Agent（如"我想变得更健康"）
- **档案/目标管理** → 你自己处理，使用你的工具

## 回复要求
- 使用中文，语气亲切专业，像一位私人健康顾问
- 先理解用户的问题，再做分析和路由
- 当需要调用专业 Agent 时，说明你将如何帮助用户
- 在回复末尾标注 [Consultation Agent]

## 重要
你的回复需要包含一个路由决策的标记（不要展示给用户看），格式为：
<<ROUTE: [diet] [exercise] [sleep]>>
如果需要调用饮食 Agent 就包含 diet，运动包含 exercise，睡眠包含 sleep。
例如：<<ROUTE: [diet] [exercise]>> 表示调用饮食和运动 Agent。
如果不需要路由到任何专业 Agent，标记为：<<ROUTE: []>>
"""

    def detect_route(self, user_message: str) -> SupervisorDecision:
        """
        Analyze user intent and decide which specialist agents to invoke.
        Uses keyword matching + LLM-based intent analysis.
        """
        user_lower = user_message.lower()
        targets: list[AgentRole] = []

        for role, keywords in self.ROUTING_KEYWORDS.items():
            if any(kw in user_lower or kw in user_message for kw in keywords):
                if role not in targets:
                    targets.append(role)

        if not targets:
            # If no keywords match, default to consultation only (no routing)
            return SupervisorDecision(
                should_route=False,
                target_agents=[],
                reasoning="用户问题为一般性咨询，无需调用专业 Agent",
                user_message="让我来帮您分析和解答这个问题。",
            )

        reasoning_parts = []
        for t in targets:
            if t == AgentRole.DIET:
                reasoning_parts.append("涉及饮食营养问题 → 路由到 DietAgent")
            elif t == AgentRole.EXERCISE:
                reasoning_parts.append("涉及运动健身问题 → 路由到 ExerciseAgent")
            elif t == AgentRole.SLEEP:
                reasoning_parts.append("涉及睡眠健康问题 → 路由到 SleepAgent")

        return SupervisorDecision(
            should_route=True,
            target_agents=targets,
            reasoning="\n".join(reasoning_parts),
            user_message=f"我将协调 {len(targets)} 个专业 Agent 来为您提供全面的建议。",
        )

    def synthesize(
        self,
        user_message: str,
        agent_outputs: dict[str, AgentMessage],
        decision: SupervisorDecision,
        user_profile: dict | None = None,
    ) -> AgentResponse:
        """
        Synthesize outputs from specialist agents into a cohesive final response.
        Uses the LLM to weave together multiple agent contributions.
        """
        contributions: dict[str, str] = {}
        plans: dict[str, Any] = {}

        for role_key, agent_msg in agent_outputs.items():
            contributions[role_key] = agent_msg.content
            if agent_msg.plan:
                plans[role_key] = agent_msg.plan

        # Build synthesis prompt
        agent_text = ""
        for role_key, content in contributions.items():
            role_names = {
                "diet": "🥗 饮食 Agent",
                "exercise": "🏃 运动 Agent",
                "sleep": "😴 睡眠 Agent",
            }
            display = role_names.get(role_key, role_key)
            agent_text += f"\n### {display} 的建议：\n{content}\n"

        profile_context = ""
        if user_profile:
            bmi = round(user_profile["weight_kg"] / ((user_profile["height_cm"] / 100) ** 2), 1)
            profile_context = f"用户 BMI: {bmi}, 目标: {user_profile.get('goals', [])}"

        synthesis_prompt = f"""你是一位健康顾问总控，请将以下多个专业 Agent 的建议整合成一个连贯、完整的回复。

## 用户问题
{user_message}

## 用户概况
{profile_context}

## 各专业 Agent 的建议
{agent_text}

## 整合要求
1. 开头友好问候并总结用户的问题
2. 将饮食、运动、睡眠建议有机整合（而非简单拼接）
3. 指出各建议之间的协同关系（如饮食+运动如何配合）
4. 按重要性排序，给出优先级建议
5. 提供具体的下一步行动计划
6. 语气专业温暖，像私人健康顾问
7. 使用 emoji 增强可读性（适度使用）
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是健康顾问总控，用中文回复，语气专业温暖。回复中不要提及 <<ROUTE>> 标记。"},
                    {"role": "user", "content": synthesis_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            synthesized = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            # Fallback: simple concatenation
            parts = [f"### {k}\n{v}" for k, v in contributions.items()]
            synthesized = "## 您的健康建议汇总\n\n" + "\n\n".join(parts)

        # Generate next steps
        next_steps = self._generate_next_steps(
            decision.target_agents, agent_outputs
        )

        return AgentResponse(
            message=synthesized,
            agent_contributions=contributions,
            plans=plans,
            next_steps=next_steps,
            timestamp=datetime.now().isoformat(),
        )

    def _generate_next_steps(
        self,
        target_agents: list[AgentRole],
        agent_outputs: dict[str, AgentMessage],
    ) -> list[str]:
        """Generate suggested next steps for the user."""
        steps = []
        if AgentRole.DIET in target_agents:
            steps.append("📝 记录接下来 3 天的饮食，看看是否符合建议的热量范围")
        if AgentRole.EXERCISE in target_agents:
            steps.append("🏃 按照运动计划完成本周至少 3 次训练，记录实际完成情况")
        if AgentRole.SLEEP in target_agents:
            steps.append("🌙 今晚开始按照睡眠计划调整作息，连续记录 3 天睡眠质量")
        steps.append("📊 一周后回来复查，我可以帮你追踪进展并调整计划")
        return steps if steps else ["如有任何健康问题，随时回来咨询"]
