"""
Intent Router — 用户意图分析与 Agent 路由决策。

职责单一：仅负责分析用户输入并决定调用哪些 Agent。
不涉及 LLM 调用、提示词构建或回复合成。
"""

from __future__ import annotations

import logging

from src.models import AgentRole, SupervisorDecision

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Keyword Registry — 每个 Agent 的关键词表
# ═══════════════════════════════════════════════════════════════════════════════

ROUTING_KEYWORDS: dict[AgentRole, list[str]] = {
    AgentRole.DIET: [
        "饮食", "吃饭", "食物", "热量", "卡路里", "营养", "减重", "减肥", "增肌",
        "蛋白质", "碳水", "脂肪", "食谱", "meal", "diet", "food", "calorie",
        "nutrition", "weight", "lose weight", "protein", "meal plan", "吃什么",
        "早餐", "午餐", "晚餐", "零食", "食材", "烹饪", "素食", "忌口", "过敏",
        "节食", "暴食", "食欲", "饥饿", "饱腹", "代餐", "补剂", "维生素",
        "矿物质", "膳食纤维", "血糖", "血脂", "胆固醇", "BMR", "TDEE",
        "碳水循环", "生酮", "低碳", "断食", "轻断食", "辟谷",
    ],
    AgentRole.EXERCISE: [
        "运动", "锻炼", "健身", "跑步", "游泳", "瑜伽", "力量", "有氧",
        "深蹲", "俯卧撑", "哑铃", "workout", "exercise", "fitness", "gym",
        "cardio", "strength", "训练", "HIIT", "拉伸", "热量消耗", "燃脂",
        "瘦身", "塑形", "马甲线", "腹肌", "增肌训练", "举铁", "硬拉",
        "卧推", "跳绳", "骑行", "登山", "马拉松", "私教", "热身", "拉伤",
        "肌肉酸痛", "平台期", "体能", "心率", "核心", "普拉提", "CrossFit",
    ],
    AgentRole.SLEEP: [
        "睡眠", "失眠", "入睡", "熬夜", "早醒", "打鼾", "sleep", "insomnia",
        "睡不着", "多梦", "易醒", "困", "疲惫", "精力", "作息", "生物钟",
        "午睡", "打盹", "睡前", "起床", "闹钟", "褪黑素", "安眠药",
        "夜猫子", "倒时差", "夜班", "昼夜颠倒", "浅眠", "噩梦", "梦游",
        "睡眠呼吸暂停", "打呼", "嗜睡", "过度睡眠", "休息", "放松",
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Agent Synergy Triggers — 某些关键词自动触发多 Agent 协作
# ═══════════════════════════════════════════════════════════════════════════════

SYNERGY_TRIGGERS: dict[str, list[AgentRole]] = {
    "减重": [AgentRole.DIET, AgentRole.EXERCISE],
    "减肥": [AgentRole.DIET, AgentRole.EXERCISE],
    "增肌": [AgentRole.DIET, AgentRole.EXERCISE],
    "塑形": [AgentRole.DIET, AgentRole.EXERCISE],
    "健康": [AgentRole.DIET, AgentRole.EXERCISE, AgentRole.SLEEP],
    "改善": [AgentRole.DIET, AgentRole.EXERCISE, AgentRole.SLEEP],
    "全面": [AgentRole.DIET, AgentRole.EXERCISE, AgentRole.SLEEP],
    "累": [AgentRole.SLEEP, AgentRole.EXERCISE],
    "压力": [AgentRole.EXERCISE, AgentRole.SLEEP],
    "疲惫": [AgentRole.SLEEP, AgentRole.EXERCISE],
}

# ═══════════════════════════════════════════════════════════════════════════════
# Critical Safety Keywords — 立即建议就医，不自行处理
# ═══════════════════════════════════════════════════════════════════════════════

CRITICAL_KEYWORDS: list[str] = [
    "胸痛", "呼吸困难", "晕倒", "昏厥", "剧烈疼痛", "吐血", "便血",
    "自杀", "自残", "伤害自己", "伤害他人", "chest pain", "suicide",
    "过敏反应", "休克", "中毒",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Router Class
# ═══════════════════════════════════════════════════════════════════════════════

class IntentRouter:
    """
    意图路由器 — 分析用户消息并生成路由决策。

    决策流程：
    1. 安全筛查（危急关键词 → 立即就医）
    2. 关键词匹配（计算每个 Agent 的得分）
    3. 协同触发检查（某些词自动触发多 Agent）
    4. 决策汇总（是否需要路由、目标 Agent 列表）
    """

    def __init__(
        self,
        keywords: dict[AgentRole, list[str]] | None = None,
        synergy_triggers: dict[str, list[AgentRole]] | None = None,
        critical_keywords: list[str] | None = None,
    ):
        self.keywords = keywords or ROUTING_KEYWORDS
        self.synergy_triggers = synergy_triggers or SYNERGY_TRIGGERS
        self.critical_keywords = critical_keywords or CRITICAL_KEYWORDS

    def route(self, user_message: str) -> SupervisorDecision:
        """
        分析用户意图，返回路由决策。

        Args:
            user_message: 用户的原始输入

        Returns:
            SupervisorDecision 包含是否路由、目标Agent列表、推理说明
        """
        # Step 1: 安全筛查
        if self._is_critical(user_message):
            return SupervisorDecision(
                should_route=False,
                target_agents=[],
                reasoning="检测到危急症状关键词，建议用户立即就医",
                user_message="⚠️ 根据您的描述，建议您立即就医或拨打急救电话。",
                priority="urgent",
            )

        # Step 2: 关键词得分
        scores: dict[AgentRole, int] = {}
        for role, keywords in self.keywords.items():
            score = sum(1 for kw in keywords if kw in user_message)
            if score > 0:
                scores[role] = score

        targets = list(scores.keys())

        # Step 3: 协同触发
        for trigger_word, syn_agents in self.synergy_triggers.items():
            if trigger_word in user_message:
                for agent in syn_agents:
                    if agent not in targets:
                        targets.append(agent)

        # Step 4: 决策
        if not targets:
            return SupervisorDecision(
                should_route=False,
                target_agents=[],
                reasoning="用户问题为一般性健康咨询，自行处理",
                user_message="我来帮您分析和解答。",
                priority="routine",
            )

        reasoning = self._build_reasoning(targets)
        requires_reflection = len(targets) >= 2

        return SupervisorDecision(
            should_route=True,
            target_agents=targets,
            reasoning=reasoning,
            user_message=f"我将协调 {len(targets)} 位健康专家为您服务...",
            priority="routine",
            requires_reflection=requires_reflection,
        )

    # ── Private helpers ──────────────────────────────────────────────────

    def _is_critical(self, message: str) -> bool:
        """检查是否包含危急关键词。"""
        return any(kw in message for kw in self.critical_keywords)

    @staticmethod
    def _build_reasoning(targets: list[AgentRole]) -> str:
        """构建路由推理说明。"""
        labels = {
            AgentRole.DIET: "饮食营养 → DietAgent",
            AgentRole.EXERCISE: "运动健身 → ExerciseAgent",
            AgentRole.SLEEP: "睡眠健康 → SleepAgent",
        }
        return "\n".join(f"涉及{labels.get(t, str(t))}" for t in targets)


# Module-level singleton
_router: IntentRouter | None = None


def get_router() -> IntentRouter:
    global _router
    if _router is None:
        _router = IntentRouter()
    return _router
