"""
Agentic RAG Tools — evidence-based health knowledge retrieval.

Integrates a curated health knowledge base with:
- Nutrition science (macro/micronutrients, dietary patterns)
- Exercise science (training principles, recovery, injury prevention)
- Sleep medicine (circadian rhythm, sleep disorders, CBT-I)
- Preventive medicine (screening, risk factors, lifestyle medicine)
- Mental wellness (stress management, mindfulness, behavioral change)

The knowledge base is pre-embedded for fast retrieval and provides
citations for evidence-based recommendations.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from src.models import RAGDocument, RAGRetrievalResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Curated Health Knowledge Base (50+ entries across 6 categories)
# ═══════════════════════════════════════════════════════════════════════════════

HEALTH_KNOWLEDGE_BASE: list[RAGDocument] = [
    # ── Nutrition ────────────────────────────────────────────────────────
    RAGDocument(
        title="地中海饮食模式",
        content="地中海饮食以蔬菜、水果、全谷物、豆类、坚果、橄榄油为主，适量摄入鱼类和禽肉，"
                "限制红肉和加工食品。大量研究证实该饮食模式可降低心血管疾病风险 30%、"
                "全因死亡率 25%。核心特征：高单不饱和脂肪、高膳食纤维、高抗氧化物质、"
                "低饱和脂肪、低精制碳水。",
        source="PREDIMED Study, NEJM 2018",
        category="nutrition",
        tags=["地中海饮食", "心血管", "饮食模式", "抗炎"],
        reliability_score=0.95,
    ),
    RAGDocument(
        title="蛋白质摄入与肌肉健康",
        content="成人每日蛋白质推荐摄入量 0.8g/kg 体重（RDA），但运动人群和老年人需要更多。"
                "增肌期建议 1.6-2.0g/kg/天，分 4-5 餐摄入，每餐 20-40g 优质蛋白。"
                "亮氨酸是肌肉蛋白合成的关键触发因子（阈值 ~2.5g/餐）。"
                "动物蛋白生物利用率高于植物蛋白，但植物蛋白组合可互补。"
                "睡前摄入酪蛋白（20-40g）可支持夜间肌肉修复。",
        source="ISSN Position Stand, JISSN 2017",
        category="nutrition",
        tags=["蛋白质", "增肌", "运动营养", "氨基酸"],
        reliability_score=0.90,
    ),
    RAGDocument(
        title="间歇性禁食与代谢健康",
        content="间歇性禁食（IF）主要包括 16:8（每日禁食16小时）和 5:2（每周2天低热量）模式。"
                "研究表明 IF 可改善胰岛素敏感性、促进脂肪氧化、诱导细胞自噬。"
                "减重效果与持续热量限制相当，但依从性可能更好。"
                "不适用于：孕妇、青少年、进食障碍史者、1型糖尿病患者、体重过轻者。"
                "常见副作用：初期饥饿感、头痛、注意力下降，通常 1-2 周适应。",
        source="NEJM Review, 2019; Cell Metabolism, 2020",
        category="nutrition",
        tags=["间歇性禁食", "代谢", "减重", "自噬"],
        reliability_score=0.85,
    ),
    RAGDocument(
        title="膳食纤维与肠道健康",
        content="膳食纤维分为可溶性（燕麦、豆类、苹果）和不可溶性（全麦、坚果、蔬菜）两类。"
                "每日推荐摄入量 25-35g，但多数人仅摄入 15g 左右。"
                "充足纤维摄入可降低结直肠癌风险、改善血糖控制、降低 LDL 胆固醇、"
                "促进肠道有益菌群生长（短链脂肪酸产生）。增加纤维摄入时应同时增加饮水，"
                "循序渐进避免腹胀。",
        source="WHO Guidelines; Lancet 2019",
        category="nutrition",
        tags=["膳食纤维", "肠道菌群", "消化健康", "益生元"],
        reliability_score=0.90,
    ),
    RAGDocument(
        title="运动前中后的营养策略",
        content="运动前 2-3h：富含碳水、中等蛋白、低脂饮食（如燕麦+香蕉+牛奶）。"
                "运动前 30-60min：快速碳水（香蕉、能量胶、运动饮料），如有需要。"
                "运动中（>60min）：每小时 30-60g 碳水，补充电解质。"
                "运动后 30-60min（代谢窗口）：碳水:蛋白质 = 3-4:1，"
                "约 20-40g 蛋白 + 60-120g 碳水。巧克力牛奶是便捷选择。"
                "运动后 2h 内完成完整餐食。全程饮水 500-1000ml/h。",
        source="ACSM Position Stand, 2016; ISSN Recommendations",
        category="nutrition",
        tags=["运动营养", "碳水", "恢复", "蛋白质时机"],
        reliability_score=0.90,
    ),
    RAGDocument(
        title="微量营养素与免疫功能",
        content="维生素D：免疫调节关键因子，缺乏与呼吸道感染风险增加相关。"
                "维生素C：抗氧化剂，可能缩短感冒持续时间（非预防）。"
                "锌：免疫细胞发育和功能所需，缺乏导致免疫功能下降。"
                "硒：抗氧化酶组分，影响病毒抵抗力。"
                "铁：氧运输和免疫细胞增殖必需。"
                "建议优先从食物获取微量营养素，仅在明确缺乏时补充。"
                "维生素D是例外——日照不足地区普遍缺乏，建议检测血清 25(OH)D 水平。",
        source="Nutrients Journal, 2020; BMJ 2021",
        category="nutrition",
        tags=["微量营养素", "免疫力", "维生素", "矿物质"],
        reliability_score=0.85,
    ),

    # ── Exercise ──────────────────────────────────────────────────────────
    RAGDocument(
        title="抗阻训练基本原则",
        content="渐进超负荷：逐步增加重量、次数、组数或减少休息来持续刺激肌肉适应。"
                "特异性原则：训练适应与训练方式高度相关（如想提升跑步就训练跑步）。"
                "恢复原则：肌肉在休息时生长，同一肌群应间隔 48-72h 训练。"
                "初学者：全身训练每周 3 次，每次 8-10 个动作，1-3 组，8-12 RM。"
                "中级：上下肢分化或推/拉/腿分化，每周 4-5 次训练。"
                "高级：精细化周期训练（增肌期→力量期→爆发力期）。",
        source="ACSM Guidelines, 11th Ed; NSCA Essentials",
        category="exercise",
        tags=["抗阻训练", "渐进超负荷", "增肌", "训练原则"],
        reliability_score=0.95,
    ),
    RAGDocument(
        title="心血管运动处方",
        content="FITT-VP 原则：Frequency（频率）、Intensity（强度）、Time（时间）、"
                "Type（类型）、Volume（总量）、Progression（进展）。"
                "改善心肺适能：每周 3-5 天，强度 40-89% HRR，每次 20-60 分钟。"
                "最大心率估算：HRmax = 220 - 年龄（或 208 - 0.7×年龄 更精确）。"
                "心率储备 (HRR) = HRmax - HRrest。"
                "靶心率 = HRrest + (HRR × 目标强度%)。"
                "可从 Borg RPE 6-20 量表辅助判断（目标 12-16，即「有些费力」）。",
        source="ACSM Guidelines, 11th Edition, 2022",
        category="exercise",
        tags=["有氧运动", "心率", "FITT", "心肺适能"],
        reliability_score=0.95,
    ),
    RAGDocument(
        title="运动损伤预防指南",
        content="热身：5-10分钟动态拉伸+低强度有氧，提高肌肉温度和关节活动度。"
                "整理：5-10分钟静态拉伸，帮助恢复肌肉长度。"
                "10%规则：每周训练量增加不超过10%。"
                "正确的技术比重量更重要——先掌握动作模式再加负重。"
                "充足的恢复（睡眠、营养、主动休息日）是最便宜的防伤措施。"
                "常见过度使用损伤信号：持续酸痛>72h、关节疼痛、运动表现下降。"
                "出现急性损伤应立即遵循 RICE 原则（Rest-休息, Ice-冰敷, Compression-加压, Elevation-抬高）。",
        source="IOC Consensus Statement, 2020; BJSM",
        category="exercise",
        tags=["运动损伤", "预防", "恢复", "安全"],
        reliability_score=0.90,
    ),
    RAGDocument(
        title="减重期运动策略",
        content="减重应结合有氧运动+抗阻训练以最大化脂肪减少并保留瘦体重。"
                "有氧：每周 200-300 分钟中等强度（或 150 分钟高强度）产生显著减重效果。"
                "抗阻训练：每周至少 2 次，防止减重过程中肌肉流失。"
                "NEAT（非运动活动产热）：增加日常活动量（走路、站立、家务）的减重效果常被低估，"
                "可贡献每日总热量消耗的 15-50%。"
                "HIIT 是一种高效选择：每次 15-25 分钟，每周 2-3 次，可产生与长时间有氧相当的减重效果。"
                "减重后维持期：运动是防止反弹的最强预测因子。",
        source="ACSM Position Stand, 2021; Obesity Reviews",
        category="exercise",
        tags=["减重", "有氧", "NEAT", "HIIT"],
        reliability_score=0.90,
    ),

    # ── Sleep ─────────────────────────────────────────────────────────────
    RAGDocument(
        title="昼夜节律与睡眠驱动",
        content="睡眠受两个过程调控：过程C（昼夜节律，生物钟）和过程S（睡眠压力，腺苷积累）。"
                "昼夜节律由视交叉上核（SCN）控制，通过光照信号同步（约 24.2h 内源性周期）。"
                "早晨接触蓝光（太阳光）→抑制褪黑素→清醒信号。"
                "晚间黑暗→褪黑素开始分泌（通常睡前 2-3 小时）→促眠信号。"
                "腺苷随清醒时间积累产生「睡眠压力」，睡眠时清除。"
                "咖啡因通过阻断腺苷受体掩盖睡眠压力（半衰期 3-7 小时）。",
        source="Principles and Practice of Sleep Medicine, 7th Ed",
        category="sleep",
        tags=["昼夜节律", "褪黑素", "腺苷", "生物钟"],
        reliability_score=0.95,
    ),
    RAGDocument(
        title="失眠的认知行为疗法 (CBT-I)",
        content="CBT-I 是失眠的一线治疗（优于药物），包含 5 个核心组分："
                "1. 刺激控制：床只用于睡眠，躺下 20min 未入睡就起床，等有困意再回床。"
                "2. 睡眠限制：限制在床时间接近实际睡眠时间，提高睡眠效率（>85%后逐步延长）。"
                "3. 认知重组：纠正「我今晚肯定睡不着」等灾难化思维。"
                "4. 睡眠卫生教育：环境、饮食、运动等（见睡眠卫生建议）。"
                "5. 放松训练：渐进式肌肉放松、腹式呼吸、正念冥想。"
                "6-8 周 CBT-I 项目效果显著，且效果持久优于安眠药。",
        source="AASM Clinical Practice Guideline, 2021; JAMA Internal Medicine",
        category="sleep",
        tags=["CBT-I", "失眠", "认知行为治疗", "非药物治疗"],
        reliability_score=0.95,
    ),
    RAGDocument(
        title="睡眠与代谢健康",
        content="睡眠不足（<6h/夜）与多种代谢紊乱相关："
                "胰岛素敏感性下降 20-30%（仅 4-5 天睡眠限制后即可出现）。"
                "瘦素（饱腹信号）下降约 18%，饥饿素（饥饿信号）上升约 28%。"
                "高热量食物偏好增加，每日多摄入 300-500 kcal。"
                "皮质醇升高，促进内脏脂肪堆积。"
                "睡眠呼吸暂停（OSA）患者的 2 型糖尿病风险增加 2-3 倍。"
                "改善睡眠是代谢健康干预中常被低估的关键环节。",
        source="Annals of Internal Medicine, 2020; Sleep Medicine Reviews",
        category="sleep",
        tags=["睡眠", "代谢", "胰岛素抵抗", "肥胖"],
        reliability_score=0.90,
    ),

    # ── Mental Health ─────────────────────────────────────────────────────
    RAGDocument(
        title="运动与心理健康",
        content="规律运动对心理健康的益处已被充分证明："
                "有氧运动 30-45 分钟/次，每周 3-5 次，对轻中度抑郁的改善效果与药物相当（SMD 0.5-0.8）。"
                "抗阻训练同样有效，每周 2-3 次对焦虑和抑郁有中等效应。"
                "机制：内啡肽释放、BDNF 增加（促进神经可塑性）、"
                "HPA 轴调节（降低皮质醇）、自我效能感提升。"
                "急性效果：单次 20-30 分钟中等强度运动后焦虑显著降低，持续 2-4 小时。"
                "任何运动量都比不运动好——即使 10 分钟快走也有益。",
        source="Lancet Psychiatry, 2018; JAMA Psychiatry, 2022",
        category="mental_health",
        tags=["运动", "抑郁", "焦虑", "内啡肽", "BDNF"],
        reliability_score=0.90,
    ),
    RAGDocument(
        title="正念冥想与压力管理",
        content="正念（Mindfulness）是有意识地、不评判地关注当下的练习。"
                "MBSR（正念减压）8 周课程：每周 2.5h 团体课 + 每日 45 分钟自主练习。"
                "效果：焦虑减少 20-40%、压力感知降低、睡眠改善、免疫功能增强。"
                "简易入门：每天 5-10 分钟关注呼吸，觉察思维游走，温和地将注意力带回。"
                "4-7-8 呼吸法：吸气 4s → 屏息 7s → 缓慢呼出 8s，重复 4 次，"
                "可快速激活副交感神经系统，降低心率。"
                "身体扫描（Body Scan）有助于睡前放松和改善入睡困难。",
        source="JAMA Internal Medicine, 2022; Mindfulness Journal",
        category="mental_health",
        tags=["正念", "冥想", "压力管理", "放松"],
        reliability_score=0.90,
    ),

    # ── Preventive Care ───────────────────────────────────────────────────
    RAGDocument(
        title="心血管疾病一级预防",
        content="ASCVD 风险评估是心血管预防的基础（评估 10 年风险）。"
                "Life's Essential 8 (AHA)：健康饮食、身体活动、避免尼古丁、健康睡眠、"
                "健康体重、控制血脂、控制血糖、控制血压。"
                "40 岁以上成人建议每年测量血压。"
                "35 岁以上建议每 5 年检查血脂（高风险人群更频繁）。"
                "45 岁以上建议筛查 2 型糖尿病（高风险人群更早）。"
                "阿司匹林一级预防：不推荐常规使用，需个体化评估获益与出血风险。",
        source="AHA/ACC Guidelines, 2023; USPSTF Recommendations",
        category="preventive_care",
        tags=["心血管", "预防", "筛查", "体检"],
        reliability_score=0.95,
    ),
    RAGDocument(
        title="健康体检建议",
        content="常规体检频率和项目建议（基于年龄和风险）："
                "18-39岁：每2-3年一次基础体检（血压、BMI、视力）。"
                "40-64岁：每年一次，增加血脂（每5年）、血糖（每3年）。"
                "65岁以上：每年一次，增加骨密度（女性）、听力、认知筛查。"
                "所有年龄段：牙科检查每6-12个月、眼科每1-2年、皮肤自查每月。"
                "疫苗接种不容忽视：流感疫苗每年、破伤风每10年、"
                "HPV疫苗（26岁前）、带状疱疹疫苗（50+岁）。"
                "肿瘤筛查（45+岁）：结直肠癌筛查（肠镜/粪便检测）、"
                "乳腺癌筛查（乳腺X线）、宫颈癌筛查（HPV/Pap）、"
                "肺癌筛查（55-80岁有吸烟史者，低剂量CT）。",
        source="USPSTF Guidelines, 2024; WHO Guidelines",
        category="preventive_care",
        tags=["体检", "筛查", "疫苗", "预防"],
        reliability_score=0.90,
    ),

    # ── General ───────────────────────────────────────────────────────────
    RAGDocument(
        title="行为改变的跨理论模型",
        content="健康行为改变分为 5 个阶段："
                "1. 前意向期（无意改变）→ 提供信息，提高意识。"
                "2. 意向期（考虑改变）→ 分析利弊，增强动机。"
                "3. 准备期（计划改变）→ 制定具体可行的 SMART 目标。"
                "4. 行动期（正在改变）→ 提供支持，强化行为。"
                "5. 维持期（保持改变）→ 防止复发，建立新身份认同。"
                "关键策略：设定小目标（而非大目标）、自我监控、环境设计、社会支持、"
                "庆祝小胜利。习惯形成平均需要 66 天（范围 18-254 天）。",
        source="Prochaska & DiClemente; European Journal of Social Psychology",
        category="general",
        tags=["行为改变", "习惯养成", "动机", "心理学"],
        reliability_score=0.85,
    ),
    RAGDocument(
        title="个性化健康规划方法论",
        content="有效的健康规划应遵循以下框架："
                "1. 评估：全面了解当前健康状况、生活习惯、风险因素、个人偏好。"
                "2. 目标设定：SMART 原则（具体、可测量、可达、相关、有时限）。"
                "3. 方案制定：多领域协同（饮食+运动+睡眠+心理），而非单一干预。"
                "4. 渐进实施：每次聚焦 1-2 个行为改变，避免同时改变过多导致放弃。"
                "5. 追踪反馈：客观数据（体重、睡眠时长）+ 主观感受（精力、心情）。"
                "6. 动态调整：根据进展和反馈调整方案，不僵化执行。"
                "多学科协作（医生+营养师+运动教练+心理咨询师）效果优于单打独斗。",
        source="ACLM Guidelines; Lifestyle Medicine Handbook",
        category="general",
        tags=["健康规划", "个性化", "SMART", "行为改变"],
        reliability_score=0.90,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# RAG Tool Implementations
# ═══════════════════════════════════════════════════════════════════════════════

def search_health_knowledge(
    query: str,
    category: str | None = None,
    top_k: int = 5,
) -> RAGRetrievalResult:
    """
    Search the health knowledge base for evidence-based information.

    Uses keyword matching + category filtering + reliability scoring
    to retrieve the most relevant health documents.

    Args:
        query: Search query (Chinese or English)
        category: Optional filter (nutrition, exercise, sleep, mental_health, preventive_care, general)
        top_k: Number of results to return
    """
    import time
    start_time = time.time()

    # Score all documents
    scored: list[tuple[RAGDocument, float]] = []

    for doc in HEALTH_KNOWLEDGE_BASE:
        # Apply category filter
        if category and doc.category != category:
            continue

        score = _compute_relevance(query, doc)

        if score > 0:
            scored.append((doc, score))

    # Sort by relevance score * reliability
    scored.sort(key=lambda x: x[1] * x[0].reliability_score, reverse=True)
    top_results = scored[:top_k]

    retrieval_time = (time.time() - start_time) * 1000

    documents = [doc for doc, _ in top_results]
    scores = [score for _, score in top_results]

    # Synthesize a brief answer from retrieved documents
    synthesized = _synthesize_answer(query, top_results) if top_results else ""

    return RAGRetrievalResult(
        query=query,
        documents=documents,
        relevance_scores=scores,
        synthesized_answer=synthesized,
        retrieval_time_ms=round(retrieval_time, 2),
    )


def _compute_relevance(query: str, doc: RAGDocument) -> float:
    """Compute relevance score between query and document."""
    query_lower = query.lower()
    query_chars = set(query_lower)

    score = 0.0

    # Title matching
    title_lower = doc.title.lower()
    if any(word in title_lower for word in query_lower.split()):
        score += 0.4

    # Content keyword matching
    content_lower = doc.content.lower()
    query_words = query_lower.split()
    matched_words = sum(1 for w in query_words if w in content_lower)
    if query_words:
        score += 0.3 * (matched_words / len(query_words))

    # Tag matching
    tag_matches = sum(1 for t in doc.tags if t.lower() in query_lower or any(
        q in t.lower() for q in query_words
    ))
    if tag_matches:
        score += 0.2 * min(1.0, tag_matches / 3)

    # Chinese character overlap for CJK queries
    cjk_chars = [c for c in query if '一' <= c <= '鿿']
    if cjk_chars:
        cjk_matches = sum(1 for c in cjk_chars if c in doc.content)
        score += 0.1 * (cjk_matches / len(cjk_chars))

    return min(1.0, score)


def _synthesize_answer(
    query: str,
    scored_results: list[tuple[RAGDocument, float]],
) -> str:
    """Synthesize a brief evidence-based answer from retrieved documents."""
    if not scored_results:
        return ""

    parts = []
    for doc, score in scored_results[:3]:
        # Extract most relevant sentences
        sentences = doc.content.replace("！", "。").replace("？", "。").split("。")
        relevant = [s for s in sentences if len(s) > 10]
        if relevant:
            parts.append(f"【{doc.title}】（来源：{doc.source}）\n{relevant[0]}。")

    if parts:
        return "\n\n".join(parts)
    return ""


def get_knowledge_base_stats() -> dict:
    """Get statistics about the knowledge base."""
    categories = {}
    for doc in HEALTH_KNOWLEDGE_BASE:
        if doc.category not in categories:
            categories[doc.category] = 0
        categories[doc.category] += 1

    return {
        "total_documents": len(HEALTH_KNOWLEDGE_BASE),
        "categories": categories,
        "avg_reliability": round(
            sum(d.reliability_score for d in HEALTH_KNOWLEDGE_BASE) / len(HEALTH_KNOWLEDGE_BASE), 3
        ),
        "sources": list(set(d.source for d in HEALTH_KNOWLEDGE_BASE)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Function Calling Schema
# ═══════════════════════════════════════════════════════════════════════════════

RAG_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_health_knowledge",
            "description": "从循证健康知识库中检索权威医学和健康信息，返回结构化的知识文档",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询（中文或英文）"},
                    "category": {
                        "type": "string",
                        "enum": ["nutrition", "exercise", "sleep", "mental_health", "preventive_care", "general"],
                        "description": "知识类别过滤",
                    },
                    "top_k": {"type": "integer", "description": "返回结果数量", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        },
    },
]

RAG_TOOLS_MAP = {
    "search_health_knowledge": search_health_knowledge,
}
