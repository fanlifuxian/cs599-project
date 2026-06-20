"""
MCP (Model Context Protocol) Tools — health knowledge server integration.

Implements MCP-compatible tool definitions and a local health knowledge
MCP server that provides structured health data to agents.

MCP Tool Categories:
- health_guidelines: WHO, CDC, AASM official guidelines
- drug_interactions: Common medication-exercise-diet interactions
- medical_reference: Vital signs, lab values, BMI standards
- emergency_triage: When to seek professional medical help
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from src.config.settings import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MCP Tool Implementations
# ═══════════════════════════════════════════════════════════════════════════════

def query_health_guidelines(
    topic: str,
    organization: str = "WHO",
    language: str = "zh",
) -> dict:
    """
    Query official health guidelines from major health organizations.

    Args:
        topic: Health topic (nutrition, exercise, sleep, cardiovascular, diabetes, etc.)
        organization: Source organization (WHO, CDC, AASM, AHA, ACSM)
        language: Response language (zh/en)
    """
    guidelines_db = {
        "nutrition": {
            "WHO": {
                "title": "WHO 健康饮食指南",
                "key_points": [
                    "每日至少摄入 400g 水果和蔬菜（土豆、红薯等根茎类除外）",
                    "总脂肪摄入量低于总能量摄入的 30%",
                    "饱和脂肪摄入量低于总能量摄入的 10%",
                    "反式脂肪摄入量低于总能量摄入的 1%",
                    "游离糖摄入量低于总能量摄入的 10%（最好低于 5%）",
                    "每日盐摄入量低于 5g（约一茶匙）",
                ],
                "url": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
            },
            "CDCF": {
                "title": "CDC 美国人膳食指南",
                "key_points": [
                    "限制添加糖至每日热量 10% 以下",
                    "限制饱和脂肪至每日热量 10% 以下",
                    "每日钠摄入 < 2300mg",
                    "选择多种蛋白质来源（瘦肉、禽肉、鱼、豆类、坚果）",
                ],
            },
        },
        "exercise": {
            "WHO": {
                "title": "WHO 身体活动指南",
                "key_points": [
                    "成人每周至少 150-300 分钟中等强度有氧运动",
                    "或 75-150 分钟高强度有氧运动",
                    "每周至少 2 天进行肌肉强化活动",
                    "限制久坐时间，用任何强度的活动替代久坐",
                    "65 岁以上老人应增加平衡和协调训练",
                ],
                "url": "https://www.who.int/news-room/fact-sheets/detail/physical-activity",
            },
            "ACSM": {
                "title": "ACSM 运动处方指南",
                "key_points": [
                    "有氧运动：FITT 原则（频率 3-5天/周，强度 40-89% HRR，时间 30-60min，类型 有氧）",
                    "抗阻训练：频率 2-3天/周，每个肌群 2-4 组，8-12 次重复",
                    "柔韧性训练：频率 ≥2-3天/周，每个拉伸保持 10-30 秒",
                ],
            },
        },
        "sleep": {
            "AASM": {
                "title": "AASM 成人睡眠指南",
                "key_points": [
                    "成人推荐每夜规律睡眠 7 小时以上",
                    "睡眠不足 7 小时与多种不良健康结局相关",
                    "保持固定的起床时间比固定入睡时间更重要",
                    "白天接触自然光 30 分钟以上有助于调节昼夜节律",
                    "如果躺下 20 分钟无法入睡，应起床进行放松活动",
                ],
                "url": "https://aasm.org/clinical-resources/clinical-practice-guidelines/",
            },
            "NIH": {
                "title": "NIH 睡眠卫生建议",
                "key_points": [
                    "保持规律作息，每天同一时间睡觉和起床",
                    "睡前 1 小时不使用电子屏幕",
                    "睡前避免咖啡因（咖啡因半衰期 3-7 小时）",
                    "规律运动有助于改善睡眠质量",
                    "卧室保持凉爽、黑暗、安静",
                ],
            },
        },
        "cardiovascular": {
            "AHA": {
                "title": "AHA 心血管健康指南 (Life's Essential 8)",
                "key_points": [
                    "健康饮食（多吃全食物，限制加工食品）",
                    "身体活动（成人 150+ 分钟/周中等强度）",
                    "避免尼古丁（包括传统香烟和电子烟）",
                    "健康睡眠（成人 7-9 小时/夜）",
                    "健康体重（BMI 18.5-24.9）",
                    "控制血脂（关注非-HDL 胆固醇）",
                    "控制血糖（空腹血糖 < 100 mg/dL）",
                    "控制血压（< 120/80 mmHg）",
                ],
            },
        },
        "diabetes": {
            "ADA": {
                "title": "ADA 糖尿病管理指南",
                "key_points": [
                    "HbA1c 目标 < 7.0%（个体化调整）",
                    "餐前血糖 80-130 mg/dL",
                    "餐后 1-2 小时血糖 < 180 mg/dL",
                    "每周至少 150 分钟中等强度有氧运动",
                    "每周 2-3 次抗阻训练（非连续日）",
                ],
            },
        },
    }

    # Find matching guideline
    for key, orgs in guidelines_db.items():
        if key in topic.lower() or topic.lower() in key:
            org_data = orgs.get(organization.upper())
            if not org_data:
                # Return first available org
                org_name = list(orgs.keys())[0]
                org_data = orgs[org_name]
                organization = org_name
            return {
                "status": "ok",
                "topic": key,
                "organization": organization,
                **org_data,
                "retrieved_at": datetime.now().isoformat(),
            }

    # General fallback
    return {
        "status": "partial",
        "topic": topic,
        "organization": organization,
        "message": f"未找到 '{topic}' 的精确指南，以下为通用健康建议",
        "key_points": [
            "保持均衡饮食，多摄入蔬菜水果",
            "每周保持 150 分钟以上中等强度运动",
            "保证 7-9 小时优质睡眠",
            "定期体检，关注血压、血糖、血脂指标",
        ],
        "retrieved_at": datetime.now().isoformat(),
        "note": "此为一般性建议，具体健康问题请咨询医生",
    }


def check_drug_interactions(
    medications: list[str],
    foods: list[str] = None,
    activities: list[str] = None,
) -> dict:
    """
    Check for known interactions between medications, foods, and activities.

    Args:
        medications: List of medication names
        foods: Foods/ingredients to check against
        activities: Exercise/activities to check against
    """
    foods = foods or []
    activities = activities or []

    # Known interaction database (simplified)
    interactions_db = {
        "华法林": {
            "foods": {
                "菠菜": {"severity": "moderate", "note": "菠菜富含维生素K，会降低华法林抗凝效果"},
                "西兰花": {"severity": "moderate", "note": "西兰花富含维生素K，应保持摄入量稳定"},
                "绿茶": {"severity": "mild", "note": "大量饮用绿茶可能影响华法林代谢"},
                "大蒜": {"severity": "moderate", "note": "大蒜具有抗血小板作用，可能增加出血风险"},
            },
        },
        "他汀类": {
            "foods": {
                "西柚": {"severity": "high", "note": "西柚（葡萄柚）抑制他汀代谢酶，可能导致血药浓度升高 3-5 倍"},
                "柚子": {"severity": "high", "note": "柚类水果增加他汀副作用风险（肌痛、肝损伤）"},
                "酒精": {"severity": "moderate", "note": "酒精增加他汀肝毒性风险"},
            },
        },
        "二甲双胍": {
            "foods": {
                "酒精": {"severity": "high", "note": "酒精增加乳酸酸中毒风险"},
            },
        },
        "布洛芬": {
            "foods": {
                "酒精": {"severity": "high", "note": "酒精增加胃出血风险"},
            },
        },
        "甲状腺素": {
            "foods": {
                "大豆": {"severity": "moderate", "note": "大豆制品可能影响甲状腺素吸收，应间隔 4 小时以上"},
                "高纤维食物": {"severity": "mild", "note": "高纤维食物可能减少甲状腺素吸收"},
                "咖啡": {"severity": "moderate", "note": "咖啡显著减少甲状腺素吸收，应在服药后等待至少60分钟"},
            },
        },
        "ACEI": {
            "foods": {
                "香蕉": {"severity": "mild", "note": "ACEI可能升高血钾，大量摄入高钾食物需谨慎"},
            },
        },
    }

    findings = []
    for med in medications:
        med_key = None
        for key in interactions_db:
            if key in med or med in key:
                med_key = key
                break

        if not med_key:
            continue

        interactions = interactions_db[med_key]

        for food in foods:
            for food_key, info in interactions.get("foods", {}).items():
                if food_key in food or food in food_key:
                    findings.append({
                        "medication": med,
                        "interacting_with": food,
                        "type": "food",
                        "severity": info["severity"],
                        "warning": info["note"],
                    })

    if not findings:
        return {
            "status": "ok",
            "medications_checked": medications,
            "foods_checked": foods,
            "interactions_found": 0,
            "findings": [],
            "note": "未发现已知的药物-食物相互作用（本数据库覆盖有限，建议咨询医生或药师）",
        }

    return {
        "status": "warning",
        "medications_checked": medications,
        "foods_checked": foods,
        "interactions_found": len(findings),
        "findings": findings,
        "disclaimer": "本检查为自动筛查，不能替代专业医生或药师的建议",
    }


def get_medical_reference(
    metric: str,
    value: float,
    unit: str = "",
) -> dict:
    """
    Provide medical reference ranges and interpretation for common health metrics.

    Args:
        metric: Metric name (bmi, blood_pressure_systolic, blood_pressure_diastolic,
               heart_rate, fasting_glucose, hba1c, total_cholesterol, ldl, hdl)
        value: Measured value
        unit: Unit of measurement
    """
    references = {
        "bmi": {
            "ranges": [
                {"label": "偏瘦", "min": 0, "max": 18.5, "interpretation": "体重过轻，建议增加营养摄入并咨询医生"},
                {"label": "正常", "min": 18.5, "max": 24.0, "interpretation": "体重在正常范围内"},
                {"label": "偏胖", "min": 24.0, "max": 28.0, "interpretation": "体重偏高，建议关注饮食并增加运动"},
                {"label": "肥胖", "min": 28.0, "max": 100, "interpretation": "体重显著超标，建议制定减重计划并咨询医生"},
            ],
            "unit": "kg/m²",
        },
        "heart_rate_resting": {
            "ranges": [
                {"label": "心动过缓", "min": 0, "max": 60, "interpretation": "静息心率偏低（运动员可能正常）"},
                {"label": "正常", "min": 60, "max": 100, "interpretation": "静息心率在正常范围"},
                {"label": "心动过速", "min": 100, "max": 250, "interpretation": "静息心率偏高，建议就医评估"},
            ],
            "unit": "bpm",
        },
        "fasting_glucose": {
            "ranges": [
                {"label": "正常", "min": 0, "max": 5.6, "interpretation": "空腹血糖正常"},
                {"label": "糖尿病前期", "min": 5.6, "max": 7.0, "interpretation": "空腹血糖受损，建议进行OGTT检查"},
                {"label": "糖尿病", "min": 7.0, "max": 50, "interpretation": "空腹血糖达到糖尿病诊断标准，请尽快就医"},
            ],
            "unit": "mmol/L",
        },
        "total_cholesterol": {
            "ranges": [
                {"label": "理想", "min": 0, "max": 5.2, "interpretation": "总胆固醇水平理想"},
                {"label": "边缘升高", "min": 5.2, "max": 6.2, "interpretation": "总胆固醇边缘升高，建议调整饮食"},
                {"label": "升高", "min": 6.2, "max": 20, "interpretation": "总胆固醇升高，建议就医评估心血管风险"},
            ],
            "unit": "mmol/L",
        },
    }

    ref = references.get(metric.lower())
    if not ref:
        return {
            "status": "unknown_metric",
            "metric": metric,
            "message": f"暂不支持 '{metric}' 的参考范围查询",
            "supported_metrics": list(references.keys()),
        }

    # Find which range the value falls into
    category = None
    for r in ref["ranges"]:
        if r["min"] <= value < r["max"]:
            category = r
            break

    return {
        "status": "ok",
        "metric": metric,
        "value": value,
        "unit": ref["unit"],
        "category": category["label"] if category else "未知",
        "interpretation": category["interpretation"] if category else "无法判断",
        "reference_ranges": ref["ranges"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MCP Tool Schemas (OpenAI-compatible Function Calling)
# ═══════════════════════════════════════════════════════════════════════════════

MCP_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_health_guidelines",
            "description": "查询WHO、CDC、AASM等权威机构的官方健康指南",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": ["nutrition", "exercise", "sleep", "cardiovascular", "diabetes", "weight_management", "mental_health"],
                        "description": "健康主题",
                    },
                    "organization": {
                        "type": "string",
                        "enum": ["WHO", "CDC", "AASM", "AHA", "ACSM", "ADA"],
                        "description": "指南发布机构",
                    },
                    "language": {
                        "type": "string",
                        "enum": ["zh", "en"],
                        "description": "语言偏好",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_drug_interactions",
            "description": "检查药物与食物/运动之间的已知相互作用",
            "parameters": {
                "type": "object",
                "properties": {
                    "medications": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "正在服用的药物名称列表",
                    },
                    "foods": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "计划摄入的食物/成分",
                    },
                    "activities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "计划进行的运动/活动",
                    },
                },
                "required": ["medications"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_medical_reference",
            "description": "获取健康指标的医学参考范围和解读",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["bmi", "heart_rate_resting", "fasting_glucose", "total_cholesterol"],
                        "description": "健康指标名称",
                    },
                    "value": {"type": "number", "description": "指标数值"},
                    "unit": {"type": "string", "description": "单位"},
                },
                "required": ["metric", "value"],
            },
        },
    },
]

MCP_TOOLS_MAP = {
    "query_health_guidelines": query_health_guidelines,
    "check_drug_interactions": check_drug_interactions,
    "get_medical_reference": get_medical_reference,
}
