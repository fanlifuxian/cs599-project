"""
Sleep Agent Tools — sleep quality analysis, sleep planning, hygiene tips.
All tools expose OpenAI-compatible function calling schemas.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Implementations
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_sleep_quality(
    sleep_hours: float,
    fall_asleep_minutes: int = 20,
    wake_ups: int = 0,
    sleep_time: str = "23:00",
    wake_time: str = "07:00",
    feel_rested: int = 3,  # 1-5 scale
    screen_time_before_bed_minutes: int = 30,
    caffeine_after_4pm: bool = False,
    exercise_today: bool = False,
) -> dict:
    """
    Analyze sleep quality based on self-reported metrics.

    Args:
        sleep_hours: Total hours of sleep
        fall_asleep_minutes: How long it took to fall asleep
        wake_ups: Number of times waking up during the night
        sleep_time: Bedtime (HH:MM 24h format)
        wake_time: Wake time (HH:MM 24h format)
        feel_rested: How rested do you feel (1=exhausted, 5=fully rested)
        screen_time_before_bed_minutes: Screen time in the 1h before bed
        caffeine_after_4pm: Whether caffeine was consumed after 4pm
        exercise_today: Whether user exercised today
    """
    score = 100

    # Sleep duration scoring (optimal 7-9 hours)
    if sleep_hours < 5:
        score -= 30
        duration_feedback = "严重不足，长期睡眠不足会显著影响健康"
    elif sleep_hours < 6:
        score -= 20
        duration_feedback = "不足，建议至少保证 7 小时睡眠"
    elif sleep_hours < 7:
        score -= 8
        duration_feedback = "略低于推荐值，争取再增加 30 分钟"
    elif sleep_hours <= 9:
        duration_feedback = "✅ 睡眠时长在理想范围内"
    else:
        score -= 5
        duration_feedback = "睡眠时间偏长，过长睡眠也可能导致困倦"

    # Sleep onset latency
    if fall_asleep_minutes > 30:
        score -= 10
        onset_feedback = "入睡时间过长，可能存在入睡困难"
    elif fall_asleep_minutes > 20:
        score -= 5
        onset_feedback = "入睡时间稍长，可以尝试睡前放松练习"
    elif fall_asleep_minutes < 5:
        score -= 2
        onset_feedback = "入睡极快，可能表明存在睡眠不足（身体过度疲劳）"
    else:
        onset_feedback = "✅ 入睡时间正常"

    # Sleep fragmentation
    if wake_ups > 2:
        score -= 15
        frag_feedback = "夜间醒来次数过多，睡眠连续性差"
    elif wake_ups > 0:
        score -= 5
        frag_feedback = f"夜间醒来 {wake_ups} 次，轻微影响睡眠连续性"
    else:
        frag_feedback = "✅ 睡眠连续无中断"

    # Subjective restedness
    score += (feel_rested - 3) * 5
    rested_map = {1: "非常疲惫", 2: "较疲惫", 3: "一般", 4: "较精神", 5: "精力充沛"}
    rested_feedback = f"主观感受：{rested_map.get(feel_rested, '一般')}"

    # Lifestyle factors
    lifestyle_feedback = []
    if screen_time_before_bed_minutes > 30:
        score -= 8
        lifestyle_feedback.append("睡前屏幕使用时间过长，蓝光抑制褪黑素分泌")
    if caffeine_after_4pm:
        score -= 8
        lifestyle_feedback.append("午后摄入咖啡因可能影响入睡")
    if not exercise_today:
        lifestyle_feedback.append("今天未运动，适度运动有助改善睡眠质量")
    else:
        lifestyle_feedback.append("✅ 今天有运动，有助促进深度睡眠")

    # Clamp score
    score = max(0, min(100, score))

    if score >= 85:
        grade = "优秀 (A)"
    elif score >= 70:
        grade = "良好 (B)"
    elif score >= 55:
        grade = "一般 (C)"
    elif score >= 40:
        grade = "较差 (D)"
    else:
        grade = "很差 (F)"

    return {
        "score": score,
        "grade": grade,
        "metrics": {
            "sleep_hours": sleep_hours,
            "fall_asleep_minutes": fall_asleep_minutes,
            "wake_ups": wake_ups,
            "sleep_window": f"{sleep_time} - {wake_time}",
        },
        "feedback": {
            "duration": duration_feedback,
            "onset": onset_feedback,
            "fragmentation": frag_feedback,
            "restedness": rested_feedback,
            "lifestyle": lifestyle_feedback,
        },
        "timestamp": datetime.now().isoformat(),
    }


def generate_sleep_plan(
    target_sleep_hours: float = 8.0,
    current_sleep_time: str = "23:30",
    current_wake_time: str = "07:00",
    issues: list[str] | None = None,
) -> dict:
    """
    Generate a personalized sleep improvement plan.

    Args:
        target_sleep_hours: Target hours of sleep per night
        current_sleep_time: Current typical bedtime (HH:MM)
        current_wake_time: Current typical wake time (HH:MM)
        issues: Specific sleep issues e.g. ["difficulty_falling_asleep", "waking_up_tired"]
    """
    issues = issues or []

    # Parse current times
    try:
        bed_h, bed_m = map(int, current_sleep_time.split(":"))
        wake_h, wake_m = map(int, current_wake_time.split(":"))
    except (ValueError, AttributeError):
        bed_h, bed_m = 23, 30
        wake_h, wake_m = 7, 0

    # Calculate recommended bedtime to achieve target
    wake_minutes = wake_h * 60 + wake_m
    target_bed_minutes = wake_minutes - int(target_sleep_hours * 60) - 30  # 30 min buffer

    if target_bed_minutes < 0:
        target_bed_minutes += 24 * 60

    rec_bed_h = (target_bed_minutes // 60) % 24
    rec_bed_m = target_bed_minutes % 60

    # Pre-sleep routine
    pre_sleep_routine = [
        "21:00 - 减少屏幕亮度和蓝光（开启夜间模式）",
        f"{rec_bed_h - 1:02d}:{rec_bed_m:02d} - 停止使用手机/电脑",
        f"{rec_bed_h - 1:02d}:{rec_bed_m:02d} - 温水泡脚 10 分钟",
        f"{rec_bed_h - 1:02d}:{rec_bed_m:02d} - 阅读纸质书或听轻音乐",
        f"{rec_bed_h:02d}:{max(0, rec_bed_m - 10):02d} - 调暗灯光，准备入睡",
        f"{rec_bed_h:02d}:{rec_bed_m:02d} - 上床睡觉",
    ]

    # Environment tips
    environment_tips = [
        "保持卧室温度在 18-22°C",
        "使用遮光窗帘，确保房间足够黑暗",
        "保持安静，必要时使用白噪音机",
        "床只用于睡眠，不在床上工作或玩手机",
    ]

    # Avoid items
    avoid_items = [
        "睡前 3 小时内不摄入咖啡因（咖啡/浓茶/可乐）",
        "睡前 2 小时内不大量进食",
        "睡前 1 小时不使用电子屏幕",
        "睡前不饮酒（酒精虽促进入睡但破坏深度睡眠）",
    ]

    # Issue-specific additions
    if "difficulty_falling_asleep" in issues:
        pre_sleep_routine.append("尝试 4-7-8 呼吸法：吸气 4 秒，屏息 7 秒，缓慢呼气 8 秒")
        avoid_items.append("睡前避免剧烈运动和兴奋性活动")
    if "waking_up_tired" in issues:
        environment_tips.append("尝试使用日出模拟闹钟，自然唤醒")
        pre_sleep_routine.append("睡前记录明天的待办事项，清空大脑")

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "recommended_bedtime": f"{rec_bed_h:02d}:{rec_bed_m:02d}",
        "recommended_wake_time": current_wake_time,
        "target_hours": target_sleep_hours,
        "pre_sleep_routine": pre_sleep_routine,
        "environment_tips": environment_tips,
        "avoid_items": avoid_items,
        "current_pattern": {
            "sleep_time": current_sleep_time,
            "wake_time": current_wake_time,
            "actual_hours": round(
                ((wake_h * 60 + wake_m) - (bed_h * 60 + bed_m)) / 60
                if (wake_h * 60 + wake_m) > (bed_h * 60 + bed_m)
                else ((wake_h * 60 + wake_m + 24 * 60) - (bed_h * 60 + bed_m)) / 60,
                1,
            ),
        },
        "issues_addressed": issues,
    }


def get_sleep_hygiene_tips(topic: str = "general") -> dict:
    """
    Get evidence-based sleep hygiene tips by topic.

    Args:
        topic: "general", "environment", "diet", "exercise", "stress", "schedule", "insomnia"
    """
    tips_db = {
        "general": [
            "保持固定的睡觉和起床时间，即使是周末",
            "每天保证 7-9 小时睡眠",
            "白天适当晒太阳，帮助调节昼夜节律",
            "建立放松的睡前例行程序",
        ],
        "environment": [
            "卧室温度保持在 18-22°C，凉爽环境更有助入睡",
            "使用遮光窗帘或眼罩，保持完全黑暗",
            "减少噪音干扰，可使用白噪音或耳塞",
            "选择舒适的床垫和枕头",
            "移除卧室中的电子设备（电视、手机等）",
        ],
        "diet": [
            "睡前 3 小时避免咖啡因",
            "睡前 2 小时避免大量进食",
            "晚餐选择易消化的食物",
            "可以适量饮用温牛奶、洋甘菊茶等助眠饮品",
            "避免睡前饮酒（酒精破坏深度睡眠）",
        ],
        "exercise": [
            "每周进行至少 150 分钟中等强度有氧运动",
            "早晨或下午运动最佳",
            "睡前 2 小时内避免剧烈运动",
            "瑜伽和拉伸有助放松身心",
            "规律运动者的深度睡眠时间更长",
        ],
        "stress": [
            "睡前进行 5-10 分钟冥想或正念练习",
            "写「感恩日记」或「待办清单」，清空大脑",
            "使用 4-7-8 呼吸法放松",
            "如果躺下 20 分钟仍无法入睡，起床做些放松活动，有困意再回床",
            "考虑使用渐进式肌肉放松法",
        ],
        "schedule": [
            "固定起床时间比固定入睡时间更重要",
            "每天起床后 30 分钟内接触自然光",
            "避免白天小睡超过 30 分钟",
            "下午 3 点后不小睡",
            "建立一致的睡前信号（如刷牙→阅读→关灯）",
        ],
        "insomnia": [
            "如果躺下 20 分钟无法入睡，起床离开卧室",
            "只在有困意时上床",
            "床只用于睡眠，不在床上工作、吃饭、看手机",
            "限制在床时间，提高睡眠效率",
            "考虑记录睡眠日记，了解自己的睡眠模式",
            "如果失眠持续超过 3 周，建议咨询睡眠专科医生",
        ],
    }

    topic_tips = tips_db.get(topic, tips_db["general"])

    return {
        "topic": topic,
        "tips": topic_tips,
        "source": "基于美国睡眠医学会 (AASM) 和 CBT-I 认知行为疗法建议",
        "disclaimer": "此为一般性建议，如有持续性睡眠问题请咨询专业医生",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Function Calling Schema Definitions
# ═══════════════════════════════════════════════════════════════════════════════

SLEEP_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "analyze_sleep_quality",
            "description": "根据用户自述的睡眠指标分析睡眠质量并打分",
            "parameters": {
                "type": "object",
                "properties": {
                    "sleep_hours": {"type": "number", "description": "总睡眠时长（小时）"},
                    "fall_asleep_minutes": {"type": "integer", "description": "入睡所需时间（分钟）"},
                    "wake_ups": {"type": "integer", "description": "夜间醒来次数"},
                    "sleep_time": {"type": "string", "description": "上床时间 HH:MM"},
                    "wake_time": {"type": "string", "description": "起床时间 HH:MM"},
                    "feel_rested": {"type": "integer", "description": "主观精力感 1(很累)-5(精力充沛)", "minimum": 1, "maximum": 5},
                    "screen_time_before_bed_minutes": {"type": "integer", "description": "睡前屏幕使用时间（分钟）"},
                    "caffeine_after_4pm": {"type": "boolean", "description": "下午4点后是否摄入咖啡因"},
                    "exercise_today": {"type": "boolean", "description": "今天是否运动"},
                },
                "required": ["sleep_hours"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_sleep_plan",
            "description": "生成个性化睡眠改善计划",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_sleep_hours": {"type": "number", "description": "目标睡眠时长（小时）"},
                    "current_sleep_time": {"type": "string", "description": "当前入睡时间 HH:MM"},
                    "current_wake_time": {"type": "string", "description": "当前起床时间 HH:MM"},
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "具体睡眠问题",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sleep_hygiene_tips",
            "description": "获取循证的睡眠卫生建议",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "enum": ["general", "environment", "diet", "exercise", "stress", "schedule", "insomnia"],
                        "description": "建议主题",
                    },
                },
                "required": [],
            },
        },
    },
]

SLEEP_TOOLS_MAP = {
    "analyze_sleep_quality": analyze_sleep_quality,
    "generate_sleep_plan": generate_sleep_plan,
    "get_sleep_hygiene_tips": get_sleep_hygiene_tips,
}
