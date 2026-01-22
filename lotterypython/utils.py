"""
Utility functions for LotteryPython

Includes date calculations, formatting helpers, and common utilities.
"""

from datetime import datetime, timedelta
from typing import Tuple


def get_next_draw_date(lottery_type: str) -> Tuple[str, str]:
    """
    計算下一期開獎日期

    Args:
        lottery_type: 'big' for 大樂透 or 'super' for 威力彩

    Returns:
        tuple: (draw_date, weekday_name) e.g. ("2025-01-21", "週二")

    開獎規則:
        - 大樂透: 週二(1), 週五(4)
        - 威力彩: 週一(0), 週四(3)
    """
    today = datetime.now()
    weekday = today.weekday()  # 0=Monday, 1=Tuesday, ...

    weekday_names = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"]

    if lottery_type == 'big':
        # 大樂透: 週二(1), 週五(4)
        draw_days = [1, 4]
    else:
        # 威力彩: 週一(0), 週四(3)
        draw_days = [0, 3]

    # 找最近的開獎日（包含今天）
    for i in range(7):
        check_day = (weekday + i) % 7
        if check_day in draw_days:
            next_draw = today + timedelta(days=i)
            return (
                next_draw.strftime("%Y-%m-%d"),
                weekday_names[check_day]
            )

    # Fallback (should never reach here)
    return today.strftime("%Y-%m-%d"), weekday_names[weekday]


def get_draw_info(lottery_type: str, next_period: str) -> dict:
    """
    取得完整的開獎資訊

    Args:
        lottery_type: 'big' or 'super'
        next_period: 下一期期數

    Returns:
        dict with period, date, weekday, lottery_name
    """
    draw_date, weekday = get_next_draw_date(lottery_type)
    lottery_name = "大樂透" if lottery_type == 'big' else "威力彩"
    draw_schedule = "週二、週五" if lottery_type == 'big' else "週一、週四"

    return {
        "period": next_period,
        "draw_date": draw_date,
        "weekday": weekday,
        "lottery_name": lottery_name,
        "draw_schedule": draw_schedule,
        "display": f"第 {next_period} 期 | {draw_date} ({weekday})"
    }


def format_combination_reason(predictions: list, final_numbers: list, final_special: int) -> dict:
    """
    產生號碼組合的說明

    Args:
        predictions: 各家人的預測結果列表
        final_numbers: 最終選定的號碼
        final_special: 最終特別號

    Returns:
        dict with combination explanation
    """
    from collections import Counter

    # 統計每個號碼被多少人推薦
    all_numbers = []
    all_specials = []

    for pred in predictions:
        if 'error' not in pred and 'numbers' in pred:
            all_numbers.extend(pred['numbers'])
            if 'special' in pred:
                all_specials.append(pred['special'])

    number_counts = Counter(all_numbers)
    special_counts = Counter(all_specials)

    total_profiles = len([p for p in predictions if 'error' not in p])

    # 產生每個號碼的說明
    number_reasons = []
    for num in final_numbers:
        count = number_counts.get(num, 0)
        if count >= total_profiles:
            reason = f"全員推薦"
        elif count > 1:
            reason = f"{count}/{total_profiles} 人推薦"
        else:
            reason = "單人推薦"
        number_reasons.append({
            "number": num,
            "count": count,
            "reason": reason
        })

    special_count = special_counts.get(final_special, 0)
    special_reason = f"{special_count}/{total_profiles} 人推薦" if special_count > 1 else "單人推薦"

    return {
        "method": "頻率投票法",
        "description": f"綜合 {total_profiles} 位家人的命理分析，選出最多人推薦的號碼",
        "numbers": number_reasons,
        "special": {
            "number": final_special,
            "count": special_count,
            "reason": special_reason
        }
    }
