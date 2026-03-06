"""
LINE Messaging API Push Notifier

Sends daily lottery prediction results to users via LINE Push API.
Supports per-user personalization including fortune (命理) predictions.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger("line_notifier")

LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
MAX_MESSAGE_LEN = 4500  # LINE limit is 5000, keeping buffer


class LineNotifier:
    """LINE Messaging API Push client."""

    def __init__(self, channel_access_token: str):
        self.token = channel_access_token
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {channel_access_token}",
        }

    def push_message(self, line_id: str, message: str) -> bool:
        """Push a single text message to a LINE ID.

        Args:
            line_id: LINE User ID (starts with U)
            message: Text content to send

        Returns:
            True if successful, False otherwise
        """
        payload = {
            "to": line_id,
            "messages": [{"type": "text", "text": message}],
        }
        for attempt in range(3):
            try:
                resp = requests.post(
                    LINE_PUSH_URL,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=10,
                )
                if resp.status_code == 200:
                    return True
                if resp.status_code == 429:
                    logger.warning(f"LINE rate limit hit for {line_id}, waiting 2s (attempt {attempt+1}/3)")
                    time.sleep(2 ** attempt)
                    continue
                logger.error(f"LINE push failed for {line_id}: HTTP {resp.status_code} — {resp.text}")
                return False
            except requests.RequestException as e:
                logger.error(f"LINE push request error for {line_id}: {e}")
                return False
        return False

    def push_messages(self, line_id: str, messages: list) -> bool:
        """Push multiple text messages to a LINE ID (for long content)."""
        all_ok = True
        for msg in messages:
            ok = self.push_message(line_id, msg)
            if not ok:
                all_ok = False
        return all_ok


def _format_numbers(numbers: list, special: int) -> str:
    """Format a number set: '03 12 18 25 33 38 | 特: 07'"""
    nums = " ".join(f"{n:02d}" for n in sorted(numbers))
    return f"{nums} | 特: {special:02d}"


def split_message(text: str, max_len: int = MAX_MESSAGE_LEN) -> list:
    """Split a long message into chunks under max_len characters."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:max_len])
        text = text[max_len:]
    return chunks


def build_prediction_message(
    lottery_type: str,
    lottery_name: str,
    predictions: dict,
    fortune_results: Optional[dict] = None,
) -> str:
    """Build the formatted prediction message text.

    Args:
        lottery_type: 'big' or 'super'
        lottery_name: Chinese name e.g. '大樂透'
        predictions: Dict of {algo_name: {numbers, special, next_period, draw_info, ...}}
        fortune_results: Optional dict with 'ziwei' and 'zodiac' prediction dicts

    Returns:
        Formatted message string
    """
    today = datetime.now().strftime("%Y-%m-%d")

    # Determine next period and draw_info from any valid prediction
    next_period = ""
    draw_info = None
    for result in predictions.values():
        if isinstance(result, dict) and "error" not in result:
            next_period = result.get("next_period", "")
            draw_info = result.get("draw_info")
            break

    lines = [
        f"🎰 今日彩券預測報告",
        f"📅 {today} | {lottery_name}",
    ]
    if draw_info:
        draw_date = draw_info.get("draw_date", "")
        weekday = draw_info.get("weekday", "")
        if next_period and draw_date:
            lines.append(f"🔢 第 {next_period} 期 | 📆 {draw_date} ({weekday})")
    elif next_period:
        lines.append(f"🔢 下期期號: {next_period}")
    lines.append("")

    # Algorithm predictions (max 8)
    algo_lines = []
    for algo_name, result in predictions.items():
        if not isinstance(result, dict) or "error" in result:
            continue
        numbers = result.get("numbers", [])
        special = result.get("special", 0)
        if numbers:
            algo_lines.append(f"• {algo_name}: {_format_numbers(numbers, special)}")

    if algo_lines:
        lines.append("📊 各算法預測:")
        lines.extend(algo_lines)

    # Fortune predictions
    _append_fortune_lines(lines, fortune_results, indent="")

    lines.append("")
    lines.append("🤖 LotteryPython 自動推送")

    return "\n".join(lines)


def _parse_fortune_entry(entry) -> Optional[tuple]:
    """Normalize a fortune entry to (numbers, special).

    Handles both dict format {'numbers': [...], 'special': N} and
    tuple/list format [numbers_list, special_int, details_dict].
    Returns None if entry is invalid or contains an error.
    """
    if not entry:
        return None
    if isinstance(entry, dict):
        if "error" in entry:
            return None
        nums = entry.get("numbers", [])
        sp = entry.get("special", 0)
        return (nums, sp) if nums else None
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        nums, sp = entry[0], entry[1]
        if isinstance(nums, (list, tuple)) and nums:
            return (list(nums), int(sp))
    return None


def _append_fortune_lines(lines: list, fortune_results: Optional[dict], indent: str = ""):
    """Append 命理 (fortune) section lines to lines list."""
    if not fortune_results:
        return

    ziwei_parsed = _parse_fortune_entry(fortune_results.get("ziwei"))
    zodiac_parsed = _parse_fortune_entry(fortune_results.get("zodiac"))

    if not ziwei_parsed and not zodiac_parsed:
        return

    lines.append("")
    lines.append("🔮 您的命理預測:")

    if ziwei_parsed:
        nums, sp = ziwei_parsed
        lines.append("【紫微斗數】")
        lines.append(f"{indent}• 幸運號碼: {_format_numbers(nums, sp)}")

    if zodiac_parsed:
        nums, sp = zodiac_parsed
        lines.append("【生肖運勢】")
        lines.append(f"{indent}• 今日幸運: {_format_numbers(nums, sp)}")


def _extract_draw_header(predictions: dict, lottery_name: str, emoji: str) -> list:
    """Extract period/date header lines from a predictions dict."""
    next_period = ""
    draw_info = None
    for result in predictions.values():
        if isinstance(result, dict) and "error" not in result:
            next_period = result.get("next_period", "")
            draw_info = result.get("draw_info")
            break

    lines = [f"{emoji} {lottery_name}"]
    if draw_info:
        draw_date = draw_info.get("draw_date", "")
        weekday = draw_info.get("weekday", "")
        if next_period and draw_date:
            lines.append(f"🔢 第 {next_period} 期 | 📆 {draw_date} ({weekday})")
    elif next_period:
        lines.append(f"🔢 下期: {next_period}")
    return lines


def build_combined_message(
    predictions_big: dict,
    predictions_super: dict,
    fortune_results_big: Optional[dict] = None,
    fortune_results_super: Optional[dict] = None,
) -> str:
    """Build a combined prediction message for both lottery types with draw dates and 命理.

    Args:
        predictions_big: Dict of algorithm predictions for 大樂透
        predictions_super: Dict of algorithm predictions for 威力彩
        fortune_results_big: Optional fortune dict for 大樂透
        fortune_results_super: Optional fortune dict for 威力彩

    Returns:
        Formatted combined message string
    """
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [
        "🎰 今日彩券預測報告",
        f"📅 {today}",
        "",
    ]

    for predictions, lottery_name, emoji, fortune in [
        (predictions_big, "大樂透", "🔵", fortune_results_big),
        (predictions_super, "威力彩", "🟡", fortune_results_super),
    ]:
        lines.extend(_extract_draw_header(predictions, lottery_name, emoji))

        algo_lines = []
        for algo_name, result in predictions.items():
            if not isinstance(result, dict) or "error" in result:
                continue
            numbers = result.get("numbers", [])
            special = result.get("special", 0)
            if numbers:
                algo_lines.append(f"  • {algo_name}: {_format_numbers(numbers, special)}")

        if algo_lines:
            lines.append("📊 算法預測:")
            lines.extend(algo_lines)

        # Fortune (命理) for this lottery type
        _append_fortune_lines(lines, fortune, indent="  ")

        lines.append("")

    lines.append("🤖 LotteryPython 自動推送")
    return "\n".join(lines)


def notify_all_users_both(
    predictions_big: dict,
    predictions_super: dict,
    df_big=None,
    df_super=None,
    dry_run: bool = False,
) -> dict:
    """Push combined predictions for both lottery types to all enabled users.

    Args:
        predictions_big: Algorithm predictions for 大樂透
        predictions_super: Algorithm predictions for 威力彩
        df_big: Historical DataFrame for 大樂透 (used for fortune predictions)
        df_super: Historical DataFrame for 威力彩
        dry_run: If True, log without calling LINE API

    Returns:
        Summary dict: {'notified': int, 'failed': int, 'skipped': int}
    """
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    if not token:
        logger.warning("LINE_CHANNEL_ACCESS_TOKEN not set — skipping LINE notifications")
        return {"notified": 0, "failed": 0, "skipped": 0}

    from predict.astrology.profiles import UserManager, DB_PATH
    user_manager = UserManager(DB_PATH)
    notifiable = user_manager.get_notifiable_users()

    if not notifiable:
        logger.info("No users with enable_line_notify=1 — nothing to push")
        return {"notified": 0, "failed": 0, "skipped": 0}

    notifier = LineNotifier(token) if not dry_run else None
    summary = {"notified": 0, "failed": 0, "skipped": 0}

    for user in notifiable:
        line_ids = user.get("line_ids", [])
        if not line_ids:
            logger.info(f"User '{user['username']}' has no line_ids — skipping")
            summary["skipped"] += 1
            continue

        # Build fortune (命理) separately for each lottery type if enabled
        fortune_big = None
        fortune_super = None
        if user.get("enable_fortune"):
            if df_big is not None:
                fortune_big = run_fortune_for_user(user["id"], "big", df_big)
            if df_super is not None:
                fortune_super = run_fortune_for_user(user["id"], "super", df_super)

        message = build_combined_message(
            predictions_big=predictions_big,
            predictions_super=predictions_super,
            fortune_results_big=fortune_big,
            fortune_results_super=fortune_super,
        )
        chunks = split_message(message)

        for line_id in line_ids:
            if dry_run:
                logger.info(f"[DRY RUN] Would push to {user['username']} ({line_id}):")
                for chunk in chunks:
                    logger.info(f"\n{chunk}")
                summary["notified"] += 1
            else:
                ok = notifier.push_messages(line_id, chunks)
                if ok:
                    logger.info(f"Pushed to {user['username']} ({line_id})")
                    summary["notified"] += 1
                else:
                    logger.error(f"Failed to push to {user['username']} ({line_id})")
                    summary["failed"] += 1

    return summary


def build_update_message(lottery_name: str, draws: list) -> str:
    """Build a short message announcing new draw results.

    Args:
        lottery_name: e.g. '大樂透' or '威力彩'
        draws: List of draw dicts with period, date, numbers, special

    Returns:
        Formatted message string
    """
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"📢 {lottery_name} 開獎通知", f"📅 {today}", ""]
    for draw in draws:
        nums = draw.get("numbers", [])
        special = draw.get("special", 0)
        period = draw.get("period", "")
        date = draw.get("date", "")
        lines.append(f"期號 {period} ({date})")
        lines.append(f"號碼: {_format_numbers(nums, special)}")
        lines.append("")
    lines.append("🤖 LotteryPython 自動推送")
    return "\n".join(lines)


def notify_data_update(
    lottery_type: str,
    lottery_name: str,
    draws: list,
    dry_run: bool = False,
) -> dict:
    """Push new draw results to all notifiable users immediately after data update.

    Args:
        lottery_type: 'big' or 'super'
        lottery_name: Chinese name
        draws: List of new draw dicts from update_data.main()
        dry_run: If True, log without calling API

    Returns:
        Summary dict: {'notified': int, 'failed': int, 'skipped': int}
    """
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    if not token:
        logger.warning("LINE_CHANNEL_ACCESS_TOKEN not set — skipping update notification")
        return {"notified": 0, "failed": 0, "skipped": 0}

    from predict.astrology.profiles import UserManager, DB_PATH
    user_manager = UserManager(DB_PATH)
    notifiable = user_manager.get_notifiable_users()

    if not notifiable:
        return {"notified": 0, "failed": 0, "skipped": 0}

    notifier = LineNotifier(token) if not dry_run else None
    message = build_update_message(lottery_name, draws)
    summary = {"notified": 0, "failed": 0, "skipped": 0}

    for user in notifiable:
        line_ids = user.get("line_ids", [])
        if not line_ids:
            summary["skipped"] += 1
            continue
        for line_id in line_ids:
            if dry_run:
                logger.info(f"[DRY RUN] Update notify to {user['username']} ({line_id}):\n{message}")
                summary["notified"] += 1
            else:
                ok = notifier.push_message(line_id, message)
                if ok:
                    logger.info(f"Update notified: {user['username']} ({line_id})")
                    summary["notified"] += 1
                else:
                    logger.error(f"Update notify failed: {user['username']} ({line_id})")
                    summary["failed"] += 1

    return summary


def run_fortune_for_user(user_id: int, lottery_type: str, df) -> dict:
    """Run fortune (命理) predictions for a specific user.

    Args:
        user_id: User's database ID
        lottery_type: 'big' or 'super'
        df: Historical lottery DataFrame

    Returns:
        Dict with 'ziwei' and 'zodiac' results, or empty dict on failure
    """
    try:
        from predict.lotto_predict_astrology import predict_ziwei, predict_zodiac, has_profiles
        if not has_profiles(user_id):
            logger.warning(f"User {user_id} has enable_fortune=1 but no birth profiles — skipping fortune")
            return {}
        results = {}
        try:
            results["ziwei"] = predict_ziwei(lottery_type, user_id=user_id)
        except Exception as e:
            logger.warning(f"predict_ziwei failed for user {user_id}: {e}")
        try:
            results["zodiac"] = predict_zodiac(lottery_type, user_id=user_id)
        except Exception as e:
            logger.warning(f"predict_zodiac failed for user {user_id}: {e}")
        return results
    except Exception as e:
        logger.warning(f"Fortune prediction setup failed for user {user_id}: {e}")
        return {}


def notify_all_users(
    predictions: dict,
    lottery_type: str,
    lottery_name: str,
    df=None,
    dry_run: bool = False,
) -> dict:
    """Push predictions to all users with LINE notifications enabled.

    Args:
        predictions: Dict of algorithm predictions from run_predictions()
        lottery_type: 'big' or 'super'
        lottery_name: Chinese lottery name
        df: Historical DataFrame (needed for fortune predictions)
        dry_run: If True, log messages without actually calling LINE API

    Returns:
        Summary dict: {'notified': int, 'failed': int, 'skipped': int}
    """
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    if not token:
        logger.warning("LINE_CHANNEL_ACCESS_TOKEN not set — skipping LINE notifications")
        return {"notified": 0, "failed": 0, "skipped": 0}

    from predict.astrology.profiles import UserManager, DB_PATH
    user_manager = UserManager(DB_PATH)
    notifiable = user_manager.get_notifiable_users()

    if not notifiable:
        logger.info("No users with enable_line_notify=1 — nothing to push")
        return {"notified": 0, "failed": 0, "skipped": 0}

    notifier = LineNotifier(token) if not dry_run else None
    summary = {"notified": 0, "failed": 0, "skipped": 0}

    for user in notifiable:
        line_ids = user.get("line_ids", [])
        if not line_ids:
            logger.info(f"User '{user['username']}' has no line_ids — skipping")
            summary["skipped"] += 1
            continue

        # Build fortune if enabled
        fortune_results = None
        if user.get("enable_fortune") and df is not None:
            fortune_results = run_fortune_for_user(user["id"], lottery_type, df)

        message = build_prediction_message(
            lottery_type=lottery_type,
            lottery_name=lottery_name,
            predictions=predictions,
            fortune_results=fortune_results,
        )
        chunks = split_message(message)

        for line_id in line_ids:
            if dry_run:
                logger.info(f"[DRY RUN] Would push to {user['username']} ({line_id}):")
                for chunk in chunks:
                    logger.info(f"\n{chunk}")
                summary["notified"] += 1
            else:
                ok = notifier.push_messages(line_id, chunks)
                if ok:
                    logger.info(f"Pushed to {user['username']} ({line_id})")
                    summary["notified"] += 1
                else:
                    logger.error(f"Failed to push to {user['username']} ({line_id})")
                    summary["failed"] += 1

    return summary
