from __future__ import annotations

from datetime import date
import gspread
from oauth2client.service_account import ServiceAccountCredentials


HEADER = ["預測期數", "日期", "彩種", "演算法", "號碼", "特別號"]


def _normalize_row(row: list[str]) -> list[str]:
    """Ensure an existing worksheet row has the expected number of columns."""

    target_len = len(HEADER)
    normalized = list(row)
    if len(normalized) < target_len:
        normalized = [""] * (target_len - len(normalized)) + normalized
    elif len(normalized) > target_len:
        normalized = normalized[:target_len]
    return normalized


def _sort_key(row: list[str]) -> tuple[int, int | str]:
    """Return a key for sorting rows by predicted period descending."""

    period = row[0].strip()
    if period.isdigit():
        return (1, int(period))
    return (0, period)


def append_analysis_results(
    predictions: list[tuple[str, str, list[int], int]],
    lotto_type: str,
) -> None:
    """Append prediction rows to the "分析結果" worksheet.

    Parameters
    ----------
    predictions:
        A list of tuples ``(algorithm, period, numbers, special)``.
    lotto_type:
        Either ``"big"`` or ``"super"``.
    """
    scope = ["https://spreadsheets.google.com/feeds"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    book = client.open_by_key("1WApSh6XbBkcjAhDUyO8IvufhPHUX40MOIskl1qL89hQ")
    try:
        sheet = book.worksheet("分析結果")
    except getattr(gspread, "WorksheetNotFound", Exception):
        sheet = book.add_worksheet(title="分析結果", rows="100", cols="6")

    today = date.today().isoformat()
    rows = []
    existing_values = sheet.get_all_values()

    existing_rows: list[list[str]] = []
    if existing_values:
        if existing_values[0] == HEADER:
            values_iter = existing_values[1:]
        else:
            values_iter = existing_values
        for row in values_iter:
            if not any(cell.strip() for cell in row):
                continue
            existing_rows.append(_normalize_row(row))

    for algorithm, period, numbers, special in predictions:
        num_str = " ".join(str(n) for n in numbers)
        rows.append([str(period), today, lotto_type, algorithm, num_str, str(special)])

    if not rows and not existing_rows:
        return

    merged_rows = existing_rows + rows
    merged_rows.sort(key=_sort_key, reverse=True)

    sheet.clear()
    sheet.update("A1", [HEADER] + merged_rows)
