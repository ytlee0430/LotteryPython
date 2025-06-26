from __future__ import annotations

from datetime import date
import gspread
from oauth2client.service_account import ServiceAccountCredentials


def append_analysis_results(predictions: list[tuple[str, list[int], int]], lotto_type: str) -> None:
    """Append prediction rows to the "分析結果" worksheet.

    Parameters
    ----------
    predictions:
        A list of tuples ``(algorithm, numbers, special)``.
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
        sheet = book.add_worksheet(title="分析結果", rows="100", cols="5")

    today = date.today().isoformat()
    rows = []
    for algorithm, numbers, special in predictions:
        num_str = " ".join(str(n) for n in numbers)
        rows.append([today, lotto_type, algorithm, num_str, special])

    if rows:
        sheet.append_rows(rows)
