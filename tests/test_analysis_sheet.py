from datetime import date
from unittest.mock import MagicMock, patch

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lotterypython.analysis_sheet import append_analysis_results


def test_append_analysis_results_formats_rows_and_appends():
    preds = [
        ("Algo1", "123", [1, 2, 3, 4, 5, 6], 7),
        ("Algo2", "124", [11, 12, 13, 14, 15, 16], 8),
    ]

    with patch("lotterypython.analysis_sheet.gspread") as gs, patch(
        "lotterypython.analysis_sheet.ServiceAccountCredentials"
    ) as sac:
        gs.WorksheetNotFound = Exception
        client = MagicMock()
        worksheet = MagicMock()
        workbook = MagicMock()
        client.open_by_key.return_value = workbook
        workbook.worksheet.return_value = worksheet
        gs.authorize.return_value = client
        sac.from_json_keyfile_name.return_value = "creds"

        append_analysis_results(preds, "big")

        sac.from_json_keyfile_name.assert_called_once()
        client.open_by_key.assert_called_once_with(
            "1WApSh6XbBkcjAhDUyO8IvufhPHUX40MOIskl1qL89hQ"
        )
        workbook.worksheet.assert_called_once_with("分析結果")
        worksheet.get_all_values.assert_called_once_with()
        today = date.today().isoformat()
        header = ["預測期數", "日期", "彩種", "演算法", "號碼", "特別號"]
        expected_rows = [
            ["124", today, "big", "Algo2", "11 12 13 14 15 16", "8"],
            ["123", today, "big", "Algo1", "1 2 3 4 5 6", "7"],
        ]
        worksheet.clear.assert_called_once_with()
        worksheet.update.assert_called_once_with("A1", [header] + expected_rows)


def test_append_analysis_results_normalizes_existing_rows():
    preds = [("Algo", "200", [1, 2, 3, 4, 5, 6], 7)]

    with patch("lotterypython.analysis_sheet.gspread") as gs, patch(
        "lotterypython.analysis_sheet.ServiceAccountCredentials"
    ) as sac:
        gs.WorksheetNotFound = Exception
        client = MagicMock()
        worksheet = MagicMock()
        workbook = MagicMock()
        client.open_by_key.return_value = workbook
        workbook.worksheet.return_value = worksheet
        worksheet.get_all_values.return_value = [["2023-01-01", "big", "Old", "1 2 3 4 5 6", "7"]]
        gs.authorize.return_value = client
        sac.from_json_keyfile_name.return_value = "creds"

        append_analysis_results(preds, "big")

        header = ["預測期數", "日期", "彩種", "演算法", "號碼", "特別號"]
        today = date.today().isoformat()
        expected_rows = [
            ["200", today, "big", "Algo", "1 2 3 4 5 6", "7"],
            ["", "2023-01-01", "big", "Old", "1 2 3 4 5 6", "7"],
        ]
        worksheet.update.assert_called_once_with("A1", [header] + expected_rows)
