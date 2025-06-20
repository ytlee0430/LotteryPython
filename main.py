# -*- coding: utf-8 -*-
"""Main entry for lottery utilities.

This script can update lottery draw data and also contains legacy
analysis code. Use ``--update`` with ``--type`` to append the latest
results to Google Sheets.
"""

import argparse
from lotterypython.update_data import main as update_lottery_data

def _legacy_analysis(lotto_type: str) -> None:
    """Run prediction scripts using Google Sheet data.

    This function downloads draw history from Google Sheets for the given
    ``lotto_type`` (``"big"`` or ``"super"``) and feeds the data to every
    predictor in the :mod:`predict` package. The results of each predictor are
    printed to stdout.
    """

    from pathlib import Path

    import pandas as pd
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials

    from predict.lotto_predict_hot_50 import predict_hot50
    from predict.lotto_predict_rf_gb_knn import predict_algorithms
    from predict.lotto_predict_lstm import predict_lstm
    from predict.lotto_predict_LSTMRF import predict_lstm_rf
    from predict import lotto_predict_radom

    from lotterypython.update_data import lotteryTypeAndTitleDict

    scope = ["https://spreadsheets.google.com/feeds"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)

    sheet = client.open_by_key("1WApSh6XbBkcjAhDUyO8IvufhPHUX40MOIskl1qL89hQ").worksheet(
        lotteryTypeAndTitleDict[lotto_type] + "-" + "落球順"
    )
    records = sheet.get_all_records()
    if not records:
        print("No data found in Google Sheet")
        return

    df = pd.DataFrame(records)
    for col in ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Special"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    today_index = len(df)

    main_nums, special = predict_hot50(df, today_index)
    print("===== Hot-50 prediction =====")
    print("numbers:", sorted(main_nums))
    print("special:", special)

    results, sp_rf = predict_algorithms(df)
    print("\n===== RF/GB/KNN prediction =====")
    for name, nums in results.items():
        print(f"{name}: {sorted(nums)} + SP:{sp_rf}")

    print("\n===== Random prediction =====")
    print("numbers:", sorted(lotto_predict_radom.numbers_1_to_38))
    print("special:", lotto_predict_radom.number_1_to_7)

    csv_path = Path("lotterypython") / (
        "big_sequence.csv" if lotto_type == "big" else "super_sequence.csv"
    )
    df.to_csv(csv_path, index=False)

    nums_lstm, sp_lstm = predict_lstm(df)
    print("\n===== LSTM prediction =====")
    print("numbers:", nums_lstm)
    print("special:", sp_lstm)

    power_csv = Path("power_lottery.csv")
    df.to_csv(power_csv, index=False)
    nums_ai, sp_ai = predict_lstm_rf(df)
    print("\n===== AI result =====")
    print("numbers :", nums_ai)
    print("sp  :", sp_ai)
    print("=======================")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lottery utilities")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Fetch latest draws and append to Google Sheets",
    )
    parser.add_argument(
        "--type",
        choices=["big", "super"],
        default="big",
        help="Lottery type: big (lotto649) or super (superlotto638)",
    )
    args = parser.parse_args()

    if args.update:
        update_lottery_data(args.type)
    else:
        _legacy_analysis(args.type)


if __name__ == "__main__":
    main()
