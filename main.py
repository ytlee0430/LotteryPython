# -*- coding: utf-8 -*-
"""Main entry for lottery utilities.

This script can update lottery draw data and also contains legacy
analysis code. Use ``--update`` with ``--type`` to append the latest
results to Google Sheets.
"""

import argparse
from lotterypython.update_data import main as update_lottery_data

def _legacy_analysis(lotto_type: str, *, save_results: bool = False) -> None:
    """Run prediction scripts using Google Sheet data.

    This function downloads draw history from Google Sheets for the given
    ``lotto_type`` (``"big"`` or ``"super"``) and feeds the data to every
    predictor in the :mod:`predict` package. The results of each predictor are
    printed to stdout.
    """
    from pathlib import Path
    from lotterypython.analysis_sheet import append_analysis_results
    from lotterypython.logic import get_data_from_gsheet, run_predictions

    df = get_data_from_gsheet(lotto_type)
    if df.empty:
        print("No data found in Google Sheet")
        return

    # Save CSVs to maintain legacy side effects
    csv_path = Path("lotterypython") / (
        "big_sequence.csv" if lotto_type == "big" else "super_sequence.csv"
    )
    df.to_csv(csv_path, index=False)

    power_csv = Path("power_lottery.csv")
    df.to_csv(power_csv, index=False)

    # Run all predictions
    results = run_predictions(df)

    predictions: list[tuple[str, str, list[int], int]] = []

    # 1. Hot-50
    if "Hot-50" in results and "error" not in results["Hot-50"]:
        r = results["Hot-50"]
        print("===== Hot-50 prediction =====")
        print("numbers:", r["numbers"])
        print("special:", r["special"])
        predictions.append(("Hot-50", r["next_period"], r["numbers"], r["special"]))

    # 2. RF/GB/KNN
    print("\n===== RF/GB/KNN prediction =====")
    for key in ["RandomForest", "GradientBoosting", "KNN"]:
        if key in results and "error" not in results[key]:
            r = results[key]
            print(f"{key}: {r['numbers']} + SP:{r['special']}")
            predictions.append((key, r["next_period"], r["numbers"], r["special"]))

    # 3. Random
    if "Random" in results and "error" not in results["Random"]:
        r = results["Random"]
        print("\n===== Random prediction =====")
        print("numbers:", r["numbers"])
        print("special:", r["special"])
        predictions.append(
            (
                "Random",
                r["next_period"],
                r["numbers"],
                r["special"],
            )
        )

    # 4. LSTM
    if "LSTM" in results and "error" not in results["LSTM"]:
        r = results["LSTM"]
        print("\n===== LSTM prediction =====")
        print("numbers:", r["numbers"])
        print("special:", r["special"])
        predictions.append(("LSTM", r["next_period"], r["numbers"], r["special"]))

    # 5. LSTM-RF
    if "LSTM-RF" in results and "error" not in results["LSTM-RF"]:
        r = results["LSTM-RF"]
        print("\n===== AI result =====")
        print("numbers :", r["numbers"])
        print("sp  :", r["special"])
        print("=======================")
        predictions.append(("LSTM-RF", r["next_period"], r["numbers"], r["special"]))

    if save_results:
        append_analysis_results(predictions, lotto_type)


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
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Append prediction results to the Google Sheet",
    )
    args = parser.parse_args()

    if args.update:
        update_lottery_data(args.type)
    else:
        _legacy_analysis(args.type, save_results=args.save_results)


if __name__ == "__main__":
    main()
