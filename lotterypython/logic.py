import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from lotterypython.update_data import lotteryTypeAndTitleDict
from pathlib import Path

from predict.lotto_predict_hot_50 import predict_hot50
from predict.lotto_predict_rf_gb_knn import predict_algorithms
from predict.lotto_predict_lstm import predict_lstm
from predict.lotto_predict_LSTMRF import predict_lstm_rf
from predict import lotto_predict_radom

def get_data_from_gsheet(lotto_type: str) -> pd.DataFrame:
    """Fetch lottery data from Google Sheets."""
    scope = ["https://spreadsheets.google.com/feeds"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)

    sheet_title = lotteryTypeAndTitleDict.get(lotto_type, "big-lottery") + "-" + "落球順"
    try:
        sheet = client.open_by_key("1WApSh6XbBkcjAhDUyO8IvufhPHUX40MOIskl1qL89hQ").worksheet(sheet_title)
    except Exception as e:
        print(f"Error opening sheet {sheet_title}: {e}")
        return pd.DataFrame()

    records = sheet.get_all_records()
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    for col in ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Special"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    
    return df

def run_predictions(df: pd.DataFrame) -> dict:
    """Run all prediction algorithms on the dataframe."""
    if df.empty:
        return {}

    today_index = len(df)
    period_values = (
        df["Period"].dropna().astype(str).str.strip() if "Period" in df.columns else []
    )
    numeric_periods = []
    for value in period_values:
        digits_only = "".join(ch for ch in str(value) if ch.isdigit())
        if digits_only:
            numeric_periods.append(digits_only)
    if numeric_periods:
        max_len = max(len(p) for p in numeric_periods)
        next_period = f"{max(int(p) for p in numeric_periods) + 1:0{max_len}d}"
    else:
        next_period = ""

    results = {}
    
    # Hot-50
    try:
        main_nums, special = predict_hot50(df, today_index)
        results["Hot-50"] = {
            "next_period": next_period,
            "numbers": sorted(main_nums),
            "special": int(special)
        }
    except Exception as e:
        results["Hot-50"] = {"error": str(e)}

    # RF/GB/KNN
    try:
        alg_results, sp_rf = predict_algorithms(df)
        for name, nums in alg_results.items():
            results[name] = {
                "next_period": next_period,
                "numbers": sorted(nums),
                "special": int(sp_rf)
            }
    except Exception as e:
        results["RF_GB_KNN_Error"] = str(e)


    # LSTM
    try:
        nums_lstm, sp_lstm = predict_lstm(df)
        results["LSTM"] = {
            "next_period": next_period,
            "numbers": sorted(nums_lstm),
            "special": int(sp_lstm)
        }
    except Exception as e:
        results["LSTM"] = {"error": str(e)}

    # LSTM-RF
    try:
        nums_ai, sp_ai = predict_lstm_rf(df)
        results["LSTM-RF"] = {
            "next_period": next_period,
            "numbers": sorted(nums_ai),
            "special": int(sp_ai)
        }
    except Exception as e:
        results["LSTM-RF"] = {"error": str(e)}
    
    return results
