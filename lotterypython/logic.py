import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from lotterypython.update_data import lotteryTypeAndTitleDict
from pathlib import Path

from predict.lotto_predict_hot_50 import predict_hot50
from predict.lotto_predict_cold_50 import predict_cold50
from predict.lotto_predict_rf_gb_knn import predict_algorithms
from predict.lotto_predict_xgboost import predict_xgboost
from predict.lotto_predict_lstm import predict_lstm
from predict.lotto_predict_LSTMRF import predict_lstm_rf
from predict.lotto_predict_markov import predict_markov
from predict.lotto_predict_pattern import predict_pattern
from predict.lotto_predict_ensemble import predict_ensemble
from predict.lotto_predict_astrology import predict_ziwei, predict_zodiac, has_profiles
from predict import lotto_predict_radom
from predict.astrology.profiles import AllPredictionsCacheManager

# Singleton cache manager
_all_cache_manager = None

def get_all_cache_manager():
    """Get or create the all predictions cache manager singleton."""
    global _all_cache_manager
    if _all_cache_manager is None:
        _all_cache_manager = AllPredictionsCacheManager()
    return _all_cache_manager

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

def run_predictions(df: pd.DataFrame, use_cache: bool = True) -> dict:
    """Run all prediction algorithms on the dataframe.

    Args:
        df: DataFrame with lottery history
        use_cache: Whether to use cached results if available

    Returns:
        dict with all algorithm predictions
    """
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

    # Determine lottery type
    max_num = int(df[['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth']].max().max())
    lottery_type = 'big' if max_num > 38 else 'super'

    # Check cache first
    if use_cache and next_period:
        cache_manager = get_all_cache_manager()
        cached = cache_manager.get_cached_predictions(lottery_type, next_period)
        if cached:
            # Add from_cache flag to each result
            for key in cached:
                if isinstance(cached[key], dict) and 'error' not in cached[key]:
                    cached[key]['from_cache'] = True
            return cached

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

    # Cold-50
    try:
        nums_cold, sp_cold = predict_cold50(df, today_index)
        results["Cold-50"] = {
            "next_period": next_period,
            "numbers": sorted(nums_cold),
            "special": int(sp_cold)
        }
    except Exception as e:
        results["Cold-50"] = {"error": str(e)}

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

    # XGBoost
    try:
        nums_xgb, sp_xgb = predict_xgboost(df, today_index)
        results["XGBoost"] = {
            "next_period": next_period,
            "numbers": sorted(nums_xgb),
            "special": int(sp_xgb)
        }
    except Exception as e:
        results["XGBoost"] = {"error": str(e)}

    # Markov Chain
    try:
        nums_markov, sp_markov = predict_markov(df, today_index)
        results["Markov"] = {
            "next_period": next_period,
            "numbers": sorted(nums_markov),
            "special": int(sp_markov)
        }
    except Exception as e:
        results["Markov"] = {"error": str(e)}

    # Pattern Analysis
    try:
        nums_pattern, sp_pattern = predict_pattern(df, today_index)
        results["Pattern"] = {
            "next_period": next_period,
            "numbers": sorted(nums_pattern),
            "special": int(sp_pattern)
        }
    except Exception as e:
        results["Pattern"] = {"error": str(e)}

    # Astrology predictions (only if profiles exist)
    if has_profiles():
        # Astrology-Ziwei (紫微斗數)
        try:
            nums_ziwei, sp_ziwei, details_ziwei = predict_ziwei(lottery_type)
            results["Astrology-Ziwei"] = {
                "next_period": next_period,
                "numbers": sorted(nums_ziwei),
                "special": int(sp_ziwei),
                "details": details_ziwei.get("predictions", [])
            }
        except Exception as e:
            results["Astrology-Ziwei"] = {"error": str(e)}

        # Astrology-Zodiac (西洋星座)
        try:
            nums_zodiac, sp_zodiac, details_zodiac = predict_zodiac(lottery_type)
            results["Astrology-Zodiac"] = {
                "next_period": next_period,
                "numbers": sorted(nums_zodiac),
                "special": int(sp_zodiac),
                "details": details_zodiac.get("predictions", [])
            }
        except Exception as e:
            results["Astrology-Zodiac"] = {"error": str(e)}

    # Ensemble Voting
    try:
        nums_ensemble, sp_ensemble = predict_ensemble(df, today_index, previous_results=results)
        results["Ensemble"] = {
            "next_period": next_period,
            "numbers": sorted(nums_ensemble),
            "special": int(sp_ensemble)
        }
    except Exception as e:
        results["Ensemble"] = {"error": str(e)}

    # Add from_cache=False to all fresh results
    for key in results:
        if isinstance(results[key], dict) and 'error' not in results[key]:
            results[key]['from_cache'] = False

    # Save to cache
    if use_cache and next_period:
        cache_manager = get_all_cache_manager()
        cache_manager.save_predictions(lottery_type, next_period, results)

    return results
