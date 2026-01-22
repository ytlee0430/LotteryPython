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
from lotterypython.utils import get_draw_info, format_combination_reason

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

def run_predictions(df: pd.DataFrame, use_cache: bool = True, user_id: int = None) -> dict:
    """Run all prediction algorithms on the dataframe.

    Args:
        df: DataFrame with lottery history
        use_cache: Whether to use cached results if available
        user_id: Owner user's ID for user-specific cache

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

    # Get draw info for display
    draw_info = get_draw_info(lottery_type, next_period)

    # Check cache first
    if use_cache and next_period:
        cache_manager = get_all_cache_manager()
        cached = cache_manager.get_cached_predictions(lottery_type, next_period, user_id)
        if cached:
            # Add from_cache flag and draw_info to each result
            for key in cached:
                if isinstance(cached[key], dict) and 'error' not in cached[key]:
                    cached[key]['from_cache'] = True
                    cached[key]['draw_info'] = draw_info
            return cached

    results = {}
    
    # Hot-50
    try:
        main_nums, special, details = predict_hot50(df, today_index)
        results["Hot-50"] = {
            "next_period": next_period,
            "numbers": sorted(main_nums),
            "special": int(special),
            "details": details
        }
    except Exception as e:
        results["Hot-50"] = {"error": str(e)}

    # Cold-50
    try:
        nums_cold, sp_cold, details = predict_cold50(df, today_index, lottery_type=lottery_type)
        results["Cold-50"] = {
            "next_period": next_period,
            "numbers": sorted(nums_cold),
            "special": int(sp_cold),
            "details": details
        }
    except Exception as e:
        results["Cold-50"] = {"error": str(e)}

    # RF/GB/KNN
    try:
        alg_results, sp_rf, alg_details = predict_algorithms(df)
        for name, nums in alg_results.items():
            results[name] = {
                "next_period": next_period,
                "numbers": sorted(nums),
                "special": int(sp_rf),
                "details": alg_details.get(name, {})
            }
    except Exception as e:
        results["RF_GB_KNN_Error"] = str(e)


    # LSTM
    try:
        nums_lstm, sp_lstm, details = predict_lstm(df, lottery_type=lottery_type)
        results["LSTM"] = {
            "next_period": next_period,
            "numbers": sorted(nums_lstm),
            "special": int(sp_lstm),
            "details": details
        }
    except Exception as e:
        results["LSTM"] = {"error": str(e)}

    # LSTM-RF
    try:
        nums_ai, sp_ai, details = predict_lstm_rf(df, lottery_type=lottery_type)
        results["LSTM-RF"] = {
            "next_period": next_period,
            "numbers": sorted(nums_ai),
            "special": int(sp_ai),
            "details": details
        }
    except Exception as e:
        results["LSTM-RF"] = {"error": str(e)}

    # XGBoost
    try:
        nums_xgb, sp_xgb, details = predict_xgboost(df, today_index)
        results["XGBoost"] = {
            "next_period": next_period,
            "numbers": sorted(nums_xgb),
            "special": int(sp_xgb),
            "details": details
        }
    except Exception as e:
        results["XGBoost"] = {"error": str(e)}

    # Markov Chain
    try:
        nums_markov, sp_markov, details = predict_markov(df, today_index)
        results["Markov"] = {
            "next_period": next_period,
            "numbers": sorted(nums_markov),
            "special": int(sp_markov),
            "details": details
        }
    except Exception as e:
        results["Markov"] = {"error": str(e)}

    # Pattern Analysis
    try:
        nums_pattern, sp_pattern, details = predict_pattern(df, today_index)
        results["Pattern"] = {
            "next_period": next_period,
            "numbers": sorted(nums_pattern),
            "special": int(sp_pattern),
            "details": details
        }
    except Exception as e:
        results["Pattern"] = {"error": str(e)}

    # Astrology predictions (only if profiles exist for this user)
    if has_profiles(user_id):
        # Astrology-Ziwei (紫微斗數)
        try:
            nums_ziwei, sp_ziwei, details_ziwei = predict_ziwei(lottery_type, user_id=user_id)
            predictions_ziwei = details_ziwei.get("predictions", [])
            combination_ziwei = format_combination_reason(predictions_ziwei, sorted(nums_ziwei), int(sp_ziwei))
            results["Astrology-Ziwei"] = {
                "next_period": next_period,
                "numbers": sorted(nums_ziwei),
                "special": int(sp_ziwei),
                "details": predictions_ziwei,
                "combination_reason": combination_ziwei
            }
        except Exception as e:
            results["Astrology-Ziwei"] = {"error": str(e)}

        # Astrology-Zodiac (西洋星座)
        try:
            nums_zodiac, sp_zodiac, details_zodiac = predict_zodiac(lottery_type, user_id=user_id)
            predictions_zodiac = details_zodiac.get("predictions", [])
            combination_zodiac = format_combination_reason(predictions_zodiac, sorted(nums_zodiac), int(sp_zodiac))
            results["Astrology-Zodiac"] = {
                "next_period": next_period,
                "numbers": sorted(nums_zodiac),
                "special": int(sp_zodiac),
                "details": predictions_zodiac,
                "combination_reason": combination_zodiac
            }
        except Exception as e:
            results["Astrology-Zodiac"] = {"error": str(e)}

    # Ensemble Voting
    try:
        nums_ensemble, sp_ensemble, details = predict_ensemble(df, today_index, previous_results=results)
        results["Ensemble"] = {
            "next_period": next_period,
            "numbers": sorted(nums_ensemble),
            "special": int(sp_ensemble),
            "details": details
        }
    except Exception as e:
        results["Ensemble"] = {"error": str(e)}

    # Add from_cache=False and draw_info to all fresh results
    for key in results:
        if isinstance(results[key], dict) and 'error' not in results[key]:
            results[key]['from_cache'] = False
            results[key]['draw_info'] = draw_info

    # Save to cache
    if use_cache and next_period:
        cache_manager = get_all_cache_manager()
        cache_manager.save_predictions(lottery_type, next_period, results, user_id)

    return results
