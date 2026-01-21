"""
Ensemble Voting Prediction Algorithm

Combines predictions from multiple models using weighted voting
to produce a final prediction that leverages collective model intelligence.
"""

from collections import Counter

from predict.lotto_predict_hot_50 import predict_hot50
from predict.lotto_predict_cold_50 import predict_cold50
from predict.lotto_predict_rf_gb_knn import predict_algorithms
from predict.lotto_predict_xgboost import predict_xgboost
from predict.lotto_predict_lstm import predict_lstm
from predict.lotto_predict_LSTMRF import predict_lstm_rf
from predict.lotto_predict_markov import predict_markov
from predict.lotto_predict_pattern import predict_pattern


# Default weights based on historical performance
# RandomForest: 222 matches, KNN: 205, GradientBoosting: 203
DEFAULT_WEIGHTS = {
    "Hot-50": 1.0,
    "Cold-50": 0.8,           # Complementary to Hot-50
    "RandomForest": 1.5,      # Highest historical match rate
    "GradientBoosting": 1.0,
    "KNN": 1.2,
    "XGBoost": 1.3,           # Enhanced gradient boosting
    "LSTM": 1.0,
    "LSTM-RF": 1.2,
    "Markov": 1.0,            # Transition probability based
    "Pattern": 0.9,           # Pattern matching
    "Astrology-Ziwei": 0.8,   # 紫微斗數 prediction
    "Astrology-Zodiac": 0.7,  # 西洋星座 prediction
}


def collect_predictions(df, today_index, previous_results=None):
    """
    Collect predictions from all available models.

    Args:
        df: DataFrame with historical lottery data
        today_index: Current index for prediction
        previous_results: Optional dict containing already calculated results

    Returns:
        list of dict: Each dict contains 'name', 'numbers', 'special', 'weight'
    """
    predictions = []

    if previous_results:
        # Use provided results instead of recalculating
        for name, result in previous_results.items():
            if name == "Ensemble" or "error" in result:
                continue
            
            # Skip if result format is not as expected
            if "numbers" not in result or "special" not in result:
                continue

            predictions.append({
                "name": name,
                "numbers": result["numbers"],
                "special": result["special"],
                "weight": DEFAULT_WEIGHTS.get(name, 1.0)
            })
        return predictions

    # Fallback to calculating if no results provided
    
    # Hot-50
    try:
        main_nums, special, _ = predict_hot50(df, today_index)
        predictions.append({
            "name": "Hot-50",
            "numbers": list(main_nums),
            "special": int(special),
            "weight": DEFAULT_WEIGHTS.get("Hot-50", 1.0)
        })
    except Exception:
        pass

    # Cold-50
    try:
        nums_cold, sp_cold, _ = predict_cold50(df, today_index)
        predictions.append({
            "name": "Cold-50",
            "numbers": list(nums_cold),
            "special": int(sp_cold),
            "weight": DEFAULT_WEIGHTS.get("Cold-50", 1.0)
        })
    except Exception:
        pass

    # RandomForest, GradientBoosting, KNN
    try:
        alg_results, sp_rf, _ = predict_algorithms(df)
        for name, nums in alg_results.items():
            predictions.append({
                "name": name,
                "numbers": list(nums),
                "special": int(sp_rf),
                "weight": DEFAULT_WEIGHTS.get(name, 1.0)
            })
    except Exception:
        pass

    # XGBoost
    try:
        nums_xgb, sp_xgb, _ = predict_xgboost(df, today_index)
        predictions.append({
            "name": "XGBoost",
            "numbers": list(nums_xgb),
            "special": int(sp_xgb),
            "weight": DEFAULT_WEIGHTS.get("XGBoost", 1.0)
        })
    except Exception:
        pass

    # LSTM
    try:
        nums_lstm, sp_lstm, _ = predict_lstm(df)
        predictions.append({
            "name": "LSTM",
            "numbers": list(nums_lstm),
            "special": int(sp_lstm),
            "weight": DEFAULT_WEIGHTS.get("LSTM", 1.0)
        })
    except Exception:
        pass

    # LSTM-RF
    try:
        nums_ai, sp_ai, _ = predict_lstm_rf(df)
        predictions.append({
            "name": "LSTM-RF",
            "numbers": list(nums_ai),
            "special": int(sp_ai),
            "weight": DEFAULT_WEIGHTS.get("LSTM-RF", 1.0)
        })
    except Exception:
        pass

    # Markov Chain
    try:
        nums_markov, sp_markov, _ = predict_markov(df, today_index)
        predictions.append({
            "name": "Markov",
            "numbers": list(nums_markov),
            "special": int(sp_markov),
            "weight": DEFAULT_WEIGHTS.get("Markov", 1.0)
        })
    except Exception:
        pass

    # Pattern Analysis
    try:
        nums_pattern, sp_pattern, _ = predict_pattern(df, today_index)
        predictions.append({
            "name": "Pattern",
            "numbers": list(nums_pattern),
            "special": int(sp_pattern),
            "weight": DEFAULT_WEIGHTS.get("Pattern", 1.0)
        })
    except Exception:
        pass

    return predictions


def weighted_vote_numbers(predictions, top_n=6):
    """
    Perform weighted voting on main numbers.

    Args:
        predictions: List of prediction dicts
        top_n: Number of top numbers to select

    Returns:
        list: Top N numbers by weighted vote
    """
    if not predictions:
        return []

    vote_counter = Counter()

    for pred in predictions:
        weight = pred.get("weight", 1.0)
        for num in pred["numbers"]:
            vote_counter[num] += weight

    # Get top N numbers by vote count
    top_numbers = [num for num, _ in vote_counter.most_common(top_n)]
    return top_numbers


def weighted_vote_special(predictions):
    """
    Perform weighted voting on special number.

    Args:
        predictions: List of prediction dicts

    Returns:
        int: Special number with highest weighted vote
    """
    if not predictions:
        return 1

    vote_counter = Counter()

    for pred in predictions:
        weight = pred.get("weight", 1.0)
        vote_counter[pred["special"]] += weight

    # Return special with highest vote
    return vote_counter.most_common(1)[0][0]


def predict_ensemble(df, today_index, weights=None, previous_results=None):
    """
    Main ensemble voting prediction function.

    Args:
        df: DataFrame with historical lottery data
        today_index: Current index for prediction
        weights: Optional custom weights dict (model_name -> weight)
        previous_results: Optional dict containing already calculated results

    Returns:
        tuple: (main_numbers, special_number)
    """
    # Update weights if custom weights provided
    if weights:
        DEFAULT_WEIGHTS.update(weights)

    # Collect all predictions
    predictions = collect_predictions(df, today_index, previous_results)

    if not predictions:
        # If no previous results and calculation failed, or just failed
        if previous_results: 
             # Try to calculate if previous_results were empty or invalid? 
             # The user asked to use passed results, so maybe fallback is not needed if intended usage is passing results.
             # But let's keep it safe. If previous_results yielded nothing, maybe we should try collecting again without it?
             # For now, let's assume if previous_results is passed, it should have data.
             # But to be robust:
             predictions = collect_predictions(df, today_index)

    if not predictions:
         raise ValueError("No predictions collected from any model")

    # Weighted voting for main numbers
    main_numbers = weighted_vote_numbers(predictions, top_n=6)

    # Weighted voting for special number
    special = weighted_vote_special(predictions)

    # Get top 10 candidates with vote weights
    vote_counter = Counter()
    for pred in predictions:
        weight = pred.get("weight", 1.0)
        for num in pred["numbers"]:
            vote_counter[num] += weight

    top_10 = [[int(num), round(float(votes), 2)] for num, votes in vote_counter.most_common(10)]

    details = {
        "type": "ensemble_voting",
        "top_10": top_10,
        "models_used": len(predictions),
        "note": "Weighted voting from multiple algorithms"
    }
    return main_numbers, special, details


def get_vote_details(df, today_index, previous_results=None):
    """
    Get detailed voting breakdown for analysis.

    Returns:
        dict: Contains predictions, vote_counts, and final_result
    """
    predictions = collect_predictions(df, today_index, previous_results)

    # Number vote breakdown
    number_votes = Counter()
    for pred in predictions:
        weight = pred.get("weight", 1.0)
        for num in pred["numbers"]:
            number_votes[num] += weight

    # Special vote breakdown
    special_votes = Counter()
    for pred in predictions:
        weight = pred.get("weight", 1.0)
        special_votes[pred["special"]] += weight

    main_numbers = weighted_vote_numbers(predictions, top_n=6)
    special = weighted_vote_special(predictions)

    return {
        "individual_predictions": predictions,
        "number_vote_counts": dict(number_votes.most_common(15)),
        "special_vote_counts": dict(special_votes.most_common(5)),
        "final_numbers": sorted(main_numbers),
        "final_special": special,
        "models_used": len(predictions)
    }


if __name__ == "__main__":
    import pandas as pd

    # Test with sample data
    CSV_FILE = "lotterypython/big_sequence.csv"
    df = pd.read_csv(CSV_FILE)

    today_index = len(df)
    main_nums, special = predict_ensemble(df, today_index)

    print("===== Ensemble Voting Prediction =====")
    print(f"Numbers: {sorted(main_nums)} + Special: {special}")

    # Show vote details
    details = get_vote_details(df, today_index)
    print(f"\nModels used: {details['models_used']}")
    print(f"Top number votes: {details['number_vote_counts']}")
