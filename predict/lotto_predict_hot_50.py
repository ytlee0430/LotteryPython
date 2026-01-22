"""
Hot-50 Prediction Algorithm

Predicts numbers that have been most frequently drawn in recent draws (hot numbers).
Based on the "momentum" hypothesis - hot numbers tend to stay hot.
"""

from collections import Counter


def predict_hot50(df, today_index, window=50):
    """
    Predict using the hottest (most frequent) numbers from recent draws.

    Args:
        df: DataFrame with historical lottery data
        today_index: Current index for prediction
        window: Number of recent draws to analyze (default 50)

    Returns:
        tuple: (main_numbers, special_number, details)
    """
    start_idx = max(0, today_index - window)
    train = df.iloc[start_idx:today_index]

    nums = train[['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth']].values.ravel()
    counter = Counter(nums)
    top_10 = counter.most_common(10)
    cnt = counter.most_common(6)
    main = [int(n) for n, _ in cnt]
    special = train['Special'].value_counts().idxmax()
    special = int(special)

    # Return details with top 10 and their frequency counts
    details = {
        "type": "frequency_ranking",
        "top_10": [[int(n), int(freq)] for n, freq in top_10],
        "window": window,
        "note": f"Based on last {window} draws"
    }
    return main, special, details
