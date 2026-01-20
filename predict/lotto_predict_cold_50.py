"""
Cold-50 Prediction Algorithm

Opposite strategy to Hot-50: predicts numbers that have been
least frequently drawn in the last 50 draws (cold numbers).
Based on the "regression to mean" hypothesis.
"""

from collections import Counter


def predict_cold50(df, today_index, window=50, lottery_type='big'):
    """
    Predict using the coldest (least frequent) numbers from recent draws.

    Args:
        df: DataFrame with historical lottery data
        today_index: Current index for prediction
        window: Number of recent draws to analyze (default 50)
        lottery_type: 'big' for 大樂透 (1-49), 'super' for 威力彩 (1-38, special 1-8)

    Returns:
        tuple: (main_numbers, special_number)
    """
    # Get the window of recent draws
    start_idx = max(0, today_index - window)
    train = df.iloc[start_idx:today_index]

    # Determine number range based on lottery type
    if lottery_type == 'super':
        max_num = 38
        max_special = 8
    else:  # big
        max_num = 49
        max_special = 49

    # Count frequency of all numbers in recent draws
    nums = train[['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth']].values.ravel()
    frequency = Counter(nums)

    # Find the coldest numbers (least frequent or never appeared)
    all_numbers = list(range(1, max_num + 1))

    # Sort by frequency (ascending) - coldest first
    # Numbers not in frequency dict have count 0
    coldest = sorted(all_numbers, key=lambda x: frequency.get(x, 0))

    # Take the 6 coldest numbers
    main = coldest[:6]

    # For special number, find the coldest special within valid range
    special_counts = Counter(train['Special'].values)
    all_specials = list(range(1, max_special + 1))
    coldest_special = min(all_specials, key=lambda x: special_counts.get(x, 0))

    return [int(n) for n in main], int(coldest_special)


def get_cold_analysis(df, today_index, window=50):
    """
    Get detailed cold number analysis.

    Returns:
        dict: Analysis details including frequency counts
    """
    start_idx = max(0, today_index - window)
    train = df.iloc[start_idx:today_index]
    max_num = int(df[['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth']].max().max())

    nums = train[['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth']].values.ravel()
    frequency = Counter(nums)

    # Find numbers that never appeared
    never_appeared = [n for n in range(1, max_num + 1) if frequency.get(n, 0) == 0]

    # Get coldest numbers with their counts
    all_numbers = list(range(1, max_num + 1))
    coldest_with_counts = sorted(
        [(n, frequency.get(n, 0)) for n in all_numbers],
        key=lambda x: x[1]
    )[:10]

    return {
        "window_size": window,
        "never_appeared": never_appeared,
        "coldest_10": coldest_with_counts,
        "total_draws_analyzed": len(train)
    }


if __name__ == "__main__":
    import pandas as pd

    CSV_FILE = "lotterypython/big_sequence.csv"
    df = pd.read_csv(CSV_FILE)
    today_index = len(df)

    main_nums, special = predict_cold50(df, today_index)
    print("===== Cold-50 Prediction =====")
    print(f"Numbers: {sorted(main_nums)} + Special: {special}")

    analysis = get_cold_analysis(df, today_index)
    print(f"\nNever appeared in last 50: {analysis['never_appeared']}")
    print(f"Coldest 10: {analysis['coldest_10']}")
