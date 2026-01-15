"""
Pattern Analysis Prediction Algorithm

Analyzes historical patterns in lottery draws including:
- Odd/Even ratio
- High/Low ratio
- Zone distribution
- Consecutive numbers
- Sum range

Generates predictions matching the most common patterns.
"""

import random
from collections import Counter
import numpy as np


def analyze_draw_pattern(numbers, max_num=49):
    """
    Analyze pattern characteristics of a single draw.

    Args:
        numbers: List of 6 lottery numbers
        max_num: Maximum lottery number

    Returns:
        dict: Pattern characteristics
    """
    numbers = [int(n) for n in numbers]
    sorted_nums = sorted(numbers)

    # Odd/Even ratio
    odd_count = sum(1 for n in numbers if n % 2 == 1)
    even_count = 6 - odd_count

    # High/Low ratio (split at midpoint)
    mid = max_num // 2
    high_count = sum(1 for n in numbers if n > mid)
    low_count = 6 - high_count

    # Zone distribution (5 zones)
    zone_size = max_num // 5 + 1
    zones = [0] * 5
    for n in numbers:
        zone_idx = min((n - 1) // zone_size, 4)
        zones[zone_idx] += 1

    # Consecutive pairs count
    consecutive = 0
    for i in range(len(sorted_nums) - 1):
        if sorted_nums[i + 1] - sorted_nums[i] == 1:
            consecutive += 1

    # Sum
    total_sum = sum(numbers)

    return {
        'odd_even': (odd_count, even_count),
        'high_low': (high_count, low_count),
        'zones': tuple(zones),
        'consecutive': consecutive,
        'sum': total_sum
    }


def get_common_patterns(df, window=100):
    """
    Analyze most common patterns from historical data.

    Args:
        df: DataFrame with historical lottery data
        window: Number of recent draws to analyze

    Returns:
        dict: Most common patterns
    """
    columns = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth']
    max_num = int(df[columns].max().max())

    recent = df.tail(window)

    patterns = []
    for _, row in recent.iterrows():
        numbers = [row[col] for col in columns]
        patterns.append(analyze_draw_pattern(numbers, max_num))

    # Find most common patterns
    odd_even_counts = Counter(p['odd_even'] for p in patterns)
    high_low_counts = Counter(p['high_low'] for p in patterns)
    consecutive_counts = Counter(p['consecutive'] for p in patterns)

    sums = [p['sum'] for p in patterns]
    avg_sum = np.mean(sums)
    std_sum = np.std(sums)

    return {
        'most_common_odd_even': odd_even_counts.most_common(3),
        'most_common_high_low': high_low_counts.most_common(3),
        'most_common_consecutive': consecutive_counts.most_common(3),
        'sum_mean': avg_sum,
        'sum_std': std_sum,
        'sum_range': (avg_sum - std_sum, avg_sum + std_sum),
        'max_num': max_num
    }


def generate_numbers_matching_pattern(target_odd_even, target_high_low,
                                      sum_range, max_num, max_attempts=1000):
    """
    Generate a set of 6 numbers matching the target pattern.

    Args:
        target_odd_even: Tuple (odd_count, even_count)
        target_high_low: Tuple (high_count, low_count)
        sum_range: Tuple (min_sum, max_sum)
        max_num: Maximum lottery number
        max_attempts: Maximum generation attempts

    Returns:
        list: 6 numbers matching pattern, or random if no match found
    """
    mid = max_num // 2

    # Separate numbers into categories
    odd_low = [n for n in range(1, mid + 1) if n % 2 == 1]
    odd_high = [n for n in range(mid + 1, max_num + 1) if n % 2 == 1]
    even_low = [n for n in range(1, mid + 1) if n % 2 == 0]
    even_high = [n for n in range(mid + 1, max_num + 1) if n % 2 == 0]

    target_odd, target_even = target_odd_even
    target_high, target_low = target_high_low
    min_sum, max_sum = sum_range

    for _ in range(max_attempts):
        numbers = set()

        # Calculate how many from each category
        # odd_high, odd_low, even_high, even_low
        odd_high_need = min(target_odd, target_high)
        odd_low_need = target_odd - odd_high_need
        even_high_need = target_high - odd_high_need
        even_low_need = target_even - even_high_need

        # Adjust if not enough in categories
        if odd_high_need > len(odd_high):
            odd_high_need = len(odd_high)
            odd_low_need = target_odd - odd_high_need
        if odd_low_need > len(odd_low):
            odd_low_need = len(odd_low)
        if even_high_need > len(even_high):
            even_high_need = len(even_high)
        if even_low_need > len(even_low):
            even_low_need = len(even_low)

        try:
            if odd_high_need > 0 and len(odd_high) >= odd_high_need:
                numbers.update(random.sample(odd_high, odd_high_need))
            if odd_low_need > 0 and len(odd_low) >= odd_low_need:
                numbers.update(random.sample(odd_low, odd_low_need))
            if even_high_need > 0 and len(even_high) >= even_high_need:
                numbers.update(random.sample(even_high, even_high_need))
            if even_low_need > 0 and len(even_low) >= even_low_need:
                numbers.update(random.sample(even_low, even_low_need))
        except ValueError:
            continue

        # Fill remaining if needed
        all_nums = list(range(1, max_num + 1))
        while len(numbers) < 6:
            n = random.choice(all_nums)
            if n not in numbers:
                numbers.add(n)

        numbers = list(numbers)[:6]

        # Check sum constraint
        if min_sum <= sum(numbers) <= max_sum:
            return sorted(numbers)

    # Fallback: return random numbers
    return sorted(random.sample(range(1, max_num + 1), 6))


def predict_pattern(df, today_index):
    """
    Predict next draw based on historical pattern analysis.

    Args:
        df: DataFrame with historical lottery data
        today_index: Current index for prediction

    Returns:
        tuple: (main_numbers, special_number)
    """
    # Analyze common patterns
    common = get_common_patterns(df.iloc[:today_index], window=100)

    # Use most common patterns as target
    target_odd_even = common['most_common_odd_even'][0][0]  # e.g., (3, 3)
    target_high_low = common['most_common_high_low'][0][0]  # e.g., (3, 3)
    sum_range = common['sum_range']
    max_num = common['max_num']

    # Generate numbers matching pattern
    predicted_nums = generate_numbers_matching_pattern(
        target_odd_even, target_high_low, sum_range, max_num
    )

    # Predict special number - use most common from recent draws
    special_counts = Counter(df.iloc[:today_index].tail(50)['Special'])
    special = special_counts.most_common(1)[0][0]

    return [int(n) for n in predicted_nums], int(special)


def get_pattern_analysis(df, today_index):
    """
    Get detailed pattern analysis for display.
    """
    common = get_common_patterns(df.iloc[:today_index], window=100)
    return {
        'odd_even_distribution': common['most_common_odd_even'],
        'high_low_distribution': common['most_common_high_low'],
        'consecutive_distribution': common['most_common_consecutive'],
        'sum_statistics': {
            'mean': round(common['sum_mean'], 1),
            'std': round(common['sum_std'], 1),
            'recommended_range': (round(common['sum_range'][0]), round(common['sum_range'][1]))
        }
    }


if __name__ == "__main__":
    import pandas as pd

    CSV_FILE = "lotterypython/big_sequence.csv"
    df = pd.read_csv(CSV_FILE)
    today_index = len(df)

    main_nums, special = predict_pattern(df, today_index)
    print("===== Pattern Analysis Prediction =====")
    print(f"Numbers: {main_nums} + Special: {special}")

    analysis = get_pattern_analysis(df, today_index)
    print(f"\nOdd/Even distribution: {analysis['odd_even_distribution']}")
    print(f"High/Low distribution: {analysis['high_low_distribution']}")
    print(f"Sum statistics: {analysis['sum_statistics']}")
