"""
Backtesting and Analysis Module

Provides historical performance analysis for lottery prediction algorithms,
including hit rate calculation and number distribution analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

# Import prediction algorithms
from predict.lotto_predict_hot_50 import predict_hot50
from predict.lotto_predict_cold_50 import predict_cold50
from predict.lotto_predict_markov import predict_markov
from predict.lotto_predict_pattern import predict_pattern


def load_historical_data(lottery_type: str = 'big') -> pd.DataFrame:
    """Load historical lottery data from CSV.

    Args:
        lottery_type: 'big' for 大樂透 or 'super' for 威力彩

    Returns:
        DataFrame with historical lottery data
    """
    csv_file = Path(__file__).parent.parent / "lotterypython" / f"{lottery_type}_sequence.csv"
    if not csv_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_file)
    # Ensure numeric columns
    for col in ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Special"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def calculate_hit_count(predicted: List[int], actual: List[int]) -> int:
    """Calculate how many numbers were hit.

    Args:
        predicted: List of predicted numbers
        actual: List of actual winning numbers

    Returns:
        Number of matching numbers
    """
    return len(set(predicted) & set(actual))


def backtest_algorithm(df: pd.DataFrame, algorithm_name: str,
                       periods: int = 50, lottery_type: str = 'big') -> Dict:
    """Run backtest for a specific algorithm.

    Args:
        df: Historical data DataFrame
        algorithm_name: Name of the algorithm to test
        periods: Number of periods to backtest
        lottery_type: 'big' or 'super'

    Returns:
        Dict with backtest results
    """
    if len(df) < periods + 50:  # Need enough history
        return {"error": "Not enough historical data"}

    max_num = 49 if lottery_type == 'big' else 38
    results = []

    # Run backtest for each period
    for i in range(periods):
        # Use data up to period (len(df) - periods + i) to predict next period
        train_end = len(df) - periods + i
        if train_end < 50:
            continue

        actual_row = df.iloc[train_end]

        # Get actual numbers
        actual_numbers = [
            actual_row['First'], actual_row['Second'], actual_row['Third'],
            actual_row['Fourth'], actual_row['Fifth'], actual_row['Sixth']
        ]
        actual_special = actual_row['Special']

        try:
            # Run prediction based on algorithm
            # Note: algorithms expect (df, today_index) where today_index is the training end
            # Most return (numbers, special, details) tuple
            if algorithm_name == 'Hot50':
                result = predict_hot50(df, train_end)
            elif algorithm_name == 'Cold50':
                result = predict_cold50(df, train_end)
            elif algorithm_name == 'Markov':
                result = predict_markov(df, train_end)
            elif algorithm_name == 'Pattern':
                result = predict_pattern(df, train_end)
            else:
                return {"error": f"Unknown algorithm: {algorithm_name}"}

            # Handle different return formats
            if isinstance(result, tuple):
                predicted_numbers = result[0]
                predicted_special = result[1]
            else:
                predicted_numbers = result.get('numbers', [])
                predicted_special = result.get('special', 0)

            # Calculate hits
            main_hits = calculate_hit_count(predicted_numbers, actual_numbers)
            special_hit = 1 if int(predicted_special) == int(actual_special) else 0

            results.append({
                'period': str(actual_row.get('Period', train_end)),
                'predicted': [int(n) for n in predicted_numbers],
                'actual': [int(n) for n in actual_numbers],
                'main_hits': int(main_hits),
                'special_hit': int(special_hit),
                'predicted_special': int(predicted_special),
                'actual_special': int(actual_special)
            })
        except Exception as e:
            results.append({
                'period': actual_row.get('Period', train_end),
                'error': str(e)
            })

    # Calculate statistics
    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        return {"error": "All predictions failed"}

    hit_counts = [r['main_hits'] for r in valid_results]
    special_hits = sum(r['special_hit'] for r in valid_results)

    # Hit distribution
    hit_distribution = Counter(hit_counts)

    return {
        'algorithm': algorithm_name,
        'periods_tested': len(valid_results),
        'average_hits': round(np.mean(hit_counts), 2),
        'max_hits': max(hit_counts),
        'min_hits': min(hit_counts),
        'hit_distribution': {str(k): v for k, v in sorted(hit_distribution.items())},
        'special_hit_rate': round(special_hits / len(valid_results) * 100, 1),
        'hit_3_or_more': sum(1 for h in hit_counts if h >= 3),
        'hit_4_or_more': sum(1 for h in hit_counts if h >= 4),
        'hit_5_or_more': sum(1 for h in hit_counts if h >= 5),
        'recent_results': valid_results[-10:]  # Last 10 results
    }


def run_full_backtest(lottery_type: str = 'big', periods: int = 50) -> Dict:
    """Run backtest for all supported algorithms.

    Args:
        lottery_type: 'big' or 'super'
        periods: Number of periods to backtest

    Returns:
        Dict with results for all algorithms
    """
    df = load_historical_data(lottery_type)
    if df.empty:
        return {"error": "No historical data found"}

    algorithms = ['Hot50', 'Cold50', 'Markov', 'Pattern']
    results = {}

    for algo in algorithms:
        results[algo] = backtest_algorithm(df, algo, periods, lottery_type)

    # Calculate ranking by average hits
    rankings = []
    for algo, result in results.items():
        if 'error' not in result:
            rankings.append({
                'algorithm': algo,
                'average_hits': result['average_hits'],
                'hit_3_plus_rate': round(result['hit_3_or_more'] / result['periods_tested'] * 100, 1)
            })

    rankings.sort(key=lambda x: x['average_hits'], reverse=True)

    return {
        'lottery_type': lottery_type,
        'periods_tested': periods,
        'total_periods_available': len(df),
        'algorithms': results,
        'ranking': rankings
    }


def analyze_number_distribution(df: pd.DataFrame, periods: int = 100) -> Dict:
    """Analyze number distribution patterns in historical data.

    Args:
        df: Historical data DataFrame
        periods: Number of recent periods to analyze

    Returns:
        Dict with distribution analysis
    """
    if df.empty:
        return {"error": "No data to analyze"}

    # Use recent periods
    recent_df = df.tail(periods)

    # Collect all main numbers
    all_numbers = []
    for _, row in recent_df.iterrows():
        numbers = [row['First'], row['Second'], row['Third'],
                   row['Fourth'], row['Fifth'], row['Sixth']]
        all_numbers.extend([n for n in numbers if n > 0])

    # Collect special numbers
    special_numbers = recent_df['Special'].tolist()

    # Determine max number (49 for big, 38 for super)
    max_num = max(all_numbers) if all_numbers else 49
    if max_num > 38:
        max_num = 49
        lottery_type = 'big'
    else:
        max_num = 38
        lottery_type = 'super'

    mid_point = max_num // 2

    # Calculate distributions
    number_freq = Counter(all_numbers)
    special_freq = Counter(special_numbers)

    # Odd/Even analysis
    odd_count = sum(1 for n in all_numbers if n % 2 == 1)
    even_count = len(all_numbers) - odd_count

    # High/Low analysis (1-24 low, 25-49 high for big)
    low_count = sum(1 for n in all_numbers if n <= mid_point)
    high_count = len(all_numbers) - low_count

    # Sum analysis per draw
    sums = []
    for _, row in recent_df.iterrows():
        numbers = [row['First'], row['Second'], row['Third'],
                   row['Fourth'], row['Fifth'], row['Sixth']]
        sums.append(sum(n for n in numbers if n > 0))

    # Consecutive number analysis
    consecutive_counts = []
    for _, row in recent_df.iterrows():
        numbers = sorted([row['First'], row['Second'], row['Third'],
                         row['Fourth'], row['Fifth'], row['Sixth']])
        consec = 0
        for i in range(len(numbers) - 1):
            if numbers[i+1] - numbers[i] == 1:
                consec += 1
        consecutive_counts.append(consec)

    # Top 10 hot and cold numbers
    all_possible = set(range(1, max_num + 1))
    hot_numbers = number_freq.most_common(10)
    cold_numbers = [(n, number_freq.get(n, 0)) for n in all_possible]
    cold_numbers.sort(key=lambda x: x[1])
    cold_numbers = cold_numbers[:10]

    return {
        'lottery_type': lottery_type,
        'periods_analyzed': len(recent_df),
        'total_numbers_drawn': len(all_numbers),

        # Odd/Even
        'odd_even_ratio': f"{odd_count}:{even_count}",
        'odd_percentage': round(odd_count / len(all_numbers) * 100, 1),
        'even_percentage': round(even_count / len(all_numbers) * 100, 1),

        # High/Low
        'high_low_ratio': f"{high_count}:{low_count}",
        'high_percentage': round(high_count / len(all_numbers) * 100, 1),
        'low_percentage': round(low_count / len(all_numbers) * 100, 1),
        'mid_point': mid_point,

        # Sum
        'sum_average': round(np.mean(sums), 1),
        'sum_min': min(sums),
        'sum_max': max(sums),
        'sum_std': round(np.std(sums), 1),

        # Consecutive
        'avg_consecutive': round(np.mean(consecutive_counts), 2),
        'max_consecutive': max(consecutive_counts),

        # Hot/Cold numbers
        'hot_numbers': [{'number': n, 'count': c} for n, c in hot_numbers],
        'cold_numbers': [{'number': n, 'count': c} for n, c in cold_numbers],

        # Special number analysis
        'special_hot': [{'number': n, 'count': c} for n, c in special_freq.most_common(10)]
    }


def get_distribution_analysis(lottery_type: str = 'big', periods: int = 100) -> Dict:
    """Get number distribution analysis for display.

    Args:
        lottery_type: 'big' or 'super'
        periods: Number of periods to analyze

    Returns:
        Dict with distribution analysis
    """
    df = load_historical_data(lottery_type)
    return analyze_number_distribution(df, periods)


def rolling_backtest(lottery_type: str = 'big', window_size: int = 20,
                     total_periods: int = 100) -> Dict:
    """
    Run rolling backtest to show performance over time.

    Tests algorithms across multiple time windows to see consistency.

    Args:
        lottery_type: 'big' or 'super'
        window_size: Size of each test window
        total_periods: Total periods to analyze

    Returns:
        Dict with rolling performance data for visualization
    """
    df = load_historical_data(lottery_type)
    if df.empty or len(df) < total_periods + 50:
        return {"error": "Not enough historical data"}

    algorithms = ['Hot50', 'Cold50', 'Markov', 'Pattern']
    num_windows = total_periods // window_size

    # Store results per window
    rolling_results = {algo: [] for algo in algorithms}
    window_labels = []

    for w in range(num_windows):
        # Calculate period range for this window
        start_offset = w * window_size
        end_idx = len(df) - total_periods + start_offset + window_size

        # Get period label
        if 'Period' in df.columns:
            period_start = str(df.iloc[end_idx - window_size].get('Period', ''))
            period_end = str(df.iloc[end_idx - 1].get('Period', ''))
            window_labels.append(f"{period_start[-3:]}-{period_end[-3:]}")
        else:
            window_labels.append(f"W{w+1}")

        # Test each algorithm on this window
        for algo in algorithms:
            try:
                result = backtest_algorithm(df, algo, window_size, lottery_type)
                if 'error' not in result:
                    rolling_results[algo].append({
                        'window': w + 1,
                        'average_hits': result['average_hits'],
                        'hit_3_plus': result['hit_3_or_more']
                    })
                else:
                    rolling_results[algo].append({
                        'window': w + 1,
                        'average_hits': 0,
                        'hit_3_plus': 0,
                        'error': result['error']
                    })
            except Exception as e:
                rolling_results[algo].append({
                    'window': w + 1,
                    'average_hits': 0,
                    'hit_3_plus': 0,
                    'error': str(e)
                })

    # Calculate overall stats per algorithm
    algorithm_summary = {}
    for algo in algorithms:
        valid_results = [r for r in rolling_results[algo] if 'error' not in r]
        if valid_results:
            avg_hits = [r['average_hits'] for r in valid_results]
            algorithm_summary[algo] = {
                'overall_average': round(np.mean(avg_hits), 2),
                'consistency': round(np.std(avg_hits), 2),  # Lower = more consistent
                'best_window': max(range(len(valid_results)),
                                   key=lambda i: valid_results[i]['average_hits']) + 1,
                'worst_window': min(range(len(valid_results)),
                                    key=lambda i: valid_results[i]['average_hits']) + 1
            }

    return {
        'lottery_type': lottery_type,
        'window_size': window_size,
        'total_periods': total_periods,
        'num_windows': num_windows,
        'window_labels': window_labels,
        'rolling_results': rolling_results,
        'algorithm_summary': algorithm_summary
    }


def optimize_window_size(lottery_type: str = 'big',
                         min_window: int = 20, max_window: int = 100,
                         step: int = 10, test_periods: int = 50) -> Dict:
    """
    Find optimal window size for Hot/Cold algorithms.

    Tests different window sizes and finds the one with best average hits.

    Args:
        lottery_type: 'big' or 'super'
        min_window: Minimum window size to test
        max_window: Maximum window size to test
        step: Step size between tests
        test_periods: Number of periods to test each window

    Returns:
        Dict with optimization results
    """
    df = load_historical_data(lottery_type)
    if df.empty:
        return {"error": "No historical data found"}

    window_sizes = list(range(min_window, max_window + 1, step))
    results = []

    for window in window_sizes:
        # Test Hot algorithm with this window
        hot_result = {'window': window, 'algorithm': 'Hot'}
        try:
            # Temporarily use different window for testing
            hot_hits = []
            for i in range(test_periods):
                train_end = len(df) - test_periods + i
                if train_end < window:
                    continue

                actual_row = df.iloc[train_end]
                actual_numbers = [
                    actual_row['First'], actual_row['Second'], actual_row['Third'],
                    actual_row['Fourth'], actual_row['Fifth'], actual_row['Sixth']
                ]

                # Predict with custom window
                predicted, _, _ = predict_hot50(df, train_end, window=window)
                hits = len(set(predicted) & set(actual_numbers))
                hot_hits.append(hits)

            if hot_hits:
                hot_result['average_hits'] = round(np.mean(hot_hits), 3)
                hot_result['hit_3_plus'] = sum(1 for h in hot_hits if h >= 3)
        except Exception as e:
            hot_result['error'] = str(e)

        results.append(hot_result)

        # Test Cold algorithm with this window
        cold_result = {'window': window, 'algorithm': 'Cold'}
        try:
            cold_hits = []
            for i in range(test_periods):
                train_end = len(df) - test_periods + i
                if train_end < window:
                    continue

                actual_row = df.iloc[train_end]
                actual_numbers = [
                    actual_row['First'], actual_row['Second'], actual_row['Third'],
                    actual_row['Fourth'], actual_row['Fifth'], actual_row['Sixth']
                ]

                predicted, _, _ = predict_cold50(df, train_end, window=window)
                hits = len(set(predicted) & set(actual_numbers))
                cold_hits.append(hits)

            if cold_hits:
                cold_result['average_hits'] = round(np.mean(cold_hits), 3)
                cold_result['hit_3_plus'] = sum(1 for h in cold_hits if h >= 3)
        except Exception as e:
            cold_result['error'] = str(e)

        results.append(cold_result)

    # Find optimal windows
    hot_results = [r for r in results if r['algorithm'] == 'Hot' and 'average_hits' in r]
    cold_results = [r for r in results if r['algorithm'] == 'Cold' and 'average_hits' in r]

    optimal_hot = max(hot_results, key=lambda x: x['average_hits']) if hot_results else None
    optimal_cold = max(cold_results, key=lambda x: x['average_hits']) if cold_results else None

    return {
        'lottery_type': lottery_type,
        'test_periods': test_periods,
        'window_range': {'min': min_window, 'max': max_window, 'step': step},
        'results': results,
        'optimal': {
            'hot_window': optimal_hot['window'] if optimal_hot else 50,
            'hot_avg_hits': optimal_hot['average_hits'] if optimal_hot else 0,
            'cold_window': optimal_cold['window'] if optimal_cold else 50,
            'cold_avg_hits': optimal_cold['average_hits'] if optimal_cold else 0
        }
    }


if __name__ == "__main__":
    # Test backtest
    print("=== Backtest Results ===")
    results = run_full_backtest('big', 30)

    if 'error' not in results:
        print(f"\nTested {results['periods_tested']} periods")
        print("\nAlgorithm Ranking:")
        for i, r in enumerate(results['ranking'], 1):
            print(f"  {i}. {r['algorithm']}: {r['average_hits']} avg hits, {r['hit_3_plus_rate']}% hit 3+")
    else:
        print(f"Error: {results['error']}")

    print("\n=== Distribution Analysis ===")
    dist = get_distribution_analysis('big', 100)

    if 'error' not in dist:
        print(f"Analyzed {dist['periods_analyzed']} periods")
        print(f"Odd/Even: {dist['odd_even_ratio']} ({dist['odd_percentage']}% odd)")
        print(f"High/Low: {dist['high_low_ratio']} ({dist['high_percentage']}% high)")
        print(f"Sum range: {dist['sum_min']} - {dist['sum_max']} (avg: {dist['sum_average']})")
        print(f"\nHot numbers: {[n['number'] for n in dist['hot_numbers'][:5]]}")
        print(f"Cold numbers: {[n['number'] for n in dist['cold_numbers'][:5]]}")
