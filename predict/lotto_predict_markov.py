"""
Markov Chain Prediction Algorithm

Uses transition probability matrix to predict which numbers
are most likely to appear based on the previous draw's numbers.
"""

import numpy as np
from collections import Counter


def build_transition_matrix(df, max_num=49):
    """
    Build a transition probability matrix from historical data.

    The matrix[i][j] represents the probability that number j+1
    appears in draw N+1 given that number i+1 appeared in draw N.

    Args:
        df: DataFrame with historical lottery data
        max_num: Maximum lottery number

    Returns:
        numpy.ndarray: Transition probability matrix (max_num x max_num)
    """
    # Initialize transition count matrix
    transition = np.zeros((max_num, max_num), dtype=np.float64)

    columns = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth']

    # Count transitions from draw i to draw i+1
    for i in range(len(df) - 1):
        current_nums = df.iloc[i][columns].values.astype(int)
        next_nums = df.iloc[i + 1][columns].values.astype(int)

        # For each number in current draw, count transitions to next draw
        for curr_num in current_nums:
            if 1 <= curr_num <= max_num:
                for next_num in next_nums:
                    if 1 <= next_num <= max_num:
                        transition[curr_num - 1][next_num - 1] += 1

    # Normalize rows to get probabilities
    row_sums = transition.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums = np.where(row_sums == 0, 1, row_sums)
    transition_prob = transition / row_sums

    return transition_prob


def predict_markov(df, today_index):
    """
    Predict next draw using Markov Chain transition probabilities.

    Args:
        df: DataFrame with historical lottery data
        today_index: Current index (uses draw at today_index-1 as current state)

    Returns:
        tuple: (main_numbers, special_number)
    """
    columns = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth']
    max_num = int(df[columns].max().max())

    # Build transition matrix from all historical data
    trans_matrix = build_transition_matrix(df.iloc[:today_index], max_num)

    # Get the last draw's numbers (current state)
    last_draw = df.iloc[today_index - 1][columns].values.astype(int)

    # Calculate probability for each number in next draw
    # Sum the transition probabilities from all numbers in current draw
    next_probs = np.zeros(max_num)
    for num in last_draw:
        if 1 <= num <= max_num:
            next_probs += trans_matrix[num - 1]

    # Get top 10 candidates with scores
    top_10_indices = np.argsort(next_probs)[-10:][::-1]
    top_10 = [[int(i + 1), round(float(next_probs[i]), 3)] for i in top_10_indices]

    # Get top 6 numbers with highest transition probability
    top_indices = np.argsort(next_probs)[-6:][::-1]
    predicted_nums = [int(i + 1) for i in top_indices]

    # For special number, use separate transition analysis
    special = predict_special_markov(df, today_index, max_num)

    details = {
        "type": "transition_probability",
        "top_10": top_10
    }
    return predicted_nums, special, details


def predict_special_markov(df, today_index, max_num):
    """
    Predict special number using Markov transition from previous special.
    """
    # Build special-to-special transition matrix
    special_trans = np.zeros((max_num, max_num), dtype=np.float64)

    for i in range(len(df) - 1):
        curr_special = int(df.iloc[i]['Special'])
        next_special = int(df.iloc[i + 1]['Special'])
        if 1 <= curr_special <= max_num and 1 <= next_special <= max_num:
            special_trans[curr_special - 1][next_special - 1] += 1

    # Normalize
    row_sums = special_trans.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    special_trans_prob = special_trans / row_sums

    # Get last special number
    last_special = int(df.iloc[today_index - 1]['Special'])

    if 1 <= last_special <= max_num:
        probs = special_trans_prob[last_special - 1]
        predicted_special = int(np.argmax(probs) + 1)
    else:
        # Fallback to most common
        predicted_special = int(Counter(df['Special']).most_common(1)[0][0])

    return predicted_special


def get_transition_analysis(df, today_index, top_n=10):
    """
    Get detailed transition analysis for debugging/visualization.
    """
    columns = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth']
    max_num = int(df[columns].max().max())
    trans_matrix = build_transition_matrix(df.iloc[:today_index], max_num)

    last_draw = df.iloc[today_index - 1][columns].values.astype(int)

    next_probs = np.zeros(max_num)
    for num in last_draw:
        if 1 <= num <= max_num:
            next_probs += trans_matrix[num - 1]

    # Get top transitions
    top_indices = np.argsort(next_probs)[-top_n:][::-1]
    top_with_probs = [(int(i + 1), float(next_probs[i])) for i in top_indices]

    return {
        "last_draw": [int(n) for n in last_draw],
        "top_transitions": top_with_probs,
        "matrix_shape": trans_matrix.shape
    }


if __name__ == "__main__":
    import pandas as pd

    CSV_FILE = "lotterypython/big_sequence.csv"
    df = pd.read_csv(CSV_FILE)
    today_index = len(df)

    main_nums, special = predict_markov(df, today_index)
    print("===== Markov Chain Prediction =====")
    print(f"Numbers: {sorted(main_nums)} + Special: {special}")

    analysis = get_transition_analysis(df, today_index)
    print(f"\nLast draw: {analysis['last_draw']}")
    print(f"Top transitions: {analysis['top_transitions']}")
