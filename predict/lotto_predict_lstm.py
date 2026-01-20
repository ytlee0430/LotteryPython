import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras import layers, models

"""Simple LSTM-based predictor for lottery numbers.

This script trains an LSTM network on historical draws found in
``lotterypython/big_sequence.csv`` and prints a predicted set of
numbers for the next draw. It uses only a small sequence of recent
draws, so the results are for demonstration purposes only.
"""

CSV_FILE = Path(__file__).resolve().parents[1] / "lotterypython" / "big_sequence.csv"
SEQ_LEN = 10
COLUMNS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"]


def encode(nums, max_num=49):
    """Return a one-hot vector for the given numbers."""
    v = np.zeros(max_num, dtype=np.float32)
    v[[n - 1 for n in nums if 1 <= n <= max_num]] = 1.0
    return v


def _features_from_df(df: pd.DataFrame, max_num=49) -> np.ndarray:
    """Return encoded number vectors for each row of ``df``."""
    feats = []
    for _, row in df.iterrows():
        nums = row[COLUMNS].tolist()
        feats.append(encode(nums, max_num))
    return np.stack(feats)


def load_features():
    df = pd.read_csv(CSV_FILE)
    return _features_from_df(df)


def build_sequences(feats):
    X, y = [], []
    for i in range(len(feats) - SEQ_LEN):
        X.append(feats[i : i + SEQ_LEN])
        nxt = feats[i + SEQ_LEN]
        y.append(nxt)
    return np.array(X), np.array(y)


def predict_lstm(df: pd.DataFrame, lottery_type='big'):
    """Train a small LSTM model on ``df`` and return predicted numbers.

    Args:
        df: DataFrame with historical lottery data
        lottery_type: 'big' for 大樂透 (1-49), 'super' for 威力彩 (1-38, special 1-8)
    """
    # Determine number ranges based on lottery type
    if lottery_type == 'super':
        max_num = 38
        max_special = 8
    else:  # big
        max_num = 49
        max_special = 49

    feats = _features_from_df(df, max_num)
    X, y = build_sequences(feats)

    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = models.Sequential(
        [
            layers.Input(shape=(SEQ_LEN, feats.shape[-1])),
            layers.LSTM(64),
            layers.Dense(128, activation="relu"),
            layers.Dense(feats.shape[-1], activation="sigmoid"),
        ]
    )
    model.compile("adam", "binary_crossentropy")
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    latest = feats[-SEQ_LEN:][np.newaxis, ...]
    probs = model.predict(latest, verbose=0)[0]

    # Get top 6 main numbers
    top6 = np.argsort(probs)[-6:][::-1] + 1
    main_numbers = sorted(top6.tolist())

    # For special number, use probability within valid special range
    # Use frequencies from historical special numbers as baseline
    special_probs = probs[:max_special]  # Only consider valid special range
    special = int(np.argmax(special_probs) + 1)

    # Fallback: if special is somehow invalid, pick randomly from valid range
    if special < 1 or special > max_special:
        special = int(np.random.randint(1, max_special + 1))

    return main_numbers, special


if __name__ == "__main__":
    df = pd.read_csv(CSV_FILE)
    nums, sp = predict_lstm(df)
    print("\n===== LSTM prediction =====")
    print("numbers:", nums)
    print("special:", sp)
    print("==========================")
