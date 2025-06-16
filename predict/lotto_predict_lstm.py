import numpy as np
import pandas as pd
from tensorflow.keras import layers, models

"""Simple LSTM-based predictor for lottery numbers.

This script trains an LSTM network on historical draws found in
``lotterypython/big_sequence.csv`` and prints a predicted set of
numbers for the next draw. It uses only a small sequence of recent
draws, so the results are for demonstration purposes only.
"""

CSV_FILE = "../lotterypython/big_sequence.csv"
SEQ_LEN = 10


def encode(nums):
    """Return a 49-dim one-hot vector for the given numbers."""
    v = np.zeros(49, dtype=np.float32)
    v[[n - 1 for n in nums]] = 1.0
    return v


def load_features():
    df = pd.read_csv(CSV_FILE)
    feats = []
    for _, row in df.iterrows():
        nums = row[["First", "Second", "Third", "Fourth", "Fifth", "Sixth"]].tolist()
        feats.append(encode(nums))
    return np.stack(feats)


def build_sequences(feats):
    X, y = [], []
    for i in range(len(feats) - SEQ_LEN):
        X.append(feats[i : i + SEQ_LEN])
        nxt = feats[i + SEQ_LEN]
        y.append(nxt)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    feats = load_features()
    X, y = build_sequences(feats)

    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = models.Sequential(
        [
            layers.Input(shape=(SEQ_LEN, 49)),
            layers.LSTM(64),
            layers.Dense(128, activation="relu"),
            layers.Dense(49, activation="sigmoid"),
        ]
    )
    model.compile("adam", "binary_crossentropy")
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    latest = feats[-SEQ_LEN:][np.newaxis, ...]
    probs = model.predict(latest, verbose=0)[0]
    top7 = np.argsort(probs)[-7:][::-1] + 1
    main_numbers = sorted(top7[:6])
    special = int(top7[6])

    print("\n===== LSTM prediction =====")
    print("numbers:", main_numbers)
    print("special:", special)
    print("==========================")
