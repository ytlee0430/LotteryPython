import pandas as pd, numpy as np
from pathlib import Path
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

CSV_FILE = Path(__file__).resolve().parents[1] / "power_lottery.csv"
SEQ_LEN = 10
COLUMNS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"]


def encode(nums, max_num=49):
    v = np.zeros(max_num, dtype=np.float32)
    v[[n - 1 for n in nums if 1 <= n <= max_num]] = 1.0
    return v


def consec(nums):
    s = sorted(nums)
    return sum(1 for x, y in zip(s, s[1:]) if y - x == 1)


def _build_features(df: pd.DataFrame, max_num=49) -> np.ndarray:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    hot_sets = []
    for i in range(len(df)):
        past = df.loc[max(0, i - 50):i - 1, COLUMNS].values.flatten()
        hot_sets.append({n for n, _ in Counter(past).most_common(10)})

    feats = []
    for i, row in df.iterrows():
        nums = row[COLUMNS].tolist()
        f = np.concatenate([
            encode(nums, max_num),
            np.eye(7)[row["Date"].weekday()],
            np.eye(6)[consec(nums)],
            [sum(1 for n in nums if n in hot_sets[i])],
        ])
        feats.append(f)
    return np.stack(feats)


def _build_sequences(feats: np.ndarray, df: pd.DataFrame, max_num=49) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(feats) - SEQ_LEN):
        X.append(feats[i : i + SEQ_LEN])
        nxt_nums = df.loc[i + SEQ_LEN, COLUMNS].tolist()
        y.append(encode(nxt_nums, max_num))
    return np.array(X), np.array(y)


def predict_lstm_rf(df: pd.DataFrame, lottery_type='big'):
    """Train the hybrid LSTM/RF model on ``df`` and return numbers.

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

    feats = _build_features(df, max_num)
    X, y = _build_sequences(feats, df, max_num)

    split = int(len(X) * 0.9)
    Xtr, Xv = X[:split], X[split:]
    ytr, yv = y[:split], y[split:]

    model = models.Sequential([
        layers.Input(shape=Xtr.shape[1:]),
        layers.LSTM(64),
        layers.Dense(128, activation="relu"),
        layers.Dense(max_num, activation="sigmoid"),
    ])
    model.compile("adam", "binary_crossentropy")
    model.fit(Xtr, ytr, epochs=25, batch_size=32, validation_data=(Xv, yv), verbose=0)

    latest_seq = feats[:SEQ_LEN][np.newaxis, ...]
    probs = model.predict(latest_seq, verbose=0)[0]

    # Get top 10 candidates with scores
    top_indices = np.argsort(probs)[-10:][::-1]
    top_10 = [[int(idx + 1), round(float(probs[idx]), 3)] for idx in top_indices]

    main6 = np.argsort(probs)[-6:][::-1] + 1

    X_spec = feats[:-1]
    y_spec = df["Special"].values[1:]
    Xtr_s, Xv_s, ytr_s, yv_s = train_test_split(X_spec, y_spec, test_size=0.1, random_state=42)
    rf_spec = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_spec.fit(Xtr_s, ytr_s)
    special = rf_spec.predict(feats[0].reshape(1, -1))[0]

    # Ensure special is within valid range
    special = int(special)
    if special < 1 or special > max_special:
        special = max(1, min(special, max_special))

    details = {
        "type": "probability_ranking",
        "top_10": top_10
    }
    return sorted(main6.tolist()), special, details


if __name__ == "__main__":
    if CSV_FILE.exists():
        df = pd.read_csv(CSV_FILE)
        nums, sp = predict_lstm_rf(df)
        print("\n===== AI result =====")
        print("numbers :", nums)
        print("sp  :", sp)
        print("=======================")
    else:
        print("CSV file not found:", CSV_FILE)
