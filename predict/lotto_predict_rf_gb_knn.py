import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier

# Input CSV with historical draws
CSV_FILE = "lotterypython/big_sequence.csv"
HISTORY = 10

COLUMNS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"]


def load_data(path=CSV_FILE):
    """Load draw history from CSV."""
    return pd.read_csv(path)


def _feature(past):
    counts = np.zeros(49, dtype=np.float32)
    for col in COLUMNS:
        counts += np.bincount(past[col], minlength=50)[1:]
    return counts


def _target(row):
    t = np.zeros(49, dtype=int)
    for n in row[COLUMNS]:
        t[n - 1] = 1
    return t


def build_dataset(df, history=HISTORY):
    X, y = [], []
    for i in range(history, len(df)):
        X.append(_feature(df.iloc[i - history:i]))
        y.append(_target(df.iloc[i]))
    return np.array(X), np.array(y)


def predict_algorithms(df):
    X, y = build_dataset(df)
    last_feat = _feature(df.iloc[-HISTORY:]).reshape(1, -1)

    models = {
        "RandomForest": MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42)),
        "GradientBoosting": MultiOutputClassifier(GradientBoostingClassifier(random_state=42)),
        "KNN": MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5)),
    }

    results = {}
    for name, model in models.items():
        model.fit(X, y)
        probs = model.predict_proba(last_feat)
        prob_vec = np.array([p[:, 1] for p in probs]).flatten()
        nums = np.argsort(prob_vec)[-6:][::-1] + 1
        results[name] = nums.tolist()

    y_spec = df["Special"].iloc[HISTORY:].values
    spec_model = RandomForestClassifier(n_estimators=100, random_state=42)
    spec_model.fit(X, y_spec)
    special = int(spec_model.predict(last_feat)[0])

    return results, special


if __name__ == "__main__":
    df = load_data()
    results, special = predict_algorithms(df)
    print("===== Predictions (for reference only) =====")
    for name, nums in results.items():
        print(f"{name}: {sorted(nums)} + SP:{special}")
