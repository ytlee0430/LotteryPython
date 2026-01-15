"""
XGBoost Prediction Algorithm

Uses XGBoost (Extreme Gradient Boosting) for lottery prediction.
XGBoost offers improved regularization, parallel processing,
and better handling of missing values compared to standard GradientBoosting.

Falls back to sklearn's GradientBoostingClassifier if xgboost is not available.
"""

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier

# Try to import xgboost, fallback to sklearn GradientBoosting
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    XGBOOST_AVAILABLE = False


HISTORY = 10
COLUMNS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"]


def _feature(past, max_num):
    """Create frequency feature vector from past draws."""
    counts = np.zeros(max_num, dtype=np.float32)
    for col in COLUMNS:
        counts += np.bincount(past[col].astype(int), minlength=max_num + 1)[1:]
    return counts


def _target(row, max_num):
    """Create binary target vector for a single draw."""
    t = np.zeros(max_num, dtype=int)
    for n in row[COLUMNS]:
        t[int(n) - 1] = 1
    return t


def build_dataset(df, history=HISTORY, max_num=49):
    """Build training dataset from historical draws."""
    X, y = [], []
    for i in range(history, len(df)):
        X.append(_feature(df.iloc[i - history:i], max_num))
        y.append(_target(df.iloc[i], max_num))
    return np.array(X), np.array(y)


def create_model():
    """Create XGBoost or fallback GradientBoosting model."""
    if XGBOOST_AVAILABLE:
        base_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
    else:
        # Fallback to GradientBoosting
        base_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )

    return MultiOutputClassifier(base_model)


def predict_xgboost(df, today_index):
    """
    Predict next draw using XGBoost.

    Args:
        df: DataFrame with historical lottery data
        today_index: Current index for prediction

    Returns:
        tuple: (main_numbers, special_number)
    """
    max_num = int(df[COLUMNS].max().max())

    # Build dataset
    X, y = build_dataset(df.iloc[:today_index], max_num=max_num)

    # Create last feature for prediction
    last_feat = _feature(df.iloc[today_index - HISTORY:today_index], max_num).reshape(1, -1)

    # Train model
    model = create_model()
    model.fit(X, y)

    # Get prediction probabilities
    probs = model.predict_proba(last_feat)

    # Extract probability for class 1 (number appearing)
    prob_vec = []
    for p in probs:
        if p.shape[1] == 2:
            prob_vec.append(p[0, 1])
        else:
            prob_vec.append(0.0)
    prob_vec = np.array(prob_vec)

    # Get top 6 numbers
    nums = np.argsort(prob_vec)[-6:][::-1] + 1
    predicted_nums = nums.tolist()

    # Predict special number
    special = predict_special_xgboost(df, today_index, max_num, X)

    return [int(n) for n in predicted_nums], int(special)


def predict_special_xgboost(df, today_index, max_num, X):
    """Predict special number using XGBoost/GradientBoosting."""
    # Convert to 0-indexed for XGBoost compatibility
    y_spec = df["Special"].iloc[HISTORY:today_index].values.astype(int) - 1

    if XGBOOST_AVAILABLE:
        spec_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0
        )
    else:
        spec_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42
        )

    last_feat = _feature(df.iloc[today_index - HISTORY:today_index], max_num).reshape(1, -1)

    spec_model.fit(X, y_spec)
    # Convert back to 1-indexed
    special = int(spec_model.predict(last_feat)[0]) + 1

    return special


def get_model_info():
    """Get information about which model is being used."""
    if XGBOOST_AVAILABLE:
        return {
            "model": "XGBoost",
            "version": xgb.__version__,
            "backend": "xgboost"
        }
    else:
        return {
            "model": "GradientBoosting (fallback)",
            "version": "sklearn",
            "backend": "sklearn.ensemble.GradientBoostingClassifier"
        }


if __name__ == "__main__":
    CSV_FILE = "lotterypython/big_sequence.csv"
    df = pd.read_csv(CSV_FILE)
    today_index = len(df)

    print(f"Model info: {get_model_info()}")

    main_nums, special = predict_xgboost(df, today_index)
    print("===== XGBoost Prediction =====")
    print(f"Numbers: {sorted(main_nums)} + Special: {special}")
