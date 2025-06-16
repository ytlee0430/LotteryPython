from pathlib import Path
import sys
import pandas as pd

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predict.lotto_predict_rf_gb_knn import predict_algorithms


def test_predict_algorithms_super():
    df = pd.read_csv(ROOT / 'lotterypython/super_sequence.csv')
    df = df.head(60)
    results, special = predict_algorithms(df)
    assert set(results.keys()) == {"RandomForest", "GradientBoosting", "KNN"}
    for nums in results.values():
        assert len(nums) == 6
        assert all(1 <= n <= 38 for n in nums)
    assert 1 <= special <= 9
