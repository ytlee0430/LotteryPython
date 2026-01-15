"""Tests for Ensemble Voting prediction algorithm."""
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predict.lotto_predict_ensemble import (
    predict_ensemble,
    weighted_vote_numbers,
    weighted_vote_special,
    collect_predictions,
    get_vote_details,
)


@pytest.fixture
def sample_df():
    """Load sample lottery data."""
    csv_path = ROOT / "lotterypython" / "big_sequence.csv"
    if not csv_path.exists():
        pytest.skip("Sample data not available")
    return pd.read_csv(csv_path)


class TestWeightedVoting:
    """Tests for weighted voting functions."""

    def test_weighted_vote_numbers_returns_top_6(self):
        predictions = [
            {"numbers": [1, 2, 3, 4, 5, 6], "special": 7, "weight": 1.0},
            {"numbers": [1, 2, 3, 7, 8, 9], "special": 7, "weight": 1.5},
        ]
        result = weighted_vote_numbers(predictions, top_n=6)
        assert len(result) == 6
        # Numbers 1, 2, 3 should be in result (appear in both)
        assert 1 in result
        assert 2 in result
        assert 3 in result

    def test_weighted_vote_numbers_respects_weights(self):
        predictions = [
            {"numbers": [10, 20, 30, 40, 41, 42], "special": 1, "weight": 3.0},
            {"numbers": [1, 2, 3, 4, 5, 6], "special": 1, "weight": 1.0},
        ]
        result = weighted_vote_numbers(predictions, top_n=6)
        # Higher weighted predictions should dominate
        assert 10 in result
        assert 20 in result

    def test_weighted_vote_numbers_empty_predictions(self):
        result = weighted_vote_numbers([], top_n=6)
        assert result == []

    def test_weighted_vote_special_picks_highest(self):
        predictions = [
            {"numbers": [1, 2, 3, 4, 5, 6], "special": 7, "weight": 1.0},
            {"numbers": [1, 2, 3, 4, 5, 6], "special": 7, "weight": 1.5},
            {"numbers": [1, 2, 3, 4, 5, 6], "special": 8, "weight": 1.0},
        ]
        result = weighted_vote_special(predictions)
        assert result == 7  # Total weight 2.5 vs 1.0

    def test_weighted_vote_special_empty_returns_default(self):
        result = weighted_vote_special([])
        assert result == 1


class TestPredictEnsemble:
    """Tests for main ensemble prediction function."""

    def test_predict_ensemble_returns_tuple(self, sample_df):
        today_index = len(sample_df)
        result = predict_ensemble(sample_df, today_index)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_ensemble_returns_6_numbers(self, sample_df):
        today_index = len(sample_df)
        numbers, special = predict_ensemble(sample_df, today_index)
        assert len(numbers) == 6

    def test_predict_ensemble_numbers_in_range(self, sample_df):
        today_index = len(sample_df)
        numbers, special = predict_ensemble(sample_df, today_index)
        for num in numbers:
            assert 1 <= num <= 49

    def test_predict_ensemble_special_in_range(self, sample_df):
        today_index = len(sample_df)
        numbers, special = predict_ensemble(sample_df, today_index)
        assert 1 <= special <= 49


class TestCollectPredictions:
    """Tests for prediction collection."""

    def test_collect_predictions_returns_list(self, sample_df):
        today_index = len(sample_df)
        predictions = collect_predictions(sample_df, today_index)
        assert isinstance(predictions, list)

    def test_collect_predictions_has_required_keys(self, sample_df):
        today_index = len(sample_df)
        predictions = collect_predictions(sample_df, today_index)
        assert len(predictions) > 0
        for pred in predictions:
            assert "name" in pred
            assert "numbers" in pred
            assert "special" in pred
            assert "weight" in pred


class TestGetVoteDetails:
    """Tests for vote details function."""

    def test_get_vote_details_returns_dict(self, sample_df):
        today_index = len(sample_df)
        details = get_vote_details(sample_df, today_index)
        assert isinstance(details, dict)

    def test_get_vote_details_has_required_keys(self, sample_df):
        today_index = len(sample_df)
        details = get_vote_details(sample_df, today_index)
        assert "individual_predictions" in details
        assert "number_vote_counts" in details
        assert "special_vote_counts" in details
        assert "final_numbers" in details
        assert "final_special" in details
        assert "models_used" in details

    def test_get_vote_details_models_count(self, sample_df):
        today_index = len(sample_df)
        details = get_vote_details(sample_df, today_index)
        # Should have multiple models
        assert details["models_used"] >= 1
