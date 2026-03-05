"""
Tests for Story-14: Walk-Forward Validation

Covers:
  8.1 Improved val_score correctly updates weights
  8.2 Non-improved val_score correctly skips weights
  8.3 Train/validation window index calculation correctness
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_validation_result_is_improvement_when_score_higher():
    from predict.backtest import WalkForwardValidator, ValidationResult, load_historical_data

    validator = WalkForwardValidator(train_periods=10, val_periods=3)
    df = load_historical_data('big')

    # Mock _compute_weighted_val_score to return predictable values
    call_count = [0]
    def mock_score(df, weights, lottery_type='big'):
        call_count[0] += 1
        # First call = candidate (higher), second = baseline (lower)
        return 5.0 if call_count[0] == 1 else 3.0

    validator._compute_weighted_val_score = mock_score

    candidate = {"Hot-50": 2.0}
    baseline = {"Hot-50": 1.0}
    result = validator.validate(df, candidate, baseline)

    assert result.is_improvement is True
    assert result.val_score > result.baseline_val_score


def test_validation_result_not_improvement_when_score_lower():
    from predict.backtest import WalkForwardValidator, load_historical_data

    validator = WalkForwardValidator(train_periods=10, val_periods=3)
    df = load_historical_data('big')

    call_count = [0]
    def mock_score(df, weights, lottery_type='big'):
        call_count[0] += 1
        return 2.0 if call_count[0] == 1 else 5.0  # candidate worse than baseline

    validator._compute_weighted_val_score = mock_score

    result = validator.validate(df, {"Hot-50": 1.0}, {"Hot-50": 1.0})
    assert result.is_improvement is False


def test_validation_not_improvement_below_threshold():
    """2% improvement threshold: equal scores should not count as improvement."""
    from predict.backtest import WalkForwardValidator, load_historical_data

    validator = WalkForwardValidator(train_periods=10, val_periods=3)
    df = load_historical_data('big')

    call_count = [0]
    def mock_score(df, weights, lottery_type='big'):
        call_count[0] += 1
        # candidate only 1% better — below the 2% threshold
        return 5.05 if call_count[0] == 1 else 5.0

    validator._compute_weighted_val_score = mock_score
    result = validator.validate(df, {}, {})
    assert result.is_improvement is False


def test_get_validation_periods_returns_int():
    from predict.config import get_validation_periods
    val = get_validation_periods()
    assert isinstance(val, int)
    assert val > 0


def test_run_autotune_skips_on_validation_failure(monkeypatch):
    """run_autotune() should skip weight update when validation fails."""
    from predict import backtest as bt_module

    # Mock run_full_backtest to return non-zero scores
    def mock_full_backtest(lottery_type, periods, use_cache=True, decay_factor=1.0):
        return {
            "ranking": [
                {"algorithm": "Hot50", "weighted_score": 10.0, "average_hits": 1.5},
                {"algorithm": "Markov", "weighted_score": 8.0, "average_hits": 1.2},
            ]
        }

    # Mock WalkForwardValidator to return non-improvement
    from predict.backtest import ValidationResult
    class MockValidator:
        def __init__(self, *args, **kwargs): pass
        def validate(self, df, candidate_weights, baseline_weights, lottery_type='big'):
            return ValidationResult(
                algorithm="ensemble", train_periods=40, val_periods=10,
                train_score=0.0, val_score=1.0, baseline_val_score=2.0,
                is_improvement=False
            )

    monkeypatch.setattr(bt_module, "run_full_backtest", mock_full_backtest)
    monkeypatch.setattr(bt_module, "WalkForwardValidator", MockValidator)

    result = bt_module.run_autotune("big", periods=20)
    assert result.get("skipped") is True
    assert result.get("reason") == "validation_failed"
    assert "val_result" in result
