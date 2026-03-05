"""
Tests for Story-15: Bayesian Weight Optimization

Covers:
  6.1 Mock optuna, test _objective() calls validator correctly
  6.2 Negative-weight algorithms stay fixed in trial
  6.3 optimizer="softmax" does not import optuna
  6.4 Smoke test: optimizer="bayesian", n_trials=3 completes without crash (skipped if no optuna)
"""

import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ─── 6.3: softmax mode does not require optuna ───────────────────────────────

def test_softmax_mode_does_not_import_optuna(monkeypatch):
    """run_autotune with optimizer=softmax should not import optuna."""
    from predict import backtest as bt_module

    # Ensure optuna is not importable during this test
    import sys
    fake_modules = dict(sys.modules)
    fake_modules['optuna'] = None  # blocks import

    def mock_full_backtest(lottery_type, periods, use_cache=True, decay_factor=1.0):
        return {
            "ranking": [
                {"algorithm": "Hot50", "weighted_score": 5.0, "average_hits": 1.0},
            ]
        }

    from predict.backtest import ValidationResult
    class MockValidator:
        def __init__(self, *args, **kwargs): pass
        def validate(self, df, candidate_weights, baseline_weights, lottery_type='big'):
            return ValidationResult(
                algorithm="ensemble", train_periods=10, val_periods=3,
                train_score=0.0, val_score=3.0, baseline_val_score=1.0,
                is_improvement=True
            )

    monkeypatch.setattr(bt_module, "run_full_backtest", mock_full_backtest)
    monkeypatch.setattr(bt_module, "WalkForwardValidator", MockValidator)

    # With optimizer=softmax, should work even with optuna blocked
    import predict.config as cfg_module

    with patch.dict(sys.modules, {'optuna': None}):
        with patch.object(cfg_module, 'update_weights_from_backtest', return_value={}):
            result = bt_module.run_autotune("big", periods=15)
            # Should not raise ImportError from optuna
            # (it may skip due to validation or other reasons but should not crash)
            assert isinstance(result, dict)


# ─── 6.1: _objective() calls validator ───────────────────────────────────────

def test_bayesian_objective_calls_validator():
    """BayesianWeightOptimizer._objective() should call WalkForwardValidator.validate()."""
    # Create mock optuna module
    mock_optuna = MagicMock()
    mock_trial = MagicMock()
    mock_trial.suggest_float.return_value = 1.5

    from predict.backtest import ValidationResult
    mock_val_result = ValidationResult(
        algorithm="ensemble", train_periods=40, val_periods=10,
        train_score=0.0, val_score=4.2, baseline_val_score=2.0,
        is_improvement=True
    )

    mock_validator = MagicMock()
    mock_validator.validate.return_value = mock_val_result

    import pandas as pd
    mock_df = pd.DataFrame({"col": [1, 2, 3]})

    with patch.dict(sys.modules, {'optuna': mock_optuna}):
        from predict.optimizer import BayesianWeightOptimizer

        opt = BayesianWeightOptimizer.__new__(BayesianWeightOptimizer)
        opt.df = mock_df
        opt.lottery_type = 'big'
        opt.validator = mock_validator
        opt._protected = set()
        opt._all_algos = ["Hot-50", "Markov"]

        score = opt._objective(mock_trial)

    assert score == 4.2
    mock_validator.validate.assert_called_once()


# ─── 6.2: Negative-weight algorithms stay fixed ──────────────────────────────

def test_negative_weight_algos_not_in_trial_suggest():
    """Algorithms with negative weights must not be passed to trial.suggest_float()."""
    mock_optuna = MagicMock()
    mock_trial = MagicMock()
    mock_trial.suggest_float.return_value = 1.0

    from predict.backtest import ValidationResult
    mock_validator = MagicMock()
    mock_validator.validate.return_value = ValidationResult(
        algorithm="ensemble", train_periods=40, val_periods=10,
        train_score=0.0, val_score=1.0, baseline_val_score=0.5,
        is_improvement=True
    )

    import pandas as pd
    mock_df = pd.DataFrame({"col": [1]})

    with patch.dict(sys.modules, {'optuna': mock_optuna}):
        from predict.optimizer import BayesianWeightOptimizer

        opt = BayesianWeightOptimizer.__new__(BayesianWeightOptimizer)
        opt.df = mock_df
        opt.lottery_type = 'big'
        opt.validator = mock_validator
        opt._protected = {"Cold-50"}  # negative weight = protected
        opt._all_algos = ["Hot-50", "Cold-50", "Markov"]

        opt._objective(mock_trial)

    # suggest_float should only be called for non-protected algos
    suggested_algos = [call.args[0] for call in mock_trial.suggest_float.call_args_list]
    assert "Cold-50" not in suggested_algos
    assert "Hot-50" in suggested_algos


# ─── 6.4: Smoke test with real optuna (skip if not installed) ─────────────────

@pytest.mark.slow
def test_bayesian_optimizer_smoke():
    """Full smoke test: 3 trials should complete without crash."""
    try:
        import optuna
    except ImportError:
        pytest.skip("optuna not installed — install with pip install optuna>=3.0.0")

    from predict.backtest import load_historical_data
    from predict.optimizer import BayesianWeightOptimizer

    df = load_historical_data('big')
    opt = BayesianWeightOptimizer(
        df, 'big', n_trials=3, timeout=30,
        train_periods=10, val_periods=3
    )
    weights = opt.optimize()

    assert isinstance(weights, dict)
    assert len(weights) > 0
    for algo, w in weights.items():
        assert isinstance(w, float), f"{algo} weight is not float"
