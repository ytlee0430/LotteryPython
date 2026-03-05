"""
Tests for Story-12: Performance-Based Auto-Tune

Covers:
  6.1 compute_softmax_weights() numerical correctness
  6.2 Negative weights are protected and not overwritten
  6.3 All-zero scores cause early return with skip flag
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ─── 6.1: compute_softmax_weights numerical correctness ──────────────────────

def test_softmax_weights_sum_to_budget():
    from predict.config import compute_softmax_weights
    scores = {"A": 10.0, "B": 5.0, "C": 2.0}
    weights = compute_softmax_weights(scores, budget=15.0, min_weight=0.0)
    assert abs(sum(weights.values()) - 15.0) < 0.01


def test_softmax_weights_higher_score_gets_higher_weight():
    from predict.config import compute_softmax_weights
    scores = {"A": 100.0, "B": 1.0}
    weights = compute_softmax_weights(scores, budget=15.0, min_weight=0.0)
    assert weights["A"] > weights["B"]


def test_softmax_weights_min_weight_floor():
    from predict.config import compute_softmax_weights
    scores = {"A": 1000.0, "B": 0.001}
    weights = compute_softmax_weights(scores, budget=15.0, min_weight=0.3)
    assert weights["B"] >= 0.3


def test_softmax_weights_uniform_when_equal_scores():
    from predict.config import compute_softmax_weights
    scores = {"A": 5.0, "B": 5.0, "C": 5.0}
    weights = compute_softmax_weights(scores, budget=15.0, min_weight=0.0)
    # Each should be equal (~5.0)
    for w in weights.values():
        assert abs(w - 5.0) < 0.01


# ─── 6.2: Negative weights are protected ─────────────────────────────────────

def test_negative_weights_not_overwritten(tmp_path, monkeypatch):
    from predict import config as cfg_module

    # Patch CONFIG_FILE to a temp file so we don't mutate real config
    import json, copy
    fake_config = {
        "hot_window": 50,
        "cold_window": 50,
        "ensemble_weights": {
            "Hot-50": 1.0,
            "Cold-50": -1.5,   # negative = protected
            "Markov": 0.8,
        },
        "auto_tune_enabled": True,
        "backtest_periods": 50,
    }
    fake_path = tmp_path / "algorithm_config.json"
    fake_path.write_text(json.dumps(fake_config))

    monkeypatch.setattr(cfg_module, "CONFIG_FILE", fake_path)
    monkeypatch.setattr(cfg_module, "_config", None)
    cfg_module.load_config()

    new_weights = {"Hot-50": 2.5, "Cold-50": 3.0, "Markov": 1.2}
    updated = cfg_module.update_weights_from_backtest(new_weights)

    # Cold-50 must remain negative
    assert updated["Cold-50"] < 0, "Negative weight must be protected"
    # Others should be updated
    assert updated["Hot-50"] == round(2.5, 3)
    assert updated["Markov"] == round(1.2, 3)


# ─── 6.3: All-zero scores skip auto-tune ─────────────────────────────────────

def test_run_autotune_skips_when_all_zero(monkeypatch):
    from predict import backtest as bt_module

    # Mock run_full_backtest to return zero weighted_scores
    def mock_full_backtest(lottery_type, periods, use_cache=True):
        return {
            "ranking": [
                {"algorithm": "Hot50", "weighted_score": 0.0, "average_hits": 0.0},
                {"algorithm": "Markov", "weighted_score": 0.0, "average_hits": 0.0},
            ]
        }

    monkeypatch.setattr(bt_module, "run_full_backtest", mock_full_backtest)

    result = bt_module.run_autotune("big", periods=5)
    assert result.get("skipped") is True
    assert result.get("reason") == "all_scores_zero"
