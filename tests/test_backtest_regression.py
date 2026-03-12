"""
Regression tests for the backtest system.

Validates that:
  - run_full_backtest() returns a dict with expected top-level keys
  - Per-algorithm results include required numeric fields
  - Scores are in reasonable range (not NaN, not negative hit rates)
  - Story-11 partial hit scoring fields are present and correct

Uses periods=3 and use_cache=False to keep the test fast and deterministic.
"""

from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ─── compute_hit_score unit tests (Task 1.3) ─────────────────────────────────

def test_compute_hit_score_zero_hits():
    from predict.backtest import compute_hit_score
    assert compute_hit_score([1, 2, 3, 4, 5, 6], 7, [10, 11, 12, 13, 14, 15], 20) == 0


def test_compute_hit_score_three_hits():
    from predict.backtest import compute_hit_score
    assert compute_hit_score([1, 2, 3, 10, 11, 12], 7, [1, 2, 3, 20, 21, 22], 20) == 3


def test_compute_hit_score_six_hits_plus_special():
    from predict.backtest import compute_hit_score
    assert compute_hit_score([1, 2, 3, 4, 5, 6], 7, [1, 2, 3, 4, 5, 6], 7) == 105  # 100 + 5


def test_compute_hit_score_special_only():
    from predict.backtest import compute_hit_score
    assert compute_hit_score([1, 2, 3, 4, 5, 6], 7, [10, 11, 12, 13, 14, 15], 7) == 5


def test_compute_hit_score_five_hits():
    from predict.backtest import compute_hit_score
    assert compute_hit_score([1, 2, 3, 4, 5, 6], 7, [1, 2, 3, 4, 5, 20], 99) == 30


# ─── Task 2.1: run_full_backtest returns expected structure ───────────────────

@pytest.mark.slow
def test_full_backtest_structure():
    from predict.backtest import run_full_backtest
    result = run_full_backtest('big', periods=3, use_cache=False)

    assert isinstance(result, dict), "run_full_backtest should return a dict"
    assert "error" not in result, f"Unexpected error: {result.get('error')}"

    # Top-level keys
    assert "ranking" in result, "result must have 'ranking' key"
    assert "algorithms" in result or any(k in result for k in ["Hot50", "Cold50", "Markov"]), \
        "result must contain algorithm results"


# ─── Task 2.2: Algorithm scores are numeric and sane ─────────────────────────

@pytest.mark.slow
def test_full_backtest_scores_are_sane():
    from predict.backtest import run_full_backtest
    result = run_full_backtest('big', periods=3, use_cache=False)

    assert "error" not in result

    # Check rankings list
    rankings = result.get("ranking", [])
    assert isinstance(rankings, list)
    for rank in rankings:
        algo = rank.get("algorithm", "unknown")
        avg_hits = rank.get("average_hits", None)
        assert avg_hits is not None, f"{algo}: missing average_hits"
        assert isinstance(avg_hits, (int, float)), f"{algo}: average_hits not numeric"
        assert avg_hits >= 0, f"{algo}: average_hits is negative ({avg_hits})"
        assert avg_hits <= 6, f"{algo}: average_hits > 6 makes no sense ({avg_hits})"
        # Story-11: weighted_score in rankings
        assert "weighted_score" in rank, f"{algo}: missing weighted_score in ranking"
        assert rank["weighted_score"] >= 0, f"{algo}: weighted_score is negative"


# ─── Story-11: Partial hit fields in per-algorithm results ───────────────────

@pytest.mark.slow
def test_backtest_algorithm_has_partial_hit_fields():
    from predict.backtest import backtest_algorithm, load_historical_data
    df = load_historical_data('big')
    result = backtest_algorithm(df, 'Hot50', periods=3, lottery_type='big')

    assert "error" not in result, f"backtest_algorithm failed: {result.get('error')}"
    assert "weighted_score" in result, "missing weighted_score"
    assert "avg_score_per_period" in result, "missing avg_score_per_period"
    assert "partial_hits" in result, "missing partial_hits"

    ph = result["partial_hits"]
    assert set(ph.keys()) == {"hit3", "hit4", "hit5", "hit6"}
    assert all(isinstance(v, int) for v in ph.values())
    assert result["weighted_score"] >= 0
    assert result["avg_score_per_period"] >= 0
