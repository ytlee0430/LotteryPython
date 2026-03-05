"""
Prediction Algorithm Configuration

Centralized configuration for all prediction algorithm parameters.
Includes default values, tunable parameters, and weight management.
"""

import math
import logging
from typing import Dict, Any
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Config file path
CONFIG_FILE = Path(__file__).parent / "algorithm_config.json"

# Default algorithm parameters
DEFAULT_CONFIG = {
    # Hot/Cold window sizes
    "hot_window": 50,
    "cold_window": 50,

    # Ensemble weights (based on historical performance)
    "ensemble_weights": {
        "Hot-50": 1.0,
        "Cold-50": 0.8,
        "RandomForest": 1.5,
        "GradientBoosting": 1.0,
        "KNN": 1.2,
        "XGBoost": 1.3,
        "LSTM": 1.0,
        "LSTM-RF": 1.2,
        "Markov": 1.0,
        "Pattern": 0.9,
        "Astrology-Ziwei": 0.8,
        "Astrology-Zodiac": 0.7,
    },

    # Auto-tune settings
    "auto_tune_enabled": False,
    "backtest_periods": 50,
}

# In-memory config (loaded at startup)
_config: Dict[str, Any] = None


def load_config() -> Dict[str, Any]:
    """Load configuration from file or use defaults."""
    global _config

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                _config = json.load(f)
            # Ensure all keys exist
            for key, value in DEFAULT_CONFIG.items():
                if key not in _config:
                    _config[key] = value
        except Exception:
            _config = DEFAULT_CONFIG.copy()
    else:
        _config = DEFAULT_CONFIG.copy()

    return _config


def save_config() -> bool:
    """Save current configuration to file."""
    global _config
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(_config, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def get_config() -> Dict[str, Any]:
    """Get current configuration."""
    global _config
    if _config is None:
        load_config()
    return _config.copy()


def update_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration with new values."""
    global _config
    if _config is None:
        load_config()

    for key, value in updates.items():
        if key in _config:
            if isinstance(_config[key], dict) and isinstance(value, dict):
                _config[key].update(value)
            else:
                _config[key] = value

    save_config()
    return get_config()


def get_hot_window() -> int:
    """Get hot algorithm window size."""
    config = get_config()
    return config.get("hot_window", 50)


def get_cold_window() -> int:
    """Get cold algorithm window size."""
    config = get_config()
    return config.get("cold_window", 50)


def get_ensemble_weights() -> Dict[str, float]:
    """Get ensemble algorithm weights."""
    config = get_config()
    return config.get("ensemble_weights", DEFAULT_CONFIG["ensemble_weights"])


def set_hot_window(window: int) -> bool:
    """Set hot algorithm window size."""
    if 10 <= window <= 200:
        update_config({"hot_window": window})
        return True
    return False


def set_cold_window(window: int) -> bool:
    """Set cold algorithm window size."""
    if 10 <= window <= 200:
        update_config({"cold_window": window})
        return True
    return False


def set_ensemble_weight(algorithm: str, weight: float) -> bool:
    """Set weight for a specific algorithm in ensemble. Negative weights enable contrarian signals."""
    if -3 <= weight <= 5:
        config = get_config()
        weights = config.get("ensemble_weights", {})
        weights[algorithm] = weight
        update_config({"ensemble_weights": weights})
        return True
    return False


def compute_softmax_weights(scores: Dict[str, float], temperature: float = 1.0,
                             budget: float = 15.0,
                             min_weight: float = 0.3) -> Dict[str, float]:
    """Compute new ensemble weights via softmax normalization.

    Args:
        scores: {algo_name: weighted_score_float}
        temperature: Higher = more uniform distribution; lower = winner-take-more
        budget: Total weight budget (sum of all weights)
        min_weight: Floor weight to prevent any algorithm from becoming negligible

    Returns:
        New weights normalized to sum=budget, floor=min_weight
    """
    names = list(scores.keys())
    values = [scores[n] / temperature for n in names]
    max_v = max(values) if values else 0.0
    exp_v = [math.exp(v - max_v) for v in values]  # numerically stable
    total = sum(exp_v) or 1.0
    raw = [budget * e / total for e in exp_v]
    weights = {n: max(r, min_weight) for n, r in zip(names, raw)}
    return weights


def update_weights_from_backtest(new_weights: Dict[str, float]) -> Dict[str, float]:
    """Update ensemble weights, protecting negative (contrarian) weights from override.

    Negative weights are manually-set contrarian signals and must not be overwritten.
    Logs old -> new weight change for each algorithm.

    Args:
        new_weights: {algo_name: new_weight} from compute_softmax_weights()

    Returns:
        Updated weights dict (as saved to config)
    """
    current = get_ensemble_weights()
    updated = current.copy()

    for algo, w in new_weights.items():
        if algo not in updated:
            continue
        if updated[algo] < 0:
            logger.info(f"  {algo}: protected (negative weight {updated[algo]}) — skipped")
            continue
        old = updated[algo]
        updated[algo] = round(w, 3)
        logger.info(f"  {algo}: {old} → {updated[algo]}")

    update_config({"ensemble_weights": updated})
    return updated


def get_decay_factor() -> float:
    """Get backtest time-decay factor (1.0 = no decay)."""
    config = get_config()
    return float(config.get("backtest_decay_factor", 1.0))


def reset_to_defaults() -> Dict[str, Any]:
    """Reset all configuration to defaults."""
    global _config
    _config = DEFAULT_CONFIG.copy()
    save_config()
    return get_config()


# Initialize config on import
load_config()
