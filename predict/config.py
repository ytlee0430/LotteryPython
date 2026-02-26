"""
Prediction Algorithm Configuration

Centralized configuration for all prediction algorithm parameters.
Includes default values, tunable parameters, and weight management.
"""

from typing import Dict, Any
import json
from pathlib import Path

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


def update_weights_from_backtest(backtest_results: Dict) -> Dict[str, float]:
    """
    Update ensemble weights based on backtest results.

    Algorithms with higher average hit counts get higher weights.

    Args:
        backtest_results: Results from run_full_backtest()

    Returns:
        Updated weights dict
    """
    if "ranking" not in backtest_results or not backtest_results["ranking"]:
        return get_ensemble_weights()

    # Calculate weight multipliers based on performance
    # Higher average_hits = higher weight
    ranking = backtest_results["ranking"]
    max_hits = max(r["average_hits"] for r in ranking) if ranking else 1
    min_hits = min(r["average_hits"] for r in ranking) if ranking else 0

    new_weights = get_ensemble_weights()

    for r in ranking:
        algo = r["algorithm"]
        avg_hits = r["average_hits"]

        # Scale weight between 0.5 and 2.0 based on relative performance
        if max_hits > min_hits:
            normalized = (avg_hits - min_hits) / (max_hits - min_hits)
            weight = 0.5 + (normalized * 1.5)  # Range: 0.5 to 2.0
        else:
            weight = 1.0

        # Map backtest algorithm names to ensemble names
        algo_mapping = {
            "Hot50": "Hot-50",
            "Cold50": "Cold-50",
            "Markov": "Markov",
            "Pattern": "Pattern",
        }

        ensemble_name = algo_mapping.get(algo, algo)
        if ensemble_name in new_weights:
            # Preserve manually-set negative weights (contrarian signals)
            if new_weights[ensemble_name] < 0:
                continue
            new_weights[ensemble_name] = round(weight, 2)

    update_config({"ensemble_weights": new_weights})
    return new_weights


def reset_to_defaults() -> Dict[str, Any]:
    """Reset all configuration to defaults."""
    global _config
    _config = DEFAULT_CONFIG.copy()
    save_config()
    return get_config()


# Initialize config on import
load_config()
