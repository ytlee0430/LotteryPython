"""
Bayesian Weight Optimizer (Story-15)

Uses optuna (optional dependency) to optimize ensemble weights via TPE (Tree-structured
Parzen Estimator). Each trial proposes a set of weights; the objective function runs
walk-forward validation and returns the val_score to maximize.

Install optional dependency:
    pip install "lotterypython[optimize]"
or:
    pip install "optuna>=3.0.0"
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class BayesianWeightOptimizer:
    """Optimize ensemble weights via Bayesian search (optuna).

    Each trial suggests weights in [0.1, 3.0] for each algorithm.
    Negative-weight algorithms are protected and kept fixed.
    The objective maximizes walk-forward val_score.
    """

    def __init__(self, df, lottery_type: str,
                 n_trials: int = 50,
                 timeout: int = 300,
                 train_periods: int = 40,
                 val_periods: int = 10,
                 decay_factor: float = 1.0):
        try:
            import optuna  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "optuna is required for Bayesian optimization. "
                "Install with: pip install 'optuna>=3.0.0'"
            )

        from predict.config import get_ensemble_weights
        from predict.backtest import WalkForwardValidator

        self.df = df
        self.lottery_type = lottery_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.validator = WalkForwardValidator(train_periods, val_periods, decay_factor)
        current_weights = get_ensemble_weights()
        self._protected = {k for k, v in current_weights.items() if v < 0}
        self._all_algos = list(current_weights.keys())

    def _objective(self, trial) -> float:
        """Optuna objective: maximize walk-forward val_score."""
        from predict.config import get_ensemble_weights

        current_weights = get_ensemble_weights()
        candidate_weights = {}
        for algo in self._all_algos:
            if algo in self._protected:
                candidate_weights[algo] = current_weights[algo]  # keep protected
            else:
                candidate_weights[algo] = trial.suggest_float(algo, 0.1, 3.0)

        # Normalize to budget
        positive_total = sum(v for v in candidate_weights.values() if v > 0) or 1.0
        normalized = {
            k: v / positive_total * 15.0
            for k, v in candidate_weights.items()
        }

        val_result = self.validator.validate(
            self.df, normalized, current_weights, self.lottery_type
        )
        return val_result.val_score

    def _normalize_weights(self, best_params: dict) -> dict:
        """Normalize best_params back to budget=15, preserving protected weights."""
        from predict.config import get_ensemble_weights

        current_weights = get_ensemble_weights()
        result = {}
        for algo in self._all_algos:
            if algo in self._protected:
                result[algo] = current_weights[algo]
            else:
                result[algo] = best_params.get(algo, current_weights.get(algo, 1.0))

        positive_total = sum(v for v in result.values() if v > 0) or 1.0
        return {k: round(v / positive_total * 15.0, 3) for k, v in result.items()}

    def optimize(self) -> Dict[str, float]:
        """Run Bayesian optimization and return best weights.

        Persists study to SQLite for cross-day continuation.
        """
        import optuna
        from pathlib import Path

        db_path = Path(__file__).parent.parent / "lotterypython" / "lottery.db"
        storage = f"sqlite:///{db_path}"

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            study_name=f"weight_opt_{self.lottery_type}",
            storage=storage,
            load_if_exists=True,
            direction="maximize",
        )

        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )

        if not study.trials:
            logger.warning("Bayesian optimization: no trials completed")
            from predict.config import get_ensemble_weights
            return get_ensemble_weights()

        best_params = study.best_params
        logger.info(
            f"Bayesian optimization complete: {len(study.trials)} trials, "
            f"best val_score={study.best_value:.4f}"
        )
        return self._normalize_weights(best_params)
