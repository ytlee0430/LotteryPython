# Story-15: Bayesian Weight Optimization（貝葉斯超參數優化）

## Status
ready

## Story
**As a** LotteryPython power user,
**I want** ensemble weights to be optimized using Bayesian search rather than simple softmax normalization,
**So that** the optimizer efficiently explores the weight space and finds combinations that maximize walk-forward validated performance.

## Background
Story-12 的 Softmax Auto-Tune 是「基於過去分數按比例分配」，屬於貪心策略，無法探索非線性的權重組合空間。
Bayesian Optimization（使用 `optuna`）透過 Tree-structured Parzen Estimator (TPE) 演算法，能在有限次試驗中找到更優的權重組合，並天然結合 Story-14 的 Walk-Forward Validation 作為目標函數。

## Acceptance Criteria
- [ ] AC1: 新增 `predict/optimizer.py` 模組，包含 `BayesianWeightOptimizer` 類別
- [ ] AC2: `BayesianWeightOptimizer.optimize()` 使用 `optuna` 執行 N 次試驗，每次試驗呼叫 walk-forward val_score 作為目標函數
- [ ] AC3: 每個演算法的搜索範圍預設為 `[0.1, 3.0]`（負權重保護：若當前為負則固定不搜索）
- [ ] AC4: `algorithm_config.json` 新增：
  - `"optimizer": "softmax"` | `"bayesian"`（切換優化策略）
  - `"bayesian_n_trials": 50`（試驗次數）
  - `"bayesian_timeout_seconds": 300`（最長執行時間，避免阻塞 daily automation）
- [ ] AC5: Bayesian 優化結果仍須通過 Walk-Forward Validation（Story-14）才會寫入 config
- [ ] AC6: optuna 的 Study 物件持久化至 SQLite（`lottery.db` 新增 `optuna_studies` table），支援跨日繼續優化
- [ ] AC7: `run_autotune()` 根據 `optimizer` 設定自動選擇 softmax 或 bayesian 策略
- [ ] AC8: 新增 `pyproject.toml` optional dependency group `[optimize]`，不影響基本安裝

## Technical Design

### BayesianWeightOptimizer
```python
# predict/optimizer.py
import optuna
from predict.backtest import WalkForwardValidator, run_backtest
from predict.config import get_ensemble_weights, get_validation_periods

class BayesianWeightOptimizer:
    def __init__(self, df, lottery_type: str,
                 n_trials: int = 50,
                 timeout: int = 300,
                 train_periods: int = 40,
                 val_periods: int = 10):
        self.df = df
        self.lottery_type = lottery_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.validator = WalkForwardValidator(train_periods, val_periods)
        self._protected = {k for k, v in get_ensemble_weights().items() if v < 0}

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective: maximize walk-forward val_score."""
        candidate_weights = {}
        for algo in get_ensemble_weights():
            if algo in self._protected:
                candidate_weights[algo] = get_ensemble_weights()[algo]
            else:
                candidate_weights[algo] = trial.suggest_float(algo, 0.1, 3.0)

        # 正規化至 budget
        total = sum(v for v in candidate_weights.values() if v > 0)
        normalized = {k: v / total * 15.0 for k, v in candidate_weights.items()}

        val_result = self.validator.validate(
            self.df, normalized, get_ensemble_weights()
        )
        return val_result.val_score

    def optimize(self) -> dict:
        """Run Bayesian optimization, return best weights dict."""
        storage = f"sqlite:///lotterypython/lottery.db"
        study = optuna.create_study(
            study_name=f"weight_opt_{self.lottery_type}",
            storage=storage,
            load_if_exists=True,  # 跨日繼續
            direction="maximize"
        )
        study.optimize(self._objective,
                       n_trials=self.n_trials,
                       timeout=self.timeout,
                       show_progress_bar=False)

        best_params = study.best_params
        return self._normalize_weights(best_params)
```

### run_autotune() 策略切換
```python
def run_autotune(lottery_type, df, periods=50):
    optimizer_type = get_config().get("optimizer", "softmax")

    if optimizer_type == "bayesian":
        from predict.optimizer import BayesianWeightOptimizer
        opt = BayesianWeightOptimizer(
            df, lottery_type,
            n_trials=get_config().get("bayesian_n_trials", 50),
            timeout=get_config().get("bayesian_timeout_seconds", 300),
        )
        candidate_weights = opt.optimize()
    else:
        # 原有 softmax 邏輯（Story-12）
        scores = run_backtest(...)
        candidate_weights = compute_softmax_weights(scores)

    # 無論哪種策略，最終都走 walk-forward 驗證
    validator = WalkForwardValidator(...)
    if validator.validate(...).is_improvement:
        update_weights_from_backtest(candidate_weights)
```

### pyproject.toml 新增 optional group
```toml
[project.optional-dependencies]
optimize = ["optuna>=3.0.0"]
```

安裝：`pip install "lotterypython[optimize]"`

## Tasks / Subtasks
- [ ] Task 1: `algorithm_config.json` 新增 `optimizer`, `bayesian_n_trials`, `bayesian_timeout_seconds`
- [ ] Task 2: `predict/config.py` 新增對應 getter 函式
- [ ] Task 3: 建立 `predict/optimizer.py` 模組
  - [ ] 3.1: `BayesianWeightOptimizer.__init__()`
  - [ ] 3.2: `_objective()` optuna trial 目標函式
  - [ ] 3.3: `optimize()` 主入口，含 SQLite 持久化 storage
  - [ ] 3.4: `_normalize_weights()` 正規化輸出
  - [ ] 3.5: 負權重保護邏輯（固定不搜索）
- [ ] Task 4: 更新 `run_autotune()` 支援策略切換（softmax / bayesian）
- [ ] Task 5: `pyproject.toml` 新增 `[optimize]` optional dependency group
- [ ] Task 6: 撰寫測試
  - [ ] 6.1: Mock `optuna`，測試 `_objective()` 呼叫 validator 正確
  - [ ] 6.2: 負權重演算法在 trial 中保持固定
  - [ ] 6.3: `optimizer="softmax"` 時不 import optuna（確認非必要依賴）
  - [ ] 6.4: Smoke test：`optimizer="bayesian"`, `n_trials=3` 能跑完不崩潰
- [ ] Task 7: 更新 `docs/prediction-algorithms.md` 說明兩種優化策略

## Dev Notes
- `optuna` 為 optional 依賴；使用前需判斷是否已安裝：`try: import optuna except ImportError: raise RuntimeError("Install with pip install lotterypython[optimize]")`
- `study_name` 含 `lottery_type`（big/super）以區分兩種彩種的 study
- `load_if_exists=True`：允許跨多天累積試驗，Bayesian 優化的效果會隨試驗次數增加
- `timeout=300` 確保 daily automation 不會因優化卡住超過 5 分鐘
- Optuna 預設使用 TPE sampler，無需額外設定
- 影響檔案：`predict/optimizer.py`（新建）, `predict/backtest.py`, `predict/config.py`, `scripts/daily_automation.py`, `algorithm_config.json`, `pyproject.toml`, `docs/prediction-algorithms.md`

## Dependencies
- **必須先完成 Story-11**（Partial Hit Scoring）
- **必須先完成 Story-12**（Auto-Tune）
- **必須先完成 Story-14**（Walk-Forward Validation）
- 外部：`optuna>=3.0.0`（optional）

## Estimated Complexity
高（新模組、外部依賴、SQLite 持久化、策略切換架構）
