# Story-14: Walk-Forward Validation（滾動前向驗證）

## Status
ready

## Story
**As a** LotteryPython developer,
**I want** algorithm weight optimization to be validated on out-of-sample data,
**So that** auto-tuned weights are less likely to overfit historical patterns.

## Background
Story-12 的 auto-tune 若直接用全部 `backtest_periods` 資料進行優化，可能導致過擬合：
優化後的權重在訓練期表現優秀，但在新開獎期表現反而更差。
Walk-Forward Validation 將資料分為「訓練窗口」與「驗證窗口」，只採用在驗證窗口表現更佳的權重。

## Acceptance Criteria
- [ ] AC1: 新增 `WalkForwardValidator` 類別於 `predict/backtest.py`
- [ ] AC2: 預設分割：訓練 40 期 + 驗證 10 期（共 50 期），可透過 config 設定
- [ ] AC3: `run_walk_forward_validation(weights_candidate, weights_baseline)` 回傳 `ValidationResult`：包含 train_score、val_score、is_improvement（bool）
- [ ] AC4: `run_autotune()` 優化完成後，先執行驗證；若 val_score 未優於現有 baseline，**不更新**權重並 log 警告
- [ ] AC5: `algorithm_config.json` 新增 `"validation_periods": 10` 設定
- [ ] AC6: `/api/backtest` 端點回傳資料新增 `validation` 欄位，含 `train_score`、`val_score`、`passed`
- [ ] AC7: Walk-forward 結果快取獨立 table：`backtest_validation_cache`

## Technical Design

### 資料分割示意
```
[ 全部 50 期 ]
|<-- 訓練 40 期 -->|<-- 驗證 10 期 -->|
   用來優化權重         用來評估新權重是否真的更好
```

### WalkForwardValidator
```python
# predict/backtest.py

@dataclass
class ValidationResult:
    algorithm: str
    train_periods: int
    val_periods: int
    train_score: float
    val_score: float
    baseline_val_score: float
    is_improvement: bool

class WalkForwardValidator:
    def __init__(self, train_periods: int = 40, val_periods: int = 10,
                 decay_factor: float = 1.0):
        self.train_periods = train_periods
        self.val_periods = val_periods
        self.decay_factor = decay_factor

    def validate(self, df, algorithm: str,
                 candidate_weights: dict,
                 baseline_weights: dict) -> ValidationResult:
        """
        1. 以 candidate_weights 跑驗證窗口（最近 val_periods 期）
        2. 以 baseline_weights 跑相同驗證窗口
        3. 比較 val_score
        """
        ...
```

### run_autotune() 整合
```python
def run_autotune(lottery_type, periods=50):
    # Step 1: 訓練窗口 backtest
    train_scores = run_backtest(periods=periods - val_periods, ...)

    # Step 2: 計算候選權重
    candidate_weights = compute_softmax_weights(train_scores)

    # Step 3: Walk-forward 驗證
    validator = WalkForwardValidator(train_periods=periods - val_periods,
                                      val_periods=val_periods)
    val_result = validator.validate(df, candidate_weights, current_weights)

    # Step 4: 只在驗證通過時更新
    if val_result.is_improvement:
        update_weights_from_backtest(candidate_weights)
        log.info("Auto-tune applied: val_score improved %.2f → %.2f",
                 val_result.baseline_val_score, val_result.val_score)
    else:
        log.warning("Auto-tune skipped: val_score %.2f did not beat baseline %.2f",
                    val_result.val_score, val_result.baseline_val_score)
```

### ValidationResult 快取
```sql
CREATE TABLE IF NOT EXISTS backtest_validation_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key TEXT UNIQUE NOT NULL,
    lottery_type TEXT NOT NULL,
    train_periods INTEGER NOT NULL,
    val_periods INTEGER NOT NULL,
    candidate_weights_hash TEXT NOT NULL,
    train_score REAL,
    val_score REAL,
    baseline_val_score REAL,
    is_improvement INTEGER,
    created_at TEXT NOT NULL
);
```

## Tasks / Subtasks
- [ ] Task 1: `algorithm_config.json` 新增 `"validation_periods": 10`
- [ ] Task 2: `predict/config.py` 新增 `get_validation_periods()` 函式
- [ ] Task 3: 新增 `ValidationResult` dataclass
- [ ] Task 4: 實作 `WalkForwardValidator` 類別
  - [ ] 4.1: `validate()` 方法：計算候選權重與 baseline 的驗證分數
  - [ ] 4.2: `is_improvement` 判斷邏輯（val_score > baseline_val_score）
- [ ] Task 5: 更新 `run_autotune()` 整合 walk-forward 驗證
  - [ ] 5.1: 訓練/驗證窗口分割
  - [ ] 5.2: 驗證失敗時 skip + log
- [ ] Task 6: 新增 `backtest_validation_cache` SQLite table
- [ ] Task 7: 更新 `/api/backtest` 回傳 `validation` 欄位
- [ ] Task 8: 撰寫測試
  - [ ] 8.1: 驗證分數提升時正確更新權重
  - [ ] 8.2: 驗證分數未提升時正確跳過
  - [ ] 8.3: 訓練/驗證窗口索引計算正確性

## Dev Notes
- `is_improvement` 可加容忍閾值：`val_score > baseline_val_score * 1.02`（高 2% 才算真正改善）
- `candidate_weights_hash`：對 weights dict 做 JSON sort + SHA256，用於 cache key
- 首次執行時（無 baseline 可比較），直接採用候選權重作為初始值
- 影響檔案：`predict/backtest.py`, `predict/config.py`, `algorithm_config.json`, `app.py`

## Dependencies
- **必須先完成 Story-11**（Partial Hit Scoring）
- **必須先完成 Story-12**（Auto-Tune）

## Estimated Complexity
中（新類別，但邏輯清晰；主要複雜度在正確切割資料窗口）
