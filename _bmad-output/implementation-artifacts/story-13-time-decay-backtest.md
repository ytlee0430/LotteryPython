# Story-13: Time-Decay Backtest Window（時間衰退回測窗口）

## Status
done

## Story
**As a** LotteryPython prediction system,
**I want** recent lottery periods to have higher weight in backtest scoring than older periods,
**So that** algorithm performance evaluation reflects current patterns rather than stale historical trends.

## Background
彩券號碼的統計規律會隨時間漂移（如熱號、冷號的週期性變化）。
目前 `run_backtest()` 對所有期數一視同仁，導致 50 期前的舊資料與最近 5 期同等影響權重優化。
Time-decay 讓最近的期數影響力更大，提升 auto-tune 的即時性。

## Acceptance Criteria
- [ ] AC1: `run_backtest()` 新增 `decay_factor: float = 1.0` 參數（預設 1.0 = 無衰退，向下相容）
- [ ] AC2: 每期分數乘以衰退係數：`score_i *= decay_factor ** (total_periods - i - 1)`（i=0 為最舊期）
- [ ] AC3: `algorithm_config.json` 新增 `"backtest_decay_factor": 0.95` 設定項
- [ ] AC4: `config.py` 新增 `get_decay_factor() -> float` 讀取函式
- [ ] AC5: `run_autotune()` 自動讀取 decay_factor 並傳入 `run_backtest()`
- [ ] AC6: 衰退因子可透過 Web UI `/settings` 頁面調整（範圍 0.8–1.0，步進 0.01）
- [ ] AC7: Cache key 加入 `decay_factor` 值，避免不同 decay 設定互相污染快取

## Technical Design

### 衰退係數計算
```python
# predict/backtest.py
def run_backtest(df, algorithm, periods=50, decay_factor=1.0):
    results = []
    for i, period_idx in enumerate(range(start_idx, end_idx)):
        predicted = run_algorithm(algorithm, df, period_idx)
        actual = get_actual(df, period_idx)
        raw_score = compute_hit_score(predicted, actual)

        # 時間衰退：i=0 最舊，i=periods-1 最新
        age = (periods - 1) - i   # age=0 為最新期
        decayed_score = raw_score * (decay_factor ** age)
        results.append(decayed_score)

    return BacktestResult(
        weighted_score=sum(results),
        # decay_factor=0.95, periods=50 時：最新期係數=1.0, 最舊期=0.95^49≈0.08
        ...
    )
```

### 衰退效果示意（decay=0.95, periods=50）
```
最新期   係數 = 0.95^0  = 1.000
-5 期前  係數 = 0.95^5  = 0.774
-10 期前 係數 = 0.95^10 = 0.599
-25 期前 係數 = 0.95^25 = 0.277
-49 期前 係數 = 0.95^49 = 0.082
```

### config.json 新增項目
```json
{
  "backtest_decay_factor": 0.95,
  "backtest_periods": 50
}
```

### Cache Key 更新
```python
decay_str = f"decay{int(decay_factor * 100)}"
cache_key = f"{lottery_type}:{algorithm}:{periods}:{data_version}:scoring_v2:{decay_str}"
```

## Tasks / Subtasks
- [ ] Task 1: `algorithm_config.json` 新增 `backtest_decay_factor: 0.95`
- [ ] Task 2: `predict/config.py` 新增 `get_decay_factor()` 函式
- [ ] Task 3: 更新 `run_backtest()` 加入 `decay_factor` 參數與衰退計算邏輯
  - [ ] 3.1: 衰退係數套用至每期 `raw_score`
  - [ ] 3.2: `BacktestResult` 新增 `decay_factor` 記錄欄位
- [ ] Task 4: 更新 Cache key 加入 `decay_factor`
- [ ] Task 5: `run_autotune()` 讀取並傳入 `decay_factor`
- [ ] Task 6: `/settings` 頁面新增 decay_factor 滑桿
  - [ ] 6.1: 範圍 0.80–1.00，步進 0.01，預設 0.95
  - [ ] 6.2: 附說明文字：「值越低代表越重視近期表現」
  - [ ] 6.3: `PUT /api/preferences` 支援儲存 `backtest_decay_factor`
- [ ] Task 7: 撰寫單元測試
  - [ ] 7.1: decay=1.0 時結果與原始 run_backtest 一致（向下相容驗證）
  - [ ] 7.2: decay=0.9 時最新期分數 > 最舊期分數

## Dev Notes
- `decay_factor=1.0` 為預設值，確保不改變現有行為（向下相容）
- 建議初始值 0.95：既有衰退效果，又不會讓舊期完全失去意義
- 衰退公式以「最新期係數=1.0」為基準，避免整體分數縮水影響 softmax 計算
- 影響檔案：`predict/backtest.py`, `predict/config.py`, `algorithm_config.json`, `templates/settings.html`, `app.py`

## Dependencies
- **必須先完成 Story-11**（Partial Hit Scoring）
- Story-12（Auto-Tune）可並行或之後整合

## Estimated Complexity
低至中（單一參數改動，但有快取和 UI 影響）
