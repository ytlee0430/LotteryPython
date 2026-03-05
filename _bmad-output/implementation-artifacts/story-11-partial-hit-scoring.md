# Story-11: Partial Hit Scoring（分層命中評分）

## Status
done

## Story
**As a** LotteryPython developer,
**I want** the backtest to score partial number matches (3/6, 4/6, 5/6) rather than only full matches,
**So that** algorithm performance metrics are meaningful and optimization targets have enough signal.

## Background
目前 `predict/backtest.py` 的命中計算只有全中（6/6）才算分，導致幾乎所有演算法的歷史分數都是零，使後續的自動調參無法區分演算法優劣。

## Acceptance Criteria
- [ ] AC1: `BacktestResult` 資料結構新增 `partial_hits: dict`，包含 `hit3`, `hit4`, `hit5`, `hit6` 各期命中次數
- [ ] AC2: `compute_hit_score(predicted, actual) -> float` 函式依照分層評分表計算：6/6=100, 5/6=30, 4/6=10, 3/6=3, <3=0
- [ ] AC3: `run_backtest()` 改用 `compute_hit_score()` 計算每期分數，總分為各期加總
- [ ] AC4: 特別號單獨計分：猜中特別號加 5 分（不影響主號分層邏輯）
- [ ] AC5: `backtest_cache` 儲存格式向下相容，cache key 加入 `scoring_version=v2` 以避免舊快取污染
- [ ] AC6: 舊有 `hit_rate` 欄位保留（向下相容），新增 `weighted_score` 與 `partial_hits` 欄位

## Technical Design

### 評分函式
```python
# predict/backtest.py
SCORING_TABLE = {6: 100, 5: 30, 4: 10, 3: 3}

def compute_hit_score(predicted_nums: list, predicted_special: int,
                      actual_nums: list, actual_special: int) -> float:
    main_hits = len(set(predicted_nums) & set(actual_nums))
    special_hit = 5 if predicted_special == actual_special else 0
    return SCORING_TABLE.get(main_hits, 0) + special_hit
```

### BacktestResult 擴充
```python
@dataclass
class BacktestResult:
    algorithm: str
    periods: int
    hit_rate: float          # 維持原有（全中率）
    weighted_score: float    # 新增：分層加權總分
    partial_hits: dict       # 新增：{"hit3": n, "hit4": n, "hit5": n, "hit6": n}
    avg_score_per_period: float  # 新增：每期平均分
```

### Cache Key 更新
```python
cache_key = f"{lottery_type}:{algorithm}:{periods}:{data_version}:scoring_v2"
```

## Tasks / Subtasks
- [ ] Task 1: 新增 `SCORING_TABLE` 常數與 `compute_hit_score()` 函式至 `predict/backtest.py`
  - [ ] 1.1: 實作主號分層計分邏輯
  - [ ] 1.2: 實作特別號加分邏輯
  - [ ] 1.3: 單元測試覆蓋邊界案例（0/6, 3/6, 6/6 + 特別號全中）
- [ ] Task 2: 擴充 `BacktestResult` dataclass
  - [ ] 2.1: 新增 `partial_hits`, `weighted_score`, `avg_score_per_period` 欄位
  - [ ] 2.2: 確保舊欄位 `hit_rate` 保留
- [ ] Task 3: 更新 `run_backtest()` 主迴圈使用新評分
  - [ ] 3.1: 每期呼叫 `compute_hit_score()` 累積分數
  - [ ] 3.2: 計算 `partial_hits` 統計
- [ ] Task 4: 更新快取 key 加入 `scoring_v2`
- [ ] Task 5: 更新 `app.py` API 回傳新欄位（`/api/backtest` 端點）
- [ ] Task 6: 更新 `tests/test_backtest_regression.py` 測試新欄位

## Dev Notes
- `SCORING_TABLE` 分數設計依據：彩券實際獎金比例縮放
- 特別號加 5 分是獨立計算，不與主號合併（避免影響分層閾值）
- `partial_hits` 格式：`{"hit3": int, "hit4": int, "hit5": int, "hit6": int}`
- 影響檔案：`predict/backtest.py`, `app.py`, `tests/test_backtest_regression.py`

## Dependencies
- 無外部新依賴

## Estimated Complexity
低（修改單一模組，不影響預測邏輯）
