# Story-12: Performance-Based Auto-Tune（基於績效自動調參）

## Status
ready

## Story
**As a** LotteryPython system,
**I want** ensemble weights to automatically update based on each algorithm's recent backtest performance,
**So that** better-performing algorithms get higher voting weights without manual intervention.

## Background
`algorithm_config.json` 中 `"auto_tune_enabled": false`，權重目前為人工靜態設定。
依賴 Story-11 的 `weighted_score` 作為優化目標（須先完成）。
現有 `predict/config.py` 已有 `update_weights_from_backtest()` 函式骨架，需完整實作。

## Acceptance Criteria
- [ ] AC1: `auto_tune_enabled: true` 時，每次 daily automation 執行後自動觸發一次 auto-tune
- [ ] AC2: Auto-tune 使用最近 `backtest_periods`（預設 40 期）的 `weighted_score` 計算新權重
- [ ] AC3: 新權重透過 softmax 正規化，使所有權重和為固定值（預設 `total_weight_budget = 15.0`）
- [ ] AC4: 負權重（人工設定）受保護：`update_weights_from_backtest()` 不覆蓋值為負數的權重
- [ ] AC5: 新增 `min_weight=0.3` 下限，避免某演算法權重趨近零而實質失效
- [ ] AC6: Auto-tune 結果寫入 `algorithm_config.json`，並於 log 輸出每個演算法的舊→新權重
- [ ] AC7: 若所有演算法 `weighted_score` 均為零（e.g. 首次執行 Story-11 前），跳過 tune 並 log 警告

## Technical Design

### Softmax Weight Normalization
```python
# predict/config.py
import math

def compute_softmax_weights(scores: dict, temperature: float = 1.0,
                             budget: float = 15.0,
                             min_weight: float = 0.3) -> dict:
    """
    scores: {algo_name: weighted_score_float}
    Returns new weights normalized to sum=budget, floor=min_weight.
    """
    names = list(scores.keys())
    values = [scores[n] / temperature for n in names]
    max_v = max(values)
    exp_v = [math.exp(v - max_v) for v in values]  # numerically stable
    total = sum(exp_v)
    raw = [budget * e / total for e in exp_v]
    # Apply floor
    weights = {n: max(r, min_weight) for n, r in zip(names, raw)}
    return weights
```

### Auto-Tune 觸發流程
```
scripts/daily_automation.py
  └─ run_autotune()
       ├─ run_backtest(periods=40) → 取得 {algo: weighted_score}
       ├─ filter out negative-weight algos (protected)
       ├─ compute_softmax_weights(scores)
       └─ save to algorithm_config.json
```

### 受保護負權重邏輯
```python
def update_weights_from_backtest(new_weights: dict):
    current = get_ensemble_weights()
    for algo, w in new_weights.items():
        if current.get(algo, 1.0) < 0:
            continue  # 保護負權重不被覆蓋
        current[algo] = round(w, 3)
    save_config({"ensemble_weights": current})
```

## Tasks / Subtasks
- [ ] Task 1: 實作 `compute_softmax_weights()` in `predict/config.py`
  - [ ] 1.1: Softmax 計算（數值穩定版本）
  - [ ] 1.2: `min_weight` 下限套用
  - [ ] 1.3: `total_weight_budget` 正規化
- [ ] Task 2: 完整實作 `update_weights_from_backtest()` in `predict/config.py`
  - [ ] 2.1: 負權重保護邏輯
  - [ ] 2.2: 寫回 `algorithm_config.json` 並 log 舊→新權重
- [ ] Task 3: 新增 `run_autotune(lottery_type, periods)` in `predict/backtest.py`
  - [ ] 3.1: 呼叫 backtest 取得各演算法分數
  - [ ] 3.2: 零分全部時 early return + log 警告
  - [ ] 3.3: 呼叫 `compute_softmax_weights()` → `update_weights_from_backtest()`
- [ ] Task 4: 在 `scripts/daily_automation.py` 中接入 `run_autotune()`
  - [ ] 4.1: 讀取 `auto_tune_enabled` flag
  - [ ] 4.2: 在每日預測後觸發
- [ ] Task 5: 更新 `algorithm_config.json`：`"auto_tune_enabled": true`
- [ ] Task 6: 撰寫測試
  - [ ] 6.1: `compute_softmax_weights()` 數值正確性
  - [ ] 6.2: 負權重保護不被覆蓋
  - [ ] 6.3: 全零分數時正確跳過

## Dev Notes
- `temperature` 參數預設 1.0；調高（如 2.0）使權重分布更平均，調低使高分演算法更集中
- `total_weight_budget=15.0`：12 個演算法 × 平均 1.25，與現有配置量級一致
- 依賴 Story-11 完成後方能取得有意義的 `weighted_score`
- 影響檔案：`predict/config.py`, `predict/backtest.py`, `scripts/daily_automation.py`, `algorithm_config.json`

## Dependencies
- **必須先完成 Story-11**（Partial Hit Scoring）

## Estimated Complexity
中（跨多個模組，但邏輯清晰）
