# LotteryPython UX 改進 Stories

> **Epic 目標**: 提升使用者對預測系統的信任感與個人化體驗
>
> **優先順序**: Story-07 → Story-08（基礎建設）→ Story-09 → Story-10（使用者體驗）
>
> **產生日期**: 2026-02-27

---

## Story-07：為每個預測演算法建立 Smoke Tests

**狀態**: `pending`

### User Story

```
As a developer maintaining LotteryPython,
I want each prediction algorithm to have a smoke test,
So that I can catch regressions immediately and trust that all 10+ algorithms produce valid output.
```

### 背景

目前只有 `rf_gb_knn`、`ensemble` 有測試，`LSTM`、`XGBoost`、`Markov`、`Pattern`、`Astrology-Ziwei`、`Astrology-Zodiac` 完全沒有測試覆蓋。當演算法輸出錯誤時，ensemble 的 `except Exception: pass` 會默默略過，導致預測品質悄悄下降。

### Acceptance Criteria

- [ ] AC1: `tests/test_algorithms_smoke.py` 存在，包含以下演算法的 smoke test：
  - `lotto_predict_hot_50`
  - `lotto_predict_cold_50`
  - `lotto_predict_xgboost`
  - `lotto_predict_lstm`
  - `lotto_predict_LSTMRF`
  - `lotto_predict_markov`
  - `lotto_predict_pattern`
- [ ] AC2: 每個 smoke test 驗證：輸出號碼數量正確（6碼）、號碼在合法範圍內、特別號在合法範圍內
- [ ] AC3: `tests/test_backtest_regression.py` 存在，驗證 `run_full_backtest()` 針對固定 fixture 資料回傳一致結果
- [ ] AC4: `predict/lotto_predict_ensemble.py` 的 `except Exception: pass` 全部改為 `except Exception as e: logger.warning(f"[{name}] failed: {e}")`
- [ ] AC5: `pytest tests/` 全部通過

### Tasks

1. 建立 `tests/test_algorithms_smoke.py`，使用 `lotterypython/big_sequence.csv` 前 60 筆作為 fixture
2. 為每個演算法寫 smoke test 函式（參考 `test_predict.py` 的既有模式）
3. 建立 `tests/test_backtest_regression.py`，使用固定 seed fixture 驗證回測一致性
4. 修復 `lotto_predict_ensemble.py` 的 silent exception swallowing
5. 在 `predict/` 加入 module-level logger

---

## Story-08：建立統一的 PredictorBase 介面

**狀態**: `pending`
**依賴**: Story-07 完成後開始（需先有 smoke tests 保護重構）

### User Story

```
As a developer adding new prediction algorithms to LotteryPython,
I want a common PredictorBase interface all algorithms implement,
So that adding, replacing, or weighting algorithms requires no changes to ensemble logic.
```

### 背景

目前 `predict/lotto_predict_ensemble.py` 必須為每個演算法寫不同的呼叫程式碼，且函式簽名各異（有些回傳 `(nums, special, detail)`，有些回傳 `(nums, special)`）。每次新增演算法都要修改 ensemble 核心邏輯。

### Acceptance Criteria

- [ ] AC1: `predict/base.py` 存在，定義 `PredictorBase` 抽象類別，包含：
  ```python
  def predict(self, df: pd.DataFrame, index: int) -> PredictResult:
      ...
  ```
  其中 `PredictResult` 包含 `numbers: list[int]`、`special: int`、`detail: dict`
- [ ] AC2: 至少 3 個現有演算法重構為 `PredictorBase` 子類別（`Hot50Predictor`、`MarkovPredictor`、`XGBoostPredictor`）
- [ ] AC3: `lotto_predict_ensemble.py` 使用統一介面呼叫已重構的演算法
- [ ] AC4: 所有現有 smoke tests（Story-07）仍然通過
- [ ] AC5: 新增演算法只需繼承 `PredictorBase` 並實作 `predict()`，無需修改 ensemble

### Tasks

1. 建立 `predict/base.py`：定義 `PredictResult` dataclass 和 `PredictorBase` ABC
2. 重構 `lotto_predict_hot_50.py` → `Hot50Predictor(PredictorBase)`
3. 重構 `lotto_predict_markov.py` → `MarkovPredictor(PredictorBase)`
4. 重構 `lotto_predict_xgboost.py` → `XGBoostPredictor(PredictorBase)`
5. 更新 `lotto_predict_ensemble.py` 使用統一介面
6. 確認所有 smoke tests 通過

---

## Story-09：預測結果可解釋性說明

**狀態**: `pending`
**依賴**: Story-08 完成（需要統一介面的 `detail` 欄位）

### User Story

```
As a lottery player using LotteryPython,
I want to see WHY each algorithm recommended specific numbers,
So that I can make informed decisions about which predictions to trust this week.
```

### 背景

使用者目前看到的只是一排數字，不知道 Ensemble 為什麼推薦這組、哪個演算法貢獻最大、這次預測的信心程度如何。缺乏可解釋性讓使用者無法建立對系統的信任。

### Acceptance Criteria

- [ ] AC1: 每個預測結果卡片顯示「推薦理由摘要」，例如：
  - `「8 個演算法中有 5 個推薦號碼 23，加權投票率 68%」`
  - `「近 50 期出現 12 次（熱號）」`
- [ ] AC2: 每個演算法面板可展開顯示該演算法的 `detail`（各號碼的信心分數或推薦原因）
- [ ] AC3: Ensemble 結果顯示**貢獻度圓餅圖或條狀圖**（哪個演算法貢獻了多少權重）
- [ ] AC4: 顯示「本次預測信心指標」：0–100 分，基於演算法間的一致性程度（越多演算法同意同一號碼，分數越高）
- [ ] AC5: 可解釋性資訊不阻擋主要流程——預設摺疊，使用者點擊展開

### Tasks

1. 在各演算法的 `PredictResult.detail` 中加入可解釋性資料（號碼頻率、投票率等）
2. 在 `lotto_predict_ensemble.py` 計算並輸出：每號碼加權票數、各演算法貢獻比例、一致性信心分數
3. 更新 `app.py` 相關 API endpoint，將 detail 資料回傳給前端
4. 更新 `templates/index.html`：加入可折疊的「推薦理由」區塊
5. 加入演算法貢獻度視覺化（使用 Chart.js 或純 CSS 條狀圖）
6. 計算並顯示信心指標（0–100）

---

## Story-10：個人化 Ensemble 權重

**狀態**: `pending`
**依賴**: Story-09 完成（使用者需先能看到每個演算法的表現才能做個人化設定）

### User Story

```
As a regular LotteryPython user,
I want to adjust which prediction algorithms I trust more,
So that my personalized recommendations reflect my own strategy and past experience.
```

### 背景

目前所有使用者共用同一組 ensemble 權重（存在 `algorithm_config.json`）。進階使用者可能有自己的策略偏好（例如相信星座但不信機器學習），或基於個人觀察想調整特定演算法的比重。

### Acceptance Criteria

- [ ] AC1: 使用者個人化權重儲存在 SQLite `users` table 的 `preferences` JSON 欄位
- [ ] AC2: 「設定」頁面提供各演算法的權重滑桿（0 = 停用, 1 = 預設, 2 = 加倍）
- [ ] AC3: 預測時若使用者有個人化權重，優先使用個人化設定；否則 fallback 至全域設定
- [ ] AC4: 使用者可一鍵「重設為預設」
- [ ] AC5: 個人化設定在跨裝置登入後仍然保留（存 DB，非 localStorage）
- [ ] AC6: 權重修改後，下一次預測立即採用新設定，無需重新整理

### Tasks

1. DB migration：在 `users` table 加入 `preferences TEXT` 欄位（JSON）
2. 更新 `UserManager`：加入 `get_preferences(user_id)`、`set_preferences(user_id, prefs)` 方法
3. 新增 API endpoints：`GET /api/preferences`、`PUT /api/preferences`
4. 更新 `lotto_predict_ensemble.py`：接受 `user_weights` 覆寫參數
5. 更新 `app.py` 各預測 endpoint：傳入 `current_user.id` 的個人化權重
6. 建立設定頁面 `templates/settings.html`，包含各演算法權重滑桿
7. 加入「重設為預設」按鈕
8. 撰寫測試：驗證個人化權重覆寫全域設定

---

## 實作建議順序

```
Story-07 (Smoke Tests)
    ↓ 有測試保護
Story-08 (PredictorBase)
    ↓ 有統一介面和 detail 欄位
Story-09 (可解釋性)
    ↓ 使用者看得到各演算法表現
Story-10 (個人化權重)
```

每個 story 獨立可交付，但後者依賴前者的基礎建設。建議一次完成一個 story 再開始下一個。

---

## 附錄：不在本 Epic 範圍內的項目

以下安全/架構項目重要但列為獨立 Epic：

| # | 項目 | 建議處理方式 |
|---|------|------------|
| 1 | SECRET_KEY 硬編碼 fallback | 獨立 hotfix，影響安全性 |
| 2 | credentials.json 在版控中 | 獨立 hotfix |
| 3 | nginx HTTPS | DevOps task |
| 4 | silent exception swallowing | Story-07 AC4 已涵蓋 |
| 5 | app.py Blueprint 拆分 | 架構重構 Epic（較大工程） |
| 6 | Backtest regression tests | Story-07 AC3 已部分涵蓋 |
| 11 | 支援更多彩券類型 | 未來 Feature Epic |
