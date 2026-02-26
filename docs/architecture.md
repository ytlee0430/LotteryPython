# 系統架構

## 架構總覽

LotteryPython 採用分層式架構設計，分為四個主要層級：

```
┌─────────────────────────────────────────────────────────────┐
│                      展示層 (Presentation)                   │
│         Flask Web UI / CLI / REST API                       │
│         index.html / profiles.html                          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      業務邏輯層 (Business Logic)             │
│         預測引擎 / 資料處理 / 排程管理                        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      快取層 (Cache Layer)                    │
│         SQLite 預測快取 / 命理快取                           │
│         all_predictions_cache / prediction_cache            │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      資料存取層 (Data Access)                │
│         Google Sheets API / 本地 CSV / SQLite               │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      外部整合層 (External Integration)       │
│         lot539.com 爬蟲 / Google Cloud / Gemini AI          │
└─────────────────────────────────────────────────────────────┘
```

## 模組架構

### 核心套件 (`lotterypython/`)

```
lotterypython/
├── __init__.py          # 套件初始化
├── __main__.py          # CLI 進入點
├── taiwan_lottery.py    # 網頁爬蟲模組
├── update_data.py       # 資料更新模組
├── logic.py             # 預測協調模組
├── analysis_sheet.py    # 分析結果儲存模組
└── *.csv                # 本地資料檔案
```

### 預測模組 (`predict/`)

```
predict/
├── lotto_predict_hot_50.py      # 頻率分析法（熱號）
├── lotto_predict_cold_50.py     # 冷號分析法
├── lotto_predict_rf_gb_knn.py   # 傳統 ML 演算法
├── lotto_predict_xgboost.py     # XGBoost 極端梯度提升
├── lotto_predict_lstm.py        # LSTM 神經網路
├── lotto_predict_LSTMRF.py      # 混合式模型
├── lotto_predict_markov.py      # 馬可夫鏈
├── lotto_predict_pattern.py     # 組合模式分析
├── lotto_predict_ensemble.py    # 集成投票法
├── lotto_predict_astrology.py   # 命理預測主模組 [NEW]
├── lotto_predict_radom.py       # 隨機基準
└── astrology/                   # 命理預測子模組 [NEW]
    ├── __init__.py
    ├── profiles.py              # 生辰資料管理 (SQLite)
    ├── gemini_client.py         # Gemini CLI 封裝
    └── birth_data.db            # SQLite 資料庫
```

## 元件職責

### `taiwan_lottery.py`
- **職責**: 網頁資料擷取
- **類別**: `TaiwanLottery`
- **功能**:
  - 使用 cloudscraper 繞過反爬蟲機制
  - BeautifulSoup 解析 HTML
  - 回傳 `Draw` dataclass 物件列表

### `update_data.py`
- **職責**: 資料同步與更新
- **功能**:
  - 從 Google Sheets 取得現有資料
  - 爬取新開獎結果
  - 同步更新 Sheets 與本地 CSV

### `logic.py`
- **職責**: 預測流程協調
- **功能**:
  - 從 Google Sheets 載入歷史資料
  - 並行執行多種預測演算法
  - 彙整預測結果

### `analysis_sheet.py`
- **職責**: 分析結果持久化
- **功能**:
  - 將預測結果寫入「分析結果」工作表
  - 自動建立缺失的工作表
  - 依期別排序資料

### `app.py`
- **職責**: Web 應用程式服務
- **功能**:
  - 提供 RESTful API 端點
  - 服務靜態網頁與模板
  - 處理 HTTP 請求與回應

## 依賴關係圖

```
                    ┌──────────────┐
                    │   app.py     │
                    │  (Web API)   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
       ┌──────────┐  ┌──────────┐  ┌──────────────┐
       │ logic.py │  │update_   │  │analysis_     │
       │          │  │data.py   │  │sheet.py      │
       └────┬─────┘  └────┬─────┘  └──────────────┘
            │             │
            │     ┌───────┴───────┐
            │     │               │
            ▼     ▼               ▼
     ┌──────────────┐     ┌──────────────┐
     │   predict/   │     │taiwan_       │
     │   (ML 模組)  │     │lottery.py    │
     └──────────────┘     └──────────────┘
```

## 技術選型理由

| 元件 | 技術選擇 | 理由 |
|------|----------|------|
| 網頁爬蟲 | cloudscraper | 自動處理 JavaScript 渲染與反爬蟲 |
| HTML 解析 | BeautifulSoup4 | 簡潔 API，適合結構化資料擷取 |
| 資料處理 | Pandas | 業界標準，強大的表格操作能力 |
| 傳統 ML | scikit-learn | 成熟穩定，豐富的演算法庫 |
| 深度學習 | TensorFlow/Keras | 序列模型支援佳，LSTM 實作完善 |
| 雲端儲存 | gspread | Google Sheets API 的 Pythonic 封裝 |
| Web 框架 | Flask | 輕量靈活，適合小型 API 服務 |
| 部署 | Docker + Gunicorn | 容器化標準方案，生產環境就緒 |

## 擴展性考量

### 水平擴展
- Web 層可透過 Docker Swarm / Kubernetes 水平擴展
- 預測模組可改為非同步任務佇列（如 Celery）

### 垂直擴展
- ML 模型可升級為更深層網路
- 可加入 GPU 支援加速深度學習推論

### 功能擴展
- 易於新增預測演算法（遵循現有模組介面）
- 可擴展支援其他彩種或國家彩券

## Ensemble 反向權重架構 (2026-02-05)

### 設計決策
集成投票系統（`lotto_predict_ensemble.py`）現支援**負權重**作為反向信號機制。
當某演算法的預測持續低於期望值時，可將其權重設為負數，使其預測的號碼在投票中被扣分。

### 架構影響
- `config.py`: `set_ensemble_weight()` 驗證範圍從 `[0, 5]` 擴展為 `[-3, 5]`
- `algorithm_config.json`: Cold-50 權重設為 `-1.0`
- `update_weights_from_backtest()`: 保護負權重不被自動調參覆蓋
- `lotto_predict_ensemble.py`: 核心投票邏輯無需修改（Counter 天然支持負值）

### 關鍵約束
- 負權重效果在 ≈ -1.0 飽和，更低值無額外收益
- 手動設定的負權重受保護，不會被 auto-tune 覆蓋
- 未來可擴展至其他低效演算法
