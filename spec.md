# LotteryPython 專案規格書

## 專案概述

LotteryPython 是一個台灣彩券資料分析與預測平台，整合網頁爬蟲、機器學習演算法與 Google Sheets 雲端儲存，提供完整的彩券數據管理與預測解決方案。

## 支援彩種

| 彩種 | 類型代碼 | 號碼範圍 | 開獎日 |
|------|----------|----------|--------|
| 大樂透 | `big` | 1-49（6碼+特別號）| 週二、週五 |
| 威力彩 | `super` | 1-38（6碼+特別號）| 週一、週四 |

## 核心功能

### 1. 資料擷取
- 從 lot539.com 自動爬取最新開獎結果
- 支援歷史資料批次下載
- 自動去重與資料驗證

### 2. 資料儲存
- **主要儲存**: Google Sheets 雲端同步
- **本地備份**: CSV 檔案（落球順序 + 排序順序）
- 雙向同步機制確保資料一致性

### 3. 預測分析
- **Hot-50**: 頻率分析法
- **RandomForest**: 隨機森林分類器
- **GradientBoosting**: 梯度提升分類器
- **KNN**: K-近鄰演算法
- **LSTM**: 長短期記憶神經網路
- **LSTM-RF**: 混合式深度學習模型

### 4. 使用介面
- **Web UI**: Flask 網頁應用程式
- **CLI**: 命令列工具
- **API**: RESTful API 端點

### 5. 自動化排程
- macOS launchd / cron 整合
- 依開獎日自動執行更新與預測

## 系統需求

### 軟體需求
- Python 3.9+
- Docker（選用，用於容器化部署）

### 外部服務
- Google Cloud Service Account（Sheets API 存取）
- 網路連線（lot539.com 資料來源）

## 專案結構

```
LotteryPython/
├── spec.md                 # 本規格文件
├── docs/                   # 詳細架構文檔
│   ├── architecture.md     # 系統架構
│   ├── data-flow.md        # 資料流程
│   ├── api-reference.md    # API 參考
│   ├── prediction-algorithms.md  # 預測演算法
│   ├── deployment.md       # 部署指南
│   ├── cli-reference.md    # CLI 參考
│   └── data-models.md      # 資料模型
├── lotterypython/          # 核心 Python 套件
├── predict/                # 預測演算法模組
├── templates/              # Web UI 模板
├── scripts/                # 自動化腳本
├── tests/                  # 測試套件
├── app.py                  # Flask 應用程式
├── Dockerfile              # Docker 映像檔
└── docker-compose.yml      # Docker 編排設定
```

## 技術棧

| 層級 | 技術 |
|------|------|
| 資料擷取 | cloudscraper, BeautifulSoup4 |
| 資料處理 | Pandas, NumPy |
| 機器學習 | scikit-learn, TensorFlow/Keras |
| 雲端儲存 | gspread, oauth2client |
| Web 框架 | Flask, Gunicorn |
| 容器化 | Docker, Nginx |

## 相關文檔

- [系統架構](docs/architecture.md)
- [資料流程](docs/data-flow.md)
- [API 參考](docs/api-reference.md)
- [預測演算法](docs/prediction-algorithms.md)
- [部署指南](docs/deployment.md)
- [CLI 參考](docs/cli-reference.md)
- [資料模型](docs/data-models.md)

## 版本資訊

- **當前版本**: 1.0.0
- **Python 版本**: 3.9+
- **最後更新**: 2026-01-15
