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

#### 現有演算法
- **Hot-50**: 頻率分析法（熱號策略）
- **RandomForest**: 隨機森林分類器
- **GradientBoosting**: 梯度提升分類器
- **KNN**: K-近鄰演算法
- **LSTM**: 長短期記憶神經網路
- **LSTM-RF**: 混合式深度學習模型

#### 新增演算法
- **Ensemble Voting**: 集成投票法，結合多模型預測結果加權投票
- **Cold-50**: 冷號分析法（Hot-50 的互補策略）
- **XGBoost**: 極端梯度提升，優化版 GradientBoosting
- **Markov Chain**: 馬可夫鏈，分析號碼轉移機率
- **Pattern Analysis**: 組合模式分析（奇偶比、高低比、區間分布）
- **Astrology-Ziwei**: 紫微斗數預測（基於生辰八字，使用 Gemini AI）
- **Astrology-Zodiac**: 西洋星座預測（基於生日，使用 Gemini AI）

### 6. 生辰八字管理
- **多人資料儲存**: SQLite 資料庫支援多人生辰資料
- **家庭分組**: 可依家庭分組管理成員（如「王家」、「李家」）
- **關係標註**: 支援家庭關係（父親、母親、長子、配偶等）
- **資料欄位**: 姓名、國曆出生年月日時、家庭群組、關係
- **Gemini AI 整合**: 透過 Gemini CLI 計算命理推薦號碼
- **Profile 管理 UI**: `/profiles-ui` 網頁介面管理生辰資料

### 7. 預測快取系統
- **SQLite 快取**: 所有預測結果存入 SQLite 資料庫
- **快取鍵值**: `user_id + lottery_type + period` 組合作為唯一識別
- **命理快取**: 依 `user_id + lottery_type + period + method + profile_ids` 快取
- **效能提升**: 首次預測 ~64 秒，快取後 ~1 秒
- **快取管理 API**: `/cache/stats` 查看統計、`/cache/clear` 清除快取
- **自動失效**: 刪除 profile 時自動清除相關快取
- **只保留最新期**: 每用戶每彩種只保留最新一期快取

### 8. 會員系統
- **用戶註冊**: 開放註冊，帳號唯一
- **密碼加密**: 使用 werkzeug.security 加密儲存
- **Session 認證**: Flask-Login + Cookie 管理登入狀態
- **資料隔離**: 每位會員只能存取自己的家人資料
- **登入頁面**: `/login` 登入、`/register` 註冊、`/logout` 登出
- **受保護路由**: 所有預測和 profile 管理 API 需登入
- **用戶表**: SQLite `users` 表儲存帳號資訊

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
│   ├── index.html          # 主頁面（預測介面）
│   ├── login.html          # 登入/註冊頁面
│   └── profiles.html       # Profile 管理頁面
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
| Web 框架 | Flask, Flask-Login, Gunicorn |
| 認證 | werkzeug.security (密碼加密) |
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

- **當前版本**: 1.2.0
- **Python 版本**: 3.9+
- **最後更新**: 2026-01-18
