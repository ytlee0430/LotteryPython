# 每日自動化排程系統

## 概述

本系統提供每日自動化排程功能，自動執行以下任務：

1. **資料更新** - 從 lot539.com 爬取最新開獎結果
2. **預測執行** - 執行所有預測算法
3. **回測分析** - 執行完整回測並快取結果
4. **參數優化** - 自動調整算法參數
5. **結果儲存** - 寫入資料庫和 Google Sheets

## 執行時機

### 依開獎日排程

| 彩種 | 開獎日 | 建議執行時間 |
|------|--------|--------------|
| 大樂透 | 週二、週五 | 21:30 (開獎後30分鐘) |
| 威力彩 | 週一、週四 | 21:30 (開獎後30分鐘) |

### 每日排程（推薦）

每天 21:30 執行，腳本會自動判斷當天應處理哪種彩種。

## 執行流程

```
┌─────────────────────────────────────────────────────────┐
│                    Daily Automation                      │
├─────────────────────────────────────────────────────────┤
│  1. 檢查今日開獎彩種                                      │
│     ├─ 週一、週四 → 威力彩 (super)                        │
│     └─ 週二、週五 → 大樂透 (big)                          │
├─────────────────────────────────────────────────────────┤
│  2. 更新開獎資料                                          │
│     └─ 爬取最新開獎結果 → CSV + Google Sheets             │
├─────────────────────────────────────────────────────────┤
│  3. 清除過期快取                                          │
│     └─ 刪除與新資料版本不符的回測快取                      │
├─────────────────────────────────────────────────────────┤
│  4. 執行回測（寫入快取）                                   │
│     ├─ 完整回測 (50期)                                    │
│     ├─ 滾動回測 (20期 × 5視窗)                            │
│     └─ 參數優化 (Hot/Cold 最佳視窗)                       │
├─────────────────────────────────────────────────────────┤
│  5. 自動調參                                              │
│     └─ 根據回測結果更新 Ensemble 權重                     │
├─────────────────────────────────────────────────────────┤
│  6. 執行預測                                              │
│     └─ 執行所有算法預測下一期號碼                         │
├─────────────────────────────────────────────────────────┤
│  7. 儲存結果                                              │
│     └─ 寫入 Google Sheets「分析結果」工作表               │
├─────────────────────────────────────────────────────────┤
│  8. 產生報告                                              │
│     └─ 輸出執行摘要到 logs/                               │
└─────────────────────────────────────────────────────────┘
```

## 腳本檔案

### 主要腳本

- `scripts/daily_automation.py` - Python 自動化主程式
- `scripts/daily_automation.sh` - Shell 包裝腳本（用於 cron/launchd）

### 設定檔

- `scripts/com.lotterypython.daily.plist` - macOS launchd 設定
- `scripts/crontab.example` - Linux/Unix crontab 範例

## 使用方式

### 手動執行

```bash
# 執行完整自動化流程
python scripts/daily_automation.py

# 指定彩種
python scripts/daily_automation.py --type big
python scripts/daily_automation.py --type super

# 只執行部分任務
python scripts/daily_automation.py --skip-update      # 跳過資料更新
python scripts/daily_automation.py --skip-backtest    # 跳過回測
python scripts/daily_automation.py --skip-predict     # 跳過預測
python scripts/daily_automation.py --skip-autotune    # 跳過自動調參

# 強制執行（忽略今天是否開獎日）
python scripts/daily_automation.py --force

# 測試模式（不實際寫入）
python scripts/daily_automation.py --dry-run
```

### API 觸發

```bash
# 透過 API 手動觸發
curl -X POST http://localhost:3000/automation/run \
  -H "Content-Type: application/json" \
  -d '{"type": "big"}'
```

### macOS launchd 設定

1. 複製設定檔：
```bash
cp scripts/com.lotterypython.daily.plist ~/Library/LaunchAgents/
```

2. 載入排程：
```bash
launchctl load ~/Library/LaunchAgents/com.lotterypython.daily.plist
```

3. 啟動排程：
```bash
launchctl start com.lotterypython.daily
```

4. 查看狀態：
```bash
launchctl list | grep lotterypython
```

5. 停止排程：
```bash
launchctl unload ~/Library/LaunchAgents/com.lotterypython.daily.plist
```

### Linux crontab 設定

編輯 crontab：
```bash
crontab -e
```

加入以下設定（每天 21:30 執行）：
```cron
30 21 * * * /path/to/LotteryPython/scripts/daily_automation.sh >> /path/to/logs/cron.log 2>&1
```

## 設定參數

### 環境變數

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `LOTTERY_PYTHON_ROOT` | 腳本所在目錄的上層 | 專案根目錄 |
| `PYTHON_BIN` | `python3` | Python 執行檔路徑 |
| `LOG_DIR` | `logs/` | 日誌輸出目錄 |
| `BACKTEST_PERIODS` | `50` | 回測期數 |
| `ROLLING_WINDOW` | `20` | 滾動回測視窗大小 |
| `ROLLING_TOTAL` | `100` | 滾動回測總期數 |

### 設定檔 (`config/automation.yaml`)

```yaml
# 執行時間設定
schedule:
  time: "21:30"
  timezone: "Asia/Taipei"

# 回測參數
backtest:
  periods: 50
  rolling_window: 20
  rolling_total: 100

# 優化參數
optimize:
  min_window: 20
  max_window: 100
  step: 10

# 通知設定（選用）
notifications:
  enabled: false
  email: ""
  slack_webhook: ""
```

## 日誌輸出

### 日誌檔案

- `logs/daily_YYYYMMDD.log` - 每日執行日誌
- `logs/errors_YYYYMMDD.log` - 錯誤日誌

### 日誌格式

```
[2026-01-24 21:30:00] INFO  === Daily Automation Started ===
[2026-01-24 21:30:00] INFO  Today is Friday, processing: 大樂透 (big)
[2026-01-24 21:30:01] INFO  Step 1/7: Updating lottery data...
[2026-01-24 21:30:05] INFO  Step 1/7: Updated 1 new records
[2026-01-24 21:30:05] INFO  Step 2/7: Clearing outdated cache...
[2026-01-24 21:30:05] INFO  Step 2/7: Cleared 15 outdated entries
[2026-01-24 21:30:06] INFO  Step 3/7: Running full backtest...
[2026-01-24 21:31:30] INFO  Step 3/7: Backtest completed (84.2s)
[2026-01-24 21:31:30] INFO  Step 4/7: Running rolling backtest...
[2026-01-24 21:33:45] INFO  Step 4/7: Rolling backtest completed (135.1s)
[2026-01-24 21:33:45] INFO  Step 5/7: Running parameter optimization...
[2026-01-24 21:35:20] INFO  Step 5/7: Optimization completed (95.3s)
[2026-01-24 21:35:20] INFO  Step 6/7: Auto-tuning ensemble weights...
[2026-01-24 21:35:21] INFO  Step 6/7: Weights updated
[2026-01-24 21:35:21] INFO  Step 7/7: Running predictions...
[2026-01-24 21:35:25] INFO  Step 7/7: Predictions completed
[2026-01-24 21:35:25] INFO  === Daily Automation Completed ===
[2026-01-24 21:35:25] INFO  Total time: 5m 25s
[2026-01-24 21:35:25] INFO  Results saved to Google Sheets
```

## 執行結果

### 成功回應

```json
{
  "status": "success",
  "lottery_type": "big",
  "timestamp": "2026-01-24T21:35:25+08:00",
  "duration_seconds": 325,
  "steps": {
    "update": {"status": "success", "new_records": 1},
    "clear_cache": {"status": "success", "cleared": 15},
    "backtest": {"status": "success", "cached": true},
    "rolling": {"status": "success", "cached": true},
    "optimize": {"status": "success", "cached": true},
    "autotune": {"status": "success", "weights_updated": true},
    "predict": {"status": "success", "algorithms": 11}
  },
  "cache_stats": {
    "total_entries": 45,
    "total_size_kb": 520.3
  }
}
```

### 錯誤處理

| 錯誤類型 | 處理方式 |
|----------|----------|
| 網路錯誤 | 重試 3 次，間隔 30 秒 |
| 資料更新失敗 | 跳過更新，繼續執行其他任務 |
| 回測失敗 | 記錄錯誤，繼續執行 |
| Google Sheets 寫入失敗 | 記錄錯誤，結果仍保存本地 |

## 監控與告警

### 健康檢查 API

```bash
# 檢查最近執行狀態
curl http://localhost:3000/automation/status
```

回應：
```json
{
  "last_run": "2026-01-24T21:35:25+08:00",
  "last_status": "success",
  "next_scheduled": "2026-01-27T21:30:00+08:00",
  "cache_health": "good"
}
```

## 相關檔案

- [API 參考](api-reference.md) - 自動化 API 端點說明
- [系統架構](architecture.md) - 系統架構圖
- [部署指南](deployment.md) - 部署與設定說明
