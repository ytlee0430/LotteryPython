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
| 大樂透 | 週二、週五 | 00:00 |
| 威力彩 | 週一、週四 | 00:00 |

### 每日排程（推薦）

每天 00:00 執行，腳本會自動判斷當天應處理哪種彩種。

### 非開獎日行為

非開獎日（週三、週六、週日）仍會執行，自動預測下一期開獎彩種：

| 非開獎日 | 預測彩種 | 說明 |
|----------|----------|------|
| 週三 | 威力彩 | 下一期為週四威力彩 |
| 週六 | 威力彩 | 下一期為週一威力彩 |
| 週日 | 威力彩 | 下一期為週一威力彩 |

非開獎日自動跳過耗時步驟（backtest、rolling backtest、autotune），僅執行：
1. 資料更新（確保不遺漏）
2. 清除過期快取
3. 預測（使用已快取的回測結果）
4. LINE 推播通知

## 執行流程

```
┌─────────────────────────────────────────────────────────┐
│                    Daily Automation                      │
├─────────────────────────────────────────────────────────┤
│  1. 檢查今日開獎彩種                                      │
│     ├─ 週一、週四 → 威力彩 (super)                        │
│     ├─ 週二、週五 → 大樂透 (big)                          │
│     └─ 週三、週六、週日 → 下一期彩種（輕量模式）          │
├─────────────────────────────────────────────────────────┤
│  2. 更新開獎資料（大樂透 + 威力彩 均更新）                 │
│     ├─ 爬取最新開獎結果 → CSV + Google Sheets             │
│     └─ 有新資料時立即推送 LINE 開獎通知                   │
├─────────────────────────────────────────────────────────┤
│  3. 清除過期快取                                          │
│     └─ 刪除與新資料版本不符的回測快取                      │
├─────────────────────────────────────────────────────────┤
│  4. 執行回測（寫入快取）                                   │
│     ├─ 完整回測 (50期)                                    │
│     ├─ 滾動回測 (20期 × 2視窗, ROLLING_TOTAL=40)          │
│     └─ 參數優化 (Hot/Cold 最佳視窗)                       │
├─────────────────────────────────────────────────────────┤
│  5. 自動調參                                              │
│     └─ 根據回測結果更新 Ensemble 權重                     │
├─────────────────────────────────────────────────────────┤
│  6. 執行預測（大樂透 + 威力彩 均執行）                     │
│     └─ 執行全部 13 個算法預測下一期號碼                   │
├─────────────────────────────────────────────────────────┤
│  7. 儲存結果                                              │
│     └─ 寫入 Google Sheets「分析結果」工作表               │
├─────────────────────────────────────────────────────────┤
│  8. LINE 推播通知                                         │
│     └─ 推送全部算法預測號碼給所有啟用通知的使用者          │
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
python scripts/daily_automation.py --skip-notify      # 跳過 LINE 通知

# 快速推送預測（不跑回測，適合手動補發通知）
python scripts/daily_automation.py --type big --skip-backtest --skip-autotune

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

> **注意：** plist 已內建 `caffeinate -i`，執行期間會自動阻止 macOS 進入睡眠，確保長達 3~5 小時的回測不被中斷。

### Linux crontab 設定

編輯 crontab：
```bash
crontab -e
```

加入以下設定（每天 00:00 執行）：
```cron
0 0 * * * /path/to/LotteryPython/scripts/daily_automation.sh >> /path/to/logs/cron.log 2>&1
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
| `ROLLING_TOTAL` | `40` | 滾動回測總期數（視窗數 = ROLLING_TOTAL ÷ ROLLING_WINDOW） |
| `LINE_CHANNEL_ACCESS_TOKEN` | 無 | LINE Messaging API Token（必填，否則跳過推播） |
| `LINE_USER_ID` | 無 | 預設 LINE 使用者 ID |

### 設定檔 (`config/automation.yaml`)

```yaml
# 執行時間設定
schedule:
  time: "00:00"
  timezone: "Asia/Taipei"

# 回測參數
backtest:
  periods: 50
  rolling_window: 20
  rolling_total: 40   # 視窗數 = 40 ÷ 20 = 2

# 優化參數
optimize:
  min_window: 20
  max_window: 100
  step: 10

# 通知設定
notifications:
  line:
    enabled: true     # 需設定 LINE_CHANNEL_ACCESS_TOKEN
```

## 日誌輸出

### 日誌檔案

- `logs/daily_YYYYMMDD.log` - 每日執行日誌
- `logs/errors_YYYYMMDD.log` - 錯誤日誌

### 日誌格式

```
[2026-01-24 21:30:00] INFO  === Daily Automation Started ===
[2026-01-24 21:30:00] INFO  Lottery type: 大樂透 (big)
[2026-01-24 21:30:01] INFO  Step 1/8: Updating lottery data...
[2026-01-24 21:30:05] INFO    大樂透: 1 new record(s)
[2026-01-24 21:30:05] INFO    LINE update notification sent for 大樂透
[2026-01-24 21:30:06] INFO    威力彩: no new records
[2026-01-24 21:30:06] INFO  Step 1/8: Data update done (1 total new records)
[2026-01-24 21:30:06] INFO  Step 2/8: Clearing outdated cache...
[2026-01-24 21:30:06] INFO  Step 2/8: Cleared 15 outdated entries
[2026-01-24 21:30:07] INFO  Step 3/8: Running full backtest...
[2026-01-24 23:12:28] INFO  Step 3/8: Backtest completed (6133.8s, from_cache=False)
[2026-01-24 23:12:28] INFO  Step 4/8: Running rolling backtest...
[2026-01-25 02:39:02] INFO  Step 4/8: Rolling backtest completed (12393.8s, from_cache=False)
[2026-01-25 02:39:02] INFO  Step 5/8: Running parameter optimization...
[2026-01-25 02:39:02] INFO  Step 5/8: Optimization completed (0.2s)
[2026-01-25 02:39:02] INFO  Step 6/8: Auto-tuning ensemble weights...
[2026-01-25 02:39:02] INFO  Step 6/8: Weights updated
[2026-01-25 02:39:02] INFO  Step 7/8: Running predictions (大樂透 + 威力彩)...
[2026-01-25 02:39:43] INFO  Step 7/8: Predictions completed (26 total across both types)
[2026-01-25 02:39:47] INFO    大樂透: results saved to Google Sheets
[2026-01-25 02:39:47] INFO  Step 8/8: Sending LINE notifications (大樂透 + 威力彩)...
[2026-01-25 02:41:50] INFO  Step 8/8: Notifications: 1 sent, 0 failed, 0 skipped
[2026-01-25 02:41:50] INFO  === Daily Automation Completed ===
[2026-01-25 02:41:50] INFO  Total time: 309m 37s
```

## 執行結果

### 成功回應

```json
{
  "status": "success",
  "lottery_type": "big",
  "timestamp": "2026-01-24T21:30:00+08:00",
  "duration_seconds": 18590,
  "steps": {
    "update": {"status": "success", "total_new": 1, "big": {"new_records": 1}, "super": {"new_records": 0}},
    "clear_cache": {"status": "success", "cleared": 15},
    "backtest": {"status": "success", "cached": false, "duration_seconds": 6133.8},
    "rolling": {"status": "success", "cached": false, "duration_seconds": 12393.8},
    "optimize": {"status": "success", "cached": false, "optimal": {"hot_window": 60, "cold_window": 60}},
    "autotune": {"status": "success", "weights_updated": true},
    "predict": {"status": "success", "algorithms_big": 13, "algorithms_super": 13, "saved_to_sheets": true},
    "line_notify": {"status": "success", "notified": 1, "failed": 0, "skipped": 0}
  },
  "cache_stats": {
    "total_entries": 28,
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
