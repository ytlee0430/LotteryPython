# API 參考

## 概述

LotteryPython 提供 RESTful API 介面，透過 Flask 框架實作。預設運行於 port 3000。

## 基礎 URL

- **開發環境**: `http://localhost:3000`
- **Docker 環境**: `http://localhost:80`（透過 Nginx 反向代理）

## 認證

本 API 使用 Session + Cookie 認證機制。大部分端點需要登入後才能存取。

### 認證流程

1. 使用 `/register` 註冊帳號
2. 使用 `/login` 登入取得 session
3. 後續請求自動帶入 session cookie
4. 使用 `/logout` 登出

---

## 認證端點

### GET /login

**說明**: 顯示登入頁面

**回應**: HTML 頁面（`templates/login.html`）

---

### POST /login

**說明**: 用戶登入

**請求格式** (form-data):
| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| username | string | 是 | 帳號 |
| password | string | 是 | 密碼 |

**成功回應**: 重導向至 `/`

**失敗回應**: 重導向至 `/login` 並顯示錯誤訊息

---

### GET /register

**說明**: 顯示註冊頁面

**回應**: HTML 頁面（`templates/login.html`，註冊 tab）

---

### POST /register

**說明**: 用戶註冊

**請求格式** (form-data):
| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| username | string | 是 | 帳號（唯一）|
| password | string | 是 | 密碼（至少4字元）|
| confirm_password | string | 是 | 確認密碼 |
| email | string | 否 | Email |

**成功回應**: 重導向至 `/login` 並顯示成功訊息

**失敗回應**: 重導向至 `/register` 並顯示錯誤訊息

---

### GET /logout

**說明**: 用戶登出（需登入）

**回應**: 重導向至 `/login`

---

## 端點列表

### GET /

**說明**: 取得主頁面 HTML（需登入，未登入重導向至 `/login`）

**回應**: HTML 頁面（`templates/index.html`）

---

### POST /predict（需登入）

**說明**: 執行預測分析

**請求格式**:
```json
{
  "type": "big" | "super"
}
```

**參數**:
| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| type | string | 是 | 彩種類型：`big`（大樂透）或 `super`（威力彩）|

**成功回應** (200):
```json
{
  "Hot-50": {
    "next_period": "113123",
    "numbers": [3, 12, 24, 35, 41, 47],
    "special": 28
  },
  "RandomForest": {
    "next_period": "113123",
    "numbers": [9, 24, 32, 40, 41, 49],
    "special": 27
  },
  "GradientBoosting": {
    "next_period": "113123",
    "numbers": [5, 18, 22, 35, 42, 48],
    "special": 15
  },
  "KNN": {
    "next_period": "113123",
    "numbers": [7, 14, 29, 33, 39, 45],
    "special": 31
  },
  "LSTM": {
    "next_period": "113123",
    "numbers": [2, 11, 25, 36, 43, 49],
    "special": 19
  },
  "LSTM-RF": {
    "next_period": "113123",
    "numbers": [8, 16, 27, 34, 40, 46],
    "special": 22
  }
}
```

**錯誤回應**:
```json
{
  "error": "錯誤訊息描述"
}
```

---

### POST /update

**說明**: 更新彩券資料（爬取最新開獎並儲存）

**請求格式**:
```json
{
  "type": "big" | "super"
}
```

**參數**:
| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| type | string | 是 | 彩種類型 |

**成功回應** (200):
```json
{
  "status": "success",
  "message": "資料更新完成",
  "new_records": 3
}
```

**錯誤回應**:
```json
{
  "status": "error",
  "message": "更新失敗：無法連接資料來源"
}
```

---

### POST /combin

**說明**: 組合操作 - 執行預測並儲存結果至 Google Sheets

**請求格式**:
```json
{
  "type": "big" | "super"
}
```

**參數**:
| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| type | string | 是 | 彩種類型 |

**成功回應** (200):
```json
{
  "Hot-50": { ... },
  "RandomForest": { ... },
  "GradientBoosting": { ... },
  "KNN": { ... },
  "LSTM": { ... },
  "LSTM-RF": { ... },
  "saved": true
}
```

**流程**:
1. 執行 `/predict` 取得所有演算法預測結果
2. 將結果寫入 Google Sheets「分析結果」工作表
3. 回傳預測結果

---

### POST /history

**說明**: 取得歷史開獎記錄

**請求格式**:
```json
{
  "type": "big" | "super"
}
```

**參數**:
| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| type | string | 是 | 彩種類型 |

**成功回應** (200):
```json
{
  "data": [
    {
      "period": "113122",
      "date": "2024-12-27",
      "numbers": [5, 12, 23, 31, 42, 49],
      "special": 18
    },
    {
      "period": "113121",
      "date": "2024-12-24",
      "numbers": [3, 17, 28, 35, 40, 47],
      "special": 25
    }
  ]
}
```

## 錯誤代碼

| HTTP 狀態碼 | 說明 |
|-------------|------|
| 200 | 成功 |
| 400 | 請求參數錯誤（缺少 type 或無效值）|
| 500 | 伺服器內部錯誤 |

## 使用範例

### cURL

```bash
# 執行大樂透預測
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"type": "big"}'

# 更新威力彩資料
curl -X POST http://localhost:3000/update \
  -H "Content-Type: application/json" \
  -d '{"type": "super"}'

# 取得大樂透歷史記錄
curl -X POST http://localhost:3000/history \
  -H "Content-Type: application/json" \
  -d '{"type": "big"}'
```

### Python

```python
import requests

# 執行預測
response = requests.post(
    'http://localhost:3000/predict',
    json={'type': 'big'}
)
predictions = response.json()

for algo, result in predictions.items():
    print(f"{algo}: {result['numbers']} + {result['special']}")
```

### JavaScript

```javascript
// 執行預測
fetch('/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ type: 'big' })
})
.then(res => res.json())
.then(data => {
  Object.entries(data).forEach(([algo, result]) => {
    console.log(`${algo}: ${result.numbers} + ${result.special}`);
  });
});
```

---

## Profile 管理端點

### GET /profiles

**說明**: 取得所有生辰 profile

**成功回應** (200):
```json
{
  "profiles": [
    {
      "id": 1,
      "name": "王小明",
      "birth_year": 1990,
      "birth_month": 5,
      "birth_day": 15,
      "birth_hour": 14,
      "family_group": "王家",
      "relationship": "長子"
    }
  ],
  "count": 1
}
```

---

### POST /profiles

**說明**: 新增生辰 profile

**請求格式**:
```json
{
  "name": "王小明",
  "birth_year": 1990,
  "birth_month": 5,
  "birth_day": 15,
  "birth_hour": 14,
  "family_group": "王家",
  "relationship": "長子"
}
```

**參數**:
| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| name | string | 是 | 姓名（唯一識別）|
| birth_year | int | 是 | 出生年（國曆）|
| birth_month | int | 是 | 出生月 (1-12) |
| birth_day | int | 是 | 出生日 (1-31) |
| birth_hour | int | 是 | 出生時 (0-23) |
| family_group | string | 否 | 家庭群組（預設 'default'）|
| relationship | string | 否 | 家庭關係 |

---

### DELETE /profiles/{name}

**說明**: 刪除指定 profile（同時清除相關快取）

---

### GET /families

**說明**: 取得所有家庭群組

**成功回應** (200):
```json
{
  "families": [
    { "name": "王家", "member_count": 3 },
    { "name": "default", "member_count": 1 }
  ]
}
```

---

## 命理預測端點

### POST /predict/astrology

**說明**: 執行命理預測（紫微斗數 + 西洋星座）

**請求格式**:
```json
{
  "type": "big" | "super",
  "profile_name": "王小明"  // 選填，不填則使用全部 profiles
}
```

**成功回應** (200):
```json
{
  "Astrology-Ziwei": {
    "numbers": [4, 9, 12, 15, 23, 41],
    "special": 9,
    "method": "紫微斗數",
    "from_cache": true,
    "period": 115005
  },
  "Astrology-Zodiac": {
    "numbers": [4, 6, 10, 19, 23, 27],
    "special": 29,
    "method": "西洋星座",
    "from_cache": true,
    "period": 115005
  }
}
```

---

## 快取管理端點

### GET /cache/stats

**說明**: 取得快取統計資訊

**成功回應** (200):
```json
{
  "astrology_cache": {
    "total_cached": 4,
    "breakdown": [
      { "lottery_type": "big", "method": "ziwei", "count": 1 }
    ]
  },
  "all_predictions_cache": {
    "total_cached": 1,
    "entries": [
      { "lottery_type": "big", "period": "115005", "created_at": "..." }
    ]
  }
}
```

---

### POST /cache/clear

**說明**: 清除所有預測快取

**成功回應** (200):
```json
{
  "message": "Cleared 4 astrology + 1 all-predictions cached entries"
}
```

---

## 回測快取管理端點

回測計算耗時較長，系統會自動快取結果。當新開獎資料加入時，快取會自動失效重新計算。

### GET /cache/backtest/stats

**說明**: 取得回測快取統計資訊

**成功回應** (200):
```json
{
  "algorithm_cache": { "count": 22, "total_size_kb": 45.2 },
  "full_cache": { "count": 4, "total_size_kb": 180.5 },
  "rolling_cache": { "count": 2, "total_size_kb": 120.3 },
  "optimize_cache": { "count": 2, "total_size_kb": 35.8 },
  "total_entries": 30,
  "total_size_kb": 381.8
}
```

---

### POST /cache/backtest/clear

**說明**: 清除回測快取

**請求格式**:
```json
{
  "type": "all" | "algorithm" | "full" | "rolling" | "optimize"
}
```

**參數**:
| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| type | string | 否 | 快取類型，預設 'all' |

- `all`: 清除所有回測快取
- `algorithm`: 只清除單一算法快取
- `full`: 只清除完整回測快取
- `rolling`: 只清除滾動回測快取
- `optimize`: 只清除優化結果快取

**成功回應** (200):
```json
{
  "message": "已清除 30 筆回測快取",
  "cleared": {
    "algorithm": 22,
    "full": 4,
    "rolling": 2,
    "optimize": 2,
    "total": 30
  }
}
```

---

### POST /cache/backtest/clear-outdated

**說明**: 清除過期的回測快取（資料版本不符的快取）

**請求格式**:
```json
{
  "type": "big" | "super" | null
}
```

**參數**:
| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| type | string | 否 | 彩種類型，不填則清除所有彩種的過期快取 |

**成功回應** (200):
```json
{
  "message": "已清除 12 筆過期回測快取",
  "cleared": {
    "algorithm": 8,
    "full": 2,
    "rolling": 1,
    "optimize": 1,
    "total": 12
  }
}
```

---

## 回測快取機制說明

### 快取策略

1. **資料版本追蹤**: 快取 key 包含 `data_version`（最新期數 + 資料筆數），確保新資料加入後自動失效
2. **分層快取**:
   - 個別算法結果（最常使用）
   - 完整回測報告
   - 滾動回測結果
   - 參數優化結果
3. **自動重算**: 快取 miss 時自動計算並儲存

### 回應欄位

回測 API 回應會包含以下快取相關欄位：

| 欄位 | 類型 | 說明 |
|------|------|------|
| from_cache | boolean | 是否來自快取 |
| cached_at | string | 快取建立時間 |
| computation_time_ms | integer | 計算耗時（毫秒）|

### 效能比較

| 情境 | 首次計算 | 快取讀取 |
|------|----------|----------|
| 完整回測 (50期) | ~30-60 秒 | <1 秒 |
| 滾動回測 (100期) | ~2-3 分鐘 | <1 秒 |
| 參數優化 | ~1-2 分鐘 | <1 秒 |

---

## 回應欄位說明

### 預測結果物件

| 欄位 | 類型 | 說明 |
|------|------|------|
| next_period | string | 下一期期別 |
| numbers | array[int] | 6 個預測號碼（已排序）|
| special | int | 特別號預測 |
| from_cache | boolean | 是否來自快取 |
| draw_info | object | 開獎資訊（見下方）|
| combination_reason | object | 號碼組合說明（命理預測專用）|

### draw_info 物件

| 欄位 | 類型 | 說明 |
|------|------|------|
| period | string | 下一期期數 |
| draw_date | string | 開獎日期 (YYYY-MM-DD) |
| weekday | string | 開獎星期 (週一～週日) |
| lottery_name | string | 彩種名稱 (大樂透/威力彩) |
| draw_schedule | string | 開獎時程 (週二、週五 或 週一、週四) |
| display | string | 完整顯示字串 |

### combination_reason 物件（命理預測專用）

| 欄位 | 類型 | 說明 |
|------|------|------|
| method | string | 組合方法名稱 (頻率投票法) |
| description | string | 組合方法說明 |
| numbers | array | 每個號碼的投票詳情 |
| special | object | 特別號的投票詳情 |

### lucky_guidance 物件（命理預測詳情內）

| 欄位 | 類型 | 說明 |
|------|------|------|
| lucky_time | string | 幸運時間 (例: 下午3-5點) |
| lucky_color | string | 幸運顏色 (例: 紅色) |
| lucky_direction | string | 幸運方位 (例: 東南方) |
| lucky_item | string | 幸運物品 (例: 紅色錢包) |

### 歷史記錄物件

| 欄位 | 類型 | 說明 |
|------|------|------|
| period | string | 期別 |
| date | string | 開獎日期 (YYYY-MM-DD) |
| numbers | array[int] | 6 個開獎號碼 |
| special | int | 特別號 |

---

## 回測與分析端點

### POST /backtest

**說明**: 執行完整回測報告，評估所有預測算法的歷史表現

**請求格式**:
```json
{
  "type": "big" | "super",
  "periods": 50  // 選填，預設 50
}
```

**成功回應** (200):
```json
{
  "lottery_type": "big",
  "periods_tested": 50,
  "total_periods_available": 1200,
  "ranking": [
    { "algorithm": "Hot-50", "average_hits": 1.52, "hit_3_plus_rate": 12.0 },
    { "algorithm": "Pattern", "average_hits": 1.48, "hit_3_plus_rate": 10.0 }
  ],
  "algorithms": {
    "Hot-50": {
      "average_hits": 1.52,
      "max_hits": 4,
      "min_hits": 0,
      "special_hit_rate": 8.0,
      "hit_3_or_more": 6,
      "hit_4_or_more": 1,
      "hit_5_or_more": 0,
      "hit_distribution": { "0": 10, "1": 20, "2": 14, "3": 5, "4": 1 }
    }
  }
}
```

---

### POST /analysis/distribution

**說明**: 分析號碼分布統計（奇偶比、高低比、熱門/冷門號碼）

**請求格式**:
```json
{
  "type": "big" | "super",
  "periods": 100  // 選填，預設 100
}
```

**成功回應** (200):
```json
{
  "lottery_type": "big",
  "periods_analyzed": 100,
  "total_numbers_drawn": 600,
  "odd_even_ratio": "312:288",
  "odd_percentage": 52.0,
  "even_percentage": 48.0,
  "high_low_ratio": "295:305",
  "high_percentage": 49.2,
  "low_percentage": 50.8,
  "mid_point": 24,
  "sum_average": 147.5,
  "sum_min": 89,
  "sum_max": 215,
  "sum_std": 28.3,
  "avg_consecutive": 0.85,
  "max_consecutive": 3,
  "hot_numbers": [
    { "number": 12, "count": 18 },
    { "number": 35, "count": 17 }
  ],
  "cold_numbers": [
    { "number": 49, "count": 3 },
    { "number": 41, "count": 4 }
  ],
  "special_hot": [
    { "number": 7, "count": 8 }
  ]
}
```

---

### POST /backtest/rolling

**說明**: 滾動回測 - 分析算法在不同時間視窗的表現一致性

**請求格式**:
```json
{
  "type": "big" | "super",
  "window": 20,   // 每個視窗期數，預設 20
  "total": 100    // 總測試期數，預設 100
}
```

**成功回應** (200):
```json
{
  "lottery_type": "big",
  "window_size": 20,
  "total_periods": 100,
  "num_windows": 5,
  "window_labels": ["001-020", "021-040", "041-060", "061-080", "081-100"],
  "rolling_results": {
    "Hot50": [
      { "window": 1, "average_hits": 1.45, "hit_3_plus": 2 },
      { "window": 2, "average_hits": 1.60, "hit_3_plus": 3 }
    ],
    "Markov": [...]
  },
  "algorithm_summary": {
    "Hot50": {
      "overall_average": 1.52,
      "consistency": 0.15,
      "best_window": 2,
      "worst_window": 4
    }
  }
}
```

---

### POST /backtest/optimize

**說明**: 參數優化 - 找出 Hot/Cold 算法的最佳視窗大小

**請求格式**:
```json
{
  "type": "big" | "super",
  "min": 20,    // 最小視窗，預設 20
  "max": 100,   // 最大視窗，預設 100
  "step": 10    // 步進值，預設 10
}
```

**成功回應** (200):
```json
{
  "lottery_type": "big",
  "test_periods": 50,
  "window_range": { "min": 20, "max": 100, "step": 10 },
  "results": [
    { "window": 20, "algorithm": "Hot", "average_hits": 0.88, "hit_3_plus": 4 },
    { "window": 20, "algorithm": "Cold", "average_hits": 0.82, "hit_3_plus": 3 }
  ],
  "optimal": {
    "hot_window": 20,
    "hot_avg_hits": 0.88,
    "cold_window": 100,
    "cold_avg_hits": 0.82
  }
}
```

---

## 每日自動化端點

### POST /automation/run

**說明**: 觸發每日自動化工作流程

**請求格式**:
```json
{
  "type": "big" | "super" | "auto",
  "skip_update": false,
  "skip_backtest": false,
  "skip_predict": false,
  "skip_autotune": false,
  "dry_run": false
}
```

**參數**:
| 參數 | 類型 | 必填 | 說明 |
|------|------|------|------|
| type | string | 否 | 彩種類型，'auto' 自動判斷今日開獎彩種 |
| skip_update | bool | 否 | 跳過資料更新步驟 |
| skip_backtest | bool | 否 | 跳過回測步驟 |
| skip_predict | bool | 否 | 跳過預測步驟 |
| skip_autotune | bool | 否 | 跳過自動調參步驟 |
| dry_run | bool | 否 | 測試模式，不實際寫入 |

**成功回應** (200):
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

**錯誤回應** (409 - 已在執行中):
```json
{
  "error": "Automation is already running"
}
```

---

### GET /automation/status

**說明**: 取得自動化執行狀態

**成功回應** (200):
```json
{
  "last_run": "2026-01-24T21:35:25+08:00",
  "last_status": "success",
  "last_lottery_type": "big",
  "is_running": false,
  "next_scheduled": "2026-01-27T21:30:00+08:00",
  "next_lottery_type": "super"
}
```

---

## 算法設定端點

### GET /config/algorithm

**說明**: 取得目前算法參數設定

**成功回應** (200):
```json
{
  "hot_window": 50,
  "cold_window": 50,
  "ensemble_weights": {
    "Hot-50": 1.0,
    "Cold-50": 0.8,
    "RandomForest": 1.5,
    "GradientBoosting": 1.0,
    "KNN": 1.2,
    "XGBoost": 1.3,
    "LSTM": 1.0,
    "LSTM-RF": 1.2,
    "Markov": 1.0,
    "Pattern": 0.9
  },
  "auto_tune_enabled": false,
  "backtest_periods": 50
}
```

---

### POST /config/algorithm

**說明**: 更新算法參數設定

**請求格式**:
```json
{
  "hot_window": 60,
  "cold_window": 40,
  "ensemble_weights": {
    "Hot-50": 1.2,
    "RandomForest": 1.8
  }
}
```

**成功回應** (200):
```json
{
  "status": "success",
  "config": { ... }
}
```

---

### POST /config/algorithm/reset

**說明**: 重設算法參數為預設值

**成功回應** (200):
```json
{
  "status": "success",
  "message": "Config reset to defaults",
  "config": { ... }
}
```

---

### POST /config/algorithm/auto-tune

**說明**: 根據回測結果自動調整 Ensemble 權重

**請求格式**:
```json
{
  "type": "big" | "super",
  "periods": 50  // 選填
}
```

**成功回應** (200):
```json
{
  "status": "success",
  "message": "Weights updated from backtest results",
  "new_weights": {
    "Hot-50": 1.52,
    "RandomForest": 1.35
  }
}
```
