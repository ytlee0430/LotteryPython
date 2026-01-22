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
