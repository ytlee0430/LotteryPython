# API 參考

## 概述

LotteryPython 提供 RESTful API 介面，透過 Flask 框架實作。預設運行於 port 3000。

## 基礎 URL

- **開發環境**: `http://localhost:3000`
- **Docker 環境**: `http://localhost:80`（透過 Nginx 反向代理）

## 端點列表

### GET /

**說明**: 取得主頁面 HTML

**回應**: HTML 頁面（`templates/index.html`）

---

### POST /predict

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

## 回應欄位說明

### 預測結果物件

| 欄位 | 類型 | 說明 |
|------|------|------|
| next_period | string | 下一期期別 |
| numbers | array[int] | 6 個預測號碼（已排序）|
| special | int | 特別號預測 |

### 歷史記錄物件

| 欄位 | 類型 | 說明 |
|------|------|------|
| period | string | 期別 |
| date | string | 開獎日期 (YYYY-MM-DD) |
| numbers | array[int] | 6 個開獎號碼 |
| special | int | 特別號 |
