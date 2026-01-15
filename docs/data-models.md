# 資料模型

## 核心資料結構

### Draw 開獎記錄

```python
from dataclasses import dataclass
from datetime import date
from typing import List

@dataclass
class Draw:
    """單期開獎記錄"""
    period: str          # 期別 (6 位數字串)
    date: date           # 開獎日期
    numbers: List[int]   # 6 個開獎號碼
    special: int         # 特別號
```

### 使用範例
```python
draw = Draw(
    period="113122",
    date=date(2024, 12, 27),
    numbers=[5, 12, 23, 31, 42, 49],
    special=18
)
```

---

## Google Sheets 結構

### 試算表資訊
- **Spreadsheet ID**: `1WApSh6XbBkcjAhDUyO8IvufhPHUX40MOIskl1qL89hQ`

### 工作表一覽

| 工作表名稱 | 用途 | 彩種 |
|-----------|------|------|
| `big-lottery-落球順` | 大樂透（落球順序）| 大樂透 |
| `big-lottery-一般順` | 大樂透（號碼排序）| 大樂透 |
| `power-lottery-落球順` | 威力彩（落球順序）| 威力彩 |
| `power-lottery-一般順` | 威力彩（號碼排序）| 威力彩 |
| `分析結果` | 預測結果儲存 | 通用 |

### 開獎記錄欄位

| 欄位 | 類型 | 說明 | 範例 |
|------|------|------|------|
| ID | integer | 流水編號 | `1` |
| Period | string | 期別（6位數）| `"113122"` |
| Date | string | 開獎日期 (YYYY-MM-DD) | `"2024-12-27"` |
| First | integer | 第一球 | `5` |
| Second | integer | 第二球 | `12` |
| Third | integer | 第三球 | `23` |
| Fourth | integer | 第四球 | `31` |
| Fifth | integer | 第五球 | `42` |
| Sixth | integer | 第六球 | `49` |
| Special | integer | 特別號 | `18` |

### 分析結果欄位

| 欄位 | 類型 | 說明 | 範例 |
|------|------|------|------|
| Period | string | 預測期別 | `"113123"` |
| Date | string | 預測日期 | `"2024-12-28"` |
| Type | string | 彩種類型 | `"big"` |
| Algorithm | string | 演算法名稱 | `"RandomForest"` |
| Numbers | string | 預測號碼（逗號分隔）| `"9,24,32,40,41,49"` |
| Special | integer | 預測特別號 | `27` |

---

## CSV 檔案結構

### 檔案位置
```
lotterypython/
├── big_sequence.csv    # 大樂透（落球順序）
├── big_sorted.csv      # 大樂透（號碼排序）
├── super_sequence.csv  # 威力彩（落球順序）
└── super_sorted.csv    # 威力彩（號碼排序）
```

### CSV 格式
```csv
ID,Period,Date,First,Second,Third,Fourth,Fifth,Sixth,Special
1,93002,2004-01-08,30,47,27,31,16,21,35
2,93003,2004-01-12,15,23,38,41,06,19,28
3,93004,2004-01-15,42,11,33,08,25,47,16
```

### 欄位說明

| 欄位 | 類型 | 約束 | 說明 |
|------|------|------|------|
| ID | int | 唯一, 遞增 | 資料列識別碼 |
| Period | str | 6 位數 | 期別編號 |
| Date | str | YYYY-MM-DD | ISO 8601 日期格式 |
| First-Sixth | int | 1-49 (大樂透), 1-38 (威力彩) | 開獎號碼 |
| Special | int | 1-49 (大樂透), 1-38 (威力彩) | 特別號 |

### Sequence vs Sorted 差異

**Sequence (落球順序)**:
- 號碼依實際落球順序排列
- 適用於分析落球位置模式

```csv
# 落球順序
1,113122,2024-12-27,42,18,7,35,3,49,22
```

**Sorted (一般順序)**:
- 號碼由小到大排序
- 適用於一般分析與對獎

```csv
# 排序後
1,113122,2024-12-27,3,7,18,35,42,49,22
```

---

## Pandas DataFrame 結構

### 載入資料
```python
import pandas as pd
from lotterypython.logic import get_data_from_gsheet

df = get_data_from_gsheet('big')
```

### DataFrame Schema
```python
df.dtypes

# Period     object (string)
# Date       object (string)
# First      int64
# Second     int64
# Third      int64
# Fourth     int64
# Fifth      int64
# Sixth      int64
# Special    int64
```

### 範例資料
```python
print(df.head())

#     Period        Date  First  Second  Third  Fourth  Fifth  Sixth  Special
# 0   93002  2004-01-08     30      47     27      31     16     21       35
# 1   93003  2004-01-12     15      23     38      41      6     19       28
# 2   93004  2004-01-15     42      11     33       8     25     47       16
```

---

## API 回應格式

### 預測結果 JSON
```json
{
  "RandomForest": {
    "next_period": "113123",
    "numbers": [9, 24, 32, 40, 41, 49],
    "special": 27
  }
}
```

### 歷史記錄 JSON
```json
{
  "data": [
    {
      "period": "113122",
      "date": "2024-12-27",
      "numbers": [5, 12, 23, 31, 42, 49],
      "special": 18
    }
  ]
}
```

---

## 期別格式

### 格式規則
- 6 位數字串
- 前 3 位：民國年份
- 後 3 位：該年度期數

### 範例
| 期別 | 解析 |
|------|------|
| `093002` | 民國 93 年第 2 期 |
| `113122` | 民國 113 年第 122 期 |

### 正規化處理
```python
def normalize_period(raw_period: str) -> str:
    """正規化期別格式"""
    # 移除引號前綴
    period = raw_period.lstrip("'\"")

    # 補足 6 位數
    period = period.zfill(6)

    return period
```

---

## 號碼範圍

### 大樂透 (big)
| 項目 | 範圍 |
|------|------|
| 主號碼 | 1 - 49 |
| 特別號 | 1 - 49 |
| 選取數 | 6 + 1 |

### 威力彩 (super)
| 項目 | 範圍 |
|------|------|
| 主號碼 | 1 - 38 |
| 特別號 | 1 - 8 |
| 選取數 | 6 + 1 |

---

## 資料驗證

### 驗證函數範例
```python
def validate_draw(draw: Draw, lotto_type: str) -> bool:
    """驗證開獎資料有效性"""

    if lotto_type == 'big':
        max_num, max_special = 49, 49
    else:  # super
        max_num, max_special = 38, 8

    # 檢查號碼數量
    if len(draw.numbers) != 6:
        return False

    # 檢查號碼範圍
    for num in draw.numbers:
        if not (1 <= num <= max_num):
            return False

    # 檢查特別號範圍
    if not (1 <= draw.special <= max_special):
        return False

    # 檢查號碼唯一性
    if len(set(draw.numbers)) != 6:
        return False

    return True
```
