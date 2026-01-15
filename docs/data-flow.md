# 資料流程

## 整體資料流

```
┌─────────────────────────────────────────────────────────────┐
│                    lot539.com (資料來源)                     │
│                    Taiwan Lottery Website                    │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP GET (cloudscraper)
                         ▼
            ┌──────────────────────────┐
            │   taiwan_lottery.py      │
            │   TaiwanLottery 類別     │
            │   - fetch_html()         │
            │   - parse_draws()        │
            └──────────────┬───────────┘
                           │ List[Draw]
                           ▼
            ┌──────────────────────────┐
            │    update_data.py        │
            │    - 期別驗證            │
            │    - 資料正規化          │
            │    - 去重處理            │
            └──────────┬───────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────────┐   ┌─────────────┐
   │ 本地 CSV │   │ Google      │   │ 記憶體      │
   │ 儲存    │   │ Sheets      │   │ DataFrame   │
   └─────────┘   └─────────────┘   └──────┬──────┘
                                          │
                                          ▼
                       ┌──────────────────────────┐
                       │      logic.py            │
                       │   run_predictions()      │
                       └──────────┬───────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │           │             │             │           │
        ▼           ▼             ▼             ▼           ▼
    ┌───────┐  ┌────────┐   ┌────────┐   ┌────────┐   ┌─────────┐
    │Hot-50 │  │   RF   │   │   GB   │   │  KNN   │   │  LSTM   │
    └───┬───┘  └───┬────┘   └───┬────┘   └───┬────┘   └────┬────┘
        │          │            │            │             │
        └──────────┴────────────┴────────────┴─────────────┘
                                  │
                                  ▼
                       ┌──────────────────────────┐
                       │   analysis_sheet.py      │
                       │   儲存預測結果           │
                       └──────────┬───────────────┘
                                  │
                                  ▼
                       ┌──────────────────────────┐
                       │    Google Sheets         │
                       │   「分析結果」工作表     │
                       └──────────────────────────┘
```

## 資料更新流程

### 步驟 1: 取得現有資料
```python
# update_data.py
existing_data = worksheet.get_all_values()
last_period = existing_data[-1][1]  # 最後一筆期別
```

### 步驟 2: 計算日期範圍
```python
last_date = parse(existing_data[-1][2])
start_date = last_date + timedelta(days=1)
end_date = datetime.now()
```

### 步驟 3: 爬取新資料
```python
# taiwan_lottery.py
lottery = TaiwanLottery(lotto_type)
new_draws = lottery.get_latest_draws(
    start_date=start_date,
    end_date=end_date
)
```

### 步驟 4: 資料驗證與正規化
```python
for draw in new_draws:
    # 期別格式正規化 (移除引號，補足6位數)
    period = normalize_period(draw.period)

    # 檢查重複
    if period in existing_periods:
        continue

    # 準備資料列
    row = [id, period, date, *numbers, special]
```

### 步驟 5: 寫入儲存
```python
# Google Sheets
worksheet.append_row(row)

# 本地 CSV（落球順序）
with open('big_sequence.csv', 'a') as f:
    writer.writerow(row)

# 本地 CSV（排序順序）
sorted_row = [id, period, date, *sorted(numbers), special]
with open('big_sorted.csv', 'a') as f:
    writer.writerow(sorted_row)
```

## 預測流程

### 步驟 1: 載入歷史資料
```python
# logic.py
df = get_data_from_gsheet(lotto_type)
# DataFrame 欄位: Period, Date, First...Sixth, Special
```

### 步驟 2: 特徵工程
各演算法有不同的特徵處理方式：

#### Hot-50
```python
# 取最近 50 期
recent = df.tail(50)
# 計算號碼頻率
frequency = Counter(all_numbers)
# 取前 6 高頻號碼
```

#### RF/GB/KNN
```python
# 建立 10 期滑動視窗
for i in range(len(df) - 10):
    window = df.iloc[i:i+10]
    # 轉換為 49 維頻率向量
    feature = np.zeros(49)
    for num in window_numbers:
        feature[num-1] += 1
```

#### LSTM
```python
# 序列編碼
sequences = []
for draw in draws:
    one_hot = np.zeros(49)
    for num in draw:
        one_hot[num-1] = 1
    sequences.append(one_hot)

# 建立時間序列
X = [sequences[i:i+10] for i in range(len-10)]
```

### 步驟 3: 模型推論
```python
# 執行所有預測器
results = {
    'Hot-50': hot50_predict(df),
    'RandomForest': rf_predict(df),
    'GradientBoosting': gb_predict(df),
    'KNN': knn_predict(df),
    'LSTM': lstm_predict(df),
    'LSTM-RF': lstmrf_predict(df)
}
```

### 步驟 4: 結果儲存
```python
# analysis_sheet.py
for algo, result in results.items():
    row = [
        next_period,
        date,
        lotto_type,
        algo,
        ','.join(map(str, result['numbers'])),
        result['special']
    ]
    worksheet.append_row(row)
```

## Google Sheets 工作表結構

| 工作表名稱 | 用途 | 欄位 |
|-----------|------|------|
| `big-lottery-落球順` | 大樂透落球順序 | ID, Period, Date, First..Sixth, Special |
| `big-lottery-一般順` | 大樂透排序 | ID, Period, Date, First..Sixth, Special |
| `power-lottery-落球順` | 威力彩落球順序 | ID, Period, Date, First..Sixth, Special |
| `power-lottery-一般順` | 威力彩排序 | ID, Period, Date, First..Sixth, Special |
| `分析結果` | 預測結果 | Period, Date, Type, Algorithm, Numbers, Special |

## CSV 檔案結構

### 檔案位置
```
lotterypython/
├── big_sequence.csv    # 大樂透落球順序
├── big_sorted.csv      # 大樂透排序
├── super_sequence.csv  # 威力彩落球順序
└── super_sorted.csv    # 威力彩排序
```

### 欄位格式
```csv
ID,Period,Date,First,Second,Third,Fourth,Fifth,Sixth,Special
1,93002,2004-01-08,30,47,27,31,16,21,35
2,93003,2004-01-12,15,23,38,41,06,19,28
```

## 同步機制

### Google Sheets → 本地 CSV
```python
def _sync_csv_with_sheet(worksheet, csv_path):
    """確保本地 CSV 與 Sheets 同步"""
    sheet_data = worksheet.get_all_values()
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(sheet_data)
```

### 本地 CSV → Google Sheets
目前為單向同步（Sheets 為主要資料來源），本地 CSV 僅作為備份。
