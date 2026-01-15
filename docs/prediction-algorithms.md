# 預測演算法

## 演算法總覽

LotteryPython 實作了 6 種預測演算法，涵蓋統計分析、傳統機器學習與深度學習方法。

| 演算法 | 類型 | 特點 | 檔案位置 |
|--------|------|------|----------|
| Hot-50 | 統計分析 | 簡單頻率分析 | `predict/lotto_predict_hot_50.py` |
| RandomForest | 傳統 ML | 集成學習 | `predict/lotto_predict_rf_gb_knn.py` |
| GradientBoosting | 傳統 ML | 梯度提升 | `predict/lotto_predict_rf_gb_knn.py` |
| KNN | 傳統 ML | 近鄰分類 | `predict/lotto_predict_rf_gb_knn.py` |
| LSTM | 深度學習 | 序列建模 | `predict/lotto_predict_lstm.py` |
| LSTM-RF | 混合式 | 深度學習+集成 | `predict/lotto_predict_LSTMRF.py` |

---

## Hot-50 頻率分析

### 原理
基於「熱號」假設：近期頻繁出現的號碼可能持續出現。

### 演算法
```python
def predict(df):
    # 取最近 50 期
    recent_draws = df.tail(50)

    # 收集所有號碼
    all_numbers = []
    for _, row in recent_draws.iterrows():
        all_numbers.extend([row['First'], row['Second'], ...])

    # 計算頻率
    frequency = Counter(all_numbers)

    # 取前 6 高頻號碼
    top_6 = [num for num, _ in frequency.most_common(6)]

    # 特別號：最常出現的特別號
    special = Counter(recent_draws['Special']).most_common(1)[0][0]

    return {'numbers': sorted(top_6), 'special': special}
```

### 優缺點
- **優點**: 簡單直觀，計算快速
- **缺點**: 假設過於簡化，無法捕捉複雜模式

---

## RandomForest 隨機森林

### 原理
使用多棵決策樹進行投票，結合 bagging 減少過擬合。

### 特徵工程
```python
def create_features(df, window=10):
    features = []
    for i in range(len(df) - window):
        # 取 10 期滑動視窗
        window_data = df.iloc[i:i+window]

        # 建立 49 維頻率向量
        freq_vector = np.zeros(49)
        for _, row in window_data.iterrows():
            for col in ['First', 'Second', ..., 'Sixth']:
                freq_vector[row[col] - 1] += 1

        features.append(freq_vector)
    return np.array(features)
```

### 模型配置
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

model = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=200, random_state=42)
)
```

### 預測流程
1. 建立 10 期歷史頻率向量
2. 訓練多輸出分類器（6 個輸出對應 6 個號碼）
3. 取各輸出機率最高的號碼
4. 特別號使用獨立的 RandomForest (100 trees)

---

## GradientBoosting 梯度提升

### 原理
逐步建立弱學習器，每一步修正前一步的殘差。

### 模型配置
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier

model = MultiOutputClassifier(
    GradientBoostingClassifier(random_state=42)
)
```

### 與 RandomForest 差異
| 特性 | RandomForest | GradientBoosting |
|------|--------------|------------------|
| 建構方式 | 並行 | 序列 |
| 偏差-方差 | 低方差 | 低偏差 |
| 過擬合風險 | 較低 | 較高 |
| 訓練速度 | 較快 | 較慢 |

---

## KNN K-近鄰演算法

### 原理
找出特徵空間中最近的 K 個鄰居，以多數決進行分類。

### 模型配置
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier

model = MultiOutputClassifier(
    KNeighborsClassifier(n_neighbors=5)
)
```

### 適用場景
- 資料分佈有明顯群聚特性
- 特徵空間維度適中
- 需要快速原型驗證

---

## LSTM 長短期記憶網路

### 原理
透過門控機制學習長期依賴關係，適合序列資料建模。

### 資料編碼
```python
def encode_draw(numbers, dim=49):
    """將一期開獎轉為 one-hot 向量"""
    one_hot = np.zeros(dim)
    for num in numbers:
        one_hot[num - 1] = 1
    return one_hot
```

### 網路架構
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(64, input_shape=(10, 49)),  # 10 期序列，49 維特徵
    Dense(128, activation='relu'),
    Dense(49, activation='sigmoid')   # 多標籤輸出
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)
```

### 訓練配置
| 參數 | 值 |
|------|-----|
| 序列長度 | 10 期 |
| LSTM 單元 | 64 |
| 隱藏層 | 128 單元 |
| 輸出層 | 49 (sigmoid) |
| 訓練週期 | 20 epochs |
| 訓練/驗證比 | 90/10 |

### 預測輸出
```python
# 取機率最高的 7 個號碼
output_probs = model.predict(X_test)
top_7_indices = np.argsort(output_probs[0])[-7:]
numbers = [i + 1 for i in top_7_indices[:6]]
special = top_7_indices[6] + 1
```

---

## LSTM-RF 混合式模型

### 原理
結合 LSTM 的時序建模能力與 RandomForest 的穩健性。

### 增強特徵
```python
def create_enhanced_features(draw, recent_50):
    features = []

    # 1. One-hot 編碼 (49 維)
    one_hot = encode_draw(draw.numbers)
    features.extend(one_hot)

    # 2. 星期幾 (7 維)
    weekday = np.zeros(7)
    weekday[draw.date.weekday()] = 1
    features.extend(weekday)

    # 3. 連續號碼數量 (6 維)
    consecutive = count_consecutive(draw.numbers)
    cons_encoding = np.zeros(6)
    cons_encoding[min(consecutive, 5)] = 1
    features.extend(cons_encoding)

    # 4. 熱號存在標記 (1 維)
    hot_numbers = get_hot_numbers(recent_50, top_n=10)
    hot_presence = any(n in hot_numbers for n in draw.numbers)
    features.append(1 if hot_presence else 0)

    return np.array(features)  # 63 維總特徵
```

### 雙模型架構
```
                    ┌─────────────────┐
                    │   增強特徵      │
                    │   (63 維)       │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐          ┌─────────────────┐
    │     LSTM        │          │  RandomForest   │
    │  (主號碼預測)   │          │  (特別號預測)   │
    └────────┬────────┘          └────────┬────────┘
             │                            │
             │  6 個號碼                  │ 1 個特別號
             └────────────┬───────────────┘
                          ▼
                   最終預測結果
```

### LSTM 配置
```python
model = Sequential([
    LSTM(64, input_shape=(10, 63)),
    Dense(128, activation='relu'),
    Dense(49, activation='sigmoid')
])
# 訓練 25 epochs
```

### 特別號 RandomForest
```python
rf_special = RandomForestClassifier(n_estimators=200, random_state=42)
# 使用相同的增強特徵訓練
```

---

## 效能比較

根據 README 記錄的預測統計：

| 演算法 | 匹配數 | 特點 |
|--------|--------|------|
| RandomForest | 222 | 最高匹配率 |
| KNN | 205 | 穩定表現 |
| GradientBoosting | 203 | 接近 KNN |

> **注意**: 彩券開獎本質上是隨機事件，任何預測演算法都無法保證中獎。以上演算法僅供研究與娛樂用途。

---

## 新增演算法指南

如需新增預測演算法，請遵循以下介面：

```python
# predict/lotto_predict_new.py

def predict(df: pd.DataFrame) -> dict:
    """
    預測演算法介面

    參數:
        df: 歷史開獎資料 DataFrame
            欄位: Period, Date, First, Second, ..., Sixth, Special

    回傳:
        dict: {
            'next_period': str,  # 下一期期別
            'numbers': list[int],  # 6 個預測號碼 (已排序)
            'special': int  # 特別號
        }
    """
    # 實作預測邏輯
    ...
    return {
        'next_period': next_period,
        'numbers': sorted(predicted_numbers),
        'special': predicted_special
    }
```

然後在 `logic.py` 中註冊：
```python
from predict import lotto_predict_new

def run_predictions(df):
    results['NewAlgorithm'] = lotto_predict_new.predict(df)
```
