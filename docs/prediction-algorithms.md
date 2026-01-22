# 預測演算法

## 演算法總覽

LotteryPython 實作了 11 種預測演算法，涵蓋統計分析、傳統機器學習、深度學習與集成方法。

### 現有演算法

| 演算法 | 類型 | 特點 | 檔案位置 |
|--------|------|------|----------|
| Hot-50 | 統計分析 | 熱號頻率分析 | `predict/lotto_predict_hot_50.py` |
| RandomForest | 傳統 ML | 集成學習 | `predict/lotto_predict_rf_gb_knn.py` |
| GradientBoosting | 傳統 ML | 梯度提升 | `predict/lotto_predict_rf_gb_knn.py` |
| KNN | 傳統 ML | 近鄰分類 | `predict/lotto_predict_rf_gb_knn.py` |
| LSTM | 深度學習 | 序列建模 | `predict/lotto_predict_lstm.py` |
| LSTM-RF | 混合式 | 深度學習+集成 | `predict/lotto_predict_LSTMRF.py` |

### 新增演算法

| 演算法 | 類型 | 特點 | 檔案位置 |
|--------|------|------|----------|
| Cold-50 | 統計分析 | 冷號頻率分析 | `predict/lotto_predict_cold_50.py` |
| XGBoost | 傳統 ML | 極端梯度提升 | `predict/lotto_predict_xgboost.py` |
| Markov Chain | 機率模型 | 號碼轉移機率 | `predict/lotto_predict_markov.py` |
| Pattern Analysis | 統計分析 | 組合模式分析 | `predict/lotto_predict_pattern.py` |
| Ensemble Voting | 集成方法 | 多模型加權投票 | `predict/lotto_predict_ensemble.py` |
| Astrology-Ziwei | AI 命理 | 紫微斗數預測 | `predict/lotto_predict_astrology.py` |
| Astrology-Zodiac | AI 命理 | 西洋星座預測 | `predict/lotto_predict_astrology.py` |

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

---

## 新增演算法詳細說明

---

## Cold-50 冷號分析

### 原理
基於「冷號回歸」假設：長期未出現的號碼可能即將出現（Hot-50 的互補策略）。

### 演算法
```python
def predict(df):
    # 取最近 50 期
    recent_draws = df.tail(50)

    # 收集所有已出現號碼
    appeared_numbers = set()
    for _, row in recent_draws.iterrows():
        appeared_numbers.update([row['First'], row['Second'], ...])

    # 計算各號碼最後出現距今期數
    all_numbers = range(1, 50)  # 大樂透 1-49
    last_seen = {}
    for num in all_numbers:
        last_seen[num] = calculate_last_appearance(df, num)

    # 取最久未出現的 6 個號碼
    coldest_6 = sorted(last_seen.keys(), key=lambda x: last_seen[x], reverse=True)[:6]

    # 特別號：最久未出現的特別號
    special = get_coldest_special(df)

    return {'numbers': sorted(coldest_6), 'special': special}
```

### 優缺點
- **優點**: 與 Hot-50 形成互補策略，覆蓋不同假設
- **缺點**: 「賭徒謬誤」- 過去不影響未來獨立事件

---

## XGBoost 極端梯度提升

### 原理
XGBoost 是 GradientBoosting 的優化實作，具有更好的正則化、並行處理與缺失值處理能力。

### 模型配置
```python
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier

model = MultiOutputClassifier(
    xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
)
```

### 與 GradientBoosting 差異
| 特性 | GradientBoosting | XGBoost |
|------|------------------|---------|
| 正則化 | 無內建 | L1/L2 正則化 |
| 並行處理 | 否 | 是 |
| 缺失值處理 | 需預處理 | 自動處理 |
| 訓練速度 | 較慢 | 較快 |
| 過擬合控制 | 較弱 | 較強 |

### 預測流程
1. 使用與 RandomForest 相同的特徵工程
2. XGBoost 多輸出分類器訓練
3. 特別號使用獨立 XGBoost 模型

---

## Markov Chain 馬可夫鏈

### 原理
將每期開獎視為狀態，分析號碼之間的轉移機率矩陣，預測下一期最可能出現的號碼。

### 演算法
```python
import numpy as np

def build_transition_matrix(df, num_range=49):
    """建立號碼轉移機率矩陣"""
    # 轉移矩陣 49x49
    transition = np.zeros((num_range, num_range))

    for i in range(len(df) - 1):
        current_draw = get_numbers(df.iloc[i])
        next_draw = get_numbers(df.iloc[i + 1])

        # 記錄從 current 到 next 的轉移
        for curr_num in current_draw:
            for next_num in next_draw:
                transition[curr_num - 1][next_num - 1] += 1

    # 正規化為機率
    row_sums = transition.sum(axis=1, keepdims=True)
    transition_prob = np.divide(transition, row_sums,
                                where=row_sums != 0)
    return transition_prob

def predict(df):
    trans_matrix = build_transition_matrix(df)

    # 取最近一期的號碼
    last_draw = get_numbers(df.iloc[-1])

    # 計算下一期各號碼的機率
    next_probs = np.zeros(49)
    for num in last_draw:
        next_probs += trans_matrix[num - 1]

    # 取機率最高的 6 個號碼
    top_6_indices = np.argsort(next_probs)[-6:]
    predicted = [i + 1 for i in top_6_indices]

    return {'numbers': sorted(predicted), 'special': predict_special(df)}
```

### 優缺點
- **優點**: 捕捉號碼間的時序關聯性
- **缺點**: 假設一階馬可夫性質（僅依賴前一期）

---

## Pattern Analysis 組合模式分析

### 原理
分析開獎號碼的組合特徵模式，包括奇偶比、高低比、區間分布、連號等。

### 特徵分析
```python
def analyze_patterns(numbers, num_range=49):
    """分析號碼組合特徵"""
    patterns = {}

    # 1. 奇偶比 (odd:even)
    odd_count = sum(1 for n in numbers if n % 2 == 1)
    patterns['odd_even'] = (odd_count, 6 - odd_count)

    # 2. 高低比 (high:low, 以 25 為界)
    high_count = sum(1 for n in numbers if n > 24)
    patterns['high_low'] = (high_count, 6 - high_count)

    # 3. 區間分布 (分 5 區)
    zones = [0] * 5  # 1-10, 11-20, 21-30, 31-40, 41-49
    for n in numbers:
        zone_idx = min((n - 1) // 10, 4)
        zones[zone_idx] += 1
    patterns['zones'] = zones

    # 4. 連號數量
    sorted_nums = sorted(numbers)
    consecutive = 0
    for i in range(len(sorted_nums) - 1):
        if sorted_nums[i + 1] - sorted_nums[i] == 1:
            consecutive += 1
    patterns['consecutive'] = consecutive

    # 5. 號碼總和
    patterns['sum'] = sum(numbers)

    return patterns

def predict(df):
    # 分析歷史模式分布
    pattern_history = [analyze_patterns(get_numbers(row))
                       for _, row in df.iterrows()]

    # 找出最常見的模式組合
    common_odd_even = most_common([p['odd_even'] for p in pattern_history])
    common_zones = most_common_distribution([p['zones'] for p in pattern_history])
    target_sum_range = get_common_sum_range(pattern_history)

    # 依據目標模式生成號碼
    predicted = generate_numbers_matching_pattern(
        odd_even=common_odd_even,
        zones=common_zones,
        sum_range=target_sum_range
    )

    return {'numbers': sorted(predicted), 'special': predict_special(df)}
```

### 常見模式統計
| 模式 | 最常見分布 | 出現機率 |
|------|-----------|----------|
| 奇偶比 | 3:3 | ~32% |
| 高低比 | 3:3 | ~31% |
| 連號 | 0-1 組 | ~75% |
| 總和範圍 | 130-170 | ~60% |

---

## Ensemble Voting 集成投票法

### 原理
結合所有現有預測模型的結果，透過加權投票機制產生最終預測，利用「群眾智慧」提高準確度。

### 演算法
```python
from collections import Counter

def ensemble_predict(df, models, weights=None):
    """
    集成投票預測

    參數:
        df: 歷史資料
        models: 預測模型列表
        weights: 各模型權重（預設等權重）
    """
    if weights is None:
        weights = [1.0] * len(models)

    # 收集所有模型的預測結果
    all_predictions = []
    for model, weight in zip(models, weights):
        result = model.predict(df)
        all_predictions.append({
            'numbers': result['numbers'],
            'special': result['special'],
            'weight': weight
        })

    # 加權投票 - 主號碼
    number_votes = Counter()
    for pred in all_predictions:
        for num in pred['numbers']:
            number_votes[num] += pred['weight']

    # 取得票最高的 6 個號碼
    top_6 = [num for num, _ in number_votes.most_common(6)]

    # 加權投票 - 特別號
    special_votes = Counter()
    for pred in all_predictions:
        special_votes[pred['special']] += pred['weight']

    special = special_votes.most_common(1)[0][0]

    return {'numbers': sorted(top_6), 'special': special}

# 使用範例
def predict(df):
    from predict import (lotto_predict_hot_50, lotto_predict_rf_gb_knn,
                         lotto_predict_lstm, lotto_predict_LSTMRF,
                         lotto_predict_cold_50, lotto_predict_xgboost)

    models = [
        lotto_predict_hot_50,
        lotto_predict_cold_50,
        lotto_predict_rf_gb_knn,  # RandomForest
        lotto_predict_xgboost,
        lotto_predict_lstm,
        lotto_predict_LSTMRF
    ]

    # 權重可依據歷史表現調整
    # 例如 RandomForest 匹配率最高，給予較高權重
    weights = [1.0, 0.8, 1.5, 1.3, 1.0, 1.2]

    return ensemble_predict(df, models, weights)
```

### 權重調整策略
| 策略 | 說明 |
|------|------|
| 等權重 | 所有模型權重相同 |
| 歷史表現 | 依據匹配率調整權重 |
| 動態權重 | 依據近期表現動態調整 |
| 衰減權重 | 近期表現權重較高 |

### 優缺點
- **優點**: 綜合多種方法優勢，減少單一模型偏差
- **缺點**: 計算成本較高（需執行所有模型）

---

## Astrology-Ziwei 紫微斗數預測

### 原理
基於中國傳統命理學「紫微斗數」，根據使用者的生辰八字（國曆年月日時），透過 Gemini AI 分析命盤，推算適合的彩券號碼。

### 資料儲存
使用 SQLite 資料庫儲存多人生辰資料：

```sql
CREATE TABLE birth_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    birth_year INTEGER NOT NULL,
    birth_month INTEGER NOT NULL,
    birth_day INTEGER NOT NULL,
    birth_hour INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Gemini CLI 整合
```python
import subprocess
import json

def call_gemini_ziwei(profile, lottery_type='big'):
    max_num = 49 if lottery_type == 'big' else 38

    prompt = f'''
    你是一位紫微斗數大師。請根據以下生辰資料分析命盤，
    並推薦最適合購買彩券的號碼。

    姓名: {profile['name']}
    出生年: {profile['birth_year']}
    出生月: {profile['birth_month']}
    出生日: {profile['birth_day']}
    出生時: {profile['birth_hour']}時

    請推薦 6 個主要號碼 (1-{max_num}) 和 1 個特別號 (1-{max_num})。

    請只回傳 JSON 格式，不要有其他文字:
    {{"numbers": [1,2,3,4,5,6], "special": 7, "analysis": "簡短命理分析"}}
    '''

    result = subprocess.run(
        ['gemini', prompt],
        capture_output=True,
        text=True,
        timeout=60
    )
    return json.loads(result.stdout)
```

### 幸運指南
Gemini AI 同時提供購買彩券的幸運指南：

| 欄位 | 說明 | 範例 |
|------|------|------|
| lucky_time | 幸運時間 | 下午3-5點 |
| lucky_color | 幸運顏色 | 紅色 |
| lucky_direction | 幸運方位 | 東南方 |
| lucky_item | 幸運物品 | 紅色錢包 |

### 優缺點
- **優點**: 結合傳統命理與現代 AI，提供個人化預測和購買建議
- **缺點**: 依賴外部 Gemini CLI，回應時間較長

---

## Astrology-Zodiac 西洋星座預測

### 原理
根據使用者的出生日期判斷西洋星座，透過 Gemini AI 分析星座運勢與幸運數字，推算適合的彩券號碼。

### 星座判斷
```python
def get_zodiac_sign(month, day):
    zodiac_dates = [
        (1, 20, "摩羯座"), (2, 19, "水瓶座"), (3, 21, "雙魚座"),
        (4, 20, "牡羊座"), (5, 21, "金牛座"), (6, 21, "雙子座"),
        (7, 23, "巨蟹座"), (8, 23, "獅子座"), (9, 23, "處女座"),
        (10, 23, "天秤座"), (11, 22, "天蠍座"), (12, 22, "射手座"),
        (12, 31, "摩羯座")
    ]
    for end_month, end_day, sign in zodiac_dates:
        if month < end_month or (month == end_month and day <= end_day):
            return sign
    return "摩羯座"
```

### Gemini CLI 整合
```python
def call_gemini_zodiac(profile, lottery_type='big'):
    max_num = 49 if lottery_type == 'big' else 38
    zodiac = get_zodiac_sign(profile['birth_month'], profile['birth_day'])

    prompt = f'''
    你是一位西洋占星術專家。請根據以下星座資料分析運勢，
    並推薦最適合購買彩券的號碼。

    姓名: {profile['name']}
    星座: {zodiac}
    出生日期: {profile['birth_year']}/{profile['birth_month']}/{profile['birth_day']}

    請推薦 6 個主要號碼 (1-{max_num}) 和 1 個特別號 (1-{max_num})。

    請只回傳 JSON 格式，不要有其他文字:
    {{"numbers": [1,2,3,4,5,6], "special": 7, "zodiac": "{zodiac}", "lucky_elements": "幸運元素"}}
    '''

    result = subprocess.run(
        ['gemini', prompt],
        capture_output=True,
        text=True,
        timeout=60
    )
    return json.loads(result.stdout)
```

### 優缺點
- **優點**: 簡單直觀，根據星座提供個人化預測
- **缺點**: 星座判斷較為通用，個人化程度低於紫微斗數

---

## 命理預測 Ensemble 整合

### 整合方式
命理預測結果可加入 Ensemble 加權投票系統：

```python
DEFAULT_WEIGHTS = {
    # ... 其他演算法權重 ...
    "Astrology-Ziwei": 0.8,   # 紫微斗數
    "Astrology-Zodiac": 0.7,  # 西洋星座
}
```

### 多人預測合併
當有多人的生辰資料時，系統會：
1. 為每個人分別計算紫微/星座預測
2. 合併所有人的推薦號碼
3. 以出現頻率決定最終推薦
4. 產生「號碼組合說明」解釋每個號碼的來源

```python
def merge_astrology_predictions(profiles, lottery_type='big'):
    all_numbers = []
    all_specials = []

    for profile in profiles:
        ziwei = call_gemini_ziwei(profile, lottery_type)
        zodiac = call_gemini_zodiac(profile, lottery_type)

        all_numbers.extend(ziwei['numbers'])
        all_numbers.extend(zodiac['numbers'])
        all_specials.extend([ziwei['special'], zodiac['special']])

    # 頻率投票
    from collections import Counter
    top_6 = [n for n, _ in Counter(all_numbers).most_common(6)]
    special = Counter(all_specials).most_common(1)[0][0]

    return top_6, special
```

### 號碼組合說明
系統會為最終號碼產生組合說明，解釋每個號碼如何被選中：

```json
{
  "method": "頻率投票法",
  "description": "綜合 4 位家人的命理分析，選出最多人推薦的號碼",
  "numbers": [
    {"number": 7, "count": 4, "reason": "全員推薦"},
    {"number": 12, "count": 3, "reason": "3/4 人推薦"},
    {"number": 23, "count": 2, "reason": "2/4 人推薦"}
  ],
  "special": {"number": 9, "count": 2, "reason": "2/4 人推薦"}
}
```

---

## 回測系統 (Backtesting)

### 原理
回測系統用於評估各預測演算法的歷史表現，透過對歷史資料進行預測並與實際結果比較，計算命中率等指標。

### 檔案位置
`predict/backtest.py`

### 主要功能

#### 1. 完整回測報告 (run_full_backtest)
```python
def run_full_backtest(lottery_type: str = 'big', periods: int = 50) -> Dict:
    """
    對所有算法執行回測分析
    
    回傳:
    - ranking: 算法排名（依平均命中數）
    - algorithms: 各算法詳細統計
      - average_hits: 平均命中數
      - max_hits / min_hits: 最高/最低命中
      - special_hit_rate: 特別號命中率
      - hit_distribution: 命中分布
    """
```

#### 2. 號碼分布分析 (analyze_number_distribution)
```python
def analyze_number_distribution(df, periods: int = 100) -> Dict:
    """
    分析歷史號碼分布特徵
    
    回傳:
    - odd_even_ratio: 奇偶比
    - high_low_ratio: 高低比
    - hot_numbers: 熱門號碼 Top 10
    - cold_numbers: 冷門號碼 Top 10
    - sum_average/min/max: 總和統計
    """
```

#### 3. 滾動回測 (rolling_backtest)
```python
def rolling_backtest(lottery_type: str = 'big', 
                     window_size: int = 20,
                     total_periods: int = 100) -> Dict:
    """
    滾動視窗回測，分析算法表現一致性
    
    回傳:
    - rolling_results: 各時間視窗的表現數據
    - algorithm_summary: 整體統計（平均、波動、最佳/最差視窗）
    """
```

#### 4. 參數優化 (optimize_window_size)
```python
def optimize_window_size(lottery_type: str = 'big',
                         min_window: int = 20,
                         max_window: int = 100,
                         step: int = 10) -> Dict:
    """
    尋找 Hot/Cold 算法的最佳視窗大小
    
    回傳:
    - optimal: 最佳 hot_window 和 cold_window
    - results: 各視窗大小的測試結果
    """
```

---

## 算法參數設定系統

### 檔案位置
- 設定模組: `predict/config.py`
- 設定檔案: `predict/algorithm_config.json`

### 可調整參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| hot_window | 50 | Hot-50 分析期數 (10-200) |
| cold_window | 50 | Cold-50 分析期數 (10-200) |
| ensemble_weights | {...} | 各算法 Ensemble 權重 (0-5) |
| backtest_periods | 50 | 回測預設期數 |

### Ensemble 預設權重
```python
DEFAULT_WEIGHTS = {
    "Hot-50": 1.0,
    "Cold-50": 0.8,
    "RandomForest": 1.5,
    "GradientBoosting": 1.0,
    "KNN": 1.2,
    "XGBoost": 1.3,
    "LSTM": 1.0,
    "LSTM-RF": 1.2,
    "Markov": 1.0,
    "Pattern": 0.9,
    "Astrology-Ziwei": 0.8,
    "Astrology-Zodiac": 0.7
}
```

### API 功能
- `GET /config/algorithm`: 取得目前設定
- `POST /config/algorithm`: 更新設定
- `POST /config/algorithm/reset`: 重設為預設值
- `POST /config/algorithm/auto-tune`: 根據回測結果自動調整權重

### 自動調權重機制
```python
def update_weights_from_backtest(backtest_results: Dict):
    """
    根據回測結果自動調整 Ensemble 權重
    
    權重計算公式:
    new_weight = base_weight * (1 + (avg_hits - baseline) * scale_factor)
    
    - baseline: 平均命中數的基準線 (1.0)
    - scale_factor: 調整幅度因子 (0.5)
    """
```
