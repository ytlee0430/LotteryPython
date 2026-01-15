# CLI 參考

## 概述

LotteryPython 提供命令列介面 (CLI) 進行資料更新與預測分析。

## 執行方式

### 方式 1: Python 模組（推薦）
```bash
python -m lotterypython [選項]
```

### 方式 2: 直接執行
```bash
python main.py [選項]
```

---

## 指令選項

| 選項 | 說明 |
|------|------|
| `--type TYPE` | 彩種類型：`big`（大樂透）或 `super`（威力彩）|
| `--update` | 更新資料（爬取最新開獎並儲存）|
| `--save-results` | 將預測結果儲存至 Google Sheets |

---

## 常用指令

### 更新大樂透資料
```bash
python -m lotterypython --update --type big
```
- 從 lot539.com 爬取最新大樂透開獎
- 更新 Google Sheets 與本地 CSV

### 更新威力彩資料
```bash
python -m lotterypython --update --type super
```
- 從 lot539.com 爬取最新威力彩開獎
- 更新 Google Sheets 與本地 CSV

### 執行大樂透預測
```bash
python -m lotterypython --type big
```
- 從 Google Sheets 載入歷史資料
- 執行所有預測演算法
- 在終端機顯示預測結果

### 執行威力彩預測
```bash
python -m lotterypython --type super
```

### 預測並儲存結果
```bash
python -m lotterypython --type big --save-results
```
- 執行預測
- 將結果寫入「分析結果」工作表

### 更新資料並預測儲存
```bash
python -m lotterypython --update --type big && \
python -m lotterypython --type big --save-results
```

---

## 輸出範例

### 預測輸出
```
=== 大樂透預測結果 ===
下一期: 113123

Hot-50:
  號碼: [3, 12, 24, 35, 41, 47]
  特別號: 28

RandomForest:
  號碼: [9, 24, 32, 40, 41, 49]
  特別號: 27

GradientBoosting:
  號碼: [5, 18, 22, 35, 42, 48]
  特別號: 15

KNN:
  號碼: [7, 14, 29, 33, 39, 45]
  特別號: 31

LSTM:
  號碼: [2, 11, 25, 36, 43, 49]
  特別號: 19

LSTM-RF:
  號碼: [8, 16, 27, 34, 40, 46]
  特別號: 22
```

### 更新輸出
```
正在更新大樂透資料...
取得現有資料: 1523 筆
爬取日期範圍: 2024-12-28 ~ 2024-12-31
新增 2 筆資料
更新完成！
```

---

## 結束代碼

| 代碼 | 說明 |
|------|------|
| 0 | 成功完成 |
| 1 | 一般錯誤 |
| 2 | 參數錯誤 |

---

## 進階用法

### 搭配 Shell Script
```bash
#!/bin/bash
# daily_lottery.sh

LOTTERY_DIR="/path/to/LotteryPython"
cd "$LOTTERY_DIR"
source venv/bin/activate

# 取得今天星期幾 (1=週一, 7=週日)
DOW=$(date +%u)

case $DOW in
    1|4)
        echo "執行威力彩更新與預測..."
        python -m lotterypython --update --type super
        python -m lotterypython --type super --save-results
        ;;
    2|5)
        echo "執行大樂透更新與預測..."
        python -m lotterypython --update --type big
        python -m lotterypython --type big --save-results
        ;;
    *)
        echo "今天沒有開獎"
        ;;
esac
```

### 搭配 Cron
```bash
# 每天晚上 10 點執行
0 22 * * * /path/to/daily_lottery.sh >> /var/log/lottery.log 2>&1
```

### 重導輸出
```bash
# 輸出到檔案
python -m lotterypython --type big > prediction.txt 2>&1

# 僅記錄錯誤
python -m lotterypython --update --type big 2>> error.log
```

---

## 環境變數

| 變數 | 說明 | 範例 |
|------|------|------|
| `PYTHON_BIN` | Python 執行檔路徑 | `/usr/bin/python3` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google 認證檔案 | `./credentials.json` |

---

## 故障排除

### 問題: ModuleNotFoundError
```bash
# 確認虛擬環境已啟動
source venv/bin/activate

# 重新安裝依賴
pip install -r requirements.txt
```

### 問題: Google 認證失敗
```bash
# 確認認證檔案存在
ls -la credentials.json

# 設定環境變數
export GOOGLE_APPLICATION_CREDENTIALS=./credentials.json
```

### 問題: 爬蟲失敗
```bash
# 測試網路連線
curl -I https://lot539.com

# 等待後重試（可能是暫時性阻擋）
sleep 60 && python -m lotterypython --update --type big
```

### 問題: TensorFlow 警告
```bash
# 設定日誌等級
export TF_CPP_MIN_LOG_LEVEL=2
python -m lotterypython --type big
```
