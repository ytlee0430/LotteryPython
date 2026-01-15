# 部署指南

## 部署選項

| 方式 | 適用場景 | 複雜度 |
|------|----------|--------|
| 本地開發 | 開發測試 | 低 |
| Docker | 單機部署 | 中 |
| Docker Compose | 完整環境 | 中 |

---

## 前置需求

### 軟體需求
- Python 3.9+
- pip 或 conda
- Docker（選用）
- Git

### 外部服務
- Google Cloud Service Account
- `credentials.json` 檔案

---

## 本地開發環境

### 1. 取得原始碼
```bash
git clone https://github.com/ytlee0430/LotteryPython.git
cd LotteryPython
```

### 2. 建立虛擬環境
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
.\venv\Scripts\activate  # Windows
```

### 3. 安裝依賴
```bash
pip install -r requirements.txt
```

### 4. 設定 Google 認證
```bash
# 將 credentials.json 放置於專案根目錄
cp /path/to/your/credentials.json ./credentials.json
```

### 5. 啟動開發伺服器
```bash
python app.py
# 伺服器運行於 http://localhost:3000
```

---

## Docker 部署

### Dockerfile 說明

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y build-essential

# 安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# 複製應用程式碼
COPY . .

EXPOSE 3000

# 使用 Gunicorn 啟動
CMD ["gunicorn", "--bind", "0.0.0.0:3000", "app:app"]
```

### 建構映像檔
```bash
docker build -t lotterypython:latest .
```

### 執行容器
```bash
docker run -d \
  --name lottery \
  -p 3000:3000 \
  -v $(pwd)/credentials.json:/app/credentials.json:ro \
  lotterypython:latest
```

---

## Docker Compose 完整部署

### docker-compose.yml

```yaml
version: '3.8'

services:
  web:
    build: .
    container_name: lottery-web
    volumes:
      - ./credentials.json:/app/credentials.json:ro
    expose:
      - "3000"
    restart: always

  nginx:
    image: nginx:alpine
    container_name: lottery-nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - web
    restart: always
```

### nginx.conf

```nginx
events {
    worker_connections 1024;
}

http {
    upstream flask {
        server web:3000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://flask;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }
}
```

### 啟動服務
```bash
# 建構並啟動
docker-compose up -d --build

# 查看日誌
docker-compose logs -f

# 停止服務
docker-compose down
```

---

## 排程設定

### macOS launchd

建立 plist 檔案 `~/Library/LaunchAgents/com.lottery.schedule.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.lottery.schedule</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/LotteryPython/scripts/run_lottery_schedule.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>22</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/tmp/lottery.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/lottery.error.log</string>
</dict>
</plist>
```

載入排程：
```bash
launchctl load ~/Library/LaunchAgents/com.lottery.schedule.plist
```

### Linux cron

```bash
# 編輯 crontab
crontab -e

# 新增排程（每天晚上 10 點執行）
0 22 * * * /path/to/LotteryPython/scripts/run_lottery_schedule.sh >> ~/lottery.log 2>&1
```

### 排程腳本說明

`scripts/run_lottery_schedule.sh`:
```bash
#!/bin/bash
cd /path/to/LotteryPython
source venv/bin/activate

DAY=$(date +%u)

case $DAY in
    1|4)  # 週一、週四：威力彩
        python -m lotterypython --update --type super
        python -m lotterypython --type super --save-results
        ;;
    2|5)  # 週二、週五：大樂透
        python -m lotterypython --update --type big
        python -m lotterypython --type big --save-results
        ;;
esac
```

---

## Google Cloud 設定

### 1. 建立服務帳戶

1. 前往 [Google Cloud Console](https://console.cloud.google.com/)
2. 建立新專案或選擇現有專案
3. 啟用 Google Sheets API
4. 建立服務帳戶
5. 下載 JSON 金鑰檔案

### 2. 設定 Sheets 權限

1. 開啟目標 Google Sheets
2. 點擊「共用」
3. 將服務帳戶 email 加入編輯者

### 3. 設定認證檔案

```bash
# 方式 1: 直接放置
cp credentials.json /path/to/LotteryPython/

# 方式 2: 環境變數
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

---

## 環境變數

| 變數 | 說明 | 預設值 |
|------|------|--------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Google 認證檔案路徑 | `./credentials.json` |
| `PYTHON_BIN` | Python 執行檔路徑 | `python` |
| `FLASK_ENV` | Flask 環境 | `production` |
| `FLASK_DEBUG` | 除錯模式 | `0` |

---

## 健康檢查

### 檢查 Web 服務
```bash
curl http://localhost:3000/
```

### 檢查 Docker 容器
```bash
docker ps
docker logs lottery-web
```

### 檢查 Google Sheets 連線
```bash
python -c "
import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://spreadsheets.google.com/feeds']
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)
print('Google Sheets 連線成功')
"
```

---

## 常見問題

### Q: Docker 建構失敗
```bash
# 清除快取重建
docker build --no-cache -t lotterypython:latest .
```

### Q: Google Sheets 認證失敗
- 確認 `credentials.json` 存在且路徑正確
- 確認服務帳戶已被授權存取目標 Sheets
- 確認 Google Sheets API 已啟用

### Q: 爬蟲被封鎖
- lot539.com 可能有反爬蟲機制
- cloudscraper 通常能處理，但可能需要等待後重試
- 避免過於頻繁的請求

### Q: TensorFlow 警告
```bash
# 忽略 GPU 警告（如果只使用 CPU）
export TF_CPP_MIN_LOG_LEVEL=2
```
