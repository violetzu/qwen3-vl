# Qwen3-VL FastAPI Server

簡介：這個 repo 提供一個 FastAPI 服務，使用 Qwen3-VL-8B-Instruct 模型進行視覺語言推論。支援文本、影像與影片輸入，可在本地或 Docker 容器中運行。

## 核心特性

- **統一端點**：單一 `POST /chat-vl` 支援同步與串流兩種模式。
- **模組化設計**：訊息解析、媒體驗證、檔案處理等分離成 helper 函式，便於維護與測試。
- **API 金鑰驗證**：所有非根路徑的請求需提供 `X-API-Key` 標頭。
- **彈性輸出**：使用 `stream=true` 參數切換至串流模式，逐漸回傳生成結果。

## 需求

- Python 相容版本（參見 `requirements.txt`）
- GPU 強烈建議（用於加速推論）
- 足夠的磁碟空間存放模型檔案（`models/` 目錄）

## 環境變數設定

在專案根目錄建立 `.env` 檔案：

```
QPIKEY=your_secret_api_key_here
```

伺服器啟動時會自動讀取此金鑰。若未設定，啟動會失敗。

## 本地運行

1. 安裝相依套件：

```bash
pip install -r requirements.txt
```

2. 啟動 FastAPI（使用 uvicorn）：

```bash
uvicorn app:app --host 0.0.0.0 --port 2333
```

啟動後存取 `http://localhost:2333/` 會看到端點資訊。

## Docker 運行

### 建置映像

```bash
cd /home/ct/qwen3-vl
docker build -f docker/Dockerfile -t qwen3vl:latest .
```

### 運行容器

```bash
docker run -d --gpus all --ipc=host --name qwen3vl \
  -v /home/ct/qwen3-vl:/workspace \
  -p 2333:2333 \
  qwen3vl:latest
```

### 使用 Docker Compose

```bash
cd docker
docker compose up -d
```

### 查看日誌與進入容器

```bash
# 查看日誌
docker logs -f -t --tail 200 qwen3vl

# 進入容器
docker exec -it qwen3vl bash
```

## API 使用說明

### 統一端點：POST /chat-vl

#### 參數說明

| 欄位 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `messages` | `str`（JSON） | 必填 | 與 Qwen3-VL 相同的 chat 格式 |
| `file` | `UploadFile` | 可選 | 若 messages 含 image/video block 時需上傳 |
| `max_new_tokens` | `int` | 256 | 模型輸出 token 上限 |
| `stream` | `bool` | false | 若為 true 開啟串流模式 |

#### 範例 1：純文本（同步）

```python
import json
import requests

API_URL = "http://localhost:2333/chat-vl"
API_KEY = "your_secret_api_key_here"
headers = {"X-API-Key": API_KEY}

messages = [
    {"role": "user", "content": "你好，幫我介紹一下你自己。"}
]

resp = requests.post(
    API_URL,
    headers=headers,
    data={"messages": json.dumps(messages)},
)

print(resp.json())
```

#### 範例 2：上傳圖片（同步）

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "這張圖片裡有什麼？"}
        ]
    }
]

with open("test.jpg", "rb") as f:
    resp = requests.post(
        API_URL,
        headers=headers,
        data={"messages": json.dumps(messages)},
        files={"file": ("test.jpg", f, "image/jpeg")}
    )

print(resp.json())
```

#### 範例 3：上傳影片（同步）

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": "請描述這段影片的內容。"}
        ]
    }
]

with open("sample.mp4", "rb") as f:
    resp = requests.post(
        API_URL,
        headers=headers,
        data={"messages": json.dumps(messages)},
        files={"file": ("sample.mp4", f, "video/mp4")}
    )

print(resp.json())
```

#### 範例 4：純文本（串流模式）

```python
# 使用 stream=true 啟用串流
with requests.post(
    API_URL,
    headers=headers,
    data={
        "messages": json.dumps(messages),
        "stream": "true"  # 啟用串流
    },
    stream=True
) as r:
    r.raise_for_status()
    for chunk in r.iter_content(decode_unicode=True):
        if chunk:
            print(chunk, end="", flush=True)
```

#### 範例 5：上傳圖片 + 串流 + 自訂 token 上限

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "詳細分析這張圖片。"}
        ]
    }
]

with open("test.jpg", "rb") as f:
    with requests.post(
        API_URL,
        headers=headers,
        data={
            "messages": json.dumps(messages),
            "max_new_tokens": 512,
            "stream": "true"
        },
        files={"file": ("test.jpg", f)},
        stream=True
    ) as r:
        r.raise_for_status()
        for chunk in r.iter_content(decode_unicode=True):
            if chunk:
                print(chunk, end="", flush=True)
```

## 程式碼架構

### Helper 函式

- `parse_messages(messages_str)`：解析 JSON 訊息字串，失敗時拋出 `HTTPException(400)`。
- `analyze_media_requirements(msgs)`：檢查訊息中是否需要 image/video，回傳 tuple `(need_video, need_image)`。
- `save_upload_tempfile(file)`：非同步函式，將上傳的檔案存至暫存位置並回傳路徑。
- `build_media_paths(temp_path, need_video, need_image)`：根據需求構建 video/image 路徑。
- `_build_inputs(messages, video_path, image_path)`：共用的 processor 輸入構建邏輯。

### 推論函式

- `run_qwen_vl_blocking(...)`：同步推論，阻塞式等待並回傳完整結果。
- `stream_qwen_vl(...)`：串流推論，使用 Thread + TextIteratorStreamer 逐步回傳文本片段。

### Middleware

- `verify_api_key`：驗證 `X-API-Key` 標頭，根路徑 `/` 不驗證。

## 常見問題 (FAQ)

**Q：如何快速測試 API？**  
A：使用上面提供的 Python 範例，或使用 curl：
```bash
curl -X POST http://localhost:2333/chat-vl \
  -H "X-API-Key: your_key" \
  -F "messages={\"role\":\"user\",\"content\":\"Hello\"}"
```

**Q：串流模式為什麼沒有輸出？**  
A：確保：
- `stream=true` 被正確設定（as string in form data）
- 在 requests 呼叫時設定 `stream=True`
- 使用 `iter_content()` 而非 `json()`

**Q：模型下載位置在哪？**  
A：預設在 `models/` 目錄（由 `MODEL_CACHE_DIR` 控制），首次執行時會自動下載。

**Q：如何增加輸出長度？**  
A：提高 `max_new_tokens` 參數值，例如設為 512 或 1024。

## 疑難排解

- **QPIKEY 未設定**：檢查 `.env` 檔案是否存在且正確。
- **GPU 記憶體不足**：確保 `device_map="auto"` 正常運作，或降低 `max_new_tokens`。
- **暫存檔未清理**：程式會在推論完成後自動刪除暫存檔，不需手動處理。

---

更多資訊請參考 `app.py` 的原始碼註解。





