# Qwen3 Server in Docker
### 1. Build Image
```sh
cd qwen3-vl
docker build -f docker/Dockerfile -t qwen3vl:latest .
```

### 2. (Optional) Run Container
```sh
docker run -d --gpus all --ipc=host --name qwen3vl \
  -v /home/ct/qwen3-vl:/workspace \
  -p 2333:2333 \
  qwen3vl:latest
```

### 2. (Optional) If Use Cloudflare Tunnels
```sh
cd qwen3-vl/docker
docker compose up -d
```

### 3. (Optional) Enter the Container
If you want to check log:
```sh
docker compose logs -t -f --tail=200
docker logs -f -t --tail 200 qwen3vl
docker logs -f -t --tail 200 cftunnel
```
If you need a new shell inside the container:
```sh
docker exec -it qwen3vl bash
```


# Client post example
## API：POST /chat-vl
| 欄位               | 類型               | 說明                                        |
| ---------------- | ---------------- | ----------------------------------------- |
| `messages`       | `str`（JSON 字串）   | 與 Qwen3-VL 相同的 chat 格式                    |
| `file`           | `UploadFile`（可選） | 若 messages 中包含 image/video block，則需上傳對應檔案 |
| `max_new_tokens` | `int`（可選）        | 模型輸出 token 數量上限（預設 256）                   |

純文字
```py
import json
import requests

API_URL = "http://localhost:2333/chat-vl"

messages = [
    {
        "role": "user",
        "content": "你好，幫我介紹一下你自己。"
    }
]

resp = requests.post(
    API_URL,
    data={"messages": json.dumps(messages)},
)

print(resp.json())
```
圖片
```py
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
        headers=HEADERS,
        data={"messages": json.dumps(messages)},
        files={"file": ("test.jpg"")}
    )

print(resp.json())
```
影片
```py
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
        headers=HEADERS,
        data={"messages": json.dumps(messages)},
        files={"file": ("sample.mp4")}
    )

print(resp.json())
```


帶 max_new_tokens
```py
resp = requests.post(
    API_URL,
    headers=HEADERS,
    data={
        "messages": json.dumps(messages),
        "max_new_tokens": 100
    }
)
```





