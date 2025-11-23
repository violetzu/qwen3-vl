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


# Client call example
| 欄位名稱             | 型態              | 說明                                        |
| ---------------- | --------------- | ----------------------------------------- |
| `file`           | UploadFile (可選) | 媒體檔案（video/image）。若 `media_type=text` 可省略 |
| `media_type`     | string          | `"video"`、`"image"` 或 `"text"`            |
| `prompt`         | string          | 給模型的問題或指令                                 |
| `max_new_tokens` | int             | 模型最大生成字數（預設：128）                          |

純文字
```sh
curl -X POST "http://localhost:2333/chat-vl" \
  -F "media_type=text" \
  -F "prompt=Explain quantum computing in simple terms." \
  -F "max_new_tokens=256"
```
影片
```sh
curl -X POST "http://localhost:2333/chat-vl" \
  -F "file=@video.mp4" \
  -F "media_type=video" \
  -F "prompt=Describe this video." \
  -F "max_new_tokens=256"
```
圖片
```sh
curl -X POST "http://localhost:2333/chat-vl" \
  -F "file=@demo.jpg" \
  -F "media_type=image" \
  -F "prompt=Describe this image." \
  -F "max_new_tokens=128"
```
python呼叫範例
```py
import requests

url = "http://localhost:2333/chat-vl"


# -------------------------------
# 若 media_type = "video" 或 "image"
# 需要提供 files
# 若 media_type = "text"
# 可以把 files 設為 None
# -------------------------------

files = {
    "file": open("test.mp4", "rb"),   # video / image 才需要
}

data = {
    "media_type": "video",            # "video" / "image" / "text"
    "prompt": "Describe this video.",
    "max_new_tokens": "256",
}

resp = requests.post(url, data=data, files=files)
files["file"].close()

print(resp.status_code, resp.text)

if resp.ok:
    print("Model output:", resp.json()["text"])
```


