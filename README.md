# qwen3 server in docker container
### build
```sh 
docker build -f docker/Dockerfile -t qwen3vl:latest .
```
### run
```sh
docker run --gpus all --ipc=host --rm --name qwen3vl -it \
  -v /home/ct/qwen3-vl:/workspace \
  -p 2333:2333 \
  qwen3vl:latest bash
```
### 進docker後
```sh
uvicorn app:app --host 0.0.0.0 --port 2333
```

# Client call example
[client.py](client.py)
```py
import requests

url = "http://localhost:2333/chat-vl"

files = {
    "file": open("test2.mp4", "rb"),
}
data = {
    "media_type": "video",
    "prompt": "Describe this video.",
    "max_new_tokens": "128",
}

resp = requests.post(url, data=data, files=files)
print(resp.status_code, resp.text)

if resp.ok:
    print("Model output:", resp.json()["text"])
```
