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