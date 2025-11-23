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

純文字
```