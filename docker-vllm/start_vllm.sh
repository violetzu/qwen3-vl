#!/usr/bin/env bash
set -e

echo "Starting vLLM Qwen3-VL-8B-Instruct server..."

vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --async-scheduling \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --max-model-len 64000 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port 2333
