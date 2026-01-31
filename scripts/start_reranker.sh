#!/bin/bash

# 设置 CUDA 设备（如果需要指定显卡）
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 设置端口
PORT=8000
HOST="0.0.0.0"

echo "Starting Qwen-VL Reranker Service on ${HOST}:${PORT}..."

# 使用 uvicorn 启动 qwen3_vl_reranker_server.py
# 假设 qwen3_vl_reranker_server.py 在当前目录下
python -m uvicorn qwen3_vl_reranker_server:app --host $HOST --port $PORT --workers 1

# 注意：由于 vLLM 的 LLM 类不是进程安全的，workers 必须设为 1。
# 如果需要更高并发，应修改 server.py 使用 AsyncLLMEngine。