#!/bin/bash

# 1. 设置 CUDA 设备：列出所有要参与并行的显卡 ID
export CUDA_VISIBLE_DEVICES=2,3

# 2. 设置 vLLM 的张量并行度 (必须与可见显卡数量一致)
export TENSOR_PARALLEL_SIZE=2

# 解决某些环境下的多进程启动问题
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 设置端口
PORT=8000
HOST="0.0.0.0"

echo "Starting Qwen-VL Reranker Service on ${HOST}:${PORT} with TP=${TENSOR_PARALLEL_SIZE}..."

# 启动命令保持不变，workers 仍然是 1
# 因为 vLLM 内部会管理多卡进程
python -m uvicorn qwen3_vl_reranker_server:app --host $HOST --port $PORT --workers 1