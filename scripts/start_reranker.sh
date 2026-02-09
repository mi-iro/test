#!/bin/bash

# 1. 设置 CUDA 设备：列出所有要参与并行的显卡 ID
# 例如使用 4 张卡：export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 2. 设置 vLLM 的张量并行度
# 在数据并行(DP)模式下，我们希望每个进程跑一个完整的模型，所以 TP 通常设为 1
export TENSOR_PARALLEL_SIZE=1

# 解决某些环境下的多进程启动问题
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 设置端口
PORT=8000
HOST="0.0.0.0"

# 计算可见 GPU 的数量，用于设置 workers 数量
# 逻辑：统计逗号分隔符的数量 + 1
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # 如果没设置，尝试使用 nvidia-smi 获取数量，或者默认 1
    WORKERS=1
else
    # 替换逗号为空格，然后计算词数
    WORKERS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)
fi

echo "Cleaning up previous locks..."
rm -f /tmp/vllm_reranker_gpu.lock
rm -f /tmp/vllm_reranker_gpu.state

echo "Starting Qwen-VL Reranker Service on ${HOST}:${PORT}..."
echo "Config: Workers(DP)=${WORKERS}, TP per Worker=${TENSOR_PARALLEL_SIZE}, Devices=${CUDA_VISIBLE_DEVICES}"

# 启动命令
# 设置 --workers 等于 GPU 数量
python -m uvicorn qwen3_vl_reranker_server:app --host $HOST --port $PORT --workers $WORKERS