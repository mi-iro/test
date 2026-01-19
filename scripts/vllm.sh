python -m vllm.entrypoints.openai.api_server \
  --served-model-name Qwen3-VL-4B-Instruct \
  --model /mnt/shared-storage-user/mineru2-shared/madongsheng/modelscope/Qwen3-VL-4B-Instruct/ \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.7