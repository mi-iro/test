#判别器judge
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/shared-storage-user/mineru2-shared/madongsheng/saves/Qwen3-VL-8B-Instruct/full/demond_0203_f_base/checkpoint-400 \
  --served-model-name sft_ckpt400 \
  --port 8003 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.7 \
  --max-num-batched-tokens 32768 \
  --mm-processor-cache-gb 0 \
  --compilation_config.cudagraph_mode PIECEWISE

#提取器extractor
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/shared-storage-user/mineru2-shared/madongsheng/saves/Qwen3-VL-8B-Instruct/full/GRPO_agent_0201/v2-20260201-080521/checkpoint-1400 \
  --served-model-name rl_ckpt1400 \
  --port 8001 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.7 \
  --max-num-batched-tokens 32768 \
  --mm-processor-cache-gb 0 \
  --compilation_config.cudagraph_mode PIECEWISE