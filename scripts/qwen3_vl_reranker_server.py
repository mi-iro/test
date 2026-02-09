import os
import sys
import uvicorn
import fcntl  # 用于文件锁
import json
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
from jinja2 import Template
# 移除顶层的 vLLM 导入，避免过早初始化 CUDA
# from vllm import LLM
# from vllm.multimodal.utils import fetch_image

# ==========================================
# 初始化 FastAPI 应用
# ==========================================
app = FastAPI(title="Qwen-VL Reranker Service")

# 全局模型变量
llm_engine = None
TEMPLATE_PATH = "/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/scripts/qwen3_vl_reranker.jinja"

# ==========================================
# 数据模型定义
# ==========================================
class Query(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None

class Document(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None

class RerankRequest(BaseModel):
    instruction: Optional[str] = "Given a search query, retrieve relevant candidates that answer the query."
    query: Query
    documents: List[Document]

# ==========================================
# 核心逻辑函数
# ==========================================
def parse_input_dict(input_data: Dict[str, Any]):
    # 延迟导入 fetch_image
    from vllm.multimodal.utils import fetch_image
    
    """解析输入字典，提取图片和文本"""
    image = input_data.get('image')
    text = input_data.get('text')

    mm_data = {'image': []}
    content = ''
    
    if image:
        content += '<|vision_start|><|image_pad|><|vision_end|>'
        if isinstance(image, str):
            if image.startswith(('http://', 'https://')):
                try:
                    image_obj = fetch_image(image)
                    mm_data['image'].append(image_obj)
                except Exception as e:
                    print(f"Warning: Failed to fetch image {image}: {e}")
            else:
                abs_image_path = os.path.abspath(image)
                if os.path.exists(abs_image_path):
                    from PIL import Image
                    image_obj = Image.open(abs_image_path)
                    mm_data['image'].append(image_obj)
                else:
                    print(f"Warning: Image file not found: {abs_image_path}")
        else:
            mm_data['image'].append(image)
    
    if text:
        content += text
    
    return content, mm_data

def format_vllm_input(query_dict: Dict, doc_dict: Dict, chat_template: str):
    """格式化为 vLLM 输入"""
    query_content, query_mm_data = parse_input_dict(query_dict)
    doc_content, doc_mm_data = parse_input_dict(doc_dict)

    mm_data = { 'image': [] }
    mm_data['image'].extend(query_mm_data['image'])
    mm_data['image'].extend(doc_mm_data['image'])

    prompt = Template(chat_template).render(
        query_content=query_content,
        doc_content=doc_content,
    )
    return {
        'prompt': prompt,
        'multi_modal_data': mm_data
    }

# ==========================================
# 辅助函数：GPU 分配
# ==========================================
def allocate_gpu_for_worker():
    """
    在多进程模式下，为当前进程分配一个独立的 GPU。
    """
    # 获取原始的可见设备列表
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_devices:
        # 如果未设置，假设所有 GPU 可用，这里简单处理，实际建议在 shell 中显式设置
        import torch
        num_gpus = torch.cuda.device_count()
        device_ids = [str(i) for i in range(num_gpus)]
    else:
        device_ids = visible_devices.split(',')
    
    num_devices = len(device_ids)
    if num_devices <= 1:
        return # 只有一个设备或没有设备，无需分配

    lock_file_path = os.path.join(tempfile.gettempdir(), "vllm_reranker_gpu.lock")
    state_file_path = os.path.join(tempfile.gettempdir(), "vllm_reranker_gpu.state")

    assigned_id = None

    with open(lock_file_path, "w") as lock_file:
        # 获取互斥锁
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            # 读取当前状态
            current_index = 0
            if os.path.exists(state_file_path):
                with open(state_file_path, "r") as f:
                    try:
                        content = f.read().strip()
                        if content:
                            current_index = int(content)
                    except ValueError:
                        pass
            
            # 分配当前索引对应的 GPU
            if current_index >= num_devices:
                # 循环分配（防止 worker 重启导致越界）
                current_index = 0
            
            assigned_id = device_ids[current_index]
            print(f"[Worker PID {os.getpid()}] Assigned Physical GPU ID: {assigned_id} (Index: {current_index})")
            
            # 更新状态
            with open(state_file_path, "w") as f:
                f.write(str(current_index + 1))
                
        finally:
            # 释放锁
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    # 关键：重写环境变量，让当前进程只看得到被分配的那张卡
    # 这样 vLLM 初始化时会认为只有这一张卡，从而实现 DP
    if assigned_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = assigned_id

# ==========================================
# API 路由
# ==========================================
@app.post("/rerank")
async def rerank(request: RerankRequest):
    global llm_engine
    if llm_engine is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    if not os.path.exists(TEMPLATE_PATH):
        template_str = "{{ instruction }} Query: {{ query_content }} Document: {{ doc_content }}"
        chat_template = Template(template_str).render(instruction=request.instruction)
    else:
        chat_template = Template(Path(TEMPLATE_PATH).read_text()).render(instruction=request.instruction)

    prompts = []
    query_dict = request.query.model_dump()
    
    for doc in request.documents:
        doc_dict = doc.model_dump()
        prompt_data = format_vllm_input(query_dict, doc_dict, chat_template)
        prompts.append(prompt_data)

    try:
        outputs = llm_engine.classify(prompts=prompts)
        scores = [float(output.outputs.probs[0]) for output in outputs]
        return {"scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    global llm_engine
    
    # 1. 动态分配 GPU
    allocate_gpu_for_worker()
    
    # 2. 延迟导入 vLLM (必须在设置 CUDA_VISIBLE_DEVICES 之后)
    from vllm import LLM
    
    print(f"Initializing vLLM Engine on PID {os.getpid()} with Device: {os.environ.get('CUDA_VISIBLE_DEVICES')}...")
    
    try:
        # 即使是 DP 模式，每个 worker 内部 TP 仍然通常设为 1
        tp_size = int(os.getenv("TENSOR_PARALLEL_SIZE", 1))
        
        llm_engine = LLM(
            model='/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Reranker-8B',
            runner='pooling',
            dtype='bfloat16',
            trust_remote_code=True,
            tensor_parallel_size=tp_size, 
            hf_overrides={
                "architectures": ["Qwen3VLForSequenceClassification"],
                "classifier_from_token": ["no", "yes"],
                "is_original_qwen3_reranker": True,
            },
            gpu_memory_utilization=0.9,
        )
        print(f"Model initialized successfully on PID {os.getpid()}!")
    except Exception as e:
        print(f"Failed to initialize model on PID {os.getpid()}: {e}")
        raise e

if __name__ == "__main__":
    # 开发调试用
    uvicorn.run(app, host="0.0.0.0", port=8003)