import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
from jinja2 import Template
from vllm import LLM
from vllm.multimodal.utils import fetch_image

# ==========================================
# 初始化 FastAPI 应用
# ==========================================
app = FastAPI(title="Qwen-VL Reranker Service")

# 全局模型变量
llm_engine = None
TEMPLATE_PATH = "/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/scripts/qwen3_vl_reranker.jinja"

# ==========================================
# 数据模型定义 (用于 HTTP 请求体验证)
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
# 核心逻辑函数 (复用原代码逻辑)
# ==========================================
def parse_input_dict(input_data: Dict[str, Any]):
    """解析输入字典，提取图片和文本"""
    image = input_data.get('image')
    text = input_data.get('text')

    mm_data = {'image': []}
    content = ''
    
    if image:
        content += '<|vision_start|><|image_pad|><|vision_end|>'
        # 处理图片加载逻辑
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
            # 假设已经是对象（但在 HTTP 请求中通常是路径或URL）
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
# API 路由
# ==========================================
@app.post("/rerank")
async def rerank(request: RerankRequest):
    global llm_engine
    if llm_engine is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    # 1. 准备模板
    if not os.path.exists(TEMPLATE_PATH):
        template_str = "{{ instruction }} Query: {{ query_content }} Document: {{ doc_content }}"
        chat_template = Template(template_str).render(instruction=request.instruction)
    else:
        chat_template = Template(Path(TEMPLATE_PATH).read_text()).render(instruction=request.instruction)

    # 2. 构造 Prompts
    prompts = []
    # 将 Pydantic 模型转换为 Dict 以适配原有函数
    query_dict = request.query.model_dump()
    
    for doc in request.documents:
        doc_dict = doc.model_dump()
        prompt_data = format_vllm_input(query_dict, doc_dict, chat_template)
        prompts.append(prompt_data)

    # 3. 执行推理
    # 注意：在 API 服务中，classify 调用是同步阻塞的。
    # 生产环境通常使用 AsyncLLMEngine，但为了保持与原代码逻辑一致且简单，这里使用 LLM 类。
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
    print("Initializing vLLM Engine...")
    
    try:
        tp_size = int(os.getenv("TENSOR_PARALLEL_SIZE", 1))
        llm_engine = LLM(
            model='/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Reranker-8B',
            # model='/mnt/shared-storage-user/mineru2-shared/madongsheng/modelscope/Qwen/Qwen3-VL-Reranker-2B',
            runner='pooling',
            dtype='bfloat16',
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            hf_overrides={
                "architectures": ["Qwen3VLForSequenceClassification"],
                "classifier_from_token": ["no", "yes"],
                "is_original_qwen3_reranker": True,
            },
            # 限制显存使用，防止 OOM (可选)
            gpu_memory_utilization=0.7,
            # 如果是多卡，建议指定分布式后端，通常 'ray' 或 'mp'
            distributed_executor_backend="ray",
        )
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        raise e

if __name__ == "__main__":
    # 开发调试用
    uvicorn.run(app, host="0.0.0.0", port=8003)