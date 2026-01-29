import os
from pathlib import Path
from typing import Dict, Any
from jinja2 import Template
from vllm import LLM
from vllm.multimodal.utils import fetch_image

# ==========================================
# 函数定义 (保持在全局作用域，供子进程调用)
# ==========================================

def parse_input_dict(input_dict: Dict[str, Any]):
    """
    Parse input dictionary to extract image and text content.
    Returns the formatted content string and multimodal data.
    """
    image = input_dict.get('image')
    text = input_dict.get('text')

    mm_data = {
        'image': []
    }
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

def format_vllm_input(
    query_dict: Dict[str, Any],
    doc_dict: Dict[str, Any],
    chat_template: str
):
    """
    Format query and document into vLLM input format.
    Combines multimodal data from both query and document.
    """
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

def get_rank_scores(
    llm,
    inputs: Dict[str, Any],
    default_instruction: str = "Given a search query, retrieve relevant candidates that answer the query.",
    template_path: str = "/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/Qwen3-VL-Embedding/examples/reranker_template.jinja"
):
    """
    Generate relevance scores for documents given a query.
    Returns a list of scores for each document.
    """
    query_dict = inputs['query']
    doc_dicts = inputs['documents']
    instruction = inputs.get('instruction') or default_instruction

    # 注意：确保 reranker_template.jinja 文件在当前目录下，或者提供绝对路径
    if not os.path.exists(template_path):
        # 如果找不到文件，提供一个简单的默认模板作为后备，防止报错
        print(f"Warning: Template file {template_path} not found. Using default string.")
        template_str = "{{ instruction }} Query: {{ query_content }} Document: {{ doc_content }}"
        chat_template = Template(template_str)
        chat_template = chat_template.render(instruction=instruction)
    else:
        chat_template = Template(Path(template_path).read_text())
        chat_template = chat_template.render(instruction=instruction)

    prompts = []

    for doc_dict in doc_dicts:
        prompt = format_vllm_input(
            query_dict, doc_dict, chat_template
        )
        prompts.append(prompt)

    outputs = llm.classify(
        prompts=prompts
    )
    scores = [ output.outputs.probs[0] for output in outputs ]
    return scores

# ==========================================
# 主执行入口 (必须放在 if __name__ == "__main__": 下)
# ==========================================

if __name__ == "__main__":
    # 1. 初始化模型
    # 注意：在 spawn 模式下，模型初始化必须在主进程保护块内
    llm = LLM(
        # model='/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Reranker-8B',
        model='/mnt/shared-storage-user/mineru2-shared/madongsheng/modelscope/Qwen/Qwen3-VL-Reranker-2B',
        runner='pooling',
        dtype='bfloat16',
        trust_remote_code=True,
        hf_overrides={
            "architectures": ["Qwen3VLForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    )

    print("Model initialized successfully!")

    # 2. 准备数据
    inputs = {
        "instruction": "Retrieve images or text relevant to the user's query.",
        "query": {
            "text": "A woman playing with her dog on a beach at sunset."
        },
        "documents": [
            {
                "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."
            },
            {
                "image": "/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/Qwen3-VL-Embedding/demo.jpeg"
            },
            {
                "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
                "image": "/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/Qwen3-VL-Embedding/demo.jpeg"
            }
        ]
    }

    print(f"Prepared query with {len(inputs['documents'])} candidate documents")

    # 3. 执行推理
    scores = get_rank_scores(llm, inputs)

    # 4. 打印结果
    print("Relevance Scores:")
    for i, score in enumerate(scores):
        print(f"Document {i+1}: {score:.4f}")