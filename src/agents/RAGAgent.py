import os
import sys
import json
import base64
import mimetypes
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI

# 确保可以导入 src 下的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.loaders.base_loader import BaseDataLoader, StandardSample, PageElement
from src.agents.utils import MinerUBboxExtractor

def local_image_to_data_url(path: str) -> str:
    """
    将本地图片转换为 Data URL (Base64)
    """
    if not path or not os.path.exists(path):
        return ""
    
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"

    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"Error encoding image {path}: {e}")
        return ""

# --- Standard RAG System Prompt ---
RAG_SYSTEM_PROMPT = """You are a helpful Document Assistant. 
Your task is to answer the user's question accurately based **ONLY** on the provided context (evidence).

### Instructions
1. Read the provided text and image evidence carefully.
2. If the answer is present in the evidence, answer strictly based on it.
3. If the provided evidence does not contain enough information to answer the question, state that you cannot find the answer.
4. Do NOT use any external knowledge not present in the context.

### Input Format
The user will provide:
- A set of evidence (images and text) retrieved from the documents.
- The question to answer.
"""

class RAGAgent:
    """
    Standard RAG Agent (Baseline).
    执行标准的 "Retrieve-then-Generate" 流程：
    1. 根据 Query 检索 Top-K 证据。
    2. 将所有证据拼接 Prompt。
    3. 调用 LLM 生成最终回答。
    包含缓存机制，支持断点续跑。
    """

    def __init__(
        self, 
        loader: BaseDataLoader, 
        base_url: str, 
        api_key: str, 
        model_name: str,
        top_k: int = 5,
        cache_dir: str = "./cache_results_rag",  # 新增缓存目录参数
        use_page: bool = False,
        use_crop: bool = True,
        use_ocr: bool = True,
        use_ocr_raw: bool = False,
        **kwargs
    ):
        """
        :param loader: 数据集加载器 (FinRAGLoader, MMLongLoader 等)，用于执行 pipeline 检索。
        :param base_url: LLM API 地址。
        :param api_key: LLM API Key。
        :param model_name: 模型名称。
        :param top_k: 单次检索的证据数量。
        :param cache_dir: 结果缓存目录。
        :param use_page: 是否在 Context 中包含整页图像。
        :param use_crop: 是否在 Context 中包含 BBox 截图。
        :param use_ocr: 是否在 Context 中包含 OCR 文本。
        :param use_ocr_raw: 是否使用 raw_content (from MinerUBboxExtractor) 替代 Agent 总结的 content。
        """
        self.loader = loader
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.top_k = top_k
        self.cache_dir = cache_dir
        
        self.use_page = use_page
        self.use_crop = use_crop
        self.use_ocr = use_ocr
        self.use_ocr_raw = use_ocr_raw
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def build_context_message(self, elements: List[PageElement]) -> List[Dict[str, Any]]:
        """
        将检索到的 PageElement 列表转换为多模态 Context 消息。
        根据 use_page, use_crop, use_ocr, use_ocr_raw 控制包含的内容。
        """
        content_list = []
        content_list.append({"type": "text", "text": "Here is the retrieved context/evidence from the documents:\n"})
        
        for i, el in enumerate(elements):
            content_list.append({"type": "text", "text": f"\n--- Evidence {i+1} ---\n"})
            
            # 1. Page Image (整页图像)
            if self.use_page:
                page_path = el.corpus_path
                if page_path and os.path.exists(page_path):
                    img_url = local_image_to_data_url(page_path)
                    if img_url:
                        content_list.append({"type": "text", "text": "Page Image:\n"})
                        content_list.append({"type": "image_url", "image_url": {"url": img_url}})

            # 2. Crop Image (证据截图)
            if self.use_crop:
                # 只有当 crop_path 存在时才展示，不再回退到 corpus_path (因为 use_page 独立控制)
                img_path = el.crop_path
                if img_path and os.path.exists(img_path):
                    img_url = local_image_to_data_url(img_path)
                    if img_url:
                        content_list.append({"type": "text", "text": "Region Crop:\n"})
                        content_list.append({"type": "image_url", "image_url": {"url": img_url}})
            
            # 3. Text Content (OCR / Summary)
            if self.use_ocr:
                text_content = ""
                bbox_extractor = MinerUBboxExtractor()
                # 如果请求 raw OCR 且元素具有 raw_content 属性
                if self.use_ocr_raw:
                    el.raw_content = bbox_extractor.extract_content_str(el.corpus_path, el.bbox)
                    text_content = el.raw_content
                else:
                    text_content = el.content
                
                if text_content:
                    content_list.append({"type": "text", "text": f"Text Content: {text_content}\n"})
        
        content_list.append({"type": "text", "text": "\n---------------------\n"})
        return content_list

    def process_sample(self, sample: StandardSample) -> StandardSample:
        """
        处理单个样本：Retrieve -> Generate
        优先读取缓存。
        """
        if sample.extra_info is None:
            sample.extra_info = {}

        cache_file = os.path.join(self.cache_dir, f"{sample.qid}.json")

        # --- 1. 尝试从缓存加载 ---
        if os.path.exists(cache_file):
            print(f"Loading cached result for Sample {sample.qid}...")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # 恢复核心字段
                sample.extra_info['final_answer'] = cached_data.get('final_answer', "")
                sample.extra_info['messages'] = cached_data.get('messages', [])
                
                if sample.extra_info['final_answer'] and "Error during" not in sample.extra_info['final_answer']:
                    # 恢复 retrieved_elements 对象列表
                    elements_dicts = cached_data.get('retrieved_elements', [])
                    restored_elements = []
                    for el_dict in elements_dicts:
                        # 过滤掉非 PageElement 字段以防止报错
                        valid_keys = PageElement.__annotations__.keys()
                        filtered_dict = {k: v for k, v in el_dict.items() if k in valid_keys}
                        
                        el_obj = PageElement(**filtered_dict)
                        restored_elements.append(el_obj)
                    
                    sample.extra_info['retrieved_elements'] = restored_elements
                    return sample
            except Exception as e:
                print(f"Error loading cache for {sample.qid}, rerunning inference. Error: {e}")

        # --- 2. 执行推理 (Retrival + Generation) ---
        print(f"Processing Sample {sample.qid} with Standard RAG...")

        # Step A: Retrieval
        try:
            image_inputs = [sample.data_source] if sample.data_source else []
            retrieved_elements = self.loader.pipeline(
                query=sample.query, 
                image_paths=image_inputs, 
                top_k=self.top_k
            )
            sample.extra_info['retrieved_elements'] = retrieved_elements
        except Exception as e:
            print(f"Error during retrieval for {sample.qid}: {e}")
            retrieved_elements = []

        # Step B: Generation
        messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
        
        user_content = []
        if retrieved_elements:
            context_content = self.build_context_message(retrieved_elements)
            user_content.extend(context_content)
        else:
            user_content.append({"type": "text", "text": "No relevant context found.\n"})

        user_content.append({"type": "text", "text": f"Question: {sample.query}"})
        messages.append({"role": "user", "content": user_content})

        final_answer = ""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0
            )
            final_answer = response.choices[0].message.content
        except Exception as e:
            print(f"LLM API Error: {e}")
            final_answer = "Error during generation."

        print(f"  [Standard RAG Answer]: {final_answer[:100]}...")

        # 记录结果
        sample.extra_info['final_answer'] = final_answer
        sample.extra_info['messages'] = messages[1:] 

        # --- 3. 写入缓存 ---
        try:
            # 序列化 retrieved_elements
            elements_to_save = []
            if 'retrieved_elements' in sample.extra_info:
                for el in sample.extra_info['retrieved_elements']:
                    if hasattr(el, 'to_dict'):
                        el_d = el.to_dict()
                        elements_to_save.append(el_d)
                    elif isinstance(el, dict):
                         elements_to_save.append(el)

            cache_data = {
                "qid": sample.qid,
                "query": sample.query,
                "gold_answer": sample.gold_answer,
                "final_answer": final_answer,
                "messages": sample.extra_info['messages'], # 注意：包含大量 base64 图片URL，文件可能会较大
                "retrieved_elements": elements_to_save
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            print(f"Saved result for Sample {sample.qid} to cache.")
            
        except Exception as e:
            print(f"Error saving cache for {sample.qid}: {e}")
        
        return sample

    def save_results(self, excel_path: str = "rag_results_summary.xlsx", json_path: str = "rag_results_summary.json"):
        """
        汇总所有样本的处理结果并保存。
        """
        if not self.loader.samples:
            print("No samples to save.")
            return

        data_rows = []
        for sample in self.loader.samples:
            final_ans = sample.extra_info.get('final_answer', "") if sample.extra_info else ""
            
            elements_to_save = []
            if sample.extra_info and 'retrieved_elements' in sample.extra_info:
                for el in sample.extra_info['retrieved_elements']:
                    if hasattr(el, 'to_dict'):
                        elements_to_save.append(el.to_dict())
                    elif isinstance(el, dict):
                         elements_to_save.append(el)

            row = {
                "QID": sample.qid,
                "Query": sample.query,
                "Gold Answer": sample.gold_answer,
                "Model Answer": final_ans,
                "Retrieved Elements": json.dumps(elements_to_save),
                "Data Source": sample.data_source
            }
            data_rows.append(row)

        # --- 保存为 Excel ---
        if excel_path:
            try:
                df = pd.DataFrame(data_rows)
                df.to_excel(excel_path, index=False)
                print(f"\n✅ Excel summary saved to: {excel_path}")
            except ImportError:
                print("Error: pandas or openpyxl not installed. Cannot save to Excel.")
            except Exception as e:
                print(f"Error saving Excel: {e}")

        # --- 保存为 JSON ---
        if json_path:
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data_rows, f, ensure_ascii=False, indent=2)
                print(f"✅ JSON summary saved to: {json_path}")
            except Exception as e:
                print(f"Error saving JSON: {e}")