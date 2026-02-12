# src/agents/RAGAgent.py

import os
import sys
import json
import base64
import mimetypes
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI
from collections import defaultdict
from io import BytesIO
from PIL import Image

# 确保可以导入 src 下的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.loaders.base_loader import BaseDataLoader, StandardSample, PageElement
from src.agents.utils import MinerUBboxExtractor

def local_image_to_data_url(path: str, max_pixels: int = None) -> str:
    """
    将本地图片转换为 Data URL (Base64)，支持最大分辨率限制 (max_pixels)。
    如果图片像素总数超过 max_pixels，将保持长宽比进行缩放。
    """
    if not path or not os.path.exists(path):
        return ""
    
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"

    try:
        with Image.open(path) as img:
            # 分辨率控制逻辑
            if max_pixels:
                w, h = img.size
                if w * h > max_pixels:
                    ratio = (max_pixels / (w * h)) ** 0.5
                    new_w = int(w * ratio)
                    new_h = int(h * ratio)
                    # 使用 BICUBIC 进行高质量缩放
                    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            
            # 保存到内存 buffer
            buffer = BytesIO()
            # 统一转为 RGB 处理 JPEG，或者保持原格式
            save_format = "PNG"
            if mime == "image/jpeg":
                save_format = "JPEG"
            elif mime == "image/webp":
                save_format = "WEBP"
                
            img.save(buffer, format=save_format)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"Error encoding image {path}: {e}")
        return ""

# --- Standard RAG System Prompt ---
RAG_SYSTEM_PROMPT = """
## ROLE
You are an expert AI assistant specializing in multimodal long document understanding. Your task is to carefully analyze the provided page images (which may contain text, figures, tables, and other content) and provide a precise answer to the user's question.
## Follow these instructions carefully:
- Core Objective: Your primary goal is to accurately and concisely answer the user's question based on the content of the provided document pages.
- Rules of numerical answers:
    - If the user asks for an absolute number (e.g., with questions like "How many...?"), you must first attempt to locate the number directly. If it cannot be found, find the relevant percentage and total count (or other necessary data) to calculate the absolute number. If the calculated absolute number for discrete entities (e.g., people, companies, objects) is a decimal, you must round it to the nearest whole number.
    - If the user asks for a percentage (or proportion), you must first attempt to locate the percentage directly. If it cannot be found, find the absolute numbers of the subgroup and the total count (or other necessary data) to calculate the percentage.
    - If the user's question is ambiguous and does not explicitly specify a number or percentage (e.g., "What's the gap between...?"), you must default to providing the absolute value. If you can only find relative values (percentages) in the chart, you must make every effort to find a total number within the provided context to calculate the absolute value. Only return the relative value as a last resort if a total number cannot be found, and explain that you cannot find total number in this case.
- Zoom-in Feature: When a page image contains figures or tables and requires closer inspection, we may provide zoomed-in images of these elements, appended after the main page image, to help you examine them closely. We will also extract text from the page image into Markdown format. Note: For questions related to page layout, you must refer to the original page image itself, not the zoomed-in images or the Markdown text, as they may lose layout information. For instance, if asked for the first figure on the page, you should consult the full page image to determine its order, not the sequence of the provided zoomed-in images.
- Page Numbering: Page numbers in the user's question typically refer to the number printed on the page image, not the page's index in the document file. For example, if a PDF's first page is the cover and the third page is the first page of content (labeled "Page 1"), a user's question about "page 1" refers to that third page. Similarly, when asked to provide a page number, you should return the printed page number from the image. Only return the page index if no number is printed on the page.
- Rule of faithfulness: Be faithful. If the provided pages do not contain sufficient information to answer the user's question, you should answer `Not answerable`. For example, if the user asks for a man in green shirts, but there are only man in red shirts in the provided pages, you should answer `Not answerable`; if the user asks for the boy playing badminton, but there are only boys playing football in the provided pages, you should answer `Not answerable`; if the user asks for a certain year's data but the provided pages only contain data for other years, you should answer `Not answerable`; if the user asks for the color of a certain object but the provided pages do not contain that object, you should answer `Not answerable`. 

### Input Format
The user will provide:
- A set of evidence (images and text) retrieved from the documents.
- The question to answer.

## Output Format:
Your entire response MUST be a single, valid JSON object and nothing else. Do not wrap it in markdown code blocks or add any other text. The JSON object must contain exactly two fields: analysis (string), and prediction (string).
- analysis field: Briefly explain your thought process. Describe how you located the answer within the document, which pages, tables, or figures you referenced, and how you connected the information to the question.
- prediction field: This must be a string containing the direct answer to the user's question.
"""

class RAGAgent:
    """
    Standard RAG Agent (Refactored).
    支持独立的 Retrieve 和 Generate 方法。
    """

    def __init__(
        self, 
        loader: BaseDataLoader, 
        base_url: str, 
        api_key: str, 
        model_name: str,
        top_k: int = 5,
        cache_dir: str = "./cache_results_rag",
        use_page: bool = False,
        use_crop: bool = True,
        use_ocr: bool = True,
        use_ocr_raw: bool = False,
        max_page_pixels: int = 1024 * 1024,
        trunc_thres: float = 0.0,
        trunc_bbox: bool = False,
        **kwargs
    ):
        self.loader = loader
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.top_k = top_k
        self.cache_dir = cache_dir
        
        self.use_page = use_page
        self.use_crop = use_crop
        self.use_ocr = use_ocr
        self.use_ocr_raw = use_ocr_raw
        self.max_page_pixels = max_page_pixels
        self.trunc_thres = trunc_thres
        self.trunc_bbox = trunc_bbox
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def retrieve(self, sample: StandardSample) -> List[PageElement]:
        """
        独立检索步骤：执行检索并返回 PageElement 列表。
        """
        print(f"Retrieving for Sample {sample.qid}...")
        try:
            image_inputs = [sample.data_source] if sample.data_source else []
            retrieved_elements = self.loader.pipeline(
                query=sample.query, 
                image_paths=image_inputs, 
                top_k=self.top_k,
                trunc_thres=self.trunc_thres,
                trunc_bbox=self.trunc_bbox,
            )
            return retrieved_elements
        except Exception as e:
            print(f"Error during retrieval for {sample.qid}: {e}")
            return []

    def generate(self, query: str, retrieved_elements: List[PageElement]) -> Dict[str, Any]:
        """
        独立生成步骤：根据 Query 和检索到的 Elements 生成回答。
        返回包含 'final_answer' 和 'messages' 的字典。
        """
        messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
        
        user_content = []
        if retrieved_elements:
            context_content = self.build_context_message(retrieved_elements)
            user_content.extend(context_content)
        else:
            user_content.append({"type": "text", "text": "No relevant context found.\n"})

        user_content.append({"type": "text", "text": f"Question: {query}"})
        messages.append({"role": "user", "content": user_content})

        final_answer = ""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                # reasoning_effort="high",
                messages=messages,
                temperature=1.0
            )
            final_answer = response.choices[0].message.content
            #print(final_answer)
        except Exception as e:
            print(f"LLM API Error: {e}")
            final_answer = "Error during generation."

        return {
            "final_answer": final_answer,
            "messages": messages[1:] # 不返回 system prompt
        }

    def build_context_message(self, elements: List[PageElement]) -> List[Dict[str, Any]]:
        """
        保持原有的 Context 构建逻辑不变。
        Modified: 按照页面文件名 (page_xx.png) 中的数字顺序组织上下文。
        """
        content_list = []
        content_list.append({"type": "text", "text": "Here is the retrieved context/evidence from the documents, grouped by page:\n"})
        
        pages_map = defaultdict(list)
        
        # 将元素按页面路径分组
        for el in elements:
            pages_map[el.corpus_path].append(el)
            
        # 定义提取页码的辅助函数 (格式: .../page_xx.png)
        def get_page_number(path: str) -> int:
            try:
                filename = os.path.basename(path)
                name_no_ext = os.path.splitext(filename)[0] # page_xx
                num_part = name_no_ext.split('_')[-1]       # xx
                return int(num_part)
            except (ValueError, IndexError, AttributeError):
                return 999999 # 无法解析时放到最后

        # 对页面路径进行排序
        page_order = sorted(pages_map.keys(), key=get_page_number)
        # print(page_order)
            
        bbox_extractor = MinerUBboxExtractor() if self.use_ocr_raw else None
        
        for p_idx, page_path in enumerate(page_order):
            page_elements = pages_map[page_path]
            file_name = os.path.basename(page_path)
            content_list.append({"type": "text", "text": f"\n=== Evidence Group {p_idx+1} (Source: {file_name}) ===\n"})
            
            if self.use_page:
                if page_path and os.path.exists(page_path):
                    img_url = local_image_to_data_url(page_path, max_pixels=self.max_page_pixels)
                    if img_url:
                        content_list.append({"type": "text", "text": "Full Page Context:\n"})
                        content_list.append({"type": "image_url", "image_url": {"url": img_url}})

            for i, el in enumerate(page_elements):
                if el.type == "page_image":
                    if self.use_ocr:
                        text_content = ""
                        if self.use_ocr_raw and bbox_extractor:
                            el.raw_content = bbox_extractor.extract_content_str(el.corpus_path, el.bbox)
                            text_content = el.raw_content
                        else:
                            text_content = el.content
                        
                        if text_content:
                            content_list.append({"type": "text", "text": f"Text Content: {text_content}\n"})
                    continue
                
                content_list.append({"type": "text", "text": f"\n-- Key Region {i+1} on this Page --\n"})
                
                if self.use_crop:
                    img_path = el.crop_path
                    if img_path and os.path.exists(img_path):
                        img_url = local_image_to_data_url(img_path)
                        if img_url:
                            content_list.append({"type": "text", "text": "Region Detail:\n"})
                            content_list.append({"type": "image_url", "image_url": {"url": img_url}})
                
                if self.use_ocr:
                    text_content = ""
                    if self.use_ocr_raw and bbox_extractor:
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
        兼容旧版的单次运行方法，依次调用 retrieve 和 generate。
        """
        if sample.extra_info is None:
            sample.extra_info = {}

        # 1. Retrieval
        retrieved_elements = self.retrieve(sample)
        sample.extra_info['retrieved_elements'] = retrieved_elements
        
        # 2. Generation
        gen_result = self.generate(sample.query, retrieved_elements)
        sample.extra_info['final_answer'] = gen_result['final_answer']
        sample.extra_info['messages'] = gen_result['messages']

        print(f"  [Standard RAG Answer]: {gen_result['final_answer'][:100]}...")
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