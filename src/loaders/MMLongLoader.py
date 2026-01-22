import os
import json
import re
import sys
import ast
import asyncio
import uuid
import torch
import collections
import math
from typing import List, Dict, Any, Optional, Callable, Union

# Adjust path to ensure we can import from src and scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.loaders.base_loader import BaseDataLoader, StandardSample, PageElement
from src.agents.ElementExtractor import ElementExtractor
from scripts.qwen3_vl_reranker import Qwen3VLReranker
from src.utils.llm_helper import create_llm_caller

# --- 新增：引入 Levenshtein 距离计算 ANLS (如果环境中没有 python-Levenshtein，可以使用 difflib 替代) ---
try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    # Fallback implementation if library is missing
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

# --- 新增：MMLongBench-Doc 定义的提取 Prompt (Appendix B.2) ---
MMLONG_EXTRACT_PROMPT_TEMPLATE = """Given the question and analysis, you are tasked to extract answers with required formats from the free-form analysis.
Your extracted answers should be one of the following formats: (1) Integer, (2) Float, (3) String and (4) List. 
If you find the analysis the question can not be answered from the given documents, type "Not answerable". 
Exception: If the analysis only tells you that it can not read/understand the images or documents type "Fail to answer".
Please make your response as concise as possible. Also note that your response should be formatted as below:

Extracted answer: [answer]
Answer format: [answer format]

Here is the example:
Question: List the primary questions asked about the services in this report.
Analysis: The primary questions asked about the services in the report for The Limes Residential Home are: 1. Is the service safe? 2. Is the service effective? 3. Is the service caring? 4. Is the service responsive? 5. Is the service well-led?
Extracted answer: ['Is the service safe?', 'Is the service effective?', 'Is the service caring?', 'Is the service responsive?', 'Is the service well-led?']
Answer format: List

Question: [question]
Analysis: [analysis]
"""

class MMLongLoader(BaseDataLoader):
    """
    MMLongBench-Doc 数据集加载器。
    用于加载 MMLongBench-Doc 中的 DocVQA 任务数据，并支持基于 LLM 的评估流程。
    """
    
    _reranker_instance = None

    def __init__(self, data_root: str, extractor: Optional[ElementExtractor] = None, reranker_model_path: str = None):
        super().__init__(data_root)
        self.extractor = extractor
        self.reranker_model_path = reranker_model_path
        
        self.json_path = os.path.join(data_root, "data", "samples.json")
        self.doc_dir = os.path.join(data_root, "data", "documents")
        self.llm_caller = None

    # ... [原有 get_reranker, _parse_assistant_content 等方法保持不变] ...
    
    @classmethod
    def get_reranker(cls, model_path: str):
        # (保持原有代码不变)
        if cls._reranker_instance is None:
            if not model_path or not os.path.exists(model_path):
                print(f"Warning: Reranker model path is invalid: {model_path}")
                return None
            print(f"⚡ Initializing Reranker Singleton from {model_path} ...")
            try:
                cls._reranker_instance = Qwen3VLReranker(
                    model_name_or_path=model_path,
                    torch_dtype=torch.float16 
                )
                print("✅ Reranker loaded successfully.")
            except Exception as e:
                print(f"❌ Failed to load Reranker: {e}")
                return None
        return cls._reranker_instance

    def _parse_assistant_content(self, content: str) -> Dict[str, Any]:
        # (保持原有代码不变)
        try:
            match = re.search(r'```json(.*?)```', content, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                return json.loads(json_str)
            else:
                return {"evidence": content, "bbox": []}
        except json.JSONDecodeError:
            return {"evidence": content, "bbox": []}

    # --- 新增：LLM 答案提取逻辑 ---
    def _extract_answer_with_llm(self, question: str, raw_response: str, llm_caller: Callable[[str], str]) -> Dict[str, Any]:
        """
        使用 LLM 从长回复中提取标准答案。
        :param llm_caller: 一个函数，输入 prompt 字符串，输出 LLM 的回复字符串。
        """
        if not raw_response:
            return {"extracted_answer": "", "answer_format": "String"}
            
        prompt = MMLONG_EXTRACT_PROMPT_TEMPLATE.replace("[question]", question).replace("[analysis]", raw_response)
        
        try:
            # 调用外部传入的 LLM 函数
            llm_output = llm_caller(prompt)
            
            # 解析输出
            extracted_answer = ""
            answer_format = "String"
            
            # 使用正则提取 Extracted answer 和 Answer format
            ans_match = re.search(r"Extracted answer:\s*(.*)", llm_output, re.IGNORECASE)
            fmt_match = re.search(r"Answer format:\s*(.*)", llm_output, re.IGNORECASE)
            
            if ans_match:
                extracted_answer = ans_match.group(1).strip()
            if fmt_match:
                answer_format = fmt_match.group(1).strip()
                
            # 简单的类型转换
            if answer_format.lower() == "list":
                try:
                    # 尝试将字符串列表转为 Python list
                    if extracted_answer.startswith("[") and extracted_answer.endswith("]"):
                         extracted_answer = ast.literal_eval(extracted_answer)
                except:
                    pass
            
            return {
                "extracted_answer": extracted_answer,
                "answer_format": answer_format
            }
            
        except Exception as e:
            print(f"Error during LLM extraction: {e}")
            return {"extracted_answer": raw_response, "answer_format": "String"}

    # --- 新增：MMLongBench-Doc 定义的评分规则 (Appendix B.3) ---
    def _compute_mmlong_score(self, pred: Any, gold: Any, fmt: str) -> float:
        """
        根据 Appendix B.3 计算分数
        """
        def normalize_str(s):
            return str(s).lower().strip()

        # 1. 预处理 Not Answerable
        if normalize_str(gold) == "not answerable":
            return 1.0 if normalize_str(pred) == "not answerable" else 0.0
        
        fmt = fmt.lower()
        
        # 2. String: ANLS with threshold 0.5
        if "string" in fmt:
            s1 = normalize_str(pred)
            s2 = normalize_str(gold)
            dist = levenshtein_distance(s1, s2)
            max_len = max(len(s1), len(s2))
            if max_len == 0: return 1.0
            nl = dist / max_len
            return 1.0 - nl if nl < 0.5 else 0.0

        # 3. Integer: Exact Match
        elif "integer" in fmt:
            try:
                # 尝试转 float 比较以处理 '10' vs '10.0'
                return 1.0 if float(pred) == float(gold) else 0.0
            except:
                return 1.0 if normalize_str(pred) == normalize_str(gold) else 0.0

        # 4. Float: 1% relative tolerance
        elif "float" in fmt:
            try:
                p_val = float(pred)
                g_val = float(gold)
                if abs(p_val - g_val) / (abs(g_val) + 1e-9) <= 0.01:
                    return 1.0
                return 0.0
            except:
                return 0.0

        # 5. List: Minimum element-wise score (Order matters in strict matching, 
        # but paper says "score each element in order", implying aligned lists)
        elif "list" in fmt:
            if not isinstance(pred, list): 
                # 尝试解析
                try: pred = ast.literal_eval(str(pred))
                except: pred = [str(pred)]
            if not isinstance(gold, list):
                try: gold = ast.literal_eval(str(gold))
                except: gold = [str(gold)]
                
            if not isinstance(pred, list) or not isinstance(gold, list):
                return 0.0
            
            if len(pred) != len(gold):
                return 0.0
            
            # Paper Eq 1: Sort lists then compare element-wise
            # "pred_list, ref_list = sorted(pred_list), sorted(ref_list)"
            try:
                pred.sort(key=lambda x: str(x))
                gold.sort(key=lambda x: str(x))
            except:
                pass # 无法排序则按原顺序
            
            scores = []
            for p, g in zip(pred, gold):
                # 递归调用评分，默认元素为 String 处理（论文未详述元素类型，通常假设为 String/Number）
                # 这里简化为 String ANLS 处理
                scores.append(self._compute_mmlong_score(p, g, "string"))
            
            return min(scores) if scores else 0.0

        # Default fallback
        return 0.0 if normalize_str(pred) != normalize_str(gold) else 1.0

    def evaluate(self) -> Dict[str, float]:
        """
        执行评估。
        
        :param llm_caller: (Optional) 用于 Answer Extraction 的 LLM 调用函数。
                           如果为 None，将跳过 LLM 提取步骤，直接使用 raw prediction 进行评分（可能导致分数偏低）。
                           Signature: func(prompt: str) -> str
        """
        total_metrics = collections.defaultdict(float)
        counts = collections.defaultdict(int)

        print(f"Starting Evaluation on {len(self.samples)} samples...")
        if self.llm_caller:
            print("Using LLM-based Answer Extraction.")
        else:
            print("Warning: No LLM caller provided. Skipping Answer Extraction (Scores might be inaccurate for long-context generation).")

        for sample in self.samples:
            if sample.extra_info is None:
                sample.extra_info = {}
            
            metrics_result = {}
            
            # --- 1. QA Evaluation (Updated with LLM Extraction) ---
            raw_pred_answer = sample.extra_info.get('final_answer', "")
            gold_answer = sample.gold_answer
            gold_format = sample.extra_info.get('answer_format', 'String') # 从数据集中获取黄金格式

            if gold_answer:
                final_pred_to_score = raw_pred_answer
                
                # A. LLM Extraction Step
                if self.llm_caller and raw_pred_answer:
                    extract_res = self._extract_answer_with_llm(sample.query, raw_pred_answer, self.llm_caller)
                    final_pred_to_score = extract_res['extracted_answer']
                    gold_format = extract_res['answer_format']
                    # 记录提取后的答案以便 debug
                    sample.extra_info['extracted_answer'] = final_pred_to_score
                
                # B. Scoring Step
                score = self._compute_mmlong_score(final_pred_to_score, gold_answer, gold_format)
                
                metrics_result['qa_score'] = score
                total_metrics['qa_score'] += score
                counts['qa'] += 1
            
            # --- 2. Page Retrieval Evaluation (Keep original logic) ---
            # 尝试获取预测的 elements
            raw_elements = sample.extra_info.get('retrieved_elements', [])
            pred_elements = []
            for el in raw_elements:
                if isinstance(el, dict):
                    valid_keys = PageElement.__annotations__.keys()
                    filtered_el = {k: v for k, v in el.items() if k in valid_keys}
                    pred_elements.append(PageElement(**filtered_el))
                elif isinstance(el, PageElement):
                    pred_elements.append(el)

            if sample.gold_pages:
                # Assuming _compute_page_metrics exists in BaseDataLoader or implemented locally
                # If not, we use a simple placeholder logic based on filename matching
                page_score = {'recall': 0.0, 'precision': 0.0}
                if hasattr(self, '_compute_page_metrics'):
                     page_score = self._compute_page_metrics(pred_elements, sample.gold_pages)
                
                metrics_result['page'] = page_score
                total_metrics['page_recall'] += page_score['recall']
                total_metrics['page_precision'] += page_score['precision']
                counts['page'] += 1

            sample.extra_info['metrics'] = metrics_result

        # --- Summary ---
        avg_results = {}
        if counts['qa'] > 0:
            avg_results['avg_qa_score'] = total_metrics['qa_score'] / counts['qa']
        
        if counts['page'] > 0:
            avg_results['avg_page_recall'] = total_metrics['page_recall'] / counts['page']
            avg_results['avg_page_precision'] = total_metrics['page_precision'] / counts['page']
            
        print(f"Evaluation Results: {avg_results}")
        return avg_results
    
    def load_data(self) -> None:
        """根据新的 samples.json 格式加载数据。"""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"MMLongBench data file not found: {self.json_path}")
        
        print(f"Loading MMLongBench data from: {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        count = 0
        for item in data:
            qid = str(count)
            doc_filename = item.get("doc_id", "")
            main_doc_path = os.path.join(self.doc_dir, doc_filename) if doc_filename else ""
            
            query_text = item.get("question", "")
            gold_answer = item.get("answer", "")
            
            evidence_pages_str = item.get("evidence_pages", "[]")
            gold_pages = []
            try:
                pages_list = ast.literal_eval(evidence_pages_str)
                if isinstance(pages_list, list):
                    gold_pages = [f"page_{str(p)}.png" for p in pages_list]
            except Exception as e:
                gold_pages = []

            extra_info = {
                "doc_type": item.get("doc_type"),
                "evidence_sources": item.get("evidence_sources"),
                "answer_format": item.get("answer_format")
            }

            sample = StandardSample(
                qid=qid,
                query=query_text,
                dataset="mmlongbench-doc",
                data_source=main_doc_path, 
                gold_answer=gold_answer,
                gold_elements=[],
                gold_pages=gold_pages,
                extra_info=extra_info
            )
            self.samples.append(sample)
            count += 1
            
        print(f"✅ Successfully loaded {count} MMLongBench samples.")

    def _pdf_to_images(self, pdf_path: str) -> Dict[int, str]:
        """将 PDF 转换为图片序列，并保存到缓存目录。"""
        if not os.path.exists(pdf_path):
             print(f"Warning: PDF not found at {pdf_path}")
             return {}

        pdf_name = os.path.basename(pdf_path).rsplit('.', 1)[0]
        workspace_dir = os.path.abspath(os.path.join(os.getcwd(), "workspace", "pdf_cache"))
        cache_dir = os.path.join(workspace_dir, pdf_name)
        os.makedirs(cache_dir, exist_ok=True)
        
        image_map = {}
        
        # 1. Check cache
        existing_files = [f for f in os.listdir(cache_dir) if f.endswith('.png')]
        if existing_files:
            temp_map = {}
            for f in existing_files:
                match = re.match(r"page_(\d+)\.png", f)
                if match:
                    idx = int(match.group(1))
                    temp_map[idx] = os.path.join(cache_dir, f)
            if temp_map:
                print(f"Using cached images for {pdf_name} ({len(temp_map)} pages)")
                return temp_map

        # 2. Convert if no cache
        print(f"Converting PDF to images: {pdf_path}")
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, dpi=200)
            for i, img in enumerate(images):
                page_num = i + 1 
                save_name = f"page_{page_num}.png"
                save_path = os.path.join(cache_dir, save_name)
                img.save(save_path, "PNG")
                image_map[page_num] = save_path
            print(f"Converted {len(image_map)} pages.")
        except ImportError:
            print("Error: `pdf2image` library is not installed.")
            return {}
        except Exception as e:
            print(f"Error converting PDF {pdf_path}: {e}")
            return {}
        return image_map

    def rerank(self, query: str, pages: List[PageElement]) -> List[PageElement]:
        """
        执行重排序。内部通过 get_reranker() 获取单例。
        """
        # 获取单例模型
        reranker = self.get_reranker(self.reranker_model_path)
        if not reranker or not pages:
            return pages
            
        print(f"Reranking {len(pages)} pages...")
        documents_input = [{"image": page.crop_path} for page in pages]
        rerank_input = {
            "instruction": "Given a search query, retrieve relevant candidates that answer the query.",
            "query": {"text": query},
            "documents": documents_input,
            "fps": 1.0 
        }
        
        try:
            scores = reranker.process(rerank_input)
            if len(scores) != len(pages):
                print(f"Warning: Reranker returned {len(scores)} scores for {len(pages)} pages.")
                return pages

            for page, score in zip(pages, scores):
                page.retrieval_score = score
                
            sorted_pages = sorted(pages, key=lambda x: x.retrieval_score, reverse=True)
            return sorted_pages
        except Exception as e:
            print(f"Error during reranking: {e}")
            return pages

    def pipeline(self, query: str, image_paths: List[str] = None,  top_k: int = 5) -> List[PageElement]:
        """
        Logic Updated:
        Lazy load and run reranker ONLY if len(pages) > top_k.
        """
        if self.extractor is None:
            print("Error: ElementExtractor is not initialized in MMLongLoader.")
            return []

        if not image_paths:
            return []

        # --- 1. Process PDF to Images ---
        processed_image_paths = []
        for path in image_paths:
            if path.lower().endswith('.pdf'):
                page_map = self._pdf_to_images(path)
                sorted_pages = sorted(page_map.keys())
                for p_num in sorted_pages:
                    processed_image_paths.append(page_map[p_num])
            else:
                processed_image_paths.append(path)
        
        if not processed_image_paths:
            return []

        # --- 2. Construct Candidate Elements ---
        candidate_pages = []
        for img_path in processed_image_paths:
            elem = PageElement(
                bbox=[0, 0, 1000, 1000],
                type="page_image",
                content="",
                corpus_id=img_path,
                crop_path=img_path
            )
            candidate_pages.append(elem)

        # --- 3. Conditional Lazy Reranking ---
        target_pages = candidate_pages
        
        # 仅当 页面数量 > Top_K 且 配置了模型路径 时，才触发重排
        if self.reranker_model_path and len(candidate_pages) > top_k:
            print(f"Page Count ({len(candidate_pages)}) > Top_K ({top_k}). Triggering Rerank...")
            ranked_pages = self.rerank(query, candidate_pages)
            target_pages = ranked_pages[:top_k]
        else:
            if len(candidate_pages) <= top_k:
                print(f"Page Count ({len(candidate_pages)}) <= Top_K ({top_k}). Skipping Rerank.")
            else:
                print(f"No Reranker configured. Taking first {top_k} pages.")
            target_pages = candidate_pages[:top_k]

        # --- 4. Element Extraction (Agent) ---
        workspace_dir = os.path.abspath(os.path.join(os.getcwd(), "workspace", "crops"))
        os.makedirs(workspace_dir, exist_ok=True)

        extracted_elements = []

        for page in target_pages:
            img_path = page.crop_path
            
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    print("Warning: Async loop running. Skipping extraction.")
                    continue
                else:
                    agent_output = asyncio.run(self.extractor.run_agent(
                        user_text=query,
                        image_paths=[img_path]  
                    ))
                
                if not agent_output:
                    continue

                predictions = agent_output.get("predictions", [])
                if not predictions:
                    continue

                last_message = predictions[-1]
                content = last_message.get("content", "")

                extracted_data = []
                try:
                    json_match = re.search(r'```json(.*?)```', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1).strip()
                    else:
                        start = content.find('[')
                        end = content.rfind(']')
                        if start != -1 and end != -1:
                            json_str = content[start:end+1]
                        else:
                            json_str = "[]"
                    extracted_data = json.loads(json_str)
                except Exception:
                    extracted_data = []

                if isinstance(extracted_data, dict):
                    extracted_data = [extracted_data]

                if isinstance(extracted_data, list):
                    current_page_image = None
                    img_w, img_h = 0, 0
                    try:
                        current_page_image = Image.open(img_path)
                        img_w, img_h = current_page_image.size
                    except Exception:
                        pass

                    for item in extracted_data:
                        bbox = item.get("bbox", [0, 0, 0, 0])
                        evidence = item.get("evidence", "")
                        
                        if not isinstance(bbox, list) or len(bbox) != 4:
                            bbox = [0, 0, 0, 0]
                        
                        current_crop_path = img_path 
                        
                        if current_page_image and bbox != [0, 0, 0, 0]:
                            try:
                                x1, y1, x2, y2 = bbox
                                x1 = int(x1 / 1000 * img_w)
                                y1 = int(y1 / 1000 * img_h)
                                x2 = int(x2 / 1000 * img_w)
                                y2 = int(y2 / 1000 * img_h)
                                
                                x1 = max(0, x1); y1 = max(0, y1); x2 = min(img_w, x2); y2 = min(img_h, y2)
                                
                                if x2 > x1 and y2 > y1:
                                    cropped_img = current_page_image.crop((x1, y1, x2, y2))
                                    filename = f"{os.path.basename(img_path).split('.')[0]}_{uuid.uuid4().hex[:8]}.jpg"
                                    save_path = os.path.join(workspace_dir, filename)
                                    cropped_img.save(save_path)
                                    current_crop_path = save_path 
                            except Exception:
                                pass

                        element = PageElement(
                            bbox=bbox,
                            type="evidence",
                            content=evidence,
                            corpus_id=img_path.split('/')[-1], 
                            crop_path=current_crop_path 
                        )
                        if hasattr(page, 'retrieval_score'):
                            element.retrieval_score = page.retrieval_score
                        extracted_elements.append(element)

            except Exception as e:
                print(f"Error during agent execution on {img_path}: {e}")

        return extracted_elements

if __name__ == "__main__":
    # Test code
    root_dir = "/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/MMLongBench-Doc"
    
    # Pass path instead of instance
    reranker_path = "/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Reranker-8B"

    loader = MMLongLoader(data_root=root_dir, reranker_model_path=reranker_path)
    try:
        loader.load_data()
        if len(loader.samples) > 0:
            s = loader.samples[0]
            print(f"\nSample 0 ID: {s.qid}, Doc: {s.data_source}")
            
            from src.agents.utils import ImageZoomOCRTool
            tool = ImageZoomOCRTool(work_dir="./workspace")
            extractor = ElementExtractor(
                base_url="http://localhost:8001/v1", 
                api_key="sk-123456", 
                model_name="MinerU-Agent-CK300",
                tool=tool
            )
            loader.extractor = extractor
            
            # This should trigger reranker ONLY if pdf pages > 2 (for test)
            if s.data_source.endswith(".pdf"):
                print("Testing Pipeline...")
                results = loader.pipeline(s.query, image_paths=[s.data_source], top_k=2)
                print(f"Extracted {len(results)} elements.")
                for res in results:
                    print(f" - Content: {res.content} \n - Crop: {res.crop_path}")
                s.extra_info['retrieved_elements'] = results
                loader.llm_caller = create_llm_caller()
                loader.evaluate()
    except Exception as e:
        print(f"Test failed: {e}")