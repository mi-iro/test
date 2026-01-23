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
from math import isclose
from collections import defaultdict
from typing import List, Dict, Any, Optional, Callable, Union

# Adjust path to ensure we can import from src and scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.loaders.base_loader import BaseDataLoader, StandardSample, PageElement
from src.agents.ElementExtractor import ElementExtractor
from scripts.qwen3_vl_reranker import Qwen3VLReranker
from src.utils.llm_helper import create_llm_caller

# --- Scoring Functions ported from mmlongbench_eval_score.py ---

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def anls_compute(groundtruth, prediction, threshold=0.5):
    dist = levenshtein_distance(groundtruth, prediction)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    if anls <= threshold:
        anls = 0.0
    return anls


def is_float_equal(reference, prediction, include_percentage: bool = False, is_close: float = False) -> bool:
    def get_precision(gt_ans: float) -> int:
        precision = 3
        if '.' in str(gt_ans):
            precision = len(str(gt_ans).split('.')[-1])
        return precision

    try:
        reference = float(str(reference).strip().rstrip("%").strip())
        prediction = float(str(prediction).strip().rstrip("%").strip())
    except:
        return False

    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        try:
            if is_close:
                if isclose(item, prediction, rel_tol=0.01):
                    return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def get_clean_string(s):
    s = str(s).lower().strip()
    if s.endswith("mile"):
        s = s[:-4].strip()
    if s.endswith("miles"):
        s = s[:-5].strip()
    if s.endswith("million"):
        s = s[:-7].strip()
    # remove parenthesis
    s = re.sub(r'\s*\([^)]*\)', '', s).strip()
    # remove quotes
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()
    s = s.strip().lstrip("$").strip()
    s = s.strip().rstrip("%").strip()
    return s


def is_exact_match(s):
    flag = False
    # Website
    if "https://" in s:
        flag = True
    # code file
    if s.endswith(".py") or s.endswith("ipynb"):
        flag = True
    if s.startswith("page"):
        flag = True
    # telephone number
    if re.fullmatch(r'\b\d+(-\d+|\s\d+)?\b', s):
        flag = True
    # time
    if "a.m." in s or "p.m." in s:
        flag = True
    # YYYY-MM-DD
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}[-\s]\d{2}\b', s):
        flag = True
    # YYYY-MM
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}\b', s):
        flag = True
    # Email address
    if re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', s):
        flag = True
    return flag


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def eval_score(gt, pred, answer_type):
    # Mapping answer formats from prompt/dataset to logic types
    if answer_type in ["Int", "Integer"]:
        try:
            gt, pred = int(gt), int(float(pred))
        except:
            pred = ""
        score = (gt == pred)
    elif answer_type == "Float":
        try:
            gt = float(get_clean_string(str(gt)))
            pred = float(get_clean_string(str(pred)))
        except:
            pred = ""
        score = is_float_equal(gt, pred, include_percentage=True, is_close=True)
    elif answer_type in ["Str", "String", "None", "Not answerable", "Fail to answer"]:
        gt = get_clean_string(gt)
        pred = get_clean_string(pred)
        if is_exact_match(gt):
            score = (gt == pred)
        else:
            score = anls_compute(gt, pred)
    else:
        # List handling
        if isinstance(gt, str) and gt.startswith("["):
            try: gt = eval(gt)
            except: pass
        if not isinstance(gt, list):
            gt = [gt]
        
        if isinstance(pred, str) and pred.startswith("["):
            try: pred = eval(pred)
            except: pass
        if not isinstance(pred, list):
            pred = [pred]
            
        # print(len(gt), len(pred))
        if len(gt) != len(pred):
            score = 0.0
        else:
            gt = sorted([get_clean_string(a) for a in gt])
            pred = sorted([get_clean_string(a) for a in pred])
            # print(gt, pred)
            if len(gt) > 0 and (isfloat(gt[0]) or is_exact_match(gt[0])):
                score = ("-".join(gt) == "-".join(pred))
            else:
                if len(gt) == 0:
                    score = 1.0 # Both empty
                else:
                    score = min([anls_compute(gt_v, pred_v) for gt_v, pred_v in zip(gt, pred)])

    return float(score)

# --- Updated Prompt Template from mmlongbench_prompt_for_answer_extraction.md ---
MMLONG_EXTRACT_PROMPT_TEMPLATE = """Given the question and analysis, you are tasked to extract answers with required formats from the free-form analysis. 
- Your extracted answers should be one of the following formats: (1) Integer, (2) Float, (3) String and (4) List. If you find the analysis the question can not be answered from the given documents, type "Not answerable". Exception: If the analysis only tells you that it can not read/understand the images or documents, type "Fail to answer".
- Please make your response as concise as possible. Also note that your response should be formatted as below:

```

Extracted answer: [answer]
Answer format: [answer format]

```

Please read the following example, then extract the answer from the model response and type it at the end of the prompt. 

---
Question: List the primary questions asked about the services in this report.
Analysis:  The primary questions asked about the services in the report for The Limes Residential Home are:\\n\\n1. Is the service safe?\\n2. Is the service effective?\\n3. Is the service caring?\\n4. Is the service responsive?\\n5. Is the service well-led?
Extracted answer: ['Is the servife safe?', 'Is the service effective', 'Is the serve caring?', 'Is the service responsive?', 'Is the service well-led?']
Answer format: List

---
Question: How many regulations of the HSCA 2008 are breached in all according to this report?
Analysis: According to the report, the provider breached 10 Health and Social Care Act 2008 (Regulated Activities) Regulations in total. Here are the specifics:\\n\\n1. Regulation 13: Safeguarding service users from abuse and improper treatment\\n2. Regulation 12: Safe care and treatment\\n3. Regulation 18: Staffing\\n4. Regulation 11: Need for consent\\n5. Regulation 10: Dignity and respect\\n6. Regulation 9: Person-centred care\\n7. Regulation 17: Good governance\\n8. Regulation 18 (CQC Registration Regulations 2009): Notification of other incidents\\n9. Regulation 18: Failure to maintain an accurate and up-to-date care plan\\n10. Regulation 11: Failure to implement the Mental Capacity Act 2005 code of practice effectively\\n\\nThese breaches involve issues concerning staffing, safeguarding, medicines management, dignity and respect, consent, care planning, governance, and failure to notify the CQC of incidents.
Extracted answer: 10
Answer format: Integer

---
Question: According to the survey that is the percentage of Chinese who are paying more or about the same attention to politics after Trump's election?
Analysis: The survey provided does not specify the percentage of Chinese individuals specifically who are paying more or about the same attention to politics after Trump's election. The report focuses primarily on American demographics and does not include specific details about the Chinese population in relation to this question. If you need information about a different demographic or a summary of the findings from the American demographic, I can certainly help with that!
Extracted answer: Not answerable
Answer format: String

---
Question: How many quotations from male respondent over 50 years old are included in this report?
Analysis: The image you've provided appears to be a screenshot of a document with multiple charts. However, the text is too small and blurry to read accurately. If you can provide a clearer image or more context, I might be able to help you with your question.
Extracted answer: Fail to answer
Answer format: String

---
Question: {question}
Analysis: {analysis}
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

    @classmethod
    def get_reranker(cls, model_path: str):
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
        try:
            match = re.search(r'```json(.*?)```', content, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                return json.loads(json_str)
            else:
                return {"evidence": content, "bbox": []}
        except json.JSONDecodeError:
            return {"evidence": content, "bbox": []}

    def _extract_answer_with_llm(self, question: str, raw_response: str, llm_caller: Callable[[str], str]) -> Dict[str, Any]:
        """
        使用 LLM 从长回复中提取标准答案。
        """
        if not raw_response:
            return {"extracted_answer": "", "answer_format": "String"}
            
        # Construct the prompt with the question and analysis appended at the end
        prompt = MMLONG_EXTRACT_PROMPT_TEMPLATE.format(question=question, analysis=raw_response)
        
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
            
            # 基础清理
            if extracted_answer.startswith("'") and extracted_answer.endswith("'"):
                extracted_answer = extracted_answer[1:-1]
            
            return {
                "extracted_answer": extracted_answer,
                "answer_format": answer_format
            }
            
        except Exception as e:
            print(f"Error during LLM extraction: {e}")
            return {"extracted_answer": raw_response, "answer_format": "String"}

    def evaluate(self) -> Dict[str, float]:
        """
        执行评估。
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
            
            # --- 1. QA Evaluation ---
            raw_pred_answer = sample.extra_info.get('final_answer', "")
            # raw_pred_answer = 'Based on the provided document, specifically the pie chart on page 4 (image 0) titled "Latinos see economic upward mobility for their children", 5% of Latinos expect their children to be **less well off** financially than they themselves are now.'
            gold_answer = sample.gold_answer
            # 默认为 String，但会尝试从数据集或提取步骤更新
            gold_format = sample.extra_info.get('answer_format', 'String') 

            if gold_answer:
                final_pred_to_score = raw_pred_answer
                
                # A. LLM Extraction Step
                if self.llm_caller and raw_pred_answer:
                    extract_res = self._extract_answer_with_llm(sample.query, raw_pred_answer, self.llm_caller)
                    final_pred_to_score = extract_res['extracted_answer']
                    # 如果提取结果给出了特定格式，优先使用提取的格式，否则回退到数据集格式
                    # 注意：eval_score 需要正确的 format 来决定比较逻辑 (Int vs String vs List)
                    if extract_res['answer_format']:
                        gold_format = extract_res['answer_format']
                    
                    sample.extra_info['extracted_answer'] = final_pred_to_score
                
                # B. Scoring Step (Use ported eval_score)
                score = eval_score(gold_answer, final_pred_to_score, gold_format)
                
                metrics_result['qa_score'] = score
                total_metrics['qa_score'] += score
                counts['qa'] += 1
            
            # --- 2. Page Retrieval Evaluation ---
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
        """根据 samples.json 格式加载数据。"""
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
                # print(f"Using cached images for {pdf_name} ({len(temp_map)} pages)")
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
        reranker = self.get_reranker(self.reranker_model_path)
        if not reranker or not pages:
            return pages
            
        # print(f"Reranking {len(pages)} pages...")
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
        if self.reranker_model_path and len(candidate_pages) > top_k:
            # print(f"Page Count ({len(candidate_pages)}) > Top_K ({top_k}). Triggering Rerank...")
            ranked_pages = self.rerank(query, candidate_pages)
            target_pages = ranked_pages[:top_k]
        else:
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
            loader.llm_caller = create_llm_caller()
            
            # Use top_k=2 for testing
            if s.data_source.endswith(".pdf"):
                print("Testing Pipeline...")
                results = loader.pipeline(s.query, image_paths=[s.data_source], top_k=2)
                s.extra_info['retrieved_elements'] = results
                loader.evaluate()
    except Exception as e:
        print(f"Test failed: {e}")