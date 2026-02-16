import os
import json
import re
import sys
import ast
import uuid
import torch
import collections
import math
from math import isclose
from collections import defaultdict
from typing import List, Dict, Any, Optional, Callable, Union
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import ast 

# Adjust path to ensure we can import from src and scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.loaders.base_loader import BaseDataLoader, StandardSample, PageElement
from src.agents.ElementExtractor import ElementExtractor
from scripts.qwen3_vl_reranker_client import Qwen3VLReranker
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
            
        if len(gt) != len(pred):
            score = 0.0
        else:
            gt = sorted([get_clean_string(a) for a in gt])
            pred = sorted([get_clean_string(a) for a in pred])
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
    
    def __init__(self, data_root: str, output_dir: str = "./", reranker: Optional[Qwen3VLReranker] = None, extractor: Optional[ElementExtractor] = None, judger: Optional[ElementExtractor] = None):
        super().__init__(data_root)
        self.extractor = extractor
        self.reranker = reranker # 取消单例，通过初始化注入对象
        self.judger = judger # 新增 judger
        
        self.json_path = os.path.join(data_root, "data", "samples.json")
        self.doc_dir = os.path.join(data_root, "data", "documents")
        self.output_dir = output_dir
        self.llm_caller = None

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
            
        prompt = MMLONG_EXTRACT_PROMPT_TEMPLATE.format(question=question, analysis=raw_response)
        
        try:
            llm_output = llm_caller(prompt)
            
            extracted_answer = ""
            answer_format = "String"
            
            ans_match = re.search(r"Extracted answer:\s*(.*)", llm_output, re.IGNORECASE)
            fmt_match = re.search(r"Answer format:\s*(.*)", llm_output, re.IGNORECASE)
            
            if ans_match:
                extracted_answer = ans_match.group(1).strip()
            if fmt_match:
                answer_format = fmt_match.group(1).strip()
            
            if extracted_answer.startswith("'") and extracted_answer.endswith("'"):
                extracted_answer = extracted_answer[1:-1]
            
            return {
                "extracted_answer": extracted_answer,
                "answer_format": answer_format
            }
            
        except Exception as e:
            print(f"Error during LLM extraction: {e}")
            return {"extracted_answer": raw_response, "answer_format": "String"}

    def _serialize_element(self, elem):
        """Helper to convert PageElement or dict to JSON-serializable dict"""
        if isinstance(elem, dict):
            return elem
        if hasattr(elem, '__dict__'):
            return elem.__dict__
        return str(elem)

    def _sanitize_filename(self, name):
        """Sanitize the source name to be safe for filenames."""
        return "".join([c if c.isalnum() else "_" for c in str(name)])

    def save_bad_cases(self, output_dir: str, task: str):
        """
        Filter and save bad cases based on evaluation metrics.
        Saves separate JSON files for each Evidence Source.
        """
        bad_case_dir = os.path.join(output_dir, "bad_cases")
        os.makedirs(bad_case_dir, exist_ok=True)

        # Global lists for all cases
        retrieval_bad_cases = []
        generation_bad_cases = []

        # Dictionary to hold lists separated by source
        retrieval_by_source = collections.defaultdict(list)
        generation_by_source = collections.defaultdict(list)

        print(f"🔍 Analyzing {len(self.samples)} samples for bad cases...")

        for sample in self.samples:
            if sample.extra_info is None:
                continue
            
            # --- Parse Evidence Sources (Logic from MMLongLoader.py) ---
            ev_sources_str = sample.extra_info.get('evidence_sources', "[]")
            try:
                ev_sources = ast.literal_eval(str(ev_sources_str))
                if not isinstance(ev_sources, list):
                    ev_sources = [str(ev_sources)]
            except:
                ev_sources = ["Unknown"]
            
            # Add "Not Answerable" if sources are empty
            if not ev_sources:
                ev_sources = ["Not Answerable"]
            # --------------------------------------------------------

            metrics = sample.extra_info.get('metrics', {})
            
            # Flatten metrics
            flat_metrics = metrics.copy()
            if 'page' in metrics and isinstance(metrics['page'], dict):
                flat_metrics['page_recall'] = metrics['page'].get('recall', 0.0)
                flat_metrics['page_precision'] = metrics['page'].get('precision', 0.0)
            
            # 1. Retrieval Bad Case Check
            is_retrieval_bad = False
            if 'page_recall' in flat_metrics and flat_metrics['page_recall'] < 1.0:
                is_retrieval_bad = True
            elif 'page' in metrics and metrics['page'].get('recall', 1.0) < 1.0:
                is_retrieval_bad = True
                flat_metrics['page_recall'] = metrics['page'].get('recall')
                flat_metrics['page_precision'] = metrics['page'].get('precision')

            # 2. Generation Bad Case Check
            is_generation_bad = False
            if 'model_eval' in metrics and metrics['model_eval'] < 1.0:
                is_generation_bad = True

            # Construct Serializable Sample Object
            sample_dict = {
                "qid": str(sample.qid),
                "query": sample.query,
                "gold_answer": sample.gold_answer,
                "gold_pages": sample.gold_pages,
                "evidence_sources": ev_sources,
                "final_answer": sample.extra_info.get("final_answer", ""),
                "metrics": flat_metrics,
                "retrieved_elements": [self._serialize_element(e) for e in sample.extra_info.get("retrieved_elements", [])],
                "doc_source": sample.data_source
            }

            # Add to collections
            if is_retrieval_bad:
                retrieval_bad_cases.append(sample_dict)
                for src in ev_sources:
                    retrieval_by_source[src].append(sample_dict)
            
            if is_generation_bad:
                generation_bad_cases.append(sample_dict)
                for src in ev_sources:
                    generation_by_source[src].append(sample_dict)

        # --- Save Retrieval Bad Cases ---
        if task in ["retrieval", "all"] and retrieval_bad_cases:
            # 1. Save All
            p_all = os.path.join(bad_case_dir, "retrieval_bad_cases_all.json")
            with open(p_all, "w", encoding="utf-8") as f:
                json.dump(retrieval_bad_cases, f, indent=2, ensure_ascii=False)
            print(f"📉 Saved {len(retrieval_bad_cases)} total retrieval bad cases to {p_all}")

            # 2. Save per Source
            for source, cases in retrieval_by_source.items():
                safe_name = self._sanitize_filename(source)
                p_src = os.path.join(bad_case_dir, f"retrieval_bad_cases_{safe_name}.json")
                with open(p_src, "w", encoding="utf-8") as f:
                    json.dump(cases, f, indent=2, ensure_ascii=False)
                print(f"   └─ Saved {len(cases)} cases for source '{source}' to {os.path.basename(p_src)}")

        # --- Save Generation Bad Cases ---
        if task in ["generation", "all"] and generation_bad_cases:
            # 1. Save All
            p_all = os.path.join(bad_case_dir, "generation_bad_cases_all.json")
            with open(p_all, "w", encoding="utf-8") as f:
                json.dump(generation_bad_cases, f, indent=2, ensure_ascii=False)
            print(f"📉 Saved {len(generation_bad_cases)} total generation bad cases to {p_all}")

            # 2. Save per Source
            for source, cases in generation_by_source.items():
                safe_name = self._sanitize_filename(source)
                p_src = os.path.join(bad_case_dir, f"generation_bad_cases_{safe_name}.json")
                with open(p_src, "w", encoding="utf-8") as f:
                    json.dump(cases, f, indent=2, ensure_ascii=False)
                print(f"   └─ Saved {len(cases)} cases for source '{source}' to {os.path.basename(p_src)}")

    def evaluate_retrieval(self) -> Dict[str, float]:
        """
        执行页面检索评估。
        新增指标：平均召回的页面数量 (avg_retrieved_page_count)
        按 Evidence Source 分组统计 (Recall, Precision, Count)
        """
        total_metrics = collections.defaultdict(float)
        counts = collections.defaultdict(int)
        
        # 分组统计字典: source -> metric_name -> value
        source_metrics = collections.defaultdict(lambda: collections.defaultdict(float))
        source_counts = collections.defaultdict(int)

        print(f"Starting Retrieval Evaluation on {len(self.samples)} samples...")

        for sample in tqdm(self.samples, desc="Evaluating Retrieval"):
            if sample.extra_info is None:
                sample.extra_info = {}
            
            metrics_result = sample.extra_info.get('metrics', {})
            
            # --- Identify Evidence Sources ---
            ev_sources_str = sample.extra_info.get('evidence_sources', "[]")
            try:
                ev_sources = ast.literal_eval(str(ev_sources_str))
                if not isinstance(ev_sources, list):
                    ev_sources = [str(ev_sources)]
            except:
                ev_sources = ["Unknown"]
            
            # Add "Not Answerable" if sources are empty
            if not ev_sources:
                ev_sources = ["Not Answerable"]

            # --- Page Retrieval Evaluation ---
            raw_elements = sample.extra_info.get('retrieved_elements', [])
            pred_elements = []
            
            # 统计当前样本召回的唯一页面数量
            unique_retrieved_pages = set()

            for el in raw_elements:
                if isinstance(el, dict):
                    valid_keys = PageElement.__annotations__.keys()
                    filtered_el = {k: v for k, v in el.items() if k in valid_keys}
                    pe = PageElement(**filtered_el)
                    pred_elements.append(pe)
                    if pe.corpus_id: unique_retrieved_pages.add(pe.corpus_id)
                elif isinstance(el, PageElement):
                    pred_elements.append(el)
                    if el.corpus_id: unique_retrieved_pages.add(el.corpus_id)

            retrieved_count = len(unique_retrieved_pages)
            total_metrics['retrieved_page_count'] += retrieved_count

            # 计算 Metrics
            # 判断是否需要计算 Recall/Precision：如果 Gold Pages 存在，或者明确是 Unanswerable（Gold Pages 为空）
            is_unanswerable = ("Not Answerable" in ev_sources)
            
            current_recall = 0.0
            current_precision = 0.0
            
            if sample.gold_pages or is_unanswerable:
                page_score = {'recall': 0.0, 'precision': 0.0}
                if hasattr(self, '_compute_page_metrics'):
                     # _compute_page_metrics 能够处理空 gold_pages (期望 pred 也为空则 P=1, R=1)
                     page_score = self._compute_page_metrics(pred_elements, sample.gold_pages)
                
                current_recall = page_score['recall']
                current_precision = page_score['precision']
                
                metrics_result['page'] = page_score
                total_metrics['page_recall'] += current_recall
                total_metrics['page_precision'] += current_precision
                counts['page'] += 1
            
            # --- Update Source-based Metrics ---
            for src in ev_sources:
                source_counts[src] += 1
                source_metrics[src]['retrieved_page_count'] += retrieved_count
                
                # 只有计算了有效 Metrics 的才计入分组 Recall/Precision 平均
                if sample.gold_pages or is_unanswerable:
                    source_metrics[src]['page_recall'] += current_recall
                    source_metrics[src]['page_precision'] += current_precision
                    source_metrics[src]['count_with_gold'] += 1

            sample.extra_info['metrics'] = metrics_result

        avg_results = {}
        if counts['page'] > 0:
            avg_results['avg_page_recall'] = total_metrics['page_recall'] / counts['page']
            avg_results['avg_page_precision'] = total_metrics['page_precision'] / counts['page']
        
        if len(self.samples) > 0:
            avg_results['avg_retrieved_page_count'] = total_metrics['retrieved_page_count'] / len(self.samples)
            
        # --- Source-based Averages ---
        for src, metrics in source_metrics.items():
            count_all = source_counts[src]
            count_gold = metrics.get('count_with_gold', 0)
            
            if count_all > 0:
                avg_results[f'avg_retrieved_page_count_{src}'] = metrics['retrieved_page_count'] / count_all
            
            if count_gold > 0:
                avg_results[f'avg_page_recall_{src}'] = metrics['page_recall'] / count_gold
                avg_results[f'avg_page_precision_{src}'] = metrics['page_precision'] / count_gold
        
        return avg_results

    def evaluate_generation(self, num_threads: int = 8) -> Dict[str, float]:
        """
        执行生成结果评估 (QA Evaluation)，支持多线程。
        新增指标：
        1. 按照 evidence_sources 分类统计 model_eval (含 Not Answerable)
        2. 统计平均 input prompt token 和 output prompt token
        """
        total_metrics = collections.defaultdict(float)
        counts = collections.defaultdict(int)

        print(f"Starting Generation Evaluation on {len(self.samples)} samples with {num_threads} workers...")
        if self.llm_caller:
            print("Using LLM-based Answer Extraction.")
        else:
            print("Warning: No LLM caller provided. Skipping Answer Extraction.")

        def process_single_sample(sample):
            """内部函数：处理单个样本的答案提取与评分"""
            if sample.extra_info is None:
                sample.extra_info = {}
            
            metrics_result = sample.extra_info.get('metrics', {})
            gen_metrics = {}
            
            raw_pred_answer = sample.extra_info.get('final_answer', "")
            gold_answer = sample.gold_answer
            gold_format = sample.extra_info.get('answer_format', 'String') 
            
            score = 0.0
            has_valid_gold = False

            if gold_answer:
                has_valid_gold = True
                final_pred_to_score = raw_pred_answer
                
                if self.llm_caller and raw_pred_answer:
                    extract_res = self._extract_answer_with_llm(sample.query, raw_pred_answer, self.llm_caller)
                    final_pred_to_score = extract_res['extracted_answer']
                    sample.extra_info['extracted_answer'] = final_pred_to_score
                
                score = eval_score(gold_answer, final_pred_to_score, gold_format)
                gen_metrics['model_eval'] = score
            
            # 更新 metrics
            metrics_result.update(gen_metrics)
            sample.extra_info['metrics'] = metrics_result
            
            return score, 1 if has_valid_gold else 0

        # 多线程执行 (保持不变)
        if num_threads > 1:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_sample = {executor.submit(process_single_sample, sample): sample for sample in self.samples}
                for future in tqdm(as_completed(future_to_sample), total=len(self.samples), desc="Evaluating Generation"):
                    try:
                        score, count = future.result()
                        sample = future_to_sample[future] # 获取对应的 sample 对象
                        if count > 0:
                            total_metrics['model_eval'] += score
                            counts['total'] += count
                            
                            # --- 新增 1: Evidence Source 分类统计 ---
                            ev_sources_str = sample.extra_info.get('evidence_sources', "[]")
                            try:
                                ev_sources = ast.literal_eval(str(ev_sources_str))
                                if not isinstance(ev_sources, list): ev_sources = [str(ev_sources)]
                            except:
                                ev_sources = ["Unknown"]
                            
                            # Add Not Answerable if empty
                            if not ev_sources: ev_sources = ["Not Answerable"]

                            for src in ev_sources:
                                total_metrics[f'model_eval_{src}'] += score
                                counts[f'count_{src}'] += 1
                            # --------------------------------------

                        # --- 新增 2: Token 统计 (无论是否有 gold answer 都统计) ---
                        total_metrics['prompt_tokens'] += sample.extra_info.get('prompt_tokens', 0)
                        total_metrics['completion_tokens'] += sample.extra_info.get('completion_tokens', 0)
                        counts['token_count_samples'] += 1
                        # ----------------------------------------------------

                    except Exception as e:
                        print(f"Error processing sample in thread: {e}")
        else:
            # 单线程回退 (添加相同的统计逻辑)
            for sample in tqdm(self.samples, desc="Evaluating Generation"):
                score, count = process_single_sample(sample)
                if count > 0:
                    total_metrics['model_eval'] += score
                    counts['total'] += count
                    
                    # --- 新增 1: Evidence Source 分类统计 ---
                    ev_sources_str = sample.extra_info.get('evidence_sources', "[]")
                    try:
                        ev_sources = ast.literal_eval(str(ev_sources_str))
                        if not isinstance(ev_sources, list): ev_sources = [str(ev_sources)]
                    except:
                        ev_sources = ["Unknown"]
                    
                    # Add Not Answerable if empty
                    if not ev_sources: ev_sources = ["Not Answerable"]
                    
                    for src in ev_sources:
                        total_metrics[f'model_eval_{src}'] += score
                        counts[f'count_{src}'] += 1
                
                # --- 新增 2: Token 统计 ---
                total_metrics['prompt_tokens'] += sample.extra_info.get('prompt_tokens', 0)
                total_metrics['completion_tokens'] += sample.extra_info.get('completion_tokens', 0)
                counts['token_count_samples'] += 1

        avg_results = {}
        if counts['total'] > 0:
            avg_results['avg_model_eval'] = total_metrics['model_eval'] / counts['total']
        
        # --- 新增：计算分类指标平均值 ---
        for key in total_metrics:
            if key.startswith('model_eval_'):
                src = key.replace('model_eval_', '')
                cnt = counts[f'count_{src}']
                if cnt > 0:
                    avg_results[f'avg_model_eval_{src}'] = total_metrics[key] / cnt
        # -----------------------------

        # --- 新增：计算 Token 平均值 ---
        if counts['token_count_samples'] > 0:
            avg_results['avg_input_tokens'] = total_metrics['prompt_tokens'] / counts['token_count_samples']
            avg_results['avg_output_tokens'] = total_metrics['completion_tokens'] / counts['token_count_samples']
        # -----------------------------
            
        return avg_results

    def evaluate(self) -> Dict[str, float]:
        """
        执行完整评估：检索 + 生成。
        """
        results = {}
        
        # 1. Page Retrieval Evaluation
        retrieval_res = self.evaluate_retrieval()
        results.update(retrieval_res)
        
        # 2. QA Evaluation
        generation_res = self.evaluate_generation()
        results.update(generation_res)
        
        print(f"Evaluation Results: {results}")
        return results
    
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
            self._pdf_to_images(main_doc_path)
            count += 1
            
        print(f"✅ Successfully loaded {count} MMLongBench samples.")

    def _pdf_to_images(self, pdf_path: str) -> Dict[int, str]:
        if not os.path.exists(pdf_path):
             print(f"Warning: PDF not found at {pdf_path}")
             return {}

        pdf_name = os.path.basename(pdf_path).rsplit('.', 1)[0]
        cache_dir = os.path.join(os.path.abspath(os.path.join(self.data_root, "pdf_cache")), pdf_name)
        os.makedirs(cache_dir, exist_ok=True)
        
        image_map = {}
        
        existing_files = [f for f in os.listdir(cache_dir) if f.endswith('.png')]
        if existing_files:
            temp_map = {}
            for f in existing_files:
                match = re.match(r"page_(\d+)\.png", f)
                if match:
                    idx = int(match.group(1))
                    temp_map[idx] = os.path.join(cache_dir, f)
            if temp_map:
                return temp_map

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
        # 修改：直接使用实例中持有的 reranker 对象
        if not self.reranker or not pages:
            return pages
            
        documents_input = [{"text": f"Page ID: {page.corpus_id}", "image": page.crop_path} for page in pages]
        rerank_input = {
            # "instruction": "Given a search query, retrieve relevant candidates that answer the query.",
            # "instruction": "Given a search query, retrieve relevant candidates that answer the query considering both the visual content and the page metadata.",
            "instruction": (
                "Given a search query, retrieve relevant candidates that answer the query. "
                "Note that 'Page ID' indicates the physical page index in the document file, "
                "which does not necessarily correspond to the logical page number printed on the page image."
            ),
            "query": {"text": query},
            "documents": documents_input,
            "fps": 1.0 
        }
        
        try:
            scores = self.reranker.process(rerank_input)
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

    def _execute_agent_and_parse_json(self, agent, query: str, image_path: str, page_id: str = "") -> List[Any]:
        """
        Refactored helper function to run an agent (judger or extractor), 
        extract JSON from its output, and handle parsing errors.
        """
        try:
            max_retry = 5
            retry_count = 0
            predictions = []

            while retry_count < max_retry:
                agent_output = agent.run_agent(
                    user_text=query,
                    image_paths=[image_path]
                )

                if not agent_output:
                    return 111, [1, 1]

                predictions = agent_output.get("predictions", [])
                if not predictions:
                    return 111, [1, 1]

                all_text = "".join(
                        item.get("content", "")
                        for item in predictions
                        if isinstance(item, dict)
                    )

                # 判断长度是否过大
                if len(all_text) <= 40000:
                    break  # 正常结果，退出重试循环

                retry_count += 1
            
            last_msg_content = predictions[-1].get("content", "")
            #print(last_msg_content)
            #json_str = "[]"
            # Try to match markdown json block
            match = re.search(r"```json\s*(.*?)\s*```", last_msg_content, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                # Fallback to finding outermost brackets
                start = last_msg_content.find('[')
                end = last_msg_content.rfind(']')
                if start != -1 and end != -1:
                    json_str = last_msg_content[start:end+1]
            

            or_json_str = json_str
            try:
                extracted_data = json.loads(json_str)
            # Fix common JSON formatting issues in LLM output
            except:
                json_str = json_str.replace("\n", "\\n") 
                json_str = json_str.replace("\t", "\\t") 
                try:
                    extracted_data = json.loads(json_str)
                except:
                    json_str = or_json_str.replace("\\", "\\\\")
                    extracted_data = json.loads(json_str)
            return last_msg_content,extracted_data
            
        except json.JSONDecodeError:
            print(f"JSON Decode Error for page {page_id}")
            return last_msg_content,[1,1]
        except Exception as e:
            print(f"Error running agent on {page_id}: {e}")
            return last_msg_content,[1,1]

    def is_valid_extracted_data(self,data):
        if not isinstance(data, list):
            return False
        return all(isinstance(item, dict) for item in data)
    def pipeline(self, query: str, image_paths: List[str] = None, top_k: int = 10, trunc_thres=0.0, trunc_bbox=False) -> List[PageElement]:
        if not image_paths:
            return []

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

        candidate_pages = []
        for img_path in processed_image_paths:
            elem = PageElement(
                bbox=[0, 0, 1000, 1000],
                type="page_image",
                content="",
                corpus_id=img_path.split('/')[-1],
                corpus_path=img_path,
                crop_path=img_path
            )
            candidate_pages.append(elem)

        target_pages = candidate_pages
        if self.reranker and len(candidate_pages) > top_k:
            ranked_pages = self.rerank(query, candidate_pages)
            target_pages = ranked_pages[:top_k]

            # --- Expansion Recall Logic ---
            # Create a new list for expanded pages to avoid modifying list while iterating (though appending is usually safe, creating new is cleaner)
            expanded_target_pages = list(target_pages)
            existing_ids = set([p.corpus_id for p in target_pages])
            
            # Map corpus_id to index in candidate_pages (which preserves document order)
            id_to_idx = {p.corpus_id: i for i, p in enumerate(candidate_pages)}
            
            for page in target_pages:
                if page.corpus_id in id_to_idx:
                    curr_idx = id_to_idx[page.corpus_id]
                    
                    # Check previous page
                    if curr_idx > 0:
                        prev_page = candidate_pages[curr_idx - 1]
                        if prev_page.corpus_id not in existing_ids:
                            expanded_target_pages.append(prev_page)
                            existing_ids.add(prev_page.corpus_id)
                    
                    # Check next page
                    if curr_idx < len(candidate_pages) - 1:
                        next_page = candidate_pages[curr_idx + 1]
                        if next_page.corpus_id not in existing_ids:
                            expanded_target_pages.append(next_page)
                            existing_ids.add(next_page.corpus_id)
            
            target_pages = expanded_target_pages
            # -----------------------------

            target_pages = [ page for page in target_pages if page.retrieval_score >= trunc_thres]
        else:
            target_pages = candidate_pages[:top_k]

        workspace_dir = os.path.abspath(os.path.join(self.output_dir, "workspace", "crops"))
        os.makedirs(workspace_dir, exist_ok=True)

        if self.extractor is None:
            return target_pages

        extracted_elements = []

        for page in target_pages:
            img_path = page.corpus_path
            MAX_RETRY = 5
            retry = 0
            try:
                # --- Refactored: Use helper function and support Judger ---
                extracted_data = []
                if self.judger is not None:
                    # 1. Run Judger Agent
                    last_msg_content, extracted_data = self._execute_agent_and_parse_json(
                        self.judger, query, img_path, page_id=page.corpus_id
                    )

                    # 2. If Judger returns data, run Extractor Agent
                    if extracted_data:
                        while retry < MAX_RETRY:
                            last_msg_content, extracted_data = self._execute_agent_and_parse_json(
                                self.extractor, query, img_path, page_id=page.corpus_id
                            )
                            if self.is_valid_extracted_data(extracted_data) and extracted_data !=[]:
                                break
                            retry += 1

                    if not self.is_valid_extracted_data(extracted_data):
                        extracted_data = []
                else:
                    # Direct Extractor run
                    while retry < MAX_RETRY:
                        last_msg_content, extracted_data = self._execute_agent_and_parse_json(
                                self.extractor, query, img_path, page_id=page.corpus_id
                            )
                        if self.is_valid_extracted_data(extracted_data) and (retry>1 or extracted_data!=[]):
                            break
                        retry += 1

                    if not self.is_valid_extracted_data(extracted_data):
                        extracted_data = []

                if not extracted_data:
                    continue
                # ---------------------------------------------------------

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
                            corpus_path=img_path,
                            crop_path=current_crop_path 
                        )
                        if hasattr(page, 'retrieval_score'):
                            element.retrieval_score = page.retrieval_score
                        extracted_elements.append(element)

            except Exception as e:
                print(f"Error during agent execution on {img_path}: {e}")

        if trunc_bbox:
            extracted_elements = extracted_elements[:top_k]
        return extracted_elements

if __name__ == "__main__":
    # Test code
    root_dir = "/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/MMLongBench-Doc"
    reranker_path = "/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Reranker-8B"

    # 修改：在外部实例化 Reranker 对象
    print(f"⚡ Initializing Reranker instance from {reranker_path} ...")
    my_reranker = Qwen3VLReranker(
        model_name_or_path=reranker_path,
        torch_dtype=torch.float16 
    )

    # 修改：初始化 Loader 时传入对象
    # 如果有 judger，可以一并传入： loader = MMLongLoader(..., judger=my_judger)
    loader = MMLongLoader(data_root=root_dir, reranker=my_reranker)
    
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
            
            if s.data_source.endswith(".pdf"):
                print("Testing Pipeline...")
                results = loader.pipeline(s.query, image_paths=[s.data_source], top_k=2)
                s.extra_info['retrieved_elements'] = results
                # 测试新拆分的 evaluate
                loader.evaluate()
    except Exception as e:
        print(f"Test failed: {e}")
