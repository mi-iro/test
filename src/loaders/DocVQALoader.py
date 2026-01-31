import os
import json
import re
import sys
import uuid
import torch
import collections
from typing import List, Dict, Any, Optional, Callable
from PIL import Image

# 调整路径以确保可以从 src 和 scripts 导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.loaders.base_loader import BaseDataLoader, StandardSample, PageElement
from src.agents.ElementExtractor import ElementExtractor
from scripts.qwen3_vl_reranker import Qwen3VLReranker
from src.utils.llm_helper import create_llm_caller

# 复用 MMLongLoader 中的评分逻辑
from src.loaders.MMLongLoader import eval_score, MMLONG_EXTRACT_PROMPT_TEMPLATE

class DocVQALoader(BaseDataLoader):
    """
    DocVQA 数据集加载器。
    适配 top3_test.jsonl 格式，支持多图检索、ElementExtractor 抽取及 LLM 评估。
    """
    
    def __init__(self, data_root: str, extractor: Optional[ElementExtractor] = None, rerank_model: Optional[Qwen3VLReranker] = None):
        super().__init__(data_root)
        self.extractor = extractor
        self.reranker = rerank_model
        
        # DocVQA 特有路径适配
        self.jsonl_path = os.path.join(data_root, "top3_test.jsonl")
        self.img_dir = os.path.join(data_root, "imgs")
        self.llm_caller = None

    def load_data(self) -> None:
        """解析 top3_test.jsonl 格式数据。"""
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"DocVQA data file not found: {self.jsonl_path}")
        
        print(f"Loading DocVQA data from: {self.jsonl_path}")
        count = 0
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                
                qid = item.get("qid", str(count))
                query_text = item.get("query", "")
                # DocVQA 的 answer 通常是列表
                gold_answers = item.get("answer", [])
                
                # 图像路径处理：jsonl 中通常是相对路径如 "imgs/4.png"
                # 根据 tree 结构，images 位于 data_root/imgs 下
                images = item.get("image", [])
                image_full_paths = [os.path.join(self.data_root, img) for img in images]
                
                # 正确答案所在的页面 (用于 Recall 评估)
                pos_imgs = item.get("posImgs", [])
                gold_pages = [os.path.basename(p) for p in pos_imgs]

                sample = StandardSample(
                    qid=qid,
                    query=query_text,
                    dataset="DocVQA",
                    data_source=image_full_paths[0] if image_full_paths else "", 
                    gold_answer=gold_answers,
                    gold_elements=[],
                    gold_pages=gold_pages,
                    extra_info={
                        # "all_images": image_full_paths,
                        # "is_sufficient": item.get("is_sufficient", True)
                    }
                )
                self.samples.append(sample)
                count += 1
            
        print(f"✅ Successfully loaded {count} DocVQA samples.")

    def _extract_answer_with_llm(self, question: str, raw_response: str) -> Dict[str, Any]:
        """
        参考 FinRAGLoader/MMLongLoader 的逻辑，利用 LLM 从自由文本分析中提取结构化答案。
        """
        if not self.llm_caller or not raw_response:
            return {"extracted_answer": raw_response, "answer_format": "String"}
            
        # 构建 Prompt，要求模型输出：Extracted answer 和 Answer format
        prompt = MMLONG_EXTRACT_PROMPT_TEMPLATE.format(question=question, analysis=raw_response)
        
        try:
            llm_output = self.llm_caller(prompt)
            
            extracted_answer = ""
            answer_format = "String"
            
            # 使用正则解析 LLM 的固定格式输出
            ans_match = re.search(r"Extracted answer:\s*(.*)", llm_output, re.IGNORECASE)
            fmt_match = re.search(r"Answer format:\s*(.*)", llm_output, re.IGNORECASE)
            
            if ans_match:
                extracted_answer = ans_match.group(1).strip()
            if fmt_match:
                answer_format = fmt_match.group(1).strip()
            
            # 去除可能存在的单引号
            if extracted_answer.startswith("'") and extracted_answer.endswith("'"):
                extracted_answer = extracted_answer[1:-1]
            
            return {
                "extracted_answer": extracted_answer,
                "answer_format": answer_format
            }
        except Exception as e:
            print(f"LLM 提取失败: {e}")
            return {"extracted_answer": raw_response, "answer_format": "String"}

    def pipeline(self, query: str, image_paths: List[str] = None, top_k: int = 3) -> List[PageElement]:
        """
        DocVQA 核心流水线：
        1. 接收候选图片。
        2. 如果提供了 reranker，对图片进行重排序。
        3. 调用 ElementExtractor (Agent) 对 Top-K 图片进行视觉搜索和答案定位。
        """
        if not image_paths: return []

        # 封装为 PageElement 格式以供 reranker 使用
        candidate_pages = []
        for img_path in image_paths:
            candidate_pages.append(PageElement(
                bbox=[0, 0, 1000, 1000],
                type="page_image",
                content="",
                corpus_id=os.path.basename(img_path),
                corpus_path=img_path,
                crop_path=img_path
            ))

        # 1. Rerank
        if self.reranker and len(candidate_pages) > top_k:
            documents_input = [{"image": page.crop_path} for page in candidate_pages]
            rerank_input = {
                "instruction": "Given a search query, retrieve relevant candidates that answer the query.",
                "query": {"text": query},
                "documents": documents_input
            }
            scores = self.reranker.process(rerank_input)
            for page, score in zip(candidate_pages, scores):
                page.retrieval_score = score
            target_pages = sorted(candidate_pages, key=lambda x: x.retrieval_score, reverse=True)[:top_k]
        else:
            target_pages = candidate_pages[:top_k]

        # 2. Element Extraction (Visual Grounding)
        extracted_elements = []
        workspace_dir = os.path.abspath(os.path.join(os.getcwd(), "workspace", "crops"))
        os.makedirs(workspace_dir, exist_ok=True)

        for page in target_pages:
            try:
                # 运行 Agent 寻找证据
                agent_output = self.extractor.run_agent(
                    user_text=query,
                    image_paths=[page.corpus_path]
                )
                
                if not agent_output or "predictions" not in agent_output:
                    continue

                content = agent_output["predictions"][-1].get("content", "")
                # 解析证据和坐标 (简化逻辑)
                json_match = re.search(r'```json(.*?)```', content, re.DOTALL)
                if json_match:
                    items = json.loads(json_match.group(1).strip())
                    if isinstance(items, dict): items = [items]
                    
                    img = Image.open(page.corpus_path)
                    w, h = img.size
                    
                    for item in items:
                        bbox = item.get("bbox", [0, 0, 0, 0])
                        element = PageElement(
                            bbox=bbox,
                            type="evidence",
                            content=item.get("evidence", ""),
                            corpus_id=page.corpus_id,
                            corpus_path=page.corpus_path,
                            crop_path=page.corpus_path # 实际应用中可按 MMLongLoader 逻辑进行 Crop 保存
                        )
                        extracted_elements.append(element)
            except Exception as e:
                print(f"Error extracting from {page.corpus_id}: {e}")

        return extracted_elements

    def evaluate(self) -> Dict[str, float]:
        """
        执行带有 LLM 辅助提取的评估。
        """
        total_metrics = collections.defaultdict(float)
        counts = collections.defaultdict(int)

        print(f"正在对 {len(self.samples)} 个样本进行评估...")

        for sample in self.samples:
            if sample.extra_info is None:
                sample.extra_info = {}
            
            # --- 提取预测答案 ---
            raw_pred = sample.extra_info.get('final_answer', "")
            gold_answers = sample.gold_answer # DocVQA 中通常是一个列表
            
            # 如果提供了 llm_caller，则先进行标准化提取
            if self.llm_caller and raw_pred:
                extract_res = self._extract_answer_with_llm(sample.query, raw_pred)
                final_pred = extract_res['extracted_answer']
                # 将提取后的结果存入 extra_info 方便后续调试查看
                sample.extra_info['extracted_answer_llm'] = final_pred
            else:
                final_pred = raw_pred

            # --- 计算 QA 得分 ---
            if gold_answers:
                # DocVQA 评分：预测值与金标准列表中的任意一个匹配即计算最高分
                best_s = 0.0
                gold_list = gold_answers if isinstance(gold_answers, list) else [gold_answers]
                
                for gold in gold_list:
                    # 调用 MMLongLoader.py 提供的 eval_score (内部包含 ANLS 逻辑)
                    s = eval_score(gold, final_pred, "String")
                    best_s = max(best_s, s)
                
                total_metrics['qa_score'] += best_s
                counts['qa'] += 1

        # 计算平均指标
        avg_results = {}
        if counts['qa'] > 0:
            avg_results['avg_qa_score'] = total_metrics['qa_score'] / counts['qa']
            
        print(f"评估完成: {avg_results}")
        return avg_results

if __name__ == "__main__":
    # 使用示例
    data_root = "/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/VisRAG/data/EVisRAG-Test-DocVQA"
    loader = DocVQALoader(data_root=data_root)
    loader.load_data()
    
    if loader.samples:
        print(f"Sample Query: {loader.samples[0].query}")
        print(f"Gold Answer: {loader.samples[0].gold_answer}")