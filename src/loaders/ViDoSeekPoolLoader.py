import os
import json
import sys
import torch
import numpy as np
import faiss
import re
import base64
import time
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from PIL import Image
import uuid

# 引入评估所需的库
try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

# Adjust path to ensure we can import from src and scripts
# Assuming this file is placed in src/loaders/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.loaders.base_loader import BaseDataLoader, StandardSample, PageElement
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
from scripts.qwen3_vl_reranker_client import Qwen3VLReranker
from src.agents.ElementExtractor import ElementExtractor
from src.utils.llm_helper import create_llm_caller

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def encode_image_to_base64(image_path):
    """Convert image to base64 encoding."""
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class ViDoSeekLoader(BaseDataLoader):
    def __init__(self, data_root: str, embedding_model=None, rerank_model=None, extractor: Optional[ElementExtractor] = None):
        super().__init__(data_root)
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.extractor = extractor
        
        # --- 路径配置 ---
        # ViDoSeek 结构: data_root 下包含 top3_test.jsonl 和 imgs 目录
        self.jsonl_path = os.path.join(data_root, "top3_test.jsonl")
        self.corpus_root = os.path.join(data_root, "imgs")
        
        # 索引路径 (存放在 data_root/indices 下)
        cache_dir = os.path.join(data_root, "indices")
        os.makedirs(cache_dir, exist_ok=True)
        self.index_path = os.path.join(cache_dir, "vidoseek_hnsw.index")
        self.doc_map_path = os.path.join(cache_dir, "vidoseek_hnsw_docmap.json")
        
        self.index = None
        self.doc_id_map = {} 
        self.llm_caller = None

    def load_data(self) -> None:
        """加载 top3_test.jsonl 数据集。"""
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"Data file not found: {self.jsonl_path}")
        
        print(f"Loading data from: {self.jsonl_path}")
        
        samples = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line[:50]}...")
                    continue

                # 解析字段
                qid = str(item.get("qid") or count)
                query_text = item.get("query", "")
                
                # 处理答案 (ViDoSeek 中通常是列表)
                answers = item.get("answer", [])
                gold_answer = ""
                if isinstance(answers, list) and len(answers) > 0:
                    gold_answer = str(answers[0])
                elif isinstance(answers, str):
                    gold_answer = answers

                # 处理 Ground Truth 图片 (posImgs)
                # 示例: "./VisRAG-Ret-Test-DocVQA/imgs/737.png" -> 需要映射到本地 self.corpus_root/737.png
                pos_imgs = item.get("posImgs", [])
                gold_pages = []
                for p_img in pos_imgs:
                    fname = os.path.basename(p_img)
                    # full_path = os.path.join(self.corpus_root, fname)
                    gold_pages.append(fname)

                # 保留额外信息
                extra_info = {
                    "source_type": item.get("source_type", "text"),
                    "query_type": item.get("query_type", "single_hop"),
                    # "is_sufficient": item.get("is_sufficient", True),
                    # "original_candidates": item.get("image", []) # 原始数据中的候选集
                }

                sample = StandardSample(
                    qid=qid, 
                    query=query_text, 
                    dataset="vidoseek",
                    data_source=self.index_path, # 标记为需从索引检索
                    gold_answer=gold_answer,
                    gold_elements=[], # ViDoSeek 暂无 bbox 级标注
                    gold_pages=gold_pages, 
                    extra_info=extra_info
                )
                samples.append(sample)
                count += 1
                
        self.samples = samples
        print(f"✅ Successfully loaded {len(self.samples)} samples from {self.jsonl_path}.")

    def _get_all_image_paths(self) -> List[str]:
        print(f"Scanning images in {self.corpus_root}...")
        image_files = []
        if not os.path.exists(self.corpus_root):
             print(f"Warning: Corpus root {self.corpus_root} does not exist.")
             return []

        for root, dirs, files in os.walk(self.corpus_root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_files.append(os.path.join(root, file))
        image_files.sort()
        print(f"Found {len(image_files)} images.")
        return image_files

    def _embed_images(self, image_paths: List[str]) -> np.ndarray:
        if self.embedding_model is None:
            raise ValueError("Embedding model is not initialized.")
        inputs = [{"image": p} for p in image_paths]
        embeddings = self.embedding_model.process(inputs)
        return embeddings.cpu().numpy().astype('float32')

    def _embed_text(self, text: str) -> np.ndarray:
        if self.embedding_model is None:
            raise ValueError("Embedding model is not initialized.")
        inputs = [{"text": text, "instruction": "Represent the user's input."}]
        embeddings = self.embedding_model.process(inputs)
        return embeddings.cpu().numpy().astype('float32')

    def build_page_vector_pool(self, batch_size=4, force_rebuild=False):
        """Build or load page-level vector index using HNSW."""
        if not force_rebuild and os.path.exists(self.index_path) and os.path.exists(self.doc_map_path):
            print(f"Loading existing HNSW index from {self.index_path}...")
            self.index = faiss.read_index(self.index_path)
            with open(self.doc_map_path, 'r', encoding='utf-8') as f:
                self.doc_id_map = {int(k): v for k, v in json.load(f).items()}
            print(f"Index loaded. Total vectors: {self.index.ntotal}")
            return

        if self.embedding_model is None:
            raise ValueError("Embedding model is required to build the index!")

        print("Building new HNSW vector index...")
        image_paths = self._get_all_image_paths()
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.corpus_root}")

        # 使用第一张图确定向量维度
        sample_emb = self._embed_images([image_paths[0]]) 
        d = sample_emb.shape[1]
        
        M = 32 
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 64 
        index.verbose = True 

        doc_id_counter = 0
        batch_paths = []
        
        for path in tqdm(image_paths, desc="Indexing Pages (HNSW)"):
            batch_paths.append(path)
            
            if len(batch_paths) >= batch_size:
                embeddings = self._embed_images(batch_paths)
                faiss.normalize_L2(embeddings) 
                index.add(embeddings)
                
                for p in batch_paths:
                    rel_path = os.path.relpath(p, self.corpus_root)
                    self.doc_id_map[doc_id_counter] = rel_path
                    doc_id_counter += 1
                
                batch_paths = []

        if batch_paths:
            embeddings = self._embed_images(batch_paths)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            for p in batch_paths:
                rel_path = os.path.relpath(p, self.corpus_root)
                self.doc_id_map[doc_id_counter] = rel_path
                doc_id_counter += 1

        self.index = index
        
        print(f"Saving HNSW index to {self.index_path}...")
        faiss.write_index(self.index, self.index_path)
        with open(self.doc_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.doc_id_map, f)
        print("HNSW Index build complete.")

    def retrieve(self, query: str, top_k: int = 5, ef_search: int = 64) -> List[PageElement]:
        """Step 1: Vector Search with HNSW"""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_page_vector_pool() first.")

        if hasattr(self.index, 'hnsw'):
             self.index.hnsw.efSearch = max(ef_search, top_k * 2)

        query_vec = self._embed_text(query)
        faiss.normalize_L2(query_vec)
        
        scores, indices = self.index.search(query_vec, top_k)
        
        retrieved_pages = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            if idx not in self.doc_id_map: continue
            
            rel_path = self.doc_id_map[idx]
            abs_path = os.path.join(self.corpus_root, rel_path)
            
            element = PageElement(
                bbox=[0, 0, 1000, 1000],
                type="page_image",
                content=f"[Page Retrieved] {rel_path}",
                corpus_id=rel_path, # 使用相对路径作为ID
                corpus_path=abs_path,
                crop_path=abs_path 
            )
            element.retrieval_score = float(score)
            retrieved_pages.append(element)
            
        return retrieved_pages

    def rerank(self, query: str, pages: List[PageElement]) -> List[PageElement]:
        """Step 2: Reranking using Qwen3-VL-Reranker"""
        if not self.rerank_model or not pages:
            return pages
        print(f"Reranking {len(pages)} pages...")
        
        documents_input = [{"image": page.corpus_path} for page in pages]
        rerank_input = {
            "instruction": "Given a search query, retrieve relevant candidates that answer the query.",
            "query": {"text": query},
            "documents": documents_input,
            "fps": 1.0 
        }
        
        scores = self.rerank_model.process(rerank_input)
        if len(scores) != len(pages):
            print(f"Warning: Reranker returned {len(scores)} scores for {len(pages)} pages.")
            return pages

        for page, score in zip(pages, scores):
            page.retrieval_score = score
            
        sorted_pages = sorted(pages, key=lambda x: x.retrieval_score, reverse=True)
        return sorted_pages

    def extract_elements_from_pages(self, pages: List[PageElement], query: str) -> List[PageElement]:
        """Step 3: Downstream Element Extraction using ElementExtractor."""
        if self.extractor is None:
            # print("Warning: ElementExtractor is not initialized, skipping fine-grained extraction.")
            return pages 

        workspace_dir = os.path.abspath(os.path.join(os.getcwd(), "workspace", "crops"))
        os.makedirs(workspace_dir, exist_ok=True)

        fine_grained_elements = []
        
        for page in tqdm(pages, desc="Extracting Elements"):
            image_path = page.crop_path
            
            if not image_path or not os.path.exists(image_path):
                print(f"Warning: Image path not found: {image_path}")
                continue

            try:
                agent_output = self.extractor.run_agent(
                    user_text=query,
                    image_paths=[image_path]
                )
                
                if not agent_output:
                    continue

                predictions = agent_output.get("predictions", [])
                if not predictions:
                    continue
                
                last_msg_content = predictions[-1].get("content", "")
                
                json_str = "[]"
                match = re.search(r'```json(.*?)```', last_msg_content, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()
                else:
                    start = last_msg_content.find('[')
                    end = last_msg_content.rfind(']')
                    if start != -1 and end != -1:
                        json_str = last_msg_content[start:end+1]

                try:
                    extracted_data = json.loads(json_str)
                except json.JSONDecodeError:
                    # print(f"JSON Decode Error for page {page.corpus_id}")
                    extracted_data = []

                if isinstance(extracted_data, list):
                    try:
                        original_img = Image.open(image_path)
                        img_w, img_h = original_img.size
                    except Exception as e:
                        print(f"Failed to open image {image_path}: {e}")
                        continue

                    for item in extracted_data:
                        bbox = item.get("bbox", [0, 0, 0, 0])
                        evidence = item.get("evidence", "")
                        
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            x1 = int(x1 / 1000 * img_w)
                            y1 = int(y1 / 1000 * img_h)
                            x2 = int(x2 / 1000 * img_w)
                            y2 = int(y2 / 1000 * img_h)
                            
                            x1 = max(0, x1); y1 = max(0, y1); x2 = min(img_w, x2); y2 = min(img_h, y2)
                            
                            current_crop_path = image_path
                            
                            if x2 > x1 and y2 > y1:
                                try:
                                    cropped_img = original_img.crop((x1, y1, x2, y2))
                                    filename = f"{os.path.basename(page.corpus_id).split('.')[0]}_{uuid.uuid4().hex[:8]}.jpg"
                                    save_path = os.path.join(workspace_dir, filename)
                                    cropped_img.save(save_path)
                                    current_crop_path = save_path
                                except Exception as crop_err:
                                    print(f"Error cropping image: {crop_err}")
                        else:
                            current_crop_path = image_path

                        element = PageElement(
                            bbox=bbox, 
                            type="evidence",
                            content=evidence,
                            corpus_id=page.corpus_id,
                            corpus_path=page.corpus_path,
                            crop_path=current_crop_path
                        )
                        
                        if hasattr(page, 'retrieval_score'):
                            element.retrieval_score = page.retrieval_score
                            
                        fine_grained_elements.append(element)
                        
            except Exception as e:
                print(f"Error during extraction on {page.corpus_id}: {e}")

        if not fine_grained_elements:
            # print("No fine-grained elements extracted.")
            pass
            
        return fine_grained_elements

    def pipeline(self, query: str, image_paths: List[str] = None, top_k: int = 10, trunc_thres=0.0, trunc_bbox=False) -> List[PageElement]:
        """Full RAG Pipeline"""
        # 1. Retrieve (Expand search space for reranking)
        pages = self.retrieve(query, top_k=top_k*4)
        
        # 2. Rerank
        ranked_pages = self.rerank(query, pages)
        ranked_pages = ranked_pages[:top_k]
        
        # Filter low scores
        ranked_pages = [ page for page in ranked_pages if page.retrieval_score >= trunc_thres]

        # 3. Extract Elements
        elements = self.extract_elements_from_pages(ranked_pages, query)
        if trunc_bbox:
            elements = elements[:top_k]
        
        # 如果没有提取到细粒度元素，回退到页面级
        if not elements and ranked_pages:
            return ranked_pages
            
        return elements

    # --------------------------------------------------------------------------------
    # Evaluation Methods
    # --------------------------------------------------------------------------------

    def _evaluate_answer_correctness(self, sample: StandardSample) -> Dict[str, Any]:
        """
        使用 LLM 进行正确性评估 (Model Eval)。
        """
        query_text = sample.query
        expected_answer = sample.gold_answer
        actual_answer = sample.extra_info.get('final_answer', "error output") if sample.extra_info else "error output"

        model_eval = 0

        if "error output" in actual_answer or not self.llm_caller:
            model_eval = 0
        else:
            evaluation_prompt = (
                f"Question: {query_text}\n"
                f"Ground_truth: {expected_answer}\n"
                f"Model_answer: {actual_answer}\n"
                f"Is the model answer correct? You only need to output 'true' for correct or 'false' for incorrect. "
                f"If the model answer does not contain any information, it should be judged as 'false'."
            )
            try:
                response_core = self.llm_caller(evaluation_prompt)
                model_eval = 1 if "true" in response_core.lower() else 0
            except Exception as e:
                print(f"Error during model eval for QID {sample.qid}: {e}")
                model_eval = 0

        return {
            "model_eval": model_eval
        }

    def _compute_element_metrics(self, pred_elements: List[PageElement], gold_elements: List[PageElement], threshold: float = 0.5) -> Dict[str, float]:
        """
        ViDoSeek 数据集目前主要关注页面检索和 QA，如果缺乏 BBox 真值，此函数可能返回空或 0。
        """
        if not gold_elements:
            return {}
        # 复用 BaseLoader 逻辑或 FinRAG 逻辑
        # ... (此处省略具体实现，因 ViDoSeek 样本中 gold_elements 为空)
        return {}

    def evaluate(self) -> Dict[str, float]:
        """
        执行评估：重点关注 Model Eval (QA) 和 Page Retrieval Recall。
        """        
        total_metrics = {
            "model_eval": 0,
            "page_recall": 0, "page_precision": 0,
        }
        counts = {
            "total": 0, "page": 0
        }

        print(f"Starting Evaluation on {len(self.samples)} samples...")
        
        for sample in tqdm(self.samples, desc="Evaluating"):
            if sample.extra_info is None:
                sample.extra_info = {}
            
            # 1. 模型答案准确性
            corr_metrics = self._evaluate_answer_correctness(sample)
            total_metrics['model_eval'] += corr_metrics['model_eval']
            counts['total'] += 1

            # 获取预测结果
            retrieved_elements = sample.extra_info.get('retrieved_elements', [])
            
            # 统一对象格式
            elements_obj = []
            for el in retrieved_elements:
                if isinstance(el, dict):
                     # 简单的 dict 转 obj
                     pe = PageElement(
                         corpus_id=el.get('corpus_id', ''),
                         corpus_path=el.get('corpus_path', ''),
                         content=el.get('content', ''),
                         bbox=el.get('bbox', [])
                     )
                     elements_obj.append(pe)
                elif isinstance(el, PageElement):
                     elements_obj.append(el)

            # 2. 页面指标计算 (Page Precision/Recall)
            # ViDoSeek: 只要检索到的页面包含在 gold_pages 中即算命中
            target_gold_pages = sample.gold_pages
            
            # 这里的匹配逻辑需要注意路径一致性
            # gold_pages 是 full path, elements_obj.corpus_path 也是 full path
            # 为了稳健，比较 basename
            if target_gold_pages:
                gold_base_names = [os.path.basename(p) for p in target_gold_pages]
                
                hits = 0
                pred_base_names = [os.path.basename(p.corpus_path) for p in elements_obj if p.corpus_path]
                pred_set = set(pred_base_names)
                
                for gn in gold_base_names:
                    if gn in pred_set:
                        hits += 1
                
                recall = hits / len(gold_base_names) if gold_base_names else 0.0
                precision = hits / len(pred_set) if pred_set else 0.0
                
                total_metrics['page_recall'] += recall
                total_metrics['page_precision'] += precision
                counts['page'] += 1

        # 计算平均分
        avg_results = {}
        if counts['total'] > 0:
            avg_results['avg_model_eval'] = total_metrics['model_eval'] / counts['total']
        if counts['page'] > 0:
            avg_results['avg_page_recall'] = total_metrics['page_recall'] / counts['page']
            avg_results['avg_page_precision'] = total_metrics['page_precision'] / counts['page']

        print(f"Evaluation Results: {avg_results}")
        return avg_results

if __name__ == "__main__":
    # 配置路径
    embedding_model_path = "/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Embedding-8B"
    reranker_model_path = "http://localhost:8000"
    
    # ViDoSeek 数据集根目录
    root_dir = "/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/VisRAG/data/EVisRAG-Test-ViDoSeek"
    
    # 模拟工具和 Extractor
    from src.agents.utils import ImageZoomOCRTool
    tool_work_dir = "./workspace" 
    
    print("Initializing Models...")
    embedder = Qwen3VLEmbedder(model_name_or_path=embedding_model_path, torch_dtype=torch.float16)
    reranker = Qwen3VLReranker(model_name_or_path=reranker_model_path, torch_dtype=torch.float16)
    
    tool = ImageZoomOCRTool(work_dir=tool_work_dir)
    extractor = ElementExtractor(
        base_url="http://localhost:8001/v1",
        api_key="sk-123456",
        model_name="MinerU-Agent-CK800",
        tool=tool
    )

    loader = ViDoSeekLoader(
        data_root=root_dir, 
        embedding_model=embedder, 
        rerank_model=reranker,
        extractor=extractor
    )
    
    loader.llm_caller = create_llm_caller()
    
    loader.load_data()
    
    # 建立索引 (扫描 imgs 目录)
    loader.build_page_vector_pool(batch_size=1)
    
    if len(loader.samples) > 0:
        test_sample = loader.samples[0]
        print(f"\nTesting Query: {test_sample.query}")
        
        # 运行 pipeline
        results = loader.pipeline(test_sample.query) 
        
        # 模拟生成
        test_sample.extra_info['final_answer'] = "Generated Answer Placeholder"
        test_sample.extra_info['retrieved_elements'] = results
        
        # 运行评估
        loader.evaluate()