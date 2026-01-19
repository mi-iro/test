import os
import json
import sys
import torch
import numpy as np
import faiss
import asyncio
import re
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from PIL import Image  # 需要安装 pillow: pip install pillow
import uuid

# Adjust path to ensure we can import from src and scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.loaders.base_loader import BaseDataLoader, StandardSample, PageElement
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
from scripts.qwen3_vl_reranker import Qwen3VLReranker
from src.agents.ElementExtractor import ElementExtractor

class FinRAGLoader(BaseDataLoader):
    def __init__(self, data_root: str, lang: str = "ch", embedding_model=None, rerank_model=None, extractor: Optional[ElementExtractor] = None):
        super().__init__(data_root)
        self.lang = lang.lower()
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.extractor = extractor
        
        # --- 路径配置 ---
        self.query_path = os.path.join(data_root, "data", "queries", f"queries_{self.lang}.json")
        self.corpus_root = os.path.join(data_root, "data", "corpus", self.lang)
        self.qrels_path = os.path.join(data_root, "data", "qrels", f"qrels_{self.lang}.tsv")
        
        # 索引路径
        cache_dir = os.path.join(data_root, "data", "indices")
        os.makedirs(cache_dir, exist_ok=True)
        self.index_path = os.path.join(cache_dir, f"finrag_{self.lang}_hnsw.index")
        self.doc_map_path = os.path.join(cache_dir, f"finrag_{self.lang}_hnsw_docmap.json")
        
        self.index = None
        self.doc_id_map = {} 

    def _load_qrels(self) -> Dict[str, List[str]]:
        """读取 qrels TSV 文件。"""
        qrels_map = {}
        if not os.path.exists(self.qrels_path):
            print(f"Warning: Qrels file not found at {self.qrels_path}. GT IDs will be empty.")
            return qrels_map
            
        print(f"Loading qrels from: {self.qrels_path}")
        with open(self.qrels_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("query-id"):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    qid = parts[0]
                    cid = parts[1]
                    if qid not in qrels_map:
                        qrels_map[qid] = []
                    qrels_map[qid].append(cid)
        return qrels_map

    def load_data(self) -> None:
        """加载 Query 数据集并关联 Qrels。"""
        if not os.path.exists(self.query_path):
            raise FileNotFoundError(f"Query file not found: {self.query_path}")
        
        qrels_map = self._load_qrels()
        print(f"Loading queries from: {self.query_path}")
        with open(self.query_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        count = 0
        for item in data:
            query_text = item.get("query", "") or item.get("question", "")
            if not query_text:
                continue
            qid = str(item.get("id") or item.get("query-id") or item.get("_id") or "")
            gold_answer = item.get("answer", "") or item.get("response", "")
            extra_info = {'category': item.get('category', None), 'answer_type': item.get('answer_type', None), 'from_pages': item.get('from_pages', None)}
            gold_pages = qrels_map.get(qid, [])
            
            sample = StandardSample(
                qid=qid, query=query_text, dataset=f"finrag-{self.lang}",
                data_source=self.index_path, gold_answer=gold_answer,
                gold_elements=None, gold_pages=gold_pages, extra_info=extra_info
            )
            self.samples.append(sample)
            count += 1
        print(f"✅ Successfully loaded {count} queries.")

    def _get_all_image_paths(self) -> List[str]:
        print(f"Scanning images in {self.corpus_root}...")
        image_files = []
        for root, dirs, files in os.walk(self.corpus_root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
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
        """
        Build or load page-level vector index using HNSW.
        """
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
        """
        Step 1: Vector Search with HNSW
        """
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
            
            rel_path = self.doc_id_map[idx]
            abs_path = os.path.join(self.corpus_root, rel_path)
            
            element = PageElement(
                bbox=[0, 0, 1000, 1000],
                type="page_image",
                content=f"[Page Retrieved] {rel_path}",
                corpus_id=rel_path,
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
        
        documents_input = [{"image": page.crop_path} for page in pages]
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

    # def extract_elements_from_pages(self, pages: List[PageElement], query: str) -> List[PageElement]:
    #     """
    #     Step 3: Downstream Element Extraction using ElementExtractor.
    #     Uses the ElementExtractor agent to find specific evidence on the retrieved pages.
    #     """
    #     if self.extractor is None:
    #         print("Warning: ElementExtractor is not initialized, skipping fine-grained extraction.")
    #         # 如果没有 extractor，可以返回原页面作为结果，或者返回空
    #         return pages 

    #     fine_grained_elements = []
        
    #     # 遍历检索并重排后的所有页面
    #     for page in tqdm(pages, desc="Extracting Elements"):
    #         image_path = page.crop_path
            
    #         # 安全检查
    #         if not image_path or not os.path.exists(image_path):
    #             print(f"Warning: Image path not found: {image_path}")
    #             continue

    #         try:
    #             # 处理异步调用。如果当前已经在 loop 中（例如 Notebook 环境），直接调用可能会报错
    #             # 这里使用简单的 asyncio.run，假设是在独立脚本中运行
    #             try:
    #                 loop = asyncio.get_running_loop()
    #             except RuntimeError:
    #                 loop = None
                
    #             if loop and loop.is_running():
    #                 print("Warning: Async event loop is already running. Cannot use asyncio.run(). Skipping this page.")
    #                 continue
    #             else:
    #                 agent_output = asyncio.run(self.extractor.run_agent(
    #                     user_text=query,
    #                     image_paths=[image_path]
    #                 ))
                
    #             if not agent_output:
    #                 continue

    #             # --- 解析 Agent 输出 ---
    #             predictions = agent_output.get("predictions", [])
    #             if not predictions:
    #                 continue
                
    #             # 获取最后一条回复内容
    #             last_msg_content = predictions[-1].get("content", "")
                
    #             # 提取 JSON 块
    #             json_str = "[]"
    #             match = re.search(r'```json(.*?)```', last_msg_content, re.DOTALL)
    #             if match:
    #                 json_str = match.group(1).strip()
    #             else:
    #                 # 尝试查找最外层列表
    #                 start = last_msg_content.find('[')
    #                 end = last_msg_content.rfind(']')
    #                 if start != -1 and end != -1:
    #                     json_str = last_msg_content[start:end+1]

    #             try:
    #                 extracted_data = json.loads(json_str)
    #             except json.JSONDecodeError:
    #                 print(f"JSON Decode Error for page {page.corpus_id}")
    #                 extracted_data = []

    #             # 转换为 PageElement
    #             if isinstance(extracted_data, list):
    #                 for item in extracted_data:
    #                     bbox = item.get("bbox", [0, 0, 0, 0])
    #                     evidence = item.get("evidence", "")
                        
    #                     # 构造新的精细化元素
    #                     element = PageElement(
    #                         bbox=bbox,
    #                         type="evidence",
    #                         content=evidence,
    #                         corpus_id=page.corpus_id,
    #                         crop_path=image_path # 保留原图路径引用
    #                     )
    #                     # 继承页面的检索分数 (可选)
    #                     if hasattr(page, 'retrieval_score'):
    #                         element.retrieval_score = page.retrieval_score
                            
    #                     fine_grained_elements.append(element)
                        
    #         except Exception as e:
    #             print(f"Error during extraction on {page.corpus_id}: {e}")

    #     # 如果没有提取到任何精细化元素，可以考虑降级返回原页面，或者就返回空列表
    #     if not fine_grained_elements:
    #         print("No fine-grained elements extracted, returning empty list.")
            
    #     return fine_grained_elements

    def extract_elements_from_pages(self, pages: List[PageElement], query: str) -> List[PageElement]:
        """
        Step 3: Downstream Element Extraction using ElementExtractor.
        Uses the ElementExtractor agent to find specific evidence on the retrieved pages.
        """
        if self.extractor is None:
            print("Warning: ElementExtractor is not initialized, skipping fine-grained extraction.")
            return pages 

        # 定义裁剪图片保存的本地工作目录
        workspace_dir = os.path.abspath(os.path.join(os.getcwd(), "workspace", "crops"))
        os.makedirs(workspace_dir, exist_ok=True)

        fine_grained_elements = []
        
        # 遍历检索并重排后的所有页面
        for page in tqdm(pages, desc="Extracting Elements"):
            image_path = page.crop_path
            
            # 安全检查
            if not image_path or not os.path.exists(image_path):
                print(f"Warning: Image path not found: {image_path}")
                continue

            try:
                # --- 1. 调用 Agent 获取 BBox ---
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                
                if loop and loop.is_running():
                    print("Warning: Async event loop is already running. Cannot use asyncio.run(). Skipping this page.")
                    continue
                else:
                    agent_output = asyncio.run(self.extractor.run_agent(
                        user_text=query,
                        image_paths=[image_path]
                    ))
                
                if not agent_output:
                    continue

                # --- 2. 解析 Agent 输出 ---
                predictions = agent_output.get("predictions", [])
                if not predictions:
                    continue
                
                last_msg_content = predictions[-1].get("content", "")
                
                # 提取 JSON 块
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
                    print(f"JSON Decode Error for page {page.corpus_id}")
                    extracted_data = []

                # --- 3. 处理每个提取的元素 (裁剪并保存) ---
                if isinstance(extracted_data, list):
                    # 打开原始图片准备裁剪
                    try:
                        original_img = Image.open(image_path)
                        img_w, img_h = original_img.size
                    except Exception as e:
                        print(f"Failed to open image {image_path}: {e}")
                        continue

                    for item in extracted_data:
                        bbox = item.get("bbox", [0, 0, 0, 0]) # 通常格式为 [x1, y1, x2, y2]
                        evidence = item.get("evidence", "")
                        
                        # 默认 VLM 输出的 bbox 是基于 1000x1000 坐标系的归一化坐标
                        # 如果模型输出是绝对像素，请注释掉下方的转换逻辑
                        # -------------------------------------------------
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            # 转换为实际像素坐标
                            x1 = int(x1 / 1000 * img_w)
                            y1 = int(y1 / 1000 * img_h)
                            x2 = int(x2 / 1000 * img_w)
                            y2 = int(y2 / 1000 * img_h)
                            
                            # 边界修正，防止越界
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(img_w, x2)
                            y2 = min(img_h, y2)
                            
                            current_crop_path = image_path # 默认回退到原图
                            
                            # 只有当 bbox 有效且有面积时才进行裁剪
                            if x2 > x1 and y2 > y1:
                                try:
                                    # 执行裁剪
                                    cropped_img = original_img.crop((x1, y1, x2, y2))
                                    
                                    # 生成唯一文件名
                                    filename = f"{os.path.basename(page.corpus_id).split('.')[0]}_{uuid.uuid4().hex[:8]}.jpg"
                                    save_path = os.path.join(workspace_dir, filename)
                                    
                                    # 保存到本地 workspace
                                    cropped_img.save(save_path)
                                    current_crop_path = save_path # 更新路径为新裁剪图片的路径
                                except Exception as crop_err:
                                    print(f"Error cropping image: {crop_err}")
                        else:
                            current_crop_path = image_path
                        # -------------------------------------------------

                        # 构造新的精细化元素
                        element = PageElement(
                            bbox=bbox, # 保留原始 bbox 数据 (通常是 1000 scale)
                            type="evidence",
                            content=evidence,
                            corpus_id=page.corpus_id,
                            crop_path=current_crop_path # <--- 这里已更新为裁剪后的本地路径
                        )
                        
                        if hasattr(page, 'retrieval_score'):
                            element.retrieval_score = page.retrieval_score
                            
                        fine_grained_elements.append(element)
                        
            except Exception as e:
                print(f"Error during extraction on {page.corpus_id}: {e}")

        if not fine_grained_elements:
            print("No fine-grained elements extracted.")
            
        return fine_grained_elements

    def pipeline(self, query: str, top_k=5) -> List[PageElement]:
        """Full RAG Pipeline"""
        # 1. 粗排检索
        pages = self.retrieve(query, top_k=top_k*2)
        # 2. 重排序
        ranked_pages = self.rerank(query, pages)
        ranked_pages = ranked_pages[:top_k]
        # 3. 精细化提取
        elements = self.extract_elements_from_pages(ranked_pages, query)
        return elements

if __name__ == "__main__":
    # 配置路径
    embedding_model_path = "/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Embedding-8B"
    reranker_model_path = "/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Reranker-8B"
    root_dir = "/mnt/shared-storage-user/mineru2-shared/jiayu/data/FinRAGBench-V"
    
    # 模拟工具和 Extractor 的初始化参数
    from src.agents.utils import ImageZoomOCRTool
    # 假设有一个工作目录
    tool_work_dir = "./workspace" 
    
    print("Initializing Models...")
    embedder = Qwen3VLEmbedder(model_name_or_path=embedding_model_path, torch_dtype=torch.float16)
    reranker = Qwen3VLReranker(model_name_or_path=reranker_model_path, torch_dtype=torch.float16)
    
    # 初始化 Extractor (需要真实可用的 API Key 和 URL)
    # 这里仅为示例代码，实际运行时需替换为有效配置
    tool = ImageZoomOCRTool(work_dir=tool_work_dir)
    extractor = ElementExtractor(
        # base_url="http://localhost:3888/v1",
        # api_key="sk-6TGzZJkJ5HfZKwnrS1A1pMb1lH5D7EDfSVC6USq24aN2JaaR",
        # model_name="qwen3-vl-plus",
        base_url="http://localhost:8001/v1",
        api_key="sk-123456",
        model_name="MinerU-Agent-CK300",
        tool=tool
    )

    loader = FinRAGLoader(
        data_root=root_dir, 
        lang="ch", 
        embedding_model=embedder, 
        rerank_model=reranker,
        extractor=extractor # 传入 extractor
    )
    
    loader.load_data()
    
    # 建立索引 (如果不存在)
    loader.build_page_vector_pool(batch_size=16)
    
    if len(loader.samples) > 0:
        test_query = loader.samples[0].query
        print(f"\nTesting Query: {test_query}")
        # 运行 pipeline
        results = loader.pipeline(test_query, top_k=2) # 仅测试 Top 2 以节省时间
        print(f"\nFinal Elements Retrieved ({len(results)}):")
        for res in results:
            print(f"- [Evidence] {res.content} (BBox: {res.bbox})")