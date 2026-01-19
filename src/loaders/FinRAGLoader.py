import os
import json
import sys
import torch
import numpy as np
import faiss
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# Adjust path to ensure we can import from src and scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.loaders.base_loader import BaseDataLoader, StandardSample, PageElement
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
from scripts.qwen3_vl_reranker import Qwen3VLReranker

class FinRAGLoader(BaseDataLoader):
    def __init__(self, data_root: str, lang: str = "ch", embedding_model=None, rerank_model=None):
        super().__init__(data_root)
        self.lang = lang.lower()
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        
        # --- 路径配置 ---
        self.query_path = os.path.join(data_root, "data", "queries", f"queries_{self.lang}.json")
        self.corpus_root = os.path.join(data_root, "data", "corpus", self.lang)
        self.qrels_path = os.path.join(data_root, "data", "qrels", f"qrels_{self.lang}.tsv")
        
        # 索引路径 (建议修改文件名以区分 HNSW 和 Flat 索引)
        cache_dir = os.path.join(data_root, "data", "indices")
        os.makedirs(cache_dir, exist_ok=True)
        # 修改索引后缀或名称，避免加载旧的 Flat 索引报错
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

        # Get dimension from a sample
        sample_emb = self._embed_images([image_paths[0]]) 
        d = sample_emb.shape[1]
        
        # --- HNSW 配置 ---
        # M: 每个节点的邻居连接数，通常在 16~64 之间，越大精度越高但构建越慢/内存占用越大
        M = 32 
        # 使用 Inner Product (内积) 度量，配合 L2 归一化等价于余弦相似度
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        
        # efConstruction: 构建时的搜索深度，建议设为 M 的 2 倍以上，影响构建时间和索引质量
        index.hnsw.efConstruction = 64 
        
        # 注意: verbose 设为 True 可以看到构建进度
        index.verbose = True 

        doc_id_counter = 0
        batch_paths = []
        
        # Faiss HNSW 不支持像 IVF 那样的 train 过程，但它是增量构建的
        # 只需要 add 即可
        
        for path in tqdm(image_paths, desc="Indexing Pages (HNSW)"):
            batch_paths.append(path)
            
            if len(batch_paths) >= batch_size:
                embeddings = self._embed_images(batch_paths)
                # 依然需要归一化，因为使用的是 Inner Product 模拟 Cosine Similarity
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

        # 设置 HNSW 检索时的 efSearch 参数
        # efSearch 越大，召回率越高，但速度越慢。通常 efSearch > top_k
        if hasattr(self.index, 'hnsw'):
             self.index.hnsw.efSearch = max(ef_search, top_k * 2)

        # 1. Encode Query
        query_vec = self._embed_text(query)
        faiss.normalize_L2(query_vec)
        
        # 2. Faiss Search
        scores, indices = self.index.search(query_vec, top_k)
        
        # 3. Encapsulate as PageElement
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

    def extract_elements_from_pages(self, pages: List[PageElement], query: str) -> List[PageElement]:
        """Step 3: Downstream Element Extraction (Mocked)"""
        fine_grained_elements = []
        for page in pages:
            e1 = PageElement(
                bbox=[100, 100, 900, 200],
                type="text",
                content=f"Extracted text from {page.corpus_id}",
                corpus_id=page.corpus_id,
                crop_path=page.crop_path 
            )
            fine_grained_elements.append(e1)
        return fine_grained_elements

    def pipeline(self, query: str, top_k=100) -> List[PageElement]:
        """Full RAG Pipeline"""
        pages = self.retrieve(query, top_k=2*top_k)
        ranked_pages = self.rerank(query, pages)
        ranked_pages = ranked_pages[:top_k]
        elements = self.extract_elements_from_pages(ranked_pages, query)
        return elements

if __name__ == "__main__":
    embedding_model_path = "/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Embedding-8B"
    reranker_model_path = "/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Reranker-8B"
    root_dir = "/mnt/shared-storage-user/mineru2-shared/jiayu/data/FinRAGBench-V"

    print("Initializing Models...")
    embedder = Qwen3VLEmbedder(model_name_or_path=embedding_model_path, torch_dtype=torch.float16)
    reranker = Qwen3VLReranker(model_name_or_path=reranker_model_path, torch_dtype=torch.float16)

    loader = FinRAGLoader(data_root=root_dir, lang="ch", embedding_model=embedder, rerank_model=reranker)
    loader.load_data()
    
    # 强制重建索引以应用 HNSW if force_rebuild=True
    loader.build_page_vector_pool(batch_size=128, force_rebuild=False)
    
    if len(loader.samples) > 0:
        test_query = loader.samples[0].query
        print(f"\nTesting Query: {test_query}")
        results = loader.pipeline(test_query, top_k=5)
        print(f"\nFinal Elements Retrieved ({len(results)}):")
        for res in results:
            print(f"- {res.content[:50]}... (Score: {getattr(res, 'retrieval_score', 0.0):.4f})")