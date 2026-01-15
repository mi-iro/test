import os
import json
import glob
import torch
import numpy as np
import faiss
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# 假设这些类定义在 src.loaders.base_loader 或当前文件中
# 为了代码独立运行，这里保留引用，实际使用请 import
from base_loader import BaseDataLoader, StandardSample, PageElement

class FinRAGLoader(BaseDataLoader):
    """
    FinRAGBench Loader: 支持中英文数据集加载、页面级向量检索与重排。
    """
    
    def __init__(self, data_root: str, lang: str = "ch", embedding_model=None, rerank_model=None):
        """
        :param data_root: FinRAGBench-V 的根目录
        :param lang: 'ch' (中文) 或 'en' (英文)
        :param embedding_model: 用于对 Query 和 Image 进行编码的模型实例
        :param rerank_model: 用于重排的模型实例
        """
        super().__init__(data_root)
        self.lang = lang.lower()
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        
        # --- 路径配置 (根据提供的目录结构) ---
        # Query 路径: data/queries/queries_ch.json
        self.query_path = os.path.join(data_root, "data", "queries", f"queries_{self.lang}.json")
        
        # Corpus 路径: data/corpus/ch/ 或 data/corpus/en/
        # 注意：目录下包含 part_0000 等子文件夹
        self.corpus_root = os.path.join(data_root, "data", "corpus", self.lang)
        
        # 索引缓存路径
        cache_dir = os.path.join(data_root, "data", "indices")
        os.makedirs(cache_dir, exist_ok=True)
        self.index_path = os.path.join(cache_dir, f"finrag_{self.lang}.index")
        self.doc_map_path = os.path.join(cache_dir, f"finrag_{self.lang}_docmap.json")
        
        self.index = None
        self.doc_id_map = {} # int -> relative_path_from_corpus_root

    def load_data(self) -> None:
        """
        加载 Query 数据集。
        """
        if not os.path.exists(self.query_path):
            raise FileNotFoundError(f"Query file not found: {self.query_path}")
            
        print(f"Loading queries from: {self.query_path}")
        with open(self.query_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # FinRAG 格式通常为 List[Dict]
        for item in data:
            # 兼容不同的 Key 命名习惯
            query_text = item.get("query", "") or item.get("question", "")
            if not query_text:
                continue

            gold_answer = item.get("answer", "") or item.get("response", "")
            
            # 尝试提取 GT BBox (如果存在)
            gold_bboxes = []
            if "evidence" in item and isinstance(item["evidence"], list):
                for ev in item["evidence"]:
                    if "bbox" in ev:
                        gold_bboxes.append(ev["bbox"])
            
            sample = StandardSample(
                query=query_text,
                dataset=f"finrag-{self.lang}",
                data_source=".png", # 这是一个视觉检索任务
                gold_answer=gold_answer,
                gold_bboxes=gold_bboxes
            )
            self.samples.append(sample)
            
        print(f"✅ Successfully loaded {len(self.samples)} queries for lang='{self.lang}'.")

    def _get_all_image_paths(self) -> List[str]:
        """
        递归扫描 corpus 目录下的所有图片文件。
        能够处理 part_0000, part_0001 等子目录结构。
        """
        print(f"Scanning images in {self.corpus_root}...")
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        image_files = []
        
        # 使用 os.walk 遍历所有子目录
        for root, dirs, files in os.walk(self.corpus_root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        # 排序以保证索引ID一致性
        image_files.sort()
        print(f"Found {len(image_files)} images.")
        return image_files

    def build_page_vector_pool(self, batch_size=32, force_rebuild=False):
        """
        构建或加载页面级向量索引。
        """
        # 1. 尝试加载现有索引
        if not force_rebuild and os.path.exists(self.index_path) and os.path.exists(self.doc_map_path):
            print(f"Loading existing index from {self.index_path}...")
            self.index = faiss.read_index(self.index_path)
            with open(self.doc_map_path, 'r', encoding='utf-8') as f:
                # JSON key 是 str，加载后转回 int
                self.doc_id_map = {int(k): v for k, v in json.load(f).items()}
            print(f"Index loaded. Total vectors: {self.index.ntotal}")
            return

        # 2. 新建索引
        if self.embedding_model is None:
            raise ValueError("Embedding model is required to build the index!")

        print("Building new vector index...")
        image_paths = self._get_all_image_paths()
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.corpus_root}")

        # 假设 Embedding 维度，需与模型一致
        # 例如 BGE-M3=1024, CLIP-L/14=768, OpenAI=1536
        # 这里先获取一个样本来确定维度
        sample_emb = self._mock_embed_images([image_paths[0]]) 
        d = sample_emb.shape[1]
        
        # 使用 Inner Product (IP) 索引，配合归一化后的向量等同于 Cosine Similarity
        index = faiss.IndexFlatIP(d) 

        # 批量处理
        doc_id_counter = 0
        batch_paths = []
        
        for path in tqdm(image_paths, desc="Indexing Pages"):
            batch_paths.append(path)
            
            if len(batch_paths) >= batch_size:
                embeddings = self._mock_embed_images(batch_paths)
                faiss.normalize_L2(embeddings) # 归一化
                index.add(embeddings)
                
                for p in batch_paths:
                    # 存储相对路径以节省空间，且方便迁移
                    rel_path = os.path.relpath(p, self.corpus_root)
                    self.doc_id_map[doc_id_counter] = rel_path
                    doc_id_counter += 1
                
                batch_paths = []

        # 处理剩余
        if batch_paths:
            embeddings = self._mock_embed_images(batch_paths)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            for p in batch_paths:
                rel_path = os.path.relpath(p, self.corpus_root)
                self.doc_id_map[doc_id_counter] = rel_path
                doc_id_counter += 1

        self.index = index
        
        # 保存
        print(f"Saving index to {self.index_path}...")
        faiss.write_index(self.index, self.index_path)
        with open(self.doc_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.doc_id_map, f)
        print("Index build complete.")

    def _mock_embed_images(self, image_paths: List[str]) -> np.ndarray:
        """
        [占位符] 实际代码中调用 self.embedding_model.encode_image(image_paths)
        必须返回 shape=(batch_size, dim) 的 float32 numpy array
        """
        if self.embedding_model:
             # 假设模型有 encode_images 方法
             # return self.embedding_model.encode_images(image_paths)
             pass
        
        # 演示用随机向量 (dim=768)
        return np.random.rand(len(image_paths), 768).astype('float32')

    def _mock_embed_text(self, text: str) -> np.ndarray:
        """
        [占位符] 实际代码中调用 self.embedding_model.encode_text(text)
        """
        # 演示用随机向量
        return np.random.rand(1, 768).astype('float32')

    def retrieve(self, query: str, top_k: int = 5) -> List[PageElement]:
        """
        Step 1: 粗排检索 (Vector Search)
        返回整个页面作为 PageElement
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_page_vector_pool() first.")

        # 1. Encode Query
        # query_vec = self.embedding_model.encode_text(query)
        query_vec = self._mock_embed_text(query)
        faiss.normalize_L2(query_vec)
        
        # 2. Faiss Search
        scores, indices = self.index.search(query_vec, top_k)
        
        # 3. 封装为 PageElement
        retrieved_pages = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            
            rel_path = self.doc_id_map[idx]
            abs_path = os.path.join(self.corpus_root, rel_path)
            
            # 创建 PageElement (此时代表整个页面)
            element = PageElement(
                bbox=[0, 0, 1000, 1000], # 全图
                type="image_page",
                content=f"[Page Retrieved] {rel_path} (Score: {score:.4f})",
                corpus_id=rel_path,
                crop_path=abs_path # 供视觉模型读取
            )
            # 临时挂载 score 属性供 rerank 使用
            element.retrieval_score = float(score)
            retrieved_pages.append(element)
            
        return retrieved_pages

    def rerank(self, query: str, pages: List[PageElement]) -> List[PageElement]:
        """
        Step 2: 重排 (Reranking)
        使用更精细的模型对 Query 和 Page 进行打分。
        """
        if not self.rerank_model or not pages:
            return pages

        print(f"Reranking {len(pages)} pages...")
        # 伪代码：
        # pairs = [(query, page.crop_path) for page in pages]
        # scores = self.rerank_model.predict(pairs)
        
        # 这里简单模拟，随机打乱顺序
        # 实际操作中请更新 page.retrieval_score 或 page.rerank_score
        sorted_pages = sorted(pages, key=lambda x: x.retrieval_score, reverse=True)
        return sorted_pages

    def extract_elements_from_pages(self, pages: List[PageElement], query: str) -> List[PageElement]:
        """
        Step 3: 神秘模块接口 (Page -> Elements)
        
        将检索到的 `Page`（整页图）交给下游模块（可能是VLM、Layout Analysis、OCR）。
        该模块会将页面拆解为更细粒度的 `PageElement`（如表格行、文本段、图表区域）。
        """
        fine_grained_elements = []
        
        for page in pages:
            # TODO: 这里调用你的神秘模块
            # elements = mysterious_module.process(page.crop_path, query)
            
            # --- 模拟神秘模块的行为 ---
            # 假设它把一个页面拆成了 1 个文本段和 1 个表格
            
            # 模拟元素 1: 文本
            e1 = PageElement(
                bbox=[100, 100, 900, 200],
                type="text",
                content=f"Extracted text from {page.corpus_id} related to {query}",
                corpus_id=page.corpus_id,
                crop_path=page.crop_path 
            )
            
            # 模拟元素 2: 表格
            e2 = PageElement(
                bbox=[100, 300, 900, 800],
                type="table",
                content=f"Extracted table data from {page.corpus_id}",
                corpus_id=page.corpus_id,
                crop_path=page.crop_path
            )
            
            fine_grained_elements.extend([e1, e2])
            
        return fine_grained_elements

    def pipeline(self, query: str, top_k=5) -> List[PageElement]:
        """
        完整的 RAG 检索流程
        Query -> Retrieve Pages -> Rerank Pages -> Extract Fine-grained Elements
        """
        # 1. 页面级检索
        pages = self.retrieve(query, top_k=top_k)
        
        # 2. 页面重排
        ranked_pages = self.rerank(query, pages)
        
        # 3. 元素提取 (神秘模块)
        elements = self.extract_elements_from_pages(ranked_pages, query)
        
        return elements

# --- 使用示例 ---
if __name__ == "__main__":
    # 配置根目录
    root_dir = "/mnt/shared-storage-user/mineru2-shared/jiayu/data/FinRAGBench-V"
    
    # 1. 初始化 Loader (选择中文 'ch')
    # 注意：你需要传入真实的 Embedding Model 才能 build index
    loader = FinRAGLoader(data_root=root_dir, lang="ch", embedding_model="MOCK_MODEL")
    
    # 2. 加载 Queries
    loader.load_data()
    
    # 3. 构建索引 (如果存在会自动加载)
    loader.build_page_vector_pool(batch_size=16)
    
    # 4. 运行单条测试
    if len(loader.samples) > 0:
        test_query = loader.samples[0].query
        print(f"\nTesting Query: {test_query}")
        
        results = loader.pipeline(test_query, top_k=3)
        
        print(f"\nFinal Elements Retrieved ({len(results)}):")
        for res in results:
            print(f"- [{res.type}] {res.content[:50]}... (Source: {res.corpus_id})")