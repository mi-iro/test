import os
import json
import torch
import sys
from tqdm import tqdm

# 假设你的目录结构，根据实际情况调整 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 引入必要的类 (根据你提供的代码段)
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
from scripts.qwen3_vl_reranker_client import Qwen3VLReranker
# 假设 FinRAGLoader 定义在名为 finrag_loader.py 的文件中，或者直接粘贴在同一个文件中
from FinRAGLoader import FinRAGLoader

# ================= Configuration =================
# 请根据你的实际环境修改以下路径
EMBEDDING_MODEL_PATH = "/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Embedding-8B"
RERANKER_MODEL_PATH = "http://localhost:8003"
DATA_ROOT_DIR = "/mnt/shared-storage-user/mineru2-shared/jiayu/data/FinRAGBench-V"
OUTPUT_FILE = "retrieved_labeled_results.json"

TOP_K_RETRIEVE = 50   # 向量检索召回数量
TOP_K_RERANK = 10     # Rerank后保留的最终数量
# =================================================

def normalize_path_id(path_str):
    """
    用于标准化路径以便比对。
    通常比对文件名(basename)比较稳健，避免绝对路径/相对路径差异。
    """
    if not path_str:
        return ""
    return os.path.basename(path_str).strip()

def main():
    print(">>> Initializing Models...")
    # 初始化 Embedder
    embedder = Qwen3VLEmbedder(
        model_name_or_path=EMBEDDING_MODEL_PATH, 
        torch_dtype=torch.float16
    )
    
    # 初始化 Reranker
    reranker = Qwen3VLReranker(
        model_name_or_path=RERANKER_MODEL_PATH, 
        torch_dtype=torch.float16
    )

    print(">>> Initializing Loader...")
    # 注意：这里不需要 Extractor，因为我们只做页面级检索
    loader = FinRAGLoader(
        data_root=DATA_ROOT_DIR, 
        lang="ch", 
        embedding_model=embedder, 
        rerank_model=reranker,
        extractor=None 
    )

    # 1. 加载数据 (Queries 和 Qrels)
    loader.load_data()
    loader.samples = loader.samples[:10]
    
    # 2. 构建或加载向量索引
    # force_rebuild=False 表示如果存在缓存索引直接加载
    loader.build_page_vector_pool(batch_size=16, force_rebuild=False)

    results_data = []

    print(f">>> Starting Retrieval for {len(loader.samples)} queries...")

    for sample in tqdm(loader.samples, desc="Processing Queries"):
        query = sample.query
        qid = sample.qid
        
        # 获取 Ground Truth 集合 (标准化为文件名)
        # sample.gold_pages 来自 qrels，可能是 ID 或 相对路径
        gold_ids = set([normalize_path_id(p) for p in sample.gold_pages])
        
        try:
            # Step 1: 向量检索 (Retrieve)
            # 扩大召回范围给 Reranker (例如取 Top 50)
            initial_pages = loader.retrieve(query, top_k=TOP_K_RETRIEVE)
            
            # Step 2: 重排序 (Rerank)
            # 使用 Qwen-VL 对图片和 Query 相关性打分
            reranked_pages = loader.rerank(query, initial_pages)
            
            # 截取最终 Top K
            final_top_pages = reranked_pages[:TOP_K_RERANK]
            
            # 构造结果列表
            page_results = []
            for page in final_top_pages:
                # 获取检索到的页面 ID (通常是文件名或相对路径)
                pred_id = normalize_path_id(page.corpus_id)
                
                # 判定正负样本
                # 如果预测的 ID 在 gold_ids 集合中，则为 1 (Positive)，否则为 0 (Negative)
                label = 1 if pred_id in gold_ids else 0
                
                page_info = {
                    "corpus_id": pred_id,          # 文件名
                    "corpus_path": page.corpus_path, # 绝对路径
                    "score": float(page.retrieval_score), # Reranker 分数
                    "label": label,                # 0 或 1
                    "is_ground_truth": bool(label)
                }
                page_results.append(page_info)

            # 记录该 Query 的完整信息
            results_data.append({
                "query_id": qid,
                "query": query,
                "gold_pages": list(sample.gold_pages), # 原始 GT 列表
                "retrieved_candidates": page_results
            })

        except Exception as e:
            print(f"Error processing query {qid}: {e}")
            continue

    # 3. 保存结果
    print(f">>> Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)
    
    # 简单的统计输出
    total_queries = len(results_data)
    queries_with_pos_retrieval = sum(1 for item in results_data if any(p['label'] == 1 for p in item['retrieved_candidates']))
    print(f"Done. Processed {total_queries} queries.")
    print(f"Queries with at least 1 correct page retrieved in Top-{TOP_K_RERANK}: {queries_with_pos_retrieval}/{total_queries}")

if __name__ == "__main__":
    main()