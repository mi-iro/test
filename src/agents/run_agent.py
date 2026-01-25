import argparse
import os
import sys
import torch
import asyncio
import json
import yaml
from PIL import Image

# 确保项目根目录在 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# --- 引入两个 Agent ---
from src.agents.AgenticRAGAgent import AgenticRAGAgent
from src.agents.RAGAgent import RAGAgent  # <--- 新增导入

from src.agents.ElementExtractor import ElementExtractor
from src.agents.utils import ImageZoomOCRTool
from src.loaders.MMLongLoader import MMLongLoader
from src.loaders.FinRAGLoader import FinRAGLoader
from src.utils.llm_helper import create_llm_caller

try:
    from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
    from scripts.qwen3_vl_reranker import Qwen3VLReranker
except ImportError:
    print("Warning: Qwen3 VL scripts not found.")

def get_parser():
    parser = argparse.ArgumentParser(description="Run RAG Evaluation (Agentic or Standard).")

    # ------------------ 核心控制参数 (新增) ------------------
    parser.add_argument("--agent_type", type=str, default="agentic", choices=["agentic", "standard"], 
                        help="Choose the type of agent: 'agentic' (ReAct loop) or 'standard' (One-pass RAG).")
    parser.add_argument("--top_k", type=int, default=5, 
                        help="Number of documents to retrieve (Only used for Standard RAG).")
    # -------------------------------------------------------

    # 配置文件路径
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    # 推理模式开关
    parser.add_argument("--infer_only", action="store_true", help="Skip evaluation step.")

    # 基础配置
    parser.add_argument("--benchmark", type=str, default="mmlong", choices=["mmlong", "finrag"], help="Target benchmark.")
    parser.add_argument("--data_root", type=str, default="/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/MMLongBench-Doc", help="Dataset root.")
    parser.add_argument("--output_dir", type=str, default="./results_rag", help="Directory to save results.")
    
    # LLM 配置
    parser.add_argument("--model_name", type=str, default="qwen3-max", help="LLM model name.")
    parser.add_argument("--base_url", type=str, default="http://localhost:3888/v1", help="LLM API Base URL.")
    parser.add_argument("--api_key", type=str, default="sk-6TGzZJkJ5HfZKwnrS1A1pMb1lH5D7EDfSVC6USq24aN2JaaR", help="LLM API Key.")
    
    # Agentic RAG 特有配置
    parser.add_argument("--max_rounds", type=int, default=5, help="Max thinking rounds (Only for Agentic RAG).")
    
    # 测试限制
    parser.add_argument("--limit", type=int, default=5, help="Limit number of samples.")

    # 模型路径 (FinRAG / Reranker)
    parser.add_argument("--embedding_model", type=str, default="/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Embedding-8B")
    parser.add_argument("--reranker_model", type=str, default="/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Reranker-8B")
    
    # Reranker Remote
    parser.add_argument("--reranker_api_base", type=str, default=None, help="vLLM API Base URL for Reranker.")
    parser.add_argument("--reranker_api_key", type=str, default="EMPTY")
    
    # MinerU / Extractor
    parser.add_argument("--mineru_server_url", type=str, default="http://10.102.250.36:8000/")
    parser.add_argument("--mineru_model_path", type=str, default="/root/checkpoints/MinerU2.5-2509-1.2B/")
    parser.add_argument("--extractor_model_name", type=str, default="MinerU-Agent-CK300")
    parser.add_argument("--extractor_base_url", type=str, default="http://localhost:8001/v1")
    parser.add_argument("--extractor_api_key", type=str, default="sk-123456")

    # FinRAG 特有
    parser.add_argument("--finrag_lang", type=str, default="ch", choices=["ch", "en", "bbox"])
    parser.add_argument("--force_rebuild_index", action="store_true")

    return parser

def parse_args_with_config():
    parser = get_parser()
    temp_args, _ = parser.parse_known_args()
    if temp_args.config and os.path.exists(temp_args.config):
        print(f"📄 Loading configuration from {temp_args.config}...")
        with open(temp_args.config, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            if config_data:
                parser.set_defaults(**config_data)
    args = parser.parse_args()
    return args

def main():
    args = parse_args_with_config()

    # 1. 目录准备
    os.makedirs(args.output_dir, exist_ok=True)
    workspace_dir = os.path.join(args.output_dir, "workspace")
    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # 保存配置
    config_path = os.path.join(args.output_dir, f"config_{args.agent_type}.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(vars(args), f, default_flow_style=False, allow_unicode=True)

    print(f"🚀 Starting Benchmark: {args.benchmark.upper()}")
    print(f"🤖 Agent Type: {args.agent_type.upper()}")
    if args.agent_type == "standard":
        print(f"🔍 Top-K: {args.top_k}")
    
    # 2. 初始化底层工具 (Extractor & Reranker)
    print("🛠️ Initializing Tools and Extractor...")
    tool = ImageZoomOCRTool(
        work_dir=os.path.join(workspace_dir, "crops"),
        mineru_server_url=args.mineru_server_url,
        mineru_model_path=args.mineru_model_path
    )
    
    extractor = ElementExtractor(
        base_url=args.extractor_base_url,
        api_key=args.extractor_api_key,
        model_name=args.extractor_model_name,
        tool=tool
    )

    reranker = None
    if args.reranker_model:
        print("🛠️ Initializing Reranker...")
        if args.reranker_api_base:
            print(f"   Mode: REMOTE (vLLM at {args.reranker_api_base})")
            reranker = Qwen3VLReranker(
                model_name_or_path=args.reranker_model,
                vllm_api_base=args.reranker_api_base,
                vllm_api_key=args.reranker_api_key
            )
        else:
            print(f"   Mode: LOCAL ({args.reranker_model})")
            reranker = Qwen3VLReranker(
                model_name_or_path=args.reranker_model, 
                torch_dtype=torch.float16
            )

    # 3. 初始化 DataLoader
    loader = None
    if args.benchmark == "mmlong":
        print("📥 Loading MMLongLoader...")
        loader = MMLongLoader(
            data_root=args.data_root, 
            extractor=extractor,
            reranker=reranker
        )
        loader.load_data()

    elif args.benchmark == "finrag":
        print("📥 Loading FinRAGLoader...")
        embedder = Qwen3VLEmbedder(model_name_or_path=args.embedding_model, torch_dtype=torch.float16)
        loader = FinRAGLoader(
            data_root=args.data_root,
            lang=args.finrag_lang,
            embedding_model=embedder,
            rerank_model=reranker,
            extractor=extractor
        )
        loader.load_data()
        loader.build_page_vector_pool(batch_size=4, force_rebuild=args.force_rebuild_index)

    loader.llm_caller = create_llm_caller()

    if args.limit and args.limit > 0:
        print(f"⚠️ Limiting samples to {args.limit}.")
        loader.samples = loader.samples[:args.limit]

    # 4. 初始化 Agent (根据类型)
    agent = None
    if args.agent_type == "agentic":
        print(f"🧠 Initializing AgenticRAGAgent (ReAct, Max Rounds: {args.max_rounds})...")
        agent = AgenticRAGAgent(
            loader=loader,
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model_name,
            top_k=args.top_k,
            cache_dir=cache_dir,
            max_rounds=args.max_rounds,
        )
    elif args.agent_type == "standard":
        print(f"📜 Initializing RAGAgent (Standard, Top-K: {args.top_k})...")
        agent = RAGAgent(
            loader=loader,
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model_name,
            top_k=args.top_k,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unknown agent type: {args.agent_type}")

    # 5. 执行处理循环
    print("\n⚡ Starting Processing Loop...")
    for i, sample in enumerate(loader.samples):
        print(f"[{i+1}/{len(loader.samples)}] Processing Sample QID: {sample.qid}")
        agent.process_sample(sample)

    # 6. 保存结果 (文件名区分 Agent 类型)
    excel_path = os.path.join(args.output_dir, f"{args.benchmark}_{args.agent_type}_results.xlsx")
    json_path = os.path.join(args.output_dir, f"{args.benchmark}_{args.agent_type}_results.json")
    agent.save_results(excel_path=excel_path, json_path=json_path)

    # 7. 执行评估
    if args.infer_only:
        print("\n⏭️  Skipping Evaluation (Infer Only).")
    else:
        print("\n📈 Starting Evaluation...")
        try:
            metrics = loader.evaluate()
            metrics_path = os.path.join(args.output_dir, f"{args.benchmark}_{args.agent_type}_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"✅ Evaluation complete. Metrics saved to {metrics_path}")
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")

if __name__ == "__main__":
    main()