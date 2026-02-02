import argparse
import os
import sys
import torch
import json
import yaml
import concurrent.futures
import traceback
from tqdm import tqdm
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.agents.AgenticRAGAgent import AgenticRAGAgent
from src.agents.RAGAgent import RAGAgent

from src.agents.ElementExtractor import ElementExtractor
from src.agents.utils import ImageZoomOCRTool
from src.loaders.MMLongLoader import MMLongLoader
from src.loaders.FinRAGLoader import FinRAGLoader
from src.loaders.ViDoSeekPoolLoader import ViDoSeekLoader
from src.loaders.DocVQALoader import DocVQALoader
from src.utils.llm_helper import create_llm_caller

try:
    from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
    from scripts.qwen3_vl_reranker_client import Qwen3VLReranker
except ImportError:
    print("Warning: Qwen3 VL scripts not found.")

def get_parser():
    parser = argparse.ArgumentParser(description="Run RAG Evaluation (Agentic or Standard).")

    parser.add_argument("--agent_type", type=str, default="standard", choices=["agentic", "standard"], 
                        help="Choose the type of agent: 'agentic' (ReAct loop) or 'standard' (One-pass RAG).")
    parser.add_argument("--top_k", type=int, default=10, 
                        help="Top-K Parameter.")
    parser.add_argument("--num_threads", type=int, default=1, 
                        help="Number of threads for parallel processing. Default is 1 (sequential).")

    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--infer_only", action="store_true", help="Skip evaluation step.")

    parser.add_argument("--benchmark", type=str, default="mmlong", choices=["mmlong", "finrag"], help="Target benchmark.")
    parser.add_argument("--data_root", type=str, default="/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/MMLongBench-Doc", help="Dataset root.")
    parser.add_argument("--output_dir", type=str, default="./results_rag", help="Directory to save results.")
    
    parser.add_argument("--model_name", type=str, default="qwen2.5-72b-instruct", help="LLM model name.")
    parser.add_argument("--base_url", type=str, default="http://localhost:3888/v1", help="LLM API Base URL.")
    parser.add_argument("--api_key", type=str, default="sk-6TGzZJkJ5HfZKwnrS1A1pMb1lH5D7EDfSVC6USq24aN2JaaR", help="LLM API Key.")
    
    parser.add_argument("--max_rounds", type=int, default=5, help="Max thinking rounds (Only for Agentic RAG).")
    
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples.")

    parser.add_argument("--embedding_model", type=str, default="/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Embedding-8B")
    parser.add_argument("--reranker_model", type=str, default="/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Reranker-8B")
    
    parser.add_argument("--reranker_api_base", type=str, default=None, help="vLLM API Base URL for Reranker.")
    parser.add_argument("--reranker_api_key", type=str, default="EMPTY")
    
    parser.add_argument("--mineru_server_url", type=str, default="http://10.102.98.181:8000/")
    parser.add_argument("--mineru_model_path", type=str, default="/root/checkpoints/MinerU2.5-2509-1.2B/")
    parser.add_argument("--extractor_model_name", type=str, default="MinerU-Agent-CK300")
    parser.add_argument("--extractor_base_url", type=str, default="http://localhost:8001/v1")
    parser.add_argument("--extractor_api_key", type=str, default="sk-123456")

    parser.add_argument("--finrag_lang", type=str, default="ch", choices=["ch", "en", "bbox"])
    parser.add_argument("--force_rebuild_index", action="store_true")

    # --- New Arguments ---
    parser.add_argument("--use_page", action="store_true", help="Include full page image in context.")
    parser.add_argument("--use_crop", action="store_true", help="Include element crop image in context.")
    parser.add_argument("--use_ocr", action="store_true", help="Include OCR text in context.")
    parser.add_argument("--use_ocr_raw", action="store_true", help="Use raw OCR content instead of Agent summary.")

    # [Added] New parameters for truncation
    parser.add_argument("--trunc_thres", type=float, default=None, help="Threshold score for truncation.")
    parser.add_argument("--trunc_bbox", action="store_true", help="Enable bounding box based truncation.")

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

def process_single_sample_safe(agent, sample, index, total):
    try:
        agent.process_sample(sample)
        return True
    except Exception as e:
        print(f"❌ Error processing sample {sample.qid}: {e}")
        traceback.print_exc()
        return False

def main():
    args = parse_args_with_config()

    os.makedirs(args.output_dir, exist_ok=True)
    workspace_dir = os.path.join(args.output_dir, "workspace")
    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    config_path = os.path.join(args.output_dir, f"config_{args.agent_type}.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(vars(args), f, default_flow_style=False, allow_unicode=True)

    print(f"🚀 Starting Benchmark: {args.benchmark.upper()}")
    print(f"🤖 Agent Type: {args.agent_type.upper()}")
    print(f"🧵 Threads: {args.num_threads}")
    if args.agent_type == "standard":
        print(f"🔍 Top-K: {args.top_k}")
        print(f"🖼️  Context Controls: Page={args.use_page}, Crop={args.use_crop}, OCR={args.use_ocr}, RawOCR={args.use_ocr_raw}")
        print(f"✂️  Truncation: Thres={args.trunc_thres}, BBox={args.trunc_bbox}")

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
    if args.reranker_api_base:
        print("🛠️ Initializing Reranker (REMOTE Mode)...")
        print(f"   Address: {args.reranker_api_base}")
        reranker = Qwen3VLReranker(
            model_name_or_path=args.reranker_api_base
        )
    elif args.reranker_model:
        print("🛠️ Initializing Reranker (LOCAL Mode)...")
        print(f"   Model Path: {args.reranker_model}")
        reranker = Qwen3VLReranker(
            model_name_or_path=args.reranker_model, 
            torch_dtype=torch.float16
        )

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
        # embedder = Qwen3VLEmbedder(model_name_or_path=args.embedding_model, torch_dtype=torch.float16)
        loader = FinRAGLoader(
            data_root=args.data_root,
            lang=args.finrag_lang,
            embedding_model=None,
            rerank_model=reranker,
            extractor=extractor
        )
        loader.load_data()
        
    elif args.benchmark == "docvqa":
        print("📥 Loading DocVQALoader...")
        loader = DocVQALoader(
            data_root=args.data_root,
            rerank_model=reranker,
            extractor=extractor
        )
        loader.load_data()
    elif args.benchmark == "vidoseek":
        print("📥 Loading ViDoSeekLoader...")
        embedder = Qwen3VLEmbedder(model_name_or_path=args.embedding_model, torch_dtype=torch.float16)
        loader = ViDoSeekLoader(
            data_root=args.data_root,
            embedding_model=embedder,
            rerank_model=reranker,
            extractor=extractor,
        )
        loader.load_data()
    else:
        raise NotImplementedError

    loader.llm_caller = create_llm_caller()

    if args.limit and args.limit > 0:
        print(f"⚠️ Limiting samples to {args.limit}.")
        loader.samples = loader.samples[:args.limit]

    agent = None
    if args.agent_type == "agentic":
        raise NotImplementedError
    elif args.agent_type == "standard":
        print(f"📜 Initializing RAGAgent (Standard, Top-K: {args.top_k})...")
        agent = RAGAgent(
            loader=loader,
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model_name,
            top_k=args.top_k,
            cache_dir=cache_dir,
            use_page=args.use_page,
            use_crop=args.use_crop,
            use_ocr=args.use_ocr,
            use_ocr_raw=args.use_ocr_raw,
            trunc_thres=args.trunc_thres, # [Added]
            trunc_bbox=args.trunc_bbox    # [Added]
        )
    else:
        raise ValueError(f"Unknown agent type: {args.agent_type}")

    print("\n⚡ Starting Processing Loop...")
    
    total_samples = len(loader.samples)
    
    if args.num_threads > 1:
        print(f"🔥 Parallel execution enabled with {args.num_threads} threads.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            future_to_sample = {
                executor.submit(process_single_sample_safe, agent, sample, i, total_samples): sample 
                for i, sample in enumerate(loader.samples)
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_sample), total=total_samples, desc="Processing"):
                sample = future_to_sample[future]
                try:
                    future.result() 
                except Exception as exc:
                    print(f"Sample {sample.qid} generated an exception: {exc}")
    else:
        print("🐢 Sequential execution (Single Thread).")
        for i, sample in enumerate(tqdm(loader.samples, desc="Processing")):
            agent.process_sample(sample)

    excel_path = os.path.join(args.output_dir, f"{args.benchmark}_{args.agent_type}_results.xlsx")
    json_path = os.path.join(args.output_dir, f"{args.benchmark}_{args.agent_type}_results.json")
    agent.save_results(excel_path=excel_path, json_path=json_path)

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

            error_analysis_dir = os.path.join(args.output_dir, "error_cases")
            os.makedirs(error_analysis_dir, exist_ok=True)
            
            print("🔍 Post-Eval: Saving error cases and updating cache with metrics...")
            
            count_updated = 0
            count_errors = 0
            
            for sample in loader.samples:
                sample_metrics = sample.extra_info.get('metrics', {})
                
                if 'model_eval' in sample_metrics and sample_metrics['model_eval'] == 0:
                    try:
                        error_file = os.path.join(error_analysis_dir, f"{sample.qid}_error.json")
                        
                        dump_data = {
                            "qid": sample.qid,
                            "query": sample.query,
                            "gold_answer": sample.gold_answer,
                            "final_answer": sample.extra_info.get('final_answer'),
                            "metrics": sample_metrics,
                            "gold_pages": sample.gold_pages,
                            "retrieved_elements": [
                                el.to_dict() if hasattr(el, 'to_dict') else el 
                                for el in sample.extra_info.get('retrieved_elements', [])
                            ],
                            "messages": sample.extra_info.get('messages', [])
                        }
                        
                        with open(error_file, 'w', encoding='utf-8') as f:
                            json.dump(dump_data, f, ensure_ascii=False, indent=2)
                        count_errors += 1
                    except Exception as e:
                        print(f"Error saving error case for {sample.qid}: {e}")

                if agent and hasattr(agent, 'cache_dir'):
                    cache_file_path = os.path.join(agent.cache_dir, f"{sample.qid}.json")
                    if os.path.exists(cache_file_path):
                        try:
                            with open(cache_file_path, 'r', encoding='utf-8') as f:
                                cache_data = json.load(f)
                            
                            cache_data['metrics'] = sample_metrics
                            cache_data['page_recall'] = sample_metrics.get('page_recall', 0.0)
                            cache_data['page_precision'] = sample_metrics.get('page_precision', 0.0)
                            cache_data['model_eval'] = sample_metrics.get('model_eval', 0)

                            with open(cache_file_path, 'w', encoding='utf-8') as f:
                                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                            count_updated += 1
                        except Exception as e:
                            print(f"Error updating cache for {sample.qid}: {e}")

            print(f"✅ Saved {count_errors} error cases to {error_analysis_dir}")
            print(f"✅ Updated {count_updated} cache files with evaluation metrics.")

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()