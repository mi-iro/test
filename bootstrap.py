# bootstrap.py

import argparse
import os
import sys
import torch
import yaml
import json  # Added import
import datetime
from omegaconf import OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.agents.RAGAgent import RAGAgent
from src.agents.ElementExtractor import ElementExtractor
from src.agents.utils import ImageZoomOCRTool
from src.loaders.MMLongLoader import MMLongLoader
from src.loaders.FinRAGLoader import FinRAGLoader
from src.loaders.DocVQALoader import DocVQALoader
from src.loaders.ViDoSeekPoolLoader import ViDoSeekLoader
from src.utils.llm_helper import create_llm_caller

try:
    from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
    from scripts.qwen3_vl_reranker_client import Qwen3VLReranker
except ImportError:
    print("Warning: Qwen3 VL scripts not found.")

def get_common_parser():
    parser = argparse.ArgumentParser(description="RAG Pipeline Component")
    
    # Common arguments
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--benchmark", type=str, default="mmlong", choices=["mmlong", "finrag", "docvqa", "vidoseek"], help="Target benchmark.")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results.")
    
    # Added Filter Argument
    parser.add_argument("--filter", type=str, default=None, help="Path to a bad_case file (JSON) or list of QIDs to filter the run.")

    # Model & API
    parser.add_argument("--model_name", type=str, default="qwen2.5-72b-instruct")
    parser.add_argument("--base_url", type=str, default="http://localhost:3888/v1")
    parser.add_argument("--api_key", type=str, default="sk-...")
    
    # Retrieval Params
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--embedding_model", type=str, default="/path/to/embedding")
    parser.add_argument("--reranker_model", type=str, default="/path/to/reranker")
    parser.add_argument("--reranker_api_base", type=str, default=None)
    parser.add_argument("--finrag_lang", type=str, default="ch")
    parser.add_argument("--trunc_thres", type=float, default=None)
    parser.add_argument("--trunc_bbox", action="store_true")

    # Extractor Params
    parser.add_argument("--mineru_server_url", type=str, default="http://10.102.98.181:8000/")
    parser.add_argument("--mineru_model_path", type=str, default="/root/checkpoints/MinerU2.5-2509-1.2B/")
    
    parser.add_argument("--judger_model_name", type=str, default="")
    parser.add_argument("--judger_base_url", type=str, default="")
    parser.add_argument("--judger_api_key", type=str, default="")
    
    parser.add_argument("--extractor_model_name", type=str, default="MinerU-Agent-CK300")
    parser.add_argument("--extractor_base_url", type=str, default="http://localhost:8001/v1")
    parser.add_argument("--extractor_api_key", type=str, default="sk-123456")

    # Generation Context Params
    parser.add_argument("--use_page", action="store_true")
    parser.add_argument("--use_crop", action="store_true")
    parser.add_argument("--use_ocr", action="store_true")
    parser.add_argument("--use_ocr_raw", action="store_true")
    
    # Limit & Threading
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads for parallel processing.")
    
    # Staged
    parser.add_argument("--evaluation_task", type=str, default="retrieval", choices=["retrieval", "generation", "all"], 
                        help="Specify which metrics to evaluate.")
    parser.add_argument("--generation_input", type=str, default=None, 
                        help="Path to the results JSON file. If None, auto-selects based on task.")
    parser.add_argument("--evaluation_input", type=str, default=None, 
                        help="Path to the results JSON file. If None, auto-selects based on task.")
    return parser

def parse_args():
    parser = get_common_parser()
    temp_args, _ = parser.parse_known_args()
    if temp_args.config and os.path.exists(temp_args.config):
        conf = OmegaConf.load(temp_args.config)
        config_data = OmegaConf.to_container(conf, resolve=True)
        if config_data:
            parser.set_defaults(**config_data)
    args = parser.parse_args()
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    return args

def save_run_config(args, stage_name="run"):
    """
    保存当前运行的配置到 output_dir，文件名包含时间戳。
    """
    if not args.output_dir:
        return
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_filename = f"config_{stage_name}_{timestamp}.yaml"
    config_path = os.path.join(args.output_dir, config_filename)
    
    try:
        # 将 args 转换为字典并保存
        config_dict = vars(args)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)
        print(f"📝 Configuration saved to {config_path}")
    except Exception as e:
        print(f"⚠️ Failed to save configuration: {e}")

def apply_qid_filter(items, filter_path, key='qid'):
    """
    通用过滤函数，用于根据 bad_case 文件过滤列表。
    items: list of dicts or list of objects
    filter_path: path to json file
    """
    if not filter_path or not os.path.exists(filter_path):
        return items
    
    print(f"🧹 Applying filter from: {filter_path}")
    try:
        with open(filter_path, 'r', encoding='utf-8') as f:
            filter_data = json.load(f)
        
        target_qids = set()
        for item in filter_data:
            if isinstance(item, dict) and 'qid' in item:
                target_qids.add(str(item['qid']))
            elif isinstance(item, (str, int)):
                target_qids.add(str(item))
                
        # items can be a list of objects (with .qid) or dicts (with ['qid'])
        filtered_items = []
        for item in items:
            # Check if item is object or dict
            item_qid = None
            if isinstance(item, dict):
                item_qid = item.get(key)
            elif hasattr(item, key):
                item_qid = getattr(item, key)
            
            if str(item_qid) in target_qids:
                filtered_items.append(item)
                
        print(f"   Reduced items from {len(items)} to {len(filtered_items)}")
        return filtered_items
    except Exception as e:
        print(f"⚠️ Filter error: {e}")
        return items

def initialize_components(args, init_retriever=True, init_generator=True):
    """
    初始化 Loader, Reranker, Agent 等组件
    """
    # 1. Extractor & Tools (Always needed for loader)
    tool = ImageZoomOCRTool(
        work_dir=os.path.join(args.output_dir, "workspace", "crops"),
        mineru_server_url=args.mineru_server_url,
        mineru_model_path=args.mineru_model_path
    )

    if args.judger_base_url and args.judger_api_key and args.judger_model_name:
        judger = ElementExtractor(
            base_url=args.judger_base_url,
            api_key=args.judger_api_key,
            model_name=args.judger_model_name,
            tool=tool
        )
    else:
        judger = None

    if args.extractor_base_url and args.extractor_api_key and args.extractor_model_name:
        extractor = ElementExtractor(
            base_url=args.extractor_base_url,
            api_key=args.extractor_api_key,
            model_name=args.extractor_model_name,
            tool=tool
        )
    else:
        extractor = None

    # 2. Reranker
    reranker = None
    if init_retriever:
        if args.reranker_api_base:
            reranker = Qwen3VLReranker(model_name_or_path=args.reranker_api_base)
        elif args.reranker_model:
            reranker = Qwen3VLReranker(model_name_or_path=args.reranker_model, torch_dtype=torch.float16)

    # 3. Loader
    loader = None
    if args.benchmark == "mmlong":
        loader = MMLongLoader(data_root=args.data_root, output_dir=args.output_dir, reranker=reranker, extractor=extractor, judger=judger)
        loader.load_data()
    elif args.benchmark == "finrag":
        loader = FinRAGLoader(data_root=args.data_root, output_dir=args.output_dir, lang=args.finrag_lang, embedding_model=None, rerank_model=reranker, extractor=extractor, judger=judger)
        loader.load_data()
    elif args.benchmark == "vidoseek":
        embedder = Qwen3VLEmbedder(model_name_or_path=args.embedding_model, torch_dtype=torch.float16) if init_retriever else None
        loader = ViDoSeekLoader(data_root=args.data_root, output_dir=args.output_dir, embedding_model=embedder, rerank_model=reranker, extractor=extractor, judger=judger)
        loader.load_data()
    else:
        raise NotImplementedError
    
    # Apply Filter (NEW)
    if args.filter:
        loader.samples = apply_qid_filter(loader.samples, args.filter)

    if args.limit and args.limit > 0:
        loader.samples = loader.samples[:args.limit]

    # 4. Agent
    agent = RAGAgent(
        loader=loader,
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        top_k=args.top_k,
        cache_dir=os.path.join(args.output_dir, "cache_agent"),
        use_page=args.use_page,
        use_crop=args.use_crop,
        use_ocr=args.use_ocr,
        use_ocr_raw=args.use_ocr_raw,
        trunc_thres=args.trunc_thres,
        trunc_bbox=args.trunc_bbox
    )

    return agent, loader