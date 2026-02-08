# run_generation.py
import json
import os
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
# Update import
from bootstrap import parse_args, initialize_components, save_run_config, apply_qid_filter
from src.loaders.base_loader import PageElement

def process_single_sample_generation(item, agent, cache_dir):
    """
    单个样本的生成处理函数，支持缓存读取
    """
    qid = str(item['qid'])
    query = item['query']
    
    # 处理特殊字符
    safe_qid = "".join([c if c.isalnum() else "_" for c in qid])
    cache_path = os.path.join(cache_dir, f"{safe_qid}.json")

    # 1. Check Cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass

    # 2. Run Generation
    # 反序列化 PageElement
    retrieved_elements_data = item.get('retrieved_elements', [])
    retrieved_elements = []
    for el_dict in retrieved_elements_data:
        valid_keys = PageElement.__annotations__.keys()
        filtered_dict = {k: v for k, v in el_dict.items() if k in valid_keys}
        retrieved_elements.append(PageElement(**filtered_dict))
    
    try:
        gen_output = agent.generate(query, retrieved_elements)
        
        # 更新结果
        item['model_answer'] = gen_output['final_answer']
        item['messages'] = gen_output['messages']
    except Exception as e:
        print(f"Error generating for {qid}: {e}")
        item['model_answer'] = "Error"
        item['messages'] = []

    # 3. Save Cache
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(item, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving cache for {qid}: {e}")

    return item

def main():
    args = parse_args()
    save_run_config(args, "generation")
    print(f"🚀 Starting Generation Stage for {args.benchmark} (Threads: {args.num_threads})...")
    
    # 这一步不需要 Retriever
    agent, _ = initialize_components(args, init_retriever=False, init_generator=True)
    
    # 读取检索阶段的结果
    retrieval_file = os.path.join(args.output_dir, "retrieval_results.json" if args.generation_input is None else args.generation_input)
    if not os.path.exists(retrieval_file):
        print(f"❌ Error: Retrieval file not found at {retrieval_file}. Run run_retrieval.py first.")
        return

    with open(retrieval_file, 'r', encoding='utf-8') as f:
        data_items = json.load(f)
    
    # Apply Filter (NEW)
    if args.filter:
        data_items = apply_qid_filter(data_items, args.filter)

    # 准备缓存目录
    cache_dir = os.path.join(args.output_dir, "cache_generation_results")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"📂 Cache directory: {cache_dir}")

    generation_results = []
    
    # 限制处理数量 (Debug 用)
    if args.limit and args.limit > 0:
        data_items = data_items[:args.limit]

    print(f"Generating answers for {len(data_items)} samples...")
    
    # 使用线程池
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        future_to_qid = {
            executor.submit(process_single_sample_generation, item, agent, cache_dir): item['qid']
            for item in data_items
        }
        
        for future in tqdm(as_completed(future_to_qid), total=len(data_items), desc="Generating"):
            try:
                result = future.result()
                if result:
                    generation_results.append(result)
            except Exception as e:
                print(f"Thread exception: {e}")

    # 排序
    try:
        generation_results.sort(key=lambda x: int(x['qid']) if str(x['qid']).isdigit() else str(x['qid']))
    except:
        pass

    # 保存最终结果
    output_file = os.path.join(args.output_dir, "generation_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(generation_results, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Generation complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()