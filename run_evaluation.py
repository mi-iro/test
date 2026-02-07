# run_evaluation.py

import json
import os
import collections
from bootstrap import parse_args, initialize_components, save_run_config
from src.loaders.base_loader import PageElement
from src.utils.llm_helper import create_llm_caller

def serialize_element(elem):
    """Helper to convert PageElement or dict to JSON-serializable dict"""
    if isinstance(elem, dict):
        return elem
    if hasattr(elem, '__dict__'):
        return elem.__dict__
    return str(elem)

def save_bad_cases(loader, output_dir, task):
    """
    Filter and save bad cases based on evaluation metrics.
    """
    bad_case_dir = os.path.join(output_dir, "bad_cases")
    os.makedirs(bad_case_dir, exist_ok=True)

    retrieval_bad_cases = []
    generation_bad_cases = []

    print(f"🔍 Analyzing {len(loader.samples)} samples for bad cases...")

    for sample in loader.samples:
        if sample.extra_info is None:
            continue
        
        metrics = sample.extra_info.get('metrics', {})
        
        # Flatten metrics for easier consumption in case_study.py
        # Loader structure: metrics['page'] = {'recall': x, ...}
        # Target structure: metrics['page_recall'] = x
        flat_metrics = metrics.copy()
        
        if 'page' in metrics and isinstance(metrics['page'], dict):
            flat_metrics['page_recall'] = metrics['page'].get('recall', 0.0)
            flat_metrics['page_precision'] = metrics['page'].get('precision', 0.0)
        
        # 1. Retrieval Bad Case Check
        # Criteria: Recall < 1.0 (some gold pages missed)
        is_retrieval_bad = False
        if 'page_recall' in flat_metrics and flat_metrics['page_recall'] < 1.0:
            is_retrieval_bad = True
        elif 'page' in metrics and metrics['page'].get('recall', 1.0) < 1.0:
            # Double check raw structure
            is_retrieval_bad = True
            flat_metrics['page_recall'] = metrics['page'].get('recall')
            flat_metrics['page_precision'] = metrics['page'].get('precision')

        # 2. Generation Bad Case Check
        # Criteria: model_eval < 1.0 (answer not perfect)
        is_generation_bad = False
        if 'model_eval' in metrics and metrics['model_eval'] < 1.0:
            is_generation_bad = True

        # Construct Serializable Sample Object
        # We assume retrieving 'final_answer' and 'retrieved_elements' from extra_info
        
        sample_dict = {
            "qid": str(sample.qid),
            "query": sample.query,
            "gold_answer": sample.gold_answer,
            "gold_pages": sample.gold_pages,
            "final_answer": sample.extra_info.get("final_answer", ""),
            "metrics": flat_metrics, # Use flattened metrics
            "retrieved_elements": [serialize_element(e) for e in sample.extra_info.get("retrieved_elements", [])],
            "doc_source": sample.data_source
        }

        if is_retrieval_bad:
            retrieval_bad_cases.append(sample_dict)
        
        if is_generation_bad:
            generation_bad_cases.append(sample_dict)

    # Save files
    if task in ["retrieval", "all"] and retrieval_bad_cases:
        p = os.path.join(bad_case_dir, "retrieval_bad_cases.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(retrieval_bad_cases, f, indent=2, ensure_ascii=False)
        print(f"📉 Saved {len(retrieval_bad_cases)} retrieval bad cases to {p}")

    if task in ["generation", "all"] and generation_bad_cases:
        p = os.path.join(bad_case_dir, "generation_bad_cases.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(generation_bad_cases, f, indent=2, ensure_ascii=False)
        print(f"📉 Saved {len(generation_bad_cases)} generation bad cases to {p}")


def main():
    args = parse_args()
    save_run_config(args, "evaluation")
    print(f"🚀 Starting Evaluation Stage for {args.benchmark} (Task: {args.evaluation_task})...")
    
    # 初始化 Loader
    _, loader = initialize_components(args, init_retriever=False, init_generator=False)
    loader.llm_caller = create_llm_caller()
    
    # 1. 确定输入文件
    input_file = args.evaluation_input
    if input_file is None:
        if args.evaluation_task == "retrieval":
            p1 = os.path.join(args.output_dir, "retrieval_results.json")
            p2 = os.path.join(args.output_dir, "generation_results.json")
            input_file = p1 if os.path.exists(p1) else p2
        else:
            input_file = os.path.join(args.output_dir, "generation_results.json")
    else:
        input_file = os.path.join(args.output_dir, input_file)
    
    if not input_file or not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        return

    print(f"📂 Loading results from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
        
    # 2. 将结果映射回 Loader 的 samples
    results_map = {str(item['qid']): item for item in results_data}
    
    matched_count = 0
    for sample in loader.samples:
        s_qid = str(sample.qid)
        if s_qid in results_map:
            res = results_map[s_qid]
            if sample.extra_info is None:
                sample.extra_info = {}
            
            if 'retrieved_elements' in res:
                sample.extra_info['retrieved_elements'] = res['retrieved_elements']
            
            # Map answer keys
            if 'model_answer' in res:
                sample.extra_info['final_answer'] = res['model_answer']
            elif 'final_answer' in res: 
                sample.extra_info['final_answer'] = res['final_answer']

            matched_count += 1
            
    print(f"✅ Mapped results for {matched_count}/{len(loader.samples)} samples.")
    
    final_metrics = {}

    # 3. 执行评估
    
    # Task: Retrieval
    if args.evaluation_task in ["retrieval", "all"]:
        try:
            print("\n--- Retrieval Metrics ---")
            r_metrics = loader.evaluate_retrieval()
            print(json.dumps(r_metrics, indent=2))
            final_metrics.update(r_metrics)
        except Exception as e:
            print(f"⚠️ Retrieval evaluation failed: {e}")

    # Task: Generation
    if args.evaluation_task in ["generation", "all"]:
        has_answers = any("final_answer" in s.extra_info for s in loader.samples if str(s.qid) in results_map)
        if has_answers:
            try:
                print("\n--- Generation Metrics ---")
                g_metrics = loader.evaluate_generation(num_threads=args.num_threads)
                print(json.dumps(g_metrics, indent=2))
                final_metrics.update(g_metrics)
            except Exception as e:
                print(f"⚠️ Generation evaluation failed: {e}")
        else:
            if args.evaluation_task == "generation":
                print("⚠️ Warning: No generation answers found in input file. Skipping generation eval.")

    # 4. 保存评估报告
    output_path = os.path.join(args.output_dir, f"evaluation_metrics_{args.evaluation_task}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\n💾 All metrics saved to {output_path}")

    # 5. 分析并保存 Bad Cases (NEW)
    print("\n--- Saving Bad Cases for Analysis ---")
    save_bad_cases(loader, args.output_dir, args.evaluation_task)


if __name__ == "__main__":
    main()