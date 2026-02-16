import json
import os
import collections
import ast
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

def sanitize_filename(name):
    """Sanitize the source name to be safe for filenames."""
    return "".join([c if c.isalnum() else "_" for c in str(name)])

def save_bad_cases(loader, output_dir, task):
    """
    Filter and save bad cases based on evaluation metrics.
    Saves separate JSON files for each Evidence Source.
    """
    bad_case_dir = os.path.join(output_dir, "bad_cases")
    os.makedirs(bad_case_dir, exist_ok=True)

    # Global lists for all cases
    retrieval_bad_cases = []
    generation_bad_cases = []

    # Dictionary to hold lists separated by source
    retrieval_by_source = collections.defaultdict(list)
    generation_by_source = collections.defaultdict(list)

    print(f"🔍 Analyzing {len(loader.samples)} samples for bad cases...")

    for sample in loader.samples:
        if sample.extra_info is None:
            continue
        
        # --- Parse Evidence Sources (Logic from MMLongLoader.py) ---
        ev_sources_str = sample.extra_info.get('evidence_sources', "[]")
        try:
            ev_sources = ast.literal_eval(str(ev_sources_str))
            if not isinstance(ev_sources, list):
                ev_sources = [str(ev_sources)]
        except:
            ev_sources = ["Unknown"]
        
        # Add "Not Answerable" if sources are empty
        if not ev_sources:
            ev_sources = ["Not Answerable"]
        # --------------------------------------------------------

        metrics = sample.extra_info.get('metrics', {})
        
        # Flatten metrics
        flat_metrics = metrics.copy()
        if 'page' in metrics and isinstance(metrics['page'], dict):
            flat_metrics['page_recall'] = metrics['page'].get('recall', 0.0)
            flat_metrics['page_precision'] = metrics['page'].get('precision', 0.0)
        
        # 1. Retrieval Bad Case Check
        is_retrieval_bad = False
        if 'page_recall' in flat_metrics and flat_metrics['page_recall'] < 1.0:
            is_retrieval_bad = True
        elif 'page' in metrics and metrics['page'].get('recall', 1.0) < 1.0:
            is_retrieval_bad = True
            flat_metrics['page_recall'] = metrics['page'].get('recall')
            flat_metrics['page_precision'] = metrics['page'].get('precision')

        # 2. Generation Bad Case Check
        is_generation_bad = False
        if 'model_eval' in metrics and metrics['model_eval'] < 1.0:
            is_generation_bad = True

        # Construct Serializable Sample Object
        sample_dict = {
            "qid": str(sample.qid),
            "query": sample.query,
            "gold_answer": sample.gold_answer,
            "gold_pages": sample.gold_pages,
            "evidence_sources": ev_sources,
            "final_answer": sample.extra_info.get("final_answer", ""),
            "metrics": flat_metrics,
            "retrieved_elements": [serialize_element(e) for e in sample.extra_info.get("retrieved_elements", [])],
            "doc_source": sample.data_source
        }

        # Add to collections
        if is_retrieval_bad:
            retrieval_bad_cases.append(sample_dict)
            for src in ev_sources:
                retrieval_by_source[src].append(sample_dict)
        
        if is_generation_bad:
            generation_bad_cases.append(sample_dict)
            for src in ev_sources:
                generation_by_source[src].append(sample_dict)

    # --- Save Retrieval Bad Cases ---
    if task in ["retrieval", "all"] and retrieval_bad_cases:
        # 1. Save All
        p_all = os.path.join(bad_case_dir, "retrieval_bad_cases_all.json")
        with open(p_all, "w", encoding="utf-8") as f:
            json.dump(retrieval_bad_cases, f, indent=2, ensure_ascii=False)
        print(f"📉 Saved {len(retrieval_bad_cases)} total retrieval bad cases to {p_all}")

        # 2. Save per Source
        for source, cases in retrieval_by_source.items():
            safe_name = sanitize_filename(source)
            p_src = os.path.join(bad_case_dir, f"retrieval_bad_cases_{safe_name}.json")
            with open(p_src, "w", encoding="utf-8") as f:
                json.dump(cases, f, indent=2, ensure_ascii=False)
            print(f"   └─ Saved {len(cases)} cases for source '{source}' to {os.path.basename(p_src)}")

    # --- Save Generation Bad Cases ---
    if task in ["generation", "all"] and generation_bad_cases:
        # 1. Save All
        p_all = os.path.join(bad_case_dir, "generation_bad_cases_all.json")
        with open(p_all, "w", encoding="utf-8") as f:
            json.dump(generation_bad_cases, f, indent=2, ensure_ascii=False)
        print(f"📉 Saved {len(generation_bad_cases)} total generation bad cases to {p_all}")

        # 2. Save per Source
        for source, cases in generation_by_source.items():
            safe_name = sanitize_filename(source)
            p_src = os.path.join(bad_case_dir, f"generation_bad_cases_{safe_name}.json")
            with open(p_src, "w", encoding="utf-8") as f:
                json.dump(cases, f, indent=2, ensure_ascii=False)
            print(f"   └─ Saved {len(cases)} cases for source '{source}' to {os.path.basename(p_src)}")


def main():
    args = parse_args()
    save_run_config(args, "evaluation")
    print(f"🚀 Starting Evaluation Stage for {args.benchmark} (Task: {args.evaluation_task})...")
    
    # Initialize Loader
    _, loader = initialize_components(args, init_retriever=False, init_generator=False)
    loader.llm_caller = create_llm_caller()
    
    # 1. Determine input file
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
        
    # 2. Map results back to Loader samples
    results_map = {str(item['qid']): item for item in results_data}
    
    valid_samples = []
    matched_count = 0
    original_count = len(loader.samples)

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

            # Map Token Stats
            sample.extra_info['prompt_tokens'] = res.get('prompt_tokens', 0)
            sample.extra_info['completion_tokens'] = res.get('completion_tokens', 0)

            matched_count += 1
            valid_samples.append(sample)
    
    # --- Strict Filtering ---
    loader.samples = valid_samples
    print(f"✅ Mapped results for {matched_count}/{original_count} samples. (Discarded {original_count - matched_count} unmatched samples)")
    
    final_metrics = {}

    # 3. Execute Evaluation
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
        has_answers = any("final_answer" in s.extra_info for s in loader.samples)
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

    # 4. Save Metrics Report
    output_path = os.path.join(args.output_dir, f"evaluation_metrics_{args.evaluation_task}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\n💾 All metrics saved to {output_path}")

    # 5. Save Bad Cases (NEW: Split by source)
    print("\n--- Saving Bad Cases for Analysis ---")
    save_bad_cases(loader, args.output_dir, args.evaluation_task)


if __name__ == "__main__":
    main()