import json
import os
import collections
import ast
from bootstrap import parse_args, initialize_components, save_run_config
from src.loaders.base_loader import PageElement
from src.utils.llm_helper import create_llm_caller

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

    # 5. Save Bad Cases (NEW: Delegated to Loader)
    print("\n--- Saving Bad Cases for Analysis ---")
    if hasattr(loader, 'save_bad_cases'):
        loader.save_bad_cases(args.output_dir, args.evaluation_task)
    else:
        print("⚠️ Loader does not support saving bad cases.")


if __name__ == "__main__":
    main()