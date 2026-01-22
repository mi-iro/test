import os
import json
import re
import argparse
import logging
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
from datetime import datetime
from collections import defaultdict
from glob import glob

def calculate_area(bbox):
    w = max(0, bbox[2] - bbox[0])
    h = max(0, bbox[3] - bbox[1])
    return w * h

def get_intersection_area(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    if x2 < x1 or y2 < y1: return 0.0
    return (x2 - x1) * (y2 - y1)

def calc_iou_min(pred, gt):
    area_p = calculate_area(pred)
    area_g = calculate_area(gt)
    if area_p == 0 or area_g == 0: return 0.0
    inter = get_intersection_area(pred, gt)
    return inter / min(area_p, area_g)

def calc_iou_standard(pred, gt):
    area_p = calculate_area(pred)
    area_g = calculate_area(gt)
    if area_p == 0 or area_g == 0: return 0.0
    inter = get_intersection_area(pred, gt)
    union = area_p + area_g - inter
    return inter / union if union > 0 else 0.0

def calculate_f_beta(precision, recall, beta):
    if precision + recall == 0: return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)

def eval_page_acc(pred_bboxes, gt_bboxes):
    """
    计算 Page Accuracy 指标：
    衡量页面级别的存在性预测是否正确（有无 GT vs 有无 Pred）
    """
    debug_logs = []
    if not pred_bboxes and not gt_bboxes:
        return 1.0, debug_logs
    elif not pred_bboxes and gt_bboxes:
        return 0.0, debug_logs
    elif not gt_bboxes and pred_bboxes:
        return 0.0, debug_logs
    else:
        # Both have bboxes
        return 1.0, debug_logs

def eval_single_sample_iou_min(pred_bboxes, gt_bboxes, threshold=0.75):
    if not pred_bboxes and not gt_bboxes: return 1.0, 1.0
    if not pred_bboxes: return 0.0, 0.0 
    if not gt_bboxes: return 0.0, 0.0   

    valid_preds = 0
    for p in pred_bboxes:
        hit = False
        for g in gt_bboxes:
            if calc_iou_min(p, g) > threshold:
                hit = True
                break
        if hit: valid_preds += 1
    precision = valid_preds / len(pred_bboxes)

    hit_gts = 0
    for g in gt_bboxes:
        hit = False
        for p in pred_bboxes:
            if calc_iou_min(p, g) > threshold:
                hit = True
                break
        if hit: hit_gts += 1
    recall = hit_gts / len(gt_bboxes)
    return precision, recall

def eval_single_sample_iou_em(pred_bboxes, gt_bboxes, threshold=0.6):
    if not pred_bboxes and not gt_bboxes: return 1.0, 1.0
    if not pred_bboxes: return 0.0, 0.0
    if not gt_bboxes: return 0.0, 0.0

    valid_pred_count = 0
    for p_box in pred_bboxes:
        is_valid = False
        for g_box in gt_bboxes:
            if calc_iou_standard(p_box, g_box) > threshold:
                is_valid = True
                break
        if is_valid: valid_pred_count += 1
    precision = valid_pred_count / len(pred_bboxes)

    hit_gt_count = 0
    for g_box in gt_bboxes:
        is_hit = False
        for p_box in pred_bboxes:
            if calc_iou_standard(p_box, g_box) > threshold:
                is_hit = True
                break
        if is_hit: hit_gt_count += 1
    recall = hit_gt_count / len(gt_bboxes)
    return precision, recall

def extract_bboxes_from_markdown_json(msg_content):
    try:
        bboxes = []
        json_objects = []
        # 策略 1: 正则查找所有的 ```json ... ``` 代码块
        code_blocks = re.findall(r"```json\s*(.*?)\s*```", msg_content, re.DOTALL)
        
        # 策略 2: 通用代码块
        if not code_blocks:
            code_blocks = re.findall(r"```\s*(.*?)\s*```", msg_content, re.DOTALL)
        
        if code_blocks:
            for block in code_blocks:
                try:
                    data = json.loads(block)
                    json_objects.append(data)
                except json.JSONDecodeError:
                    continue
        else:
            # 策略 3: 直接解析
            try:
                clean_str = msg_content.strip()
                if clean_str.startswith("```json"): clean_str = clean_str[7:]
                if clean_str.startswith("```"): clean_str = clean_str[3:]
                if clean_str.endswith("```"): clean_str = clean_str[:-3]
                data = json.loads(clean_str.strip())
                json_objects.append(data)
            except json.JSONDecodeError:
                pass

        for data in json_objects:
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "bbox" in item:
                        bboxes.append(item["bbox"])
            elif isinstance(data, dict) and "bbox" in data:
                bboxes.append(data["bbox"])
        return bboxes
    except Exception:
        return []

def convert_rel_bbox_to_abs(bbox, img_width, img_height):
    if not bbox or len(bbox) != 4: return [0, 0, 0, 0]
    x1, y1, x2, y2 = bbox
    return [
        int(max(0, (x1 / 1000.0) * img_width)), 
        int(max(0, (y1 / 1000.0) * img_height)), 
        int(min(img_width, (x2 / 1000.0) * img_width)), 
        int(min(img_height, (y2 / 1000.0) * img_height))
    ]

def visualize_sample(img_path, gt_bboxes, pred_bboxes, save_path):
    if not os.path.exists(img_path): return False
    try:
        img = cv2.imread(img_path)
        if img is None: return False
        h, w = img.shape[:2]
        
        # GT (Green)
        for box in gt_bboxes:
            x1, y1, x2, y2 = convert_rel_bbox_to_abs(box, w, h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "GT", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Pred (Red)
        for box in pred_bboxes:
            x1, y1, x2, y2 = convert_rel_bbox_to_abs(box, w, h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "Pred", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imwrite(save_path, img)
        return True
    except Exception:
        return False

def process_single_image_task(args_bundle):
    img_id, runs_list, gt_info, vis_root, betas = args_bundle
    gt_bboxes = gt_info['bboxes']
    img_path = gt_info['img_path']
    
    task_res_min = []
    task_res_em = []
    task_res_acc = []  # 存储 Accuracy 结果
    task_logs = []
    task_details = []

    id_vis_dir = os.path.join(vis_root, img_id)
    os.makedirs(id_vis_dir, exist_ok=True)

    for run_idx, run_item in enumerate(runs_list):
        if run_item is None: continue

        # 兼容 predictions 或 messages
        messages = run_item.get('predictions', run_item.get('messages', []))
        pred_bboxes = []
        
        if not messages:
            pass
        elif messages[-1]['role'] == 'user':
            pass 
        else:
            last_assistant_msg = None
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    last_assistant_msg = msg
                    break
            
            if last_assistant_msg:
                content = last_assistant_msg.get('content', '')
                pred_bboxes = extract_bboxes_from_markdown_json(content)
                

        # Metrics Calculation
        p_min, r_min = eval_single_sample_iou_min(pred_bboxes, gt_bboxes, threshold=0.75)
        p_em, r_em = eval_single_sample_iou_em(pred_bboxes, gt_bboxes, threshold=0.7)
        page_acc, _ = eval_page_acc(pred_bboxes, gt_bboxes) 

        min_f_scores = {f"F_{b}": calculate_f_beta(p_min, r_min, b) for b in betas}
        em_f_scores = {f"F_{b}": calculate_f_beta(p_em, r_em, b) for b in betas}
        
        # Log
        f_min_str = ", ".join([f"{k}={v:.4f}" for k, v in min_f_scores.items()])
        f_em_str = ", ".join([f"{k}={v:.4f}" for k, v in em_f_scores.items()])
        
        log_str = (
            f"\n--- [Eval Detail] ID: {img_id} | Run: {run_idx} ---\n"
            f"  GT BBoxes   : {gt_bboxes}\n"
            f"  Pred BBoxes : {pred_bboxes}\n"
            f"  [IoU_Min]   : Precision={p_min:.4f}, Recall={r_min:.4f} | {f_min_str}\n"
            f"  [IoU_EM]    : Precision={p_em:.4f}, Recall={r_em:.4f}  | {f_em_str}\n"
            f"  [Page_Acc]  : {page_acc:.4f}"
        )
        task_logs.append(log_str)
        
        task_res_min.append({'p': p_min, 'r': r_min})
        task_res_em.append({'p': p_em, 'r': r_em})
        task_res_acc.append({'acc': page_acc}) 

        # Detail JSON
        detail_obj = {
            "id": img_id,
            "run_index": run_idx,
            "Page_Acc": page_acc, 
            "IoU_Min": {
                "Precision": p_min,
                "Recall": r_min,
                **min_f_scores
            },
            "IoU_EM": {
                "Precision": p_em,
                "Recall": r_em,
                **em_f_scores
            }
        }
        task_details.append(detail_obj)
        
        # Vis
        vis_save_path = os.path.join(id_vis_dir, f"{run_idx}.png")
        visualize_sample(img_path, gt_bboxes, pred_bboxes, vis_save_path)
        
    return img_id, task_res_min, task_res_em, task_res_acc, task_logs, task_details

def plot_distribution(results_dict, beta_list, save_path, title_suffix=""):
    num_betas = len(beta_list)
    fig, axes = plt.subplots(1, num_betas, figsize=(5 * num_betas, 5))
    if num_betas == 1: axes = [axes]
    stats = {} 
    for idx, beta in enumerate(beta_list):
        f_scores = []
        for res in results_dict:
            f = calculate_f_beta(res['p'], res['r'], beta)
            f_scores.append(f)
        mean_f = np.mean(f_scores)
        stats[beta] = mean_f
        ax = axes[idx]
        sns.histplot(f_scores, bins=20, kde=True, ax=ax, color='salmon', edgecolor='black')
        ax.set_title(f'F_{beta} Dist ({title_suffix})\nAvg: {mean_f:.3f}')
        ax.set_xlabel('Score')
        ax.set_xlim(-0.1, 1.1)
    plt.suptitle(f'Score Distribution - {title_suffix}', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return stats

def main():
    parser = argparse.ArgumentParser(description="Parallel Eval Object Detection (Folder Input)")
    parser.add_argument('--pred_path', type=str, required=True, help="Folder containing JSON prediction files")
    parser.add_argument('--data_path', type=str, required=True, help="Original Dataset JSON path (GT source)")
    parser.add_argument('--output_dir', type=str, default="eval_result", help="Output directory")
    parser.add_argument('--betas', type=float, nargs='+', default=[1.0], help="List of Beta values") # [0.5, 1.0, 2.0]
    parser.add_argument('--workers', type=int, default=None, help="Number of CPU cores to use")
    
    args = parser.parse_args()

    vis_root = os.path.join(args.output_dir, "vis_diff")
    os.makedirs(vis_root, exist_ok=True)
    log_path = os.path.join(args.output_dir, "eval.log")
    
    with open(log_path, 'w') as f:
        f.write(f"Evaluation started at {datetime.now()}\nConfig: {args}\n")

    print("Loading GT data...")
    gt_map = {} 
    try:
        with open(args.data_path, 'r') as f:
            gt_data = json.load(f)
            for item in gt_data:
                img_id = item['id']
                img_path = item.get('images', [""])[0] if item.get('images') else ""
                conversations = item.get('conversations', [])
                gt_bboxes = []
                if conversations and conversations[-1]['role'] == 'assistant':
                    gt_bboxes = extract_bboxes_from_markdown_json(conversations[-1]['content'])
                gt_map[img_id] = {'bboxes': gt_bboxes, 'img_path': img_path}
    except Exception as e:
        print(f"Error loading GT: {e}")
        return

    print(f"Loading Pred data from directory: {args.pred_path}")
    preds_data_map = defaultdict(list)
    
    if os.path.isdir(args.pred_path):
        files = glob(os.path.join(args.pred_path, "*.json"))
        try:
            files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        except ValueError:
            files.sort()
            
        print(f"Found {len(files)} JSON files: {[os.path.basename(f) for f in files]}")
        for json_file in files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if 'id' in item:
                                preds_data_map[item['id']].append(item)
                    elif isinstance(data, dict):
                         for k, v in data.items():
                             if isinstance(v, list):
                                 preds_data_map[k].extend(v)
                             else:
                                 preds_data_map[k].append(v)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    else:
        print("Input is a file, loading directly...")
        try:
            with open(args.pred_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if 'id' in item:
                            preds_data_map[item['id']].append(item)
        except Exception as e:
             print(f"Error loading file: {e}")
             return

    task_args_list = []
    print("Preparing tasks...")
    for img_id, runs_list in preds_data_map.items():
        if img_id not in gt_map: continue
        task_args = (img_id, runs_list, gt_map[img_id], vis_root, args.betas)
        task_args_list.append(task_args)

    workers = args.workers if args.workers else multiprocessing.cpu_count()
    print(f"Starting execution with {workers} workers for {len(task_args_list)} tasks...")
    
    per_id_results_min = {}
    per_id_results_em = {}
    per_id_results_acc = {} 
    all_detailed_results = []
    
    log_file = open(log_path, 'a')

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_image_task, task) for task in task_args_list]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                r_img_id, r_min, r_em, r_acc, r_logs, r_details = future.result()
                
                per_id_results_min[r_img_id] = r_min
                per_id_results_em[r_img_id] = r_em
                per_id_results_acc[r_img_id] = r_acc # 收集 Acc
                
                for log_line in r_logs:
                    log_file.write(log_line + "\n")
                
                all_detailed_results.extend(r_details)

            except Exception as e:
                print(f"Task failed: {e}")

    log_file.close()
    
    # Save Detailed JSON
    all_detailed_results.sort(key=lambda x: x['id'])
    detailed_json_path = os.path.join(args.output_dir, "detailed_metrics.json")
    print(f"Saving detailed metrics to {detailed_json_path}...")
    with open(detailed_json_path, 'w') as f:
        json.dump(all_detailed_results, f, indent=4)

    # Aggregate & Report
    print("Aggregating summary results...")
    def aggregate_results(id_results_map, strategy='avg', beta=1.0):
        final_list = []
        for runs in id_results_map.values():
            if not runs: continue
            if strategy == 'avg':
                avg_p = np.mean([x['p'] for x in runs])
                avg_r = np.mean([x['r'] for x in runs])
                final_list.append({'p': avg_p, 'r': avg_r})
            elif strategy == 'max':
                best_run = max(runs, key=lambda x: calculate_f_beta(x['p'], x['r'], beta))
                final_list.append(best_run)
        return final_list

    # 聚合 Acc (Avg 和 Max)
    def aggregate_acc_stats(id_results_map):
        avg_of_means = []
        avg_of_maxs = []
        
        for runs in id_results_map.values():
            if not runs: continue 
            raw_accs = [x['acc'] for x in runs]
            avg_of_means.append(np.mean(raw_accs))
            avg_of_maxs.append(np.max(raw_accs))
            
        final_mean = np.mean(avg_of_means) if avg_of_means else 0.0
        final_max = np.mean(avg_of_maxs) if avg_of_maxs else 0.0
        return final_mean, final_max

    agg_min_avg = aggregate_results(per_id_results_min, 'avg')
    agg_min_max = aggregate_results(per_id_results_min, 'max', beta=1.0)
    agg_em_avg = aggregate_results(per_id_results_em, 'avg')
    agg_em_max = aggregate_results(per_id_results_em, 'max', beta=1.0)
    
    # 计算最终 Page Acc 的 Mean 和 Max
    final_acc_mean, final_acc_max = aggregate_acc_stats(per_id_results_acc)

    print("Generating plots...")
    plot_distribution(agg_min_avg, args.betas, os.path.join(args.output_dir, "dist_iou_min_avg.png"), "IoU_Min (Avg)")
    plot_distribution(agg_em_avg, args.betas, os.path.join(args.output_dir, "dist_iou_em_avg.png"), "IoU_EM (Avg)")

    table = PrettyTable()
    header = ["Metric Strategy", "Precision", "Recall"] + [f"F_{b}" for b in args.betas]
    table.field_names = header

    def add_row_to_table(name, agg_list):
        if not agg_list:
            table.add_row([name, "N/A", "N/A"] + ["N/A"]*len(args.betas))
            return
        avg_p = np.mean([x['p'] for x in agg_list])
        avg_r = np.mean([x['r'] for x in agg_list])
        f_vals = []
        for b in args.betas:
            f_scores = [calculate_f_beta(x['p'], x['r'], b) for x in agg_list]
            f_vals.append(np.mean(f_scores))
        row = [name, f"{avg_p:.4f}", f"{avg_r:.4f}"]
        for f in f_vals:
            row.append(f"{f:.4f}")
        table.add_row(row)

    add_row_to_table("IoU_Min (Avg over runs)", agg_min_avg)
    add_row_to_table("IoU_Min (Max of runs)",   agg_min_max)
    add_row_to_table("IoU_EM  (Avg over runs)", agg_em_avg)
    add_row_to_table("IoU_EM  (Max of runs)",   agg_em_max)
    
    # 添加 Page Acc 的两行 (Avg 和 Max)
    table.add_row(["Page Acc (Avg over runs)", f"{final_acc_mean:.4f}", "-"] + ["-"]*len(args.betas))
    table.add_row(["Page Acc (Max of runs)",   f"{final_acc_max:.4f}",  "-"] + ["-"]*len(args.betas))

    print("\n" + str(table))
    with open(log_path, 'a') as f:
        f.write("\nFinal Report:\n" + str(table) + "\n")

if __name__ == "__main__":
    main()