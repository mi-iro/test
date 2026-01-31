import abc
import json
import re
import string
import collections
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, Dict, Any, Tuple

@dataclass
class PageElement:
    """
    定义文档/页面中的最小检索单元。
    可以是文本段落、表格行、或者图像区域。
    """
    
    # Bounding Box 格式: [x_min, y_min, x_max, y_max]
    # 强烈建议使用归一化坐标 (0 - 1000)，以适应不同分辨率的图像
    bbox: List[int] = field(default_factory=list) 
    type: str = "text"          # 'text', 'table', 'image', 'chart' 等，可以是None
    content: str = ""                # 文本内容 (如果是纯图，可以是OCR结果或图像描述)
    raw_content: str = ""
    corpus_id: str = "" # 元素所属页面ID
    corpus_path: str = ""
    
    # 预留字段，用于存储该元素对应的原始图像裁剪路径，用于视觉模型输入
    crop_path: Optional[str] = None 

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class StandardSample:
    """
    定义一个标准的输入样本。
    所有数据集 (MMLong, FinRAG, MVTool) 都必须转换为此格式。
    """
    qid: str
    query: str
    dataset: str         # 'mmlong', 'finrag', 'mvtool'
    # --- 数据源 --- 可能是向量检索池，可能是单个PDF文档，也可能是单个图像
    data_source: str     # '.index', '.pdf', '.png'
    
    # --- Ground Truth (用于评估) ---
    gold_answer: Optional[str] = None
    
    # 答案对应的 BBox 真值 (可能有多个区域)
    gold_elements: List[PageElement] = field(default_factory=list)
    
    # 答案对应的 页面ID 真值 (可能有多个页面)
    gold_pages: List[str] = field(default_factory=list)
    
    extra_info: Optional[dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典，方便写入日志"""
        return asdict(self)

# --- Metrics Calculation Helpers (Ported from eval.py) ---

def calculate_area(bbox: List[int]) -> int:
    """计算 BBox 面积"""
    w = max(0, bbox[2] - bbox[0])
    h = max(0, bbox[3] - bbox[1])
    return w * h

def get_intersection_area(bbox1: List[int], bbox2: List[int]) -> int:
    """计算两个 BBox 的交集面积"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    if x2 < x1 or y2 < y1: return 0
    return (x2 - x1) * (y2 - y1)

def calc_iou_min(pred: List[int], gt: List[int]) -> float:
    """计算 Intersection over Minimum Area (适合包含关系检测)"""
    area_p = calculate_area(pred)
    area_g = calculate_area(gt)
    if area_p == 0 or area_g == 0: return 0.0
    inter = get_intersection_area(pred, gt)
    return inter / min(area_p, area_g)

def calc_iou_standard(pred: List[int], gt: List[int]) -> float:
    """计算标准的 Intersection over Union"""
    area_p = calculate_area(pred)
    area_g = calculate_area(gt)
    if area_p == 0 or area_g == 0: return 0.0
    inter = get_intersection_area(pred, gt)
    union = area_p + area_g - inter
    return inter / union if union > 0 else 0.0

def calculate_f_beta(precision: float, recall: float, beta: float = 1.0) -> float:
    """计算 F-beta 分数"""
    if precision + recall == 0: return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)


class BaseDataLoader(abc.ABC):
    """
    所有数据集 Loader 的抽象基类。
    强制子类实现 load_data 方法。
    """
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.samples: List[StandardSample] = []

    @abc.abstractmethod
    def load_data(self) -> None:
        """
        核心逻辑：读取原始数据，并填充 self.samples 列表。
        必须由子类 (e.g., FinRAGLoader) 实现。
        """
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> StandardSample:
        return self.samples[idx]

    def get_batch(self, batch_size: int):
        """简单的 Batch 生成器"""
        for i in range(0, len(self.samples), batch_size):
            yield self.samples[i : i + batch_size]
            
    def pipeline(self, query: str, image_paths: List[str], top_k: int) -> List[PageElement]:
        """
        根据查询检索页面元素。
        """
        raise NotImplementedError
    
    def evaluate(self) -> Dict[str, float]:
        """
        执行评估
        """
        raise NotImplementedError
        
    # --- 内部评估辅助函数 ---

    def _normalize_text(self, s: str) -> str:
        """标准化文本：小写、去标点、去多余空格"""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
            
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _compute_qa_metrics(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """计算 F1 和 Exact Match (EM)"""
        pred_norm = self._normalize_text(prediction)
        gt_norm = self._normalize_text(ground_truth)
        
        # EM
        em = 1.0 if pred_norm == gt_norm else 0.0
        
        # F1
        prediction_tokens = pred_norm.split()
        ground_truth_tokens = gt_norm.split()
        common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            f1 = 0.0
        else:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            
        return {"f1": f1, "em": em}

    def _compute_page_metrics(self, pred_elements: List[PageElement], gold_pages: List[str]) -> Dict[str, float]:
        """计算页面检索的 Recall 和 Precision"""
        # 提取预测中出现的所有页面 ID (去重)
        # 注意：这里需要确保 corpus_id 的格式与 gold_pages 一致（通常是文件名或相对路径）
        pred_page_ids = set([el.corpus_id for el in pred_elements if el.corpus_id])
        gt_page_ids = set(gold_pages)
        
        if not gt_page_ids:
            return {"recall": 1.0, "precision": 1.0 if len(pred_page_ids) == 0 else 0.0}
        
        hits = pred_page_ids & gt_page_ids
        recall = len(hits) / len(gt_page_ids)
        precision = len(hits) / len(pred_page_ids) if pred_page_ids else 0.0
        
        return {"recall": recall, "precision": precision}

    def _compute_page_accuracy(self, pred_bboxes: List[List[int]], gt_bboxes: List[List[int]]) -> float:
        """
        计算 Page Accuracy:
        衡量页面级别的存在性预测是否正确（有无 GT vs 有无 Pred）。
        逻辑同 eval.py:
        - 如果都没有: Correct (1.0)
        - 如果都有: Correct (1.0) - 此处定义较宽松，只要预测了且GT存在就算“命中”任务类型，具体质量由IoU衡量
        - 只有一个有: Incorrect (0.0)
        """
        if not pred_bboxes and not gt_bboxes:
            return 1.0
        elif not pred_bboxes and gt_bboxes:
            return 0.0
        elif not gt_bboxes and pred_bboxes:
            return 0.0
        else:
            return 1.0

    def _compute_detection_metrics(self, pred_bboxes: List[List[int]], gt_bboxes: List[List[int]], 
                                 iou_func, threshold: float) -> Tuple[float, float]:
        """
        计算基于特定 IoU 函数和阈值的 Precision 和 Recall。
        """
        if not pred_bboxes and not gt_bboxes: return 1.0, 1.0
        if not pred_bboxes: return 1.0, 0.0 
        if not gt_bboxes: return 0.0, 1.0   

        # Precision Calculation
        valid_preds = 0
        for p in pred_bboxes:
            hit = False
            for g in gt_bboxes:
                if iou_func(p, g) > threshold:
                    hit = True
                    break
            if hit: valid_preds += 1
        precision = valid_preds / len(pred_bboxes)

        # Recall Calculation
        hit_gts = 0
        for g in gt_bboxes:
            hit = False
            for p in pred_bboxes:
                if iou_func(p, g) > threshold:
                    hit = True
                    break
            if hit: hit_gts += 1
        recall = hit_gts / len(gt_bboxes)
        
        return precision, recall