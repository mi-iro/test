import abc
import json
import re
import string
import collections
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, Dict, Any

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
    corpus_id: str = "" # 元素所属页面ID
    
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
        执行评估：计算 QA 指标、页面检索指标和元素提取指标。
        评估结果将存储在每个 sample.extra_info['metrics'] 中。
        依赖 sample.extra_info 包含 'final_answer' 和可选的 'retrieved_elements'。
        
        Returns:
            Dict[str, float]: 整个数据集的平均指标 (avg_qa_f1, avg_qa_em, avg_page_recall, avg_element_iou 等)
        """
        total_metrics = collections.defaultdict(float)
        counts = collections.defaultdict(int)

        for sample in self.samples:
            if sample.extra_info is None:
                sample.extra_info = {}

            # --- 获取预测结果 ---
            pred_answer = sample.extra_info.get('final_answer', "")
            
            # 尝试获取预测的 elements (兼容 dict 列表或 PageElement 对象列表)
            raw_elements = sample.extra_info.get('retrieved_elements', [])
            pred_elements = []
            for el in raw_elements:
                if isinstance(el, dict):
                    pred_elements.append(PageElement(**{k:v for k,v in el.items() if k in PageElement.__annotations__}))
                elif isinstance(el, PageElement):
                    pred_elements.append(el)
            
            metrics_result = {}

            # 1. 计算 QA 指标 (Text Generation)
            if sample.gold_answer:
                qa_score = self._compute_qa_metrics(pred_answer, sample.gold_answer)
                metrics_result['qa'] = qa_score
                total_metrics['qa_f1'] += qa_score['f1']
                total_metrics['qa_em'] += qa_score['em']
                counts['qa'] += 1

            # 2. 计算 页面检索 指标 (Page Retrieval)
            # 仅当有 gold_pages 真值时计算
            if sample.gold_pages:
                page_score = self._compute_page_metrics(pred_elements, sample.gold_pages)
                metrics_result['page'] = page_score
                total_metrics['page_recall'] += page_score['recall']
                total_metrics['page_precision'] += page_score['precision']
                counts['page'] += 1

            # 3. 计算 元素提取 指标 (Element Extraction / BBox IoU)
            # 仅当有 gold_elements 真值时计算
            if sample.gold_elements:
                elem_score = self._compute_element_metrics(pred_elements, sample.gold_elements)
                metrics_result['element'] = elem_score
                total_metrics['element_iou'] += elem_score['mean_iou']
                counts['element'] += 1

            # 存储回 sample
            sample.extra_info['metrics'] = metrics_result

        # --- 汇总平均值 ---
        avg_results = {}
        if counts['qa'] > 0:
            avg_results['avg_qa_f1'] = total_metrics['qa_f1'] / counts['qa']
            avg_results['avg_qa_em'] = total_metrics['qa_em'] / counts['qa']
        
        if counts['page'] > 0:
            avg_results['avg_page_recall'] = total_metrics['page_recall'] / counts['page']
            avg_results['avg_page_precision'] = total_metrics['page_precision'] / counts['page']
            
        if counts['element'] > 0:
            avg_results['avg_element_iou'] = total_metrics['element_iou'] / counts['element']

        return avg_results

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
            return {"recall": 0.0, "precision": 0.0}
        
        hits = pred_page_ids & gt_page_ids
        recall = len(hits) / len(gt_page_ids)
        precision = len(hits) / len(pred_page_ids) if pred_page_ids else 0.0
        
        return {"recall": recall, "precision": precision}

    def _compute_iou(self, box1: List[int], box2: List[int]) -> float:
        """计算两个 BBox [x1, y1, x2, y2] 的 IoU"""
        # 确保 box 格式正确
        if len(box1) != 4 or len(box2) != 4:
            return 0.0
            
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - intersection_area
        
        if union_area <= 0:
            return 0.0
            
        return intersection_area / union_area

    def _compute_element_metrics(self, pred_elements: List[PageElement], gold_elements: List[PageElement]) -> Dict[str, float]:
        """
        计算元素级别的 IoU。
        逻辑：对于每个 GT BBox，找到预测中 IoU 最大的 BBox，计算这些 Max IoU 的平均值 (Recall-oriented IoU)。
        """
        if not gold_elements:
            return {"mean_iou": 0.0}
            
        total_max_iou = 0.0
        
        for gt in gold_elements:
            gt_bbox = gt.bbox
            if not gt_bbox or len(gt_bbox) != 4:
                continue
                
            max_iou_for_this_gt = 0.0
            for pred in pred_elements:
                pred_bbox = pred.bbox
                # 必须在同一页才计算 IoU
                if pred.corpus_id != gt.corpus_id:
                    continue
                    
                iou = self._compute_iou(gt_bbox, pred_bbox)
                if iou > max_iou_for_this_gt:
                    max_iou_for_this_gt = iou
            
            total_max_iou += max_iou_for_this_gt
            
        mean_iou = total_max_iou / len(gold_elements)
        return {"mean_iou": mean_iou}