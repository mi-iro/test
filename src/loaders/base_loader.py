import abc
import json
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