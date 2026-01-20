import os
import sys
from typing import List, Dict, Any, Union

# 假设文件结构，确保可以导入 src 下的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.loaders.base_loader import BaseDataLoader, StandardSample, PageElement
from src.loaders.FinRAGLoader import FinRAGLoader
from src.loaders.MVToolLoader import MVToolLoader
from src.loaders.MMLongLoader import MMLongLoader

class RAGAgent:
    """
    通用 RAG Agent，用于驱动不同的 Loader 执行 Pipeline 任务。
    它负责：
    1. 接收一个具体的数据集 Loader (FinRAG, MVTool, MMLong)。
    2. 遍历或处理 StandardSample。
    3. 根据 Loader 类型调用对应的 pipeline 接口。
    4. 将提取到的证据/答案汇总，并记录到 extra_info 字段。
    """

    def __init__(self, loader: BaseDataLoader):
        """
        初始化 Agent。
        :param loader: 已实例化的 Loader 对象 (如 FinRAGLoader, MMLongLoader 等)
        """
        self.loader = loader

    def format_answer(self, elements: List[PageElement]) -> str:
        """
        将 Pipeline 返回的多个 PageElement 整理为最终的文本答案。
        """
        if not elements:
            return "No relevant information found."
        
        # 简单拼接所有证据内容，实际场景可接入 LLM 做最终摘要生成
        answers = []
        for i, el in enumerate(elements):
            content = el.content.strip()
            if content:
                answers.append(f"Evidence {i+1}: {content}")
        
        return "\n".join(answers)

    def process_sample(self, sample: StandardSample) -> StandardSample:
        """
        处理单个样本：执行 RAG Pipeline 并更新 extra_info。
        """
        query = sample.query
        retrieved_elements: List[PageElement] = []

        try:
            source_path = sample.data_source
            if source_path:
                print(f"Running Document/Image pipeline on: {source_path}")
                retrieved_elements = self.loader.pipeline(query=query, image_paths=[source_path])
            else:
                print(f"Warning: No data_source found for sample {sample.qid}")
        except Exception as e:
            print(f"Error processing sample {sample.qid}: {e}")
            retrieved_elements = []

        # 格式化最终答案
        final_answer = self.format_answer(retrieved_elements)

        # 记录到 extra_info
        if sample.extra_info is None:
            sample.extra_info = {}
        
        # 写入查询内容和最终答案
        sample.extra_info['query_content'] = query
        sample.extra_info['final_answer'] = final_answer
        
        # 可选：也将提取到的原始 Elements 记录下来以便调试
        sample.extra_info['retrieved_elements_count'] = len(retrieved_elements)

        return sample

    def run_batch(self, batch_size: int = 1):
        """
        批量运行 Loader 中加载的所有数据。
        """
        if len(self.loader.samples) == 0:
            print("Loader has no data. Please call loader.load_data() first.")
            return

        print(f"Starting RAG task on {len(self.loader.samples)} samples...")
        
        for sample in self.loader.samples:
            self.process_sample(sample)
            print(f"Sample {sample.qid} processed. Answer: {sample.extra_info['final_answer'][:50]}...")

# --- 使用示例 ---
if __name__ == "__main__":
    # 假设环境配置
    from src.agents.ElementExtractor import ElementExtractor
    from src.agents.utils import ImageZoomOCRTool
    import torch

    # 1. 配置工具和 Extractor (Mock 配置)
    tool_work_dir = "./workspace"
    tool = ImageZoomOCRTool(work_dir=tool_work_dir)
    extractor = ElementExtractor(
        base_url="http://localhost:8001/v1",
        api_key="sk-123456",
        model_name="MinerU-Agent-CK300",
        tool=tool
    )

    # 2. 示例：使用 MMLongLoader
    # data_root 指向包含 data/samples.json 的目录
    mmlong_root = "/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/MMLongBench-Doc" 
    
    # 确保路径存在才运行，否则仅展示代码逻辑
    if os.path.exists(mmlong_root):
        loader = MMLongLoader(data_root=mmlong_root, extractor=extractor)
        loader.load_data()

        # 3. 初始化 Agent
        agent = RAGAgent(loader=loader)

        # 4. 运行单个样本测试
        if len(loader.samples) > 0:
            test_sample = loader.samples[0]
            agent.process_sample(test_sample)
            
            print("\n--- Result ---")
            print(f"Query: {test_sample.extra_info['query_content']}")
            print(f"Final Answer: {test_sample.extra_info['final_answer']}")
            print(f"Doc Type: {test_sample.extra_info.get('doc_type')}")