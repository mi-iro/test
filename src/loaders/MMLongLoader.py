import os
import json
import re
import sys
import ast
import asyncio
from typing import List, Dict, Any, Optional

# Adjust path to ensure we can import from src and scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.loaders.base_loader import BaseDataLoader, StandardSample, PageElement
from src.agents.ElementExtractor import ElementExtractor

class MMLongLoader(BaseDataLoader):
    """
    MMLongBench-Doc 数据集加载器。
    用于加载 MMLongBench-Doc 中的 DocVQA 任务数据。
    """
    def __init__(self, data_root: str, extractor: Optional[ElementExtractor] = None):
        """
        :param data_root: MMLongBench-Doc 的根目录 (包含 data/samples.json 和 data/documents)
        :param extractor: ElementExtractor 实例，用于执行 pipeline 提取任务
        """
        super().__init__(data_root)
        self.extractor = extractor
        
        # 根据提供的目录结构设置路径
        self.json_path = os.path.join(data_root, "data", "samples.json")
        self.doc_dir = os.path.join(data_root, "data", "documents")

    def _parse_assistant_content(self, content: str) -> Dict[str, Any]:
        """
        解析 Assistant 回复中的 JSON 代码块。
        复用 MVToolLoader 的逻辑。
        """
        try:
            match = re.search(r'```json(.*?)```', content, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                return json.loads(json_str)
            else:
                return {"evidence": content, "bbox": []}
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON content: {content[:50]}...")
            return {"evidence": content, "bbox": []}

    def load_data(self) -> None:
        """根据新的 samples.json 格式加载数据。"""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"MMLongBench data file not found: {self.json_path}")
        
        print(f"Loading MMLongBench data from: {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        count = 0
        for item in data:
            # 1. 基础信息提取
            # 数据中没有唯一 ID，使用计数器
            qid = str(count)
            
            # 解析文档路径
            # JSON 中的 doc_id 对应文件名 (如 "PH_2016.06.08_Economy-Final.pdf")
            doc_filename = item.get("doc_id", "")
            if doc_filename:
                main_doc_path = os.path.join(self.doc_dir, doc_filename)
            else:
                main_doc_path = ""
            
            # 2. 解析 Query 和 Answer
            query_text = item.get("question", "")
            gold_answer = item.get("answer", "")
            
            # 3. 解析 Evidence Pages (字符串格式的列表 "[5]")
            evidence_pages_str = item.get("evidence_pages", "[]")
            gold_pages = []
            try:
                # 安全解析字符串列表
                pages_list = ast.literal_eval(evidence_pages_str)
                # 转换为字符串列表
                if isinstance(pages_list, list):
                    gold_pages = [str(p) for p in pages_list]
            except Exception as e:
                # print(f"Warning: Failed to parse evidence_pages for {doc_filename}: {e}")
                gold_pages = []

            # 4. 构建 PageElement (Ground Truth)
            # 该数据集通常只提供 Text Answer，没有 BBox
            gold_elements = []
            # if gold_answer:
            #     element = PageElement(
            #         bbox=[], 
            #         type="text",
            #         content=gold_answer,
            #         corpus_id=main_doc_path,
            #         crop_path=None
            #     )
            #     gold_elements.append(element)

            # 5. 提取其他元数据
            extra_info = {
                "doc_type": item.get("doc_type"),
                "evidence_sources": item.get("evidence_sources"),
                "answer_format": item.get("answer_format")
            }

            # 6. 构建 StandardSample
            sample = StandardSample(
                qid=qid,
                query=query_text,
                dataset="mmlongbench-doc",
                data_source=main_doc_path, 
                gold_answer=gold_answer,
                gold_elements=gold_elements,
                gold_pages=gold_pages,
                extra_info=extra_info
            )
            self.samples.append(sample)
            count += 1
            
        print(f"✅ Successfully loaded {count} MMLongBench samples.")

    def pipeline(self, query: str, image_paths: List[str] = None) -> List[PageElement]:
        """
        利用 ElementExtractor 从文档/图像中提取答案。
        注意：如果输入是 PDF，需要确保 ElementExtractor 或传入的 image_paths 已经处理为图片格式。
        """
        if self.extractor is None:
            print("Error: ElementExtractor is not initialized in MMLongLoader.")
            return []

        if not image_paths:
            print("Warning: No image_paths provided for pipeline. Returning empty list.")
            return []

        # 执行 Agent
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                print("Warning: pipeline called within a running event loop. Please handle async properly.")
                return []
            else:
                agent_output = asyncio.run(self.extractor.run_agent(
                    user_text=query,
                    image_paths=image_paths
                ))
        except Exception as e:
            print(f"Agent execution failed: {e}")
            return []

        if not agent_output:
            return []

        # 解析 Agent 输出
        predictions = agent_output.get("predictions", [])
        if not predictions:
            return []

        last_message = predictions[-1]
        content = last_message.get("content", "")

        extracted_elements = []
        try:
            json_match = re.search(r'```json(.*?)```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                start = content.find('[')
                end = content.rfind(']')
                if start != -1 and end != -1:
                    json_str = content[start:end+1]
                else:
                    json_str = "[]"

            data = json.loads(json_str)

            if isinstance(data, list):
                for item in data:
                    bbox = item.get("bbox", [0, 0, 0, 0])
                    evidence = item.get("evidence", "")
                    
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        bbox = [0, 0, 0, 0]

                    element = PageElement(
                        bbox=bbox,
                        type="evidence",
                        content=evidence,
                        corpus_id=image_paths[0] if image_paths else "",
                        crop_path=None 
                    )
                    extracted_elements.append(element)
            elif isinstance(data, dict):
                 bbox = data.get("bbox", [0, 0, 0, 0])
                 evidence = data.get("evidence", "")
                 element = PageElement(
                    bbox=bbox,
                    type="evidence",
                    content=evidence,
                    corpus_id=image_paths[0] if image_paths else "",
                    crop_path=None 
                 )
                 extracted_elements.append(element)

        except Exception as e:
            print(f"Error converting agent output to PageElement: {e}")

        return extracted_elements

if __name__ == "__main__":
    # 测试代码
    # 请根据实际情况修改 root_dir
    root_dir = "/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/MMLongBench-Doc"
    
    loader = MMLongLoader(data_root=root_dir)
    try:
        loader.load_data()
        if len(loader.samples) > 0:
            s = loader.samples[0]
            print(f"\nSample 0 ID: {s.qid}")
            print(f"Query: {s.query}")
            print(f"Doc Path: {s.data_source}")
            print(f"Gold Answer: {s.gold_answer}")
            print(f"Extra Info: {s.extra_info}")
            
            # 若要测试 pipeline，需要初始化 ElementExtractor 并传入
            from src.agents.utils import ImageZoomOCRTool
            # 确保 workspace 目录存在
            tool_work_dir = "/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/src/agents/workspace"
            tool = ImageZoomOCRTool(work_dir=tool_work_dir)
            
            # 注意：这里的 API Key 和 URL 需要根据实际环境配置
            extractor = ElementExtractor(
                base_url="http://localhost:8001/v1", 
                api_key="sk-123456", 
                model_name="MinerU-Agent-CK300", 
                tool=tool
            )
            loader.extractor = extractor
            results = loader.pipeline(s.query, image_paths=[s.data_source])
            print(f"Extracted {len(results)} elements.")
            for res in results:
                print(f" - Content: {res.content[:50]}... \n - Crop: {res.crop_path}")
            
    except Exception as e:
        print(f"Test failed: {e}")