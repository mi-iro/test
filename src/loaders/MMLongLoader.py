import os
import json
import re
import sys
import ast
import asyncio
import uuid
from PIL import Image
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

    def _pdf_to_images(self, pdf_path: str) -> Dict[int, str]:
        """
        将 PDF 转换为图片序列，并保存到缓存目录。
        返回: {page_num: image_path} 映射，page_num 从 1 开始 (物理页码)。
        """
        if not os.path.exists(pdf_path):
             print(f"Warning: PDF not found at {pdf_path}")
             return {}

        # 定义缓存目录 (workspace/pdf_cache/<pdf_filename_without_ext>)
        pdf_name = os.path.basename(pdf_path).rsplit('.', 1)[0]
        workspace_dir = os.path.abspath(os.path.join(os.getcwd(), "workspace", "pdf_cache"))
        cache_dir = os.path.join(workspace_dir, pdf_name)
        os.makedirs(cache_dir, exist_ok=True)
        
        image_map = {}
        
        # 1. 检查是否存在已转换的缓存图片
        existing_files = [f for f in os.listdir(cache_dir) if f.endswith('.png')]
        if existing_files:
            # 尝试加载缓存
            valid_cache = True
            temp_map = {}
            for f in existing_files:
                # 匹配 page_1.png, page_2.png 等
                match = re.match(r"page_(\d+)\.png", f)
                if match:
                    idx = int(match.group(1))
                    temp_map[idx] = os.path.join(cache_dir, f)
            
            if temp_map:
                print(f"Using cached images for {pdf_name} ({len(temp_map)} pages)")
                return temp_map

        # 2. 如果没有缓存，执行转换
        print(f"Converting PDF to images: {pdf_path}")
        try:
            from pdf2image import convert_from_path
            # 默认使用 200 dpi，足以满足大多 OCR 需求
            images = convert_from_path(pdf_path, dpi=200)
            
            for i, img in enumerate(images):
                page_num = i + 1  # 物理页码从 1 开始
                save_name = f"page_{page_num}.png"
                save_path = os.path.join(cache_dir, save_name)
                
                img.save(save_path, "PNG")
                image_map[page_num] = save_path
                
            print(f"Converted {len(image_map)} pages.")
            
        except ImportError:
            print("Error: `pdf2image` library is not installed. Please install it to process PDF files.")
            # 如果没有库，无法处理，返回空或仅返回空字典
            return {}
        except Exception as e:
            print(f"Error converting PDF {pdf_path}: {e}")
            return {}
            
        return image_map

    def pipeline(self, query: str, image_paths: List[str] = None,  top_k: int = 5) -> List[PageElement]:
        """
        利用 ElementExtractor 从文档/图像中提取答案。
        如果输入路径包含 PDF，会自动将其拆分为图片并映射。
        每次只向 Agent 输入一张图片，并对提取到的 BBox 进行裁剪。
        """
        if self.extractor is None:
            print("Error: ElementExtractor is not initialized in MMLongLoader.")
            return []

        if not image_paths:
            print("Warning: No image_paths provided for pipeline. Returning empty list.")
            return []

        # --- 处理 PDF 输入 ---
        processed_image_paths = []
        for path in image_paths:
            if path.lower().endswith('.pdf'):
                # 调用拆分逻辑
                page_map = self._pdf_to_images(path)
                # 按页码顺序添加所有页面
                sorted_pages = sorted(page_map.keys())
                for p_num in sorted_pages:
                    processed_image_paths.append(page_map[p_num])
            else:
                processed_image_paths.append(path)
        
        if not processed_image_paths:
            print("Warning: No valid images found after processing input paths.")
            return []
            
        # 定义裁剪图片保存的本地工作目录
        workspace_dir = os.path.abspath(os.path.join(os.getcwd(), "workspace", "crops"))
        os.makedirs(workspace_dir, exist_ok=True)

        extracted_elements = []

        # 遍历所有图片，逐张调用 Agent
        for img_path in processed_image_paths:
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    print("Warning: pipeline called within a running event loop. Please handle async properly.")
                    continue
                else:
                    # 修改点：将单张图片包装成 list 传入，满足 Agent 接口要求
                    agent_output = asyncio.run(self.extractor.run_agent(
                        user_text=query,
                        image_paths=[img_path]  
                    ))
                
                if not agent_output:
                    continue

                # 解析 Agent 输出
                predictions = agent_output.get("predictions", [])
                if not predictions:
                    continue

                last_message = predictions[-1]
                content = last_message.get("content", "")

                extracted_data = []
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

                    extracted_data = json.loads(json_str)
                except Exception as e:
                    print(f"Error parsing JSON for {img_path}: {e}")
                    extracted_data = []

                # 统一处理 list 或 dict
                if isinstance(extracted_data, dict):
                    extracted_data = [extracted_data]

                if isinstance(extracted_data, list):
                    # 尝试打开当前图片，准备裁剪
                    current_page_image = None
                    img_w, img_h = 0, 0
                    try:
                        current_page_image = Image.open(img_path)
                        img_w, img_h = current_page_image.size
                    except Exception as e:
                        print(f"Failed to open image for cropping {img_path}: {e}")

                    for item in extracted_data:
                        bbox = item.get("bbox", [0, 0, 0, 0])
                        evidence = item.get("evidence", "")
                        
                        if not isinstance(bbox, list) or len(bbox) != 4:
                            bbox = [0, 0, 0, 0]
                        
                        current_crop_path = img_path # 默认 crop_path 为整页
                        
                        # --- 裁剪逻辑开始 ---
                        # 仅当 bbox 有效且图片加载成功时执行
                        if current_page_image and bbox != [0, 0, 0, 0]:
                            try:
                                x1, y1, x2, y2 = bbox
                                # 假设模型输出是 0-1000 的归一化坐标
                                x1 = int(x1 / 1000 * img_w)
                                y1 = int(y1 / 1000 * img_h)
                                x2 = int(x2 / 1000 * img_w)
                                y2 = int(y2 / 1000 * img_h)
                                
                                # 边界修正
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(img_w, x2)
                                y2 = min(img_h, y2)
                                
                                # 只有当 bbox 面积大于 0 时才裁剪
                                if x2 > x1 and y2 > y1:
                                    cropped_img = current_page_image.crop((x1, y1, x2, y2))
                                    
                                    # 生成唯一文件名
                                    filename = f"{os.path.basename(img_path).split('.')[0]}_{uuid.uuid4().hex[:8]}.jpg"
                                    save_path = os.path.join(workspace_dir, filename)
                                    
                                    cropped_img.save(save_path)
                                    current_crop_path = save_path # 更新为裁剪路径
                            except Exception as crop_err:
                                print(f"Error cropping image on {img_path}: {crop_err}")
                        # --- 裁剪逻辑结束 ---

                        # corpus_id 对应具体的单页图片路径
                        element = PageElement(
                            bbox=bbox,
                            type="evidence",
                            content=evidence,
                            corpus_id=img_path, 
                            crop_path=current_crop_path # <--- 设置 crop_path
                        )
                        extracted_elements.append(element)

            except Exception as e:
                print(f"Error during agent execution on {img_path}: {e}")

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
            
            # 测试 PDF 自动转换
            if s.data_source.endswith(".pdf"):
                print(f"Testing PDF pipeline with: {s.data_source}")
                results = loader.pipeline(s.query, image_paths=[s.data_source])
                print(f"Extracted {len(results)} elements.")
                for res in results:
                    print(f" - Content: {res.content}")
                    print(f" - Crop Path: {res.crop_path}")
            else:
                print("Skipping PDF pipeline test (sample is not PDF).")
            
    except Exception as e:
        print(f"Test failed: {e}")