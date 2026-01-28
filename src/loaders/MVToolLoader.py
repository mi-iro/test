import os
import json
import re
import sys
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from PIL import Image

# Adjust path to ensure we can import from src and scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.loaders.base_loader import BaseDataLoader, StandardSample, PageElement
from src.agents.ElementExtractor import ElementExtractor

class MVToolLoader(BaseDataLoader):
    """
    MVToolBench 数据集加载器。
    该数据集主要用于单图 VQA 任务，包含带有 BBox 的 Ground Truth。
    """
    def __init__(self, data_root: str, extractor: Optional[ElementExtractor] = None):
        """
        :param data_root: 包含 mvtoolbench_full.json 的根目录路径
        :param extractor: ElementExtractor 实例，用于执行 pipeline 提取任务
        """
        super().__init__(data_root)
        self.extractor = extractor
        
        # 尝试定位 json 文件，默认在 root 下，也可以兼容直接传入文件路径
        if data_root.endswith(".json") and os.path.isfile(data_root):
            self.json_path = data_root
            self.image_root = os.path.dirname(data_root) # 假设图片是相对路径或绝对路径
        else:
            self.json_path = os.path.join(data_root, "mvtoolbench_full.json")
            self.image_root = data_root

    def _parse_assistant_content(self, content: str) -> Dict[str, Any]:
        """
        解析 Assistant 回复中的 JSON 代码块。
        示例格式: ```json{"evidence": "...", "bbox": [...]}```
        """
        try:
            # 使用非贪婪匹配提取 ```json 和 ``` 之间的内容
            match = re.search(r'```json(.*?)```', content, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                return json.loads(json_str)
            else:
                # 如果没有代码块，尝试直接解析或返回原始内容作为 text
                return {"evidence": content, "bbox": []}
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode JSON content: {content[:50]}...")
            return {"evidence": content, "bbox": []}

    def evaluate(self, beta: float = 1.0) -> Dict[str, float]:
        """
        执行评估：计算 QA 指标、页面检索指标和元素提取指标。
        整合了 eval.py 中的 Page Accuracy, IoU Min, IoU EM 等高级指标。
        
        评估结果将存储在每个 sample.extra_info['metrics'] 中。
        依赖 sample.extra_info 包含 'final_answer' 和 'retrieved_elements'。
        
        Returns:
            Dict[str, float]: 整个数据集的平均指标
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
                    # 过滤掉非 PageElement 字段以防止 TypeError
                    valid_keys = PageElement.__annotations__.keys()
                    filtered_el = {k: v for k, v in el.items() if k in valid_keys}
                    pred_elements.append(PageElement(**filtered_el))
                elif isinstance(el, PageElement):
                    pred_elements.append(el)
            
            # 获取 BBoxes 列表 (用于 element metrics 计算)
            # 过滤掉无效的 bbox (例如全0或长度不为4)
            pred_bboxes = [p.bbox for p in pred_elements if p.bbox and len(p.bbox) == 4 and calculate_area(p.bbox) > 0]
            gt_bboxes = [g.bbox for g in sample.gold_elements if g.bbox and len(g.bbox) == 4 and calculate_area(g.bbox) > 0]
            
            metrics_result = {}

            # 1. 计算 QA 指标 (Text Generation)
            if sample.gold_answer:
                qa_score = self._compute_qa_metrics(pred_answer, sample.gold_answer)
                metrics_result['qa'] = qa_score
                total_metrics['qa_f1'] += qa_score['f1']
                total_metrics['qa_em'] += qa_score['em']
                counts['qa'] += 1

            # 2. 计算 页面检索 指标 (Page Retrieval)
            if sample.gold_pages:
                page_score = self._compute_page_metrics(pred_elements, sample.gold_pages)
                metrics_result['page'] = page_score
                total_metrics['page_recall'] += page_score['recall']
                total_metrics['page_precision'] += page_score['precision']
                counts['page'] += 1

            # 3. 计算 元素提取/检测 指标 (Element Detection - Integrated from eval.py)
            # 无论是否有 GT，都需要计算（用于处理 False Positive）
            
            # Page Accuracy (Presence Check)
            page_acc = self._compute_page_accuracy(pred_bboxes, gt_bboxes)
            metrics_result['page_acc'] = page_acc
            total_metrics['page_acc'] += page_acc
            counts['page_acc'] += 1

            if sample.gold_elements or pred_elements:
                # IoU Min Strategy (Threshold=0.75 from eval.py logic)
                min_p, min_r = self._compute_detection_metrics(pred_bboxes, gt_bboxes, iou_func=calc_iou_min, threshold=0.75)
                min_f = calculate_f_beta(min_p, min_r, beta)
                
                # IoU Standard/EM Strategy (Threshold=0.6 from eval.py default, 0.7 in usage)
                # 这里使用 0.6 作为默认标准阈值
                em_p, em_r = self._compute_detection_metrics(pred_bboxes, gt_bboxes, iou_func=calc_iou_standard, threshold=0.6)
                em_f = calculate_f_beta(em_p, em_r, beta)

                elem_score = {
                    'iou_min_precision': min_p, 'iou_min_recall': min_r, 'iou_min_f1': min_f,
                    'iou_em_precision': em_p, 'iou_em_recall': em_r, 'iou_em_f1': em_f
                }
                metrics_result['element'] = elem_score
                
                total_metrics['element_min_f1'] += min_f
                total_metrics['element_em_f1'] += em_f
                counts['element'] += 1

            # 存储回 sample
            sample.extra_info['metrics'] = metrics_result

        # --- 汇总平均值 ---
        avg_results = {}
        
        # QA
        if counts['qa'] > 0:
            avg_results['avg_qa_f1'] = total_metrics['qa_f1'] / counts['qa']
            avg_results['avg_qa_em'] = total_metrics['qa_em'] / counts['qa']
        
        # Page Retrieval
        if counts['page'] > 0:
            avg_results['avg_page_recall'] = total_metrics['page_recall'] / counts['page']
            avg_results['avg_page_precision'] = total_metrics['page_precision'] / counts['page']
        
        # Element Detection
        if counts['page_acc'] > 0:
            avg_results['avg_page_acc'] = total_metrics['page_acc'] / counts['page_acc']
            
        if counts['element'] > 0:
            avg_results['avg_element_min_f1'] = total_metrics['element_min_f1'] / counts['element']
            avg_results['avg_element_em_f1'] = total_metrics['element_em_f1'] / counts['element']

        return avg_results

    def load_data(self) -> None:
        """加载 MVToolBench JSON 数据并转换为 StandardSample 格式。"""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"MVToolBench data file not found: {self.json_path}")
        
        print(f"Loading MVToolBench data from: {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        count = 0
        for item in data:
            # 1. 基础信息提取
            qid = item.get("id", str(count))
            image_list = item.get("images", [])
            # 这里的 image_path 通常是绝对路径或相对于 dataset 的路径
            # 如果是相对路径，可能需要结合 self.image_root 拼接，这里暂按原样读取
            main_image_path = image_list[0] if image_list else ""
            
            # 2. 解析对话 (Query 和 Answer)
            conversations = item.get("conversations", [])
            query_text = ""
            gold_answer = ""
            gold_bbox = []
            
            for turn in conversations:
                role = turn.get("role", "")
                content = turn.get("content", "")
                
                if role == "user":
                    # 去除 <image> 标签，获取纯文本 Query
                    query_text = content.replace("<image>\n", "").replace("<image>", "").strip()
                
                elif role == "assistant":
                    # 解析包含 evidence 和 bbox 的 JSON
                    parsed_response = self._parse_assistant_content(content)
                    gold_answer = parsed_response.get("evidence", "")
                    gold_bbox = parsed_response.get("bbox", [])

            # 3. 构建 PageElement (作为 Ground Truth 元素)
            gold_elements = []
            if gold_bbox:
                element = PageElement(
                    bbox=gold_bbox,
                    type="text", # MVTool 主要是文本识别/VQA
                    content=gold_answer,
                    corpus_id=main_image_path,
                    crop_path=None
                )
                gold_elements.append(element)

            # 4. 构建 StandardSample
            # 注意：MVTool 是 VQA 任务，data_source 指向具体图像，而不是索引
            sample = StandardSample(
                qid=qid,
                query=query_text,
                dataset="mvtool",
                data_source=main_image_path, 
                gold_answer=gold_answer,
                gold_elements=gold_elements,
                gold_pages=image_list
            )
            self.samples.append(sample)
            count += 1
            
        print(f"✅ Successfully loaded {count} MVTool samples.")

    def pipeline(self, query: str, image_paths: List[str] = None,  top_k: int = 5) -> List[PageElement]:
        """
        利用 ElementExtractor 从给定的图像中提取能够回答 Query 的证据元素。
        
        :param query: 用户的查询文本
        :param image_paths: 待处理的图像路径列表。对于 MVTool，通常是单张图片。
        :return: 提取出的 PageElement 列表
        """
        if self.extractor is None:
            print("Error: ElementExtractor is not initialized in MVToolLoader.")
            return []

        if not image_paths:
            print("Warning: No image_paths provided for MVTool pipeline. Returning empty list.")
            return []

        # 定义裁剪图片保存的本地工作目录
        workspace_dir = os.path.abspath(os.path.join(os.getcwd(), "workspace", "crops"))
        os.makedirs(workspace_dir, exist_ok=True)

        # 执行 Agent (ElementExtractor 是异步的，这里使用 asyncio.run 封装调用)
        try:
            # 检查是否已有运行中的 loop (防止在 Jupyter 等环境中报错)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # 如果已经在异步环境中，建议由调用者直接调用 extractor，此处为了兼容同步接口只能抛出警告或尝试 nesting
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

        # 解析 Agent 的输出
        # run_agent 返回的结构是 {"predictions": [...], "images": [...]}
        predictions = agent_output.get("predictions", [])
        if not predictions:
            return []

        # 获取 Agent 的最后一条回复，其中包含 JSON 格式的证据
        last_message = predictions[-1]
        content = last_message.get("content", "")

        extracted_elements = []
        try:
            # 尝试提取 ```json ... ``` 块
            json_match = re.search(r'```json(.*?)```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # 如果没有代码块，尝试查找列表边界 [ ... ]
                start = content.find('[')
                end = content.rfind(']')
                if start != -1 and end != -1:
                    json_str = content[start:end+1]
                else:
                    # 无法解析，跳过
                    json_str = "[]"

            data = json.loads(json_str)

            # 统一将单个对象转换为列表以复用逻辑
            if isinstance(data, dict):
                data = [data]

            if isinstance(data, list):
                # 准备处理图像 (MVTool 通常是单图，取第一个)
                image_path = image_paths[0]
                original_img = None
                img_w, img_h = 0, 0
                
                try:
                    original_img = Image.open(image_path)
                    img_w, img_h = original_img.size
                except Exception as e:
                    print(f"Failed to open image {image_path}: {e}")

                for item in data:
                    bbox = item.get("bbox", [0, 0, 0, 0])
                    evidence = item.get("evidence", "")
                    
                    # 确保 bbox 格式正确
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        bbox = [0, 0, 0, 0]

                    current_crop_path = image_path # 默认回退到原图

                    # 执行坐标转换和裁剪 (仿照 FinRAGLoader)
                    if original_img:
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
                        
                        # 只有当 bbox 有效且有面积时才进行裁剪
                        if x2 > x1 and y2 > y1:
                            try:
                                # 执行裁剪
                                cropped_img = original_img.crop((x1, y1, x2, y2))
                                
                                # 生成唯一文件名
                                filename = f"{os.path.basename(image_path).split('.')[0]}_{uuid.uuid4().hex[:8]}.jpg"
                                save_path = os.path.join(workspace_dir, filename)
                                
                                # 保存到本地 workspace
                                cropped_img.save(save_path)
                                current_crop_path = save_path # 更新路径为新裁剪图片的路径
                            except Exception as crop_err:
                                print(f"Error cropping image: {crop_err}")

                    element = PageElement(
                        bbox=bbox,
                        type="evidence",
                        content=evidence,
                        corpus_id=image_path,
                        crop_path=current_crop_path 
                    )
                    extracted_elements.append(element)

        except json.JSONDecodeError:
            print(f"Failed to parse JSON from agent response: {content[-100:]}")
        except Exception as e:
            print(f"Error converting agent output to PageElement: {e}")

        return extracted_elements

if __name__ == "__main__":
    # 测试代码
    root_dir = "/mnt/shared-storage-user/mineru3-share/jiayu/newBench/dataOri/MVToolBench/mvtoolbench_benchmark"
    
    # 示例: 初始化 Loader 时不带 Extractor (仅加载数据)
    loader = MVToolLoader(data_root=root_dir)
    try:
        loader.load_data()
        if len(loader.samples) > 0:
            s = loader.samples[0]
            print(f"\nSample 0 ID: {s.qid}")
            print(f"Query: {s.query}")
            print(f"Image: {s.data_source}")
            
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
                print(f" - Content: {res.content} \n - Crop: {res.crop_path}")
            
    except FileNotFoundError as e:
        print(e)