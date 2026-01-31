import os
import json
import re
import sys
import uuid
import collections
from typing import List, Dict, Any, Optional
from PIL import Image

# 调整路径以确保可以从 src 和 scripts 导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.loaders.base_loader import BaseDataLoader, StandardSample, PageElement
from src.agents.ElementExtractor import ElementExtractor
from src.utils.llm_helper import create_llm_caller

# 复用 MMLongLoader 中的评分逻辑
from src.loaders.MMLongLoader import eval_score, MMLONG_EXTRACT_PROMPT_TEMPLATE

class DocVQALoader(BaseDataLoader):
    """
    DocVQA 数据集加载器。
    适配 top3_test.jsonl 格式，支持单图检索、ElementExtractor 抽取及 LLM 评估。
    """
    
    def __init__(self, data_root: str, extractor: Optional[ElementExtractor] = None, **kwargs):
        """
        :param data_root: 数据集根目录
        :param extractor: ElementExtractor 实例
        """
        super().__init__(data_root)
        self.extractor = extractor
        
        # DocVQA 特有路径适配
        self.jsonl_path = os.path.join(data_root, "top3_test.jsonl")
        self.llm_caller = None

    def load_data(self) -> None:
        """解析 top3_test.jsonl 格式数据。"""
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"DocVQA data file not found: {self.jsonl_path}")
        
        print(f"Loading DocVQA data from: {self.jsonl_path}")
        count = 0
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                
                qid = item.get("qid", str(count))
                query_text = item.get("query", "")
                # DocVQA 的 answer 通常是列表
                gold_answers = item.get("answer", [])
                
                # 图像路径处理
                images = item.get("posImgs", [])
                image_full_paths = [os.path.join(self.data_root, "imgs", os.path.basename(img)) for img in images]
                
                # 正确答案所在的页面 (用于 Recall 评估)
                pos_imgs = item.get("posImgs", [])
                gold_pages = [os.path.basename(p) for p in pos_imgs]

                sample = StandardSample(
                    qid=qid,
                    query=query_text,
                    dataset="DocVQA",
                    data_source=image_full_paths[0] if image_full_paths else "", 
                    gold_answer=gold_answers,
                    gold_elements=[],
                    gold_pages=gold_pages,
                    extra_info={}
                )
                self.samples.append(sample)
                count += 1
            
        print(f"✅ Successfully loaded {count} DocVQA samples.")

    def _extract_answer_with_llm(self, question: str, raw_response: str) -> Dict[str, Any]:
        """
        利用 LLM 从自由文本分析中提取结构化答案。
        """
        if not self.llm_caller or not raw_response:
            return {"extracted_answer": raw_response, "answer_format": "String"}
            
        # 构建 Prompt
        prompt = MMLONG_EXTRACT_PROMPT_TEMPLATE.format(question=question, analysis=raw_response)
        
        try:
            llm_output = self.llm_caller(prompt)
            
            extracted_answer = ""
            answer_format = "String"
            
            # 使用正则解析 LLM 的固定格式输出
            ans_match = re.search(r"Extracted answer:\s*(.*)", llm_output, re.IGNORECASE)
            fmt_match = re.search(r"Answer format:\s*(.*)", llm_output, re.IGNORECASE)
            
            if ans_match:
                extracted_answer = ans_match.group(1).strip()
            if fmt_match:
                answer_format = fmt_match.group(1).strip()
            
            # 去除可能存在的单引号
            if extracted_answer.startswith("'") and extracted_answer.endswith("'"):
                extracted_answer = extracted_answer[1:-1]
            
            return {
                "extracted_answer": extracted_answer,
                "answer_format": answer_format
            }
        except Exception as e:
            print(f"LLM 提取失败: {e}")
            return {"extracted_answer": raw_response, "answer_format": "String"}

    def pipeline(self, query: str, image_paths: List[str] = None, top_k = 10) -> List[PageElement]:
        """
        DocVQA 核心流水线：
        1. 接收指定图片（不进行重排序）。
        2. 调用 ElementExtractor (Agent) 进行视觉搜索和答案定位。
        3. 解析输出并裁剪证据区域。
        """
        if not image_paths: return []
        if self.extractor is None:
            print("Error: ElementExtractor is not initialized.")
            return []

        # 准备工作目录用于保存裁剪图片
        workspace_dir = os.path.abspath(os.path.join(os.getcwd(), "workspace", "crops"))
        os.makedirs(workspace_dir, exist_ok=True)

        extracted_elements = []

        # 遍历每一张输入图片（通常只有一张）
        for img_path in image_paths:
            try:
                # 运行 Agent (同步调用)
                agent_output = self.extractor.run_agent(
                    user_text=query,
                    image_paths=[img_path]
                )
                
                if not agent_output or "predictions" not in agent_output:
                    continue

                # 获取最后一条消息的内容
                content = agent_output["predictions"][-1].get("content", "")
                
                # --- 增强的 JSON 解析逻辑 (参考 MMLongLoader) ---
                extracted_data = []
                try:
                    json_match = re.search(r'```json(.*?)```', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1).strip()
                    else:
                        # 尝试寻找方括号
                        start = content.find('[')
                        end = content.rfind(']')
                        if start != -1 and end != -1:
                            json_str = content[start:end+1]
                        else:
                            json_str = "[]"
                    extracted_data = json.loads(json_str)
                except Exception:
                    # 解析失败则跳过或视为空
                    extracted_data = []

                if isinstance(extracted_data, dict):
                    extracted_data = [extracted_data]

                # --- 处理图片裁剪与 PageElement 封装 ---
                if isinstance(extracted_data, list):
                    current_page_image = None
                    img_w, img_h = 0, 0
                    try:
                        current_page_image = Image.open(img_path)
                        img_w, img_h = current_page_image.size
                    except Exception:
                        pass

                    for item in extracted_data:
                        bbox = item.get("bbox", [0, 0, 0, 0])
                        evidence = item.get("evidence", "")
                        
                        # 验证 bbox 格式
                        if not isinstance(bbox, list) or len(bbox) != 4:
                            bbox = [0, 0, 0, 0]
                        
                        current_crop_path = img_path 
                        
                        # 执行裁剪 (Crop)
                        if current_page_image and bbox != [0, 0, 0, 0]:
                            try:
                                x1, y1, x2, y2 = bbox
                                # 坐标归一化转换 (假设 Agent 输出为 1000x1000 坐标系)
                                x1 = int(x1 / 1000 * img_w)
                                y1 = int(y1 / 1000 * img_h)
                                x2 = int(x2 / 1000 * img_w)
                                y2 = int(y2 / 1000 * img_h)
                                
                                x1 = max(0, x1); y1 = max(0, y1); x2 = min(img_w, x2); y2 = min(img_h, y2)
                                
                                if x2 > x1 and y2 > y1:
                                    cropped_img = current_page_image.crop((x1, y1, x2, y2))
                                    # 生成唯一文件名
                                    filename = f"{os.path.basename(img_path).split('.')[0]}_{uuid.uuid4().hex[:8]}.jpg"
                                    save_path = os.path.join(workspace_dir, filename)
                                    cropped_img.save(save_path)
                                    current_crop_path = save_path 
                            except Exception as e:
                                print(f"Crop failed: {e}")

                        # 构造 PageElement
                        element = PageElement(
                            bbox=bbox,
                            type="evidence",
                            content=evidence,
                            corpus_id=os.path.basename(img_path),
                            corpus_path=img_path,
                            crop_path=current_crop_path
                        )
                        extracted_elements.append(element)

            except Exception as e:
                print(f"Error extracting from {img_path}: {e}")

        return extracted_elements

    def evaluate(self) -> Dict[str, float]:
        """
        执行带有 LLM 辅助提取的评估。
        """
        total_metrics = collections.defaultdict(float)
        counts = collections.defaultdict(int)

        print(f"正在对 {len(self.samples)} 个样本进行评估...")

        for sample in self.samples:
            if sample.extra_info is None:
                sample.extra_info = {}
            
            # --- 提取预测答案 ---
            raw_pred = sample.extra_info.get('final_answer', "")
            gold_answers = sample.gold_answer # DocVQA 中通常是一个列表
            
            # 如果提供了 llm_caller，则先进行标准化提取
            if self.llm_caller and raw_pred:
                extract_res = self._extract_answer_with_llm(sample.query, raw_pred)

                final_pred = extract_res['extracted_answer']
                sample.extra_info['extracted_answer_llm'] = final_pred
            else:
                final_pred = raw_pred

            # --- 计算 QA 得分 ---
            if gold_answers:
                # DocVQA 评分：预测值与金标准列表中的任意一个匹配即计算最高分
                best_s = 0.0
                gold_list = gold_answers if isinstance(gold_answers, list) else [gold_answers]
                
                for gold in gold_list:
                    s = eval_score(gold, final_pred, "String")
                    best_s = max(best_s, s)
                
                total_metrics['qa_score'] += best_s
                counts['qa'] += 1

        # 计算平均指标
        avg_results = {}
        if counts['qa'] > 0:
            avg_results['avg_qa_score'] = total_metrics['qa_score'] / counts['qa']
            
        print(f"评估完成: {avg_results}")
        return avg_results

if __name__ == "__main__":
    # 使用示例
    data_root = "/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/VisRAG/data/EVisRAG-Test-DocVQA"
    # 注意：此处不需要 reranker
    loader = DocVQALoader(data_root=data_root)
    loader.load_data()
    
    if loader.samples:
        print(f"Sample Query: {loader.samples[0].query}")
        print(f"Gold Answer: {loader.samples[0].gold_answer}")