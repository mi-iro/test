import os
import sys
import json
import re
import base64
import mimetypes
from typing import List, Dict, Any, Optional, Union, Tuple

# 假设文件结构，确保可以导入 src 下的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from openai import OpenAI
from src.loaders.base_loader import BaseDataLoader, StandardSample, PageElement
from src.agents.RAGAgent import RAGAgent

# 复用 utils 中的正则匹配，或者重新定义以确保健壮性
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

def local_image_to_data_url(path: str) -> str:
    """
    将本地图片转换为 Data URL (Base64)
    """
    if not path or not os.path.exists(path):
        return ""
    
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"

    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"Error encoding image {path}: {e}")
        return ""

# --- System Prompt 定义 ---
AGENTIC_SYSTEM_PROMPT = """你是一个智能文档问答助手 (Agentic RAG Agent)。你的目标是通过检索文档中的证据来准确回答用户的复杂问题。

### 你的能力与工具
你可以使用工具 **`search_evidence_tool`** 来从文档库中检索信息。

- **功能**: 输入一个具体的查询字符串（Sub-query），系统将执行检索、重排序和细粒度信息提取，并返回相关的证据片段。
- **何时使用**: 
  - 当用户的问题无法直接凭空回答时。
  - 当需要查找具体的事实、数据、条款或对比信息时。
  - 如果用户的问题包含多个方面（例如“比较 A 和 B 的收入”），请**将问题分解**为简单的子查询（例如先查“A 的收入”，再查“B 的收入”），分步调用工具。

### 工具调用格式
请严格使用以下 JSON 格式包裹在 XML 标签中进行调用：

<tool_call>
{"name": "search_evidence_tool", "arguments": {"query": "<你的搜索关键词或子问题>"}}
</tool_call>

### 思考与回复流程 (ReAct 范式)
1. **Thought**: 思考当前需要什么信息来回答用户问题，是否需要分解问题。
2. **Tool Call**: 如果需要信息，生成工具调用。
3. **Observation**: (系统会将工具返回的结果用 `<tool_response></tool_response>` 包裹后插入对话)。
4. **Repeat**: 根据观察到的结果，决定是否需要进一步搜索。
5. **Final Answer**: 当收集到足够的信息后，或者确定无法找到信息时，直接输出最终答案（不要再包含 tool_call）。

### 输出示例
User: 2023年和2024年的总收入分别是多少？
Assistant: 
Thought: 用户询问两年的收入，我需要分别检索2023年和2024年的收入数据。首先检索2023年的。
<tool_call>
{"name": "search_evidence_tool", "arguments": {"query": "2023年 总收入"}}
</tool_call>
... (等待工具返回) ...
"""

class AgenticRAGAgent(RAGAgent):
    """
    基于 ReAct 范式的 Agentic RAG。
    继承自 RAGAgent 以保持接口兼容，但重写了 process_sample 逻辑以引入 LLM 驱动的循环。
    """

    def __init__(
        self, 
        loader: BaseDataLoader, 
        base_url: str, 
        api_key: str, 
        model_name: str, 
        max_rounds: int
    ):
        """
        :param loader: 数据集加载器 (FinRAGLoader, MMLongLoader 等)，用于提供 pipeline 作为检索工具。
        :param base_url: LLM API 地址。
        :param api_key: LLM API Key。
        :param model_name: 模型名称。
        :param max_rounds: 最大交互/思考轮数，防止死循环。
        """
        super().__init__(loader)
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.max_rounds = max_rounds

    def extract_tool_call(self, text: str) -> Optional[Dict]:
        """从文本中提取 JSON 格式的工具调用"""
        match = TOOL_CALL_RE.search(text)
        if not match:
            return None
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            print(f"Error parsing tool call JSON: {match.group(1)}")
            return None

    def _execute_tool(self, tool_name: str, args: Dict) -> Union[str, List[Dict[str, Any]]]:
        """
        执行工具调用。
        将检索到的 PageElement 转换为多模态消息内容 (List[Dict])。
        包含: 页面原图(Page Image) + 证据截图(Crop) + 文本内容(Content)。
        """
        if tool_name == "search_evidence_tool":
            query = args.get("query", "")
            if not query:
                return "Error: Query cannot be empty."
            
            print(f"  [Agent Action] Searching for: {query}")
            
            try:
                image_paths = None
                if hasattr(self, 'current_data_source') and self.current_data_source:
                    image_paths = [self.current_data_source]

                elements = self.loader.pipeline(query=query, image_paths=image_paths, top_k=10)
                
                if not elements:
                    return "No relevant evidence found."
                
                # 构造多模态 Tool Response
                content_list = []
                content_list.append({"type": "text", "text": "<tool_response>\nFound the following evidence:\n"})
                
                for i, el in enumerate(elements):
                    content_list.append({"type": "text", "text": f"\n--- Evidence {i+1} ---\n"})
                    
                    # 1. 元素所在页面图像 (Page Image)
                    if el.corpus_id:
                        img_url = local_image_to_data_url(el.corpus_id)
                        if img_url:
                             content_list.append({"type": "text", "text": f"[Page Source: {os.path.basename(el.corpus_id)}]\n"})
                             content_list.append({"type": "image_url", "image_url": {"url": img_url}})
                    
                    # 2. 元素截图 (Evidence Crop)
                    # 如果 crop_path 存在且与 corpus_id 不同（避免重复展示全页），则展示
                    if el.crop_path and el.crop_path != el.corpus_id:
                        crop_url = local_image_to_data_url(el.crop_path)
                        if crop_url:
                            content_list.append({"type": "text", "text": "\n[Evidence Detail Crop]\n"})
                            content_list.append({"type": "image_url", "image_url": {"url": crop_url}})
                    
                    # 3. 元素文本内容
                    content_list.append({"type": "text", "text": f"\nContent: {el.content}\n"})
                
                content_list.append({"type": "text", "text": "\n</tool_response>"})
                return content_list
            
            except Exception as e:
                return f"Error during retrieval: {str(e)}"
        
        else:
            return f"Error: Unknown tool '{tool_name}'."

    def run_agent_loop(self, query: str) -> Tuple[str, List[Dict]]:
        """
        执行 ReAct 循环的核心异步方法。
        返回: (final_answer, full_messages_history)
        """
        messages = [
            {"role": "system", "content": AGENTIC_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]

        final_answer = ""
        
        for i in range(self.max_rounds):
            # 1. LLM 生成 Thought 和 Potential Tool Call
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=1.0
                )
                content = response.choices[0].message.content
                messages.append({"role": "assistant", "content": content})
                print(f"Round {i+1} Assistant: {content[:100]}...") 

            except Exception as e:
                print(f"LLM API Error: {e}")
                final_answer = "Error during generation."
                break

            # 2. 检查是否有工具调用
            tool_call_dict = self.extract_tool_call(content)
            
            if tool_call_dict:
                # 3. 执行工具
                tool_name = tool_call_dict.get("name")
                tool_args = tool_call_dict.get("arguments", {})
                
                tool_result = self._execute_tool(tool_name, tool_args)
                
                # 4. 构造 Tool Response
                if isinstance(tool_result, list):
                    # 如果返回是列表，说明是构造好的多模态消息
                    messages.append({"role": "user", "content": tool_result})
                else:
                    # 如果是字符串（通常是错误信息），包装为文本消息
                    tool_msg_content = f"<tool_response>\n{tool_result}\n</tool_response>"
                    messages.append({"role": "user", "content": tool_msg_content})
                
            else:
                # 没有工具调用，说明 LLM 生成了最终回复
                final_answer = content
                break
        
        if not final_answer and messages[-1]["role"] == "assistant":
            final_answer = messages[-1]["content"]

        return final_answer, messages

    def process_sample(self, sample: StandardSample) -> StandardSample:
        """
        重写父类方法。
        不再直接调用 loader.pipeline，而是启动 ReAct Agent Loop。
        """
        query = sample.query
        
        # 设置当前上下文的数据源，供 _execute_tool 使用
        self.current_data_source = sample.data_source
        
        print(f"Processing Sample {sample.qid} with Agentic Logic...")

        final_answer, history_messages = self.run_agent_loop(query)

        # 记录结果
        if sample.extra_info is None:
            sample.extra_info = {}
        
        # 记录完整的对话历史，包含多模态证据
        sample.extra_info['messages'] = history_messages
        sample.extra_info['final_answer'] = final_answer
        
        return sample

# --- 测试代码 ---
if __name__ == "__main__":
    from src.loaders.MMLongLoader import MMLongLoader
    from src.agents.ElementExtractor import ElementExtractor
    from src.agents.utils import ImageZoomOCRTool
    
    # 1. 模拟环境配置
    root_dir = "/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/MMLongBench-Doc"
    tool_work_dir = "./workspace"
    
    # 2. 初始化底层提取器 (Loader 内部需要用到的)
    tool = ImageZoomOCRTool(work_dir=tool_work_dir)
    extractor = ElementExtractor(
        base_url="http://localhost:8001/v1",
        api_key="sk-123456",
        model_name="MinerU-Agent-CK300",
        tool=tool
    )
    
    # 3. 初始化 Loader
    loader = MMLongLoader(data_root=root_dir, extractor=extractor)
    loader.load_data()

    # 4. 初始化 Agentic RAG Agent
    agent = AgenticRAGAgent(
        loader=loader,
        base_url="http://localhost:3888/v1", 
        model_name="qwen3-max",
        api_key="sk-6TGzZJkJ5HfZKwnrS1A1pMb1lH5D7EDfSVC6USq24aN2JaaR",
        max_rounds=5
    )
    
    # 5. 运行单条测试
    if len(loader.samples) > 0:
        result_sample = agent.process_sample(loader.samples[0])
        print("\n=== Final Result ===")
        print(f"Query: {result_sample.query}")
        print(f"Answer: {result_sample.extra_info['final_answer']}")
        print(f"History Steps: {len(result_sample.extra_info['messages'])}")