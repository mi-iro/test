import re
import json
import asyncio
from typing import List, Dict, Optional
from .utils import *

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

SYSTEM_PROMPT = """You are an advanced Visual Document Analysis Agent capable of precise evidence extraction from document images. Your goal is to answer user queries by locating, reading, and extracting specific information from a page.

### Your Capabilities & Tools
You have access to a powerful tool named **`image_zoom_and_ocr_tool`**.

- **Functionality**:  
  Crop a specific region of the image, optionally rotate it, and perform OCR on the cropped region.

- **When to use**:  
  - Always use this tool when the user asks for **specific text, numbers, names, dates, tables, or factual details** from the page.
  - Do NOT rely solely on the global low-resolution image when reading dense or small text.
  - If the target text is rotated, estimate and set the `angle` parameter before OCR.

- **Parameters**:
  - `label`: A short description of what you are looking for.
  - `bbox`: `[xmin, ymin, xmax, ymax]` in **0–1000 normalized coordinates**, relative to the original page.
  - `angle`: Rotation angle (counter-clockwise) applied after cropping. Default is `0`.
  - `do_ocr`: Whether to perform OCR on the cropped image.

### Tool Usage Example
Use the tool strictly in the following format:

<tool_call>
{"name": "image_zoom_and_ocr_tool", "arguments": {"label": "<A short description of what you are looking for>", "bbox": [xmin, ymin, xmax, ymax], "angle":<0/90/180/270>, "do_ocr": <true/false>}}
</tool_call>

### Your Input and Task
The user input includes:
1. One page image of a visual document.
2. The user's query intent.

Please execute the following steps:
1. **Semantic Matching**: Carefully observe the image to determine if the page content contains evidence information relevant to the user's query. If it is irrelevant, return an empty list.
2. **Precise Localization**: If relevant, extract the complete chain of visual evidence that helps to answer the query (text blocks, tables, charts or image regions).
3. **Speical Notes**: The page image may contain several evidence pieces. Pay attention to tables, charts and images, as they could also contain evidence.

### Output Format
After gathering information, output the list of relevant evidence in the following JSON format.  
If the page image is not relevant, return an empty list.

```json
[
  {
    "evidence": "<self-contained content, understandable without page context>",
    "bbox": [xmin, ymin, xmax, ymax] # 0-1000 normalized coordinates 
  }
  ...
]
```

Let us think step by step, using tool calling for better understanding of details!
"""

class ElementExtractor:
    def __init__(self, base_url: str, api_key: str, model_name: str, tool):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model_name = model_name
        self.tool = tool

    def build_multimodal_user_message(
        self,
        text: str,
        image_paths: Optional[List[str]] = None
    ) -> Dict:
        content = []

        if image_paths:
            for path in image_paths:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": local_image_to_data_url(path)
                    }
                })

        if text:
            content.append({
                "type": "text",
                "text": text
            })

        return {
            "role": "user",
            "content": content
        }

    def build_multimodal_tool_message(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None
    ) -> Dict:
        content = []
        content.append({
                "type": "text",
                "text": "<tool_response>\n"
            })

        if image_path:
            content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": local_image_to_data_url(image_path)
                    }
                })

        if text:
            content.append({
                "type": "text",
                "text": f"\n{text}"
            })

        content.append({
                "type": "text",
                "text": "\n</tool_response>"
            })

        return {
            "role": "user",
            "content": content
        }

    def extract_tool_call(self,text: str, as_json: bool = False):
        """
        提取 <tool_call> 标签内容。
        
        如果 as_json=True，则返回解析后的 dict。
        否则返回原始字符串。
        """
        match = TOOL_CALL_RE.search(text)
        if not match:
            return None
        
        tool_text = match.group(1).strip()
        
        if as_json:
            try:
                return json.loads(tool_text)
            except json.JSONDecodeError:
                return None  # 如果不是合法 JSON
        else:
            return tool_text

    # ---------- tool mock ----------

    async def _handle_tool_call(self, tool_call,image_path,step,max_rounds) -> str:
        """
        当前是测试用 mock
        后面你可以在这里真正调用 image_zoom_and_ocr_tool
        """
        if step >= max_rounds-2:
            return [False, "You have used up all the available uses of `image_zoom_and_ocr_tool`, please return you final response without use tool."]
        result_list = await self.tool.call(tool_call,image_path)
        if result_list[0]==False:
            result_list = await self.tool.call(tool_call,image_path)
            if result_list[0]==False:#重试一次
                return [False, "`image_zoom_and_ocr_tool` is wrong, you can try it again."]
        if len(result_list)==2:
            return [True, result_list[1]]
        if len(result_list)==3:
            return [True, result_list[1], f"{result_list[2]}"]

    # ---------- agent loop ----------

    async def run_agent(
        self,
        user_text: str,
        image_paths: Optional[List[str]] = None,
        max_rounds: int = 10,
        uid: int=1
    ):  
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            self.build_multimodal_user_message(user_text, image_paths),
        ]
        output = {"id":uid,"predictions":[{"role": "system", "content": SYSTEM_PROMPT},{"role": "user", "content": "<image>\n"+user_text}],
        "images":[image_paths[0]]}
        for step in range(max_rounds):
            resp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=1.0
            )

            content = resp.choices[0].message.content

            messages.append({"role": "assistant", "content": content})
            output["predictions"].append({"role": "assistant", "content": content})

            tool_call = self.extract_tool_call(content, as_json=True)
            
            # 没有 tool call → agent 结束
            if tool_call is None:
                return output
            
            tool_response_list =await self._handle_tool_call(tool_call,image_paths[0],step,max_rounds)
            if tool_response_list[0]==False:
                messages.append(
                    self.build_multimodal_tool_message(text=tool_response_list[1])
                )
                output["predictions"].append({"role": "user", "content":  "<tool_response>\n"+tool_response_list[1]+"\n</tool_response>"})

            if tool_response_list[0]==True and len(tool_response_list)==2:
                messages.append(
                    self.build_multimodal_tool_message(image_path=tool_response_list[1])
                )
                output["predictions"].append({"role": "user", "content":  "<tool_response>\n<image>\n</tool_response>"})
                output["images"].append(tool_response_list[1])

            if tool_response_list[0]==True and len(tool_response_list)==3:
                messages.append(
                    self.build_multimodal_tool_message(image_path=tool_response_list[1],text=tool_response_list[2])
                )
                output["predictions"].append({"role": "user", "content":  "<tool_response>\n<image>\n"+tool_response_list[2]+"\n</tool_response>"})
                output["images"].append(tool_response_list[1])

        print(f"[WARN] uid={uid} exceeded max_rounds, drop this sample")
        return None


if __name__ == "__main__":
    agent = ElementExtractor(
        # base_url="http://localhost:8000/v1",
        # api_key="sk-123456",
        # model_name="Qwen3-VL-4B-Instruct",
        base_url="http://35.220.164.252:3888/v1",
        api_key="sk-6TGzZJkJ5HfZKwnrS1A1pMb1lH5D7EDfSVC6USq24aN2JaaR",
        model_name="qwen3-vl-plus",
        tool=ImageZoomOCRTool(work_dir="/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/src/agents/workspace")
    )
    file_path = "/mnt/shared-storage-user/mineru2-shared/madongsheng/dataset/codevision/codevision_rl_with_ids.json"

    # 读取 JSON 文件
    with open(file_path, "r", encoding="utf-8") as f:
        inputs = json.load(f) # 会把 JSON 转换为 Python 对象（列表或字典）
    
    item = inputs[0]
    user_text=item["conversations"][0]["content"][8:]
    image_paths=[item["images"][0]]
    
    results = asyncio.run(agent.run_agent(user_text,image_paths))
    pass