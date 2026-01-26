import re
import json
import asyncio
import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Optional
from .utils import * # Assuming local_image_to_data_url and TOOL_CALL_RE are here

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
    def __init__(self, base_url: str, api_key: str, model_name: str, tool, max_image_size: int = 2048):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model_name = model_name
        self.tool = tool
        self.max_image_size = max_image_size

    def compress_image(self, image_path: str) -> str:
        """
        Compresses/Resizes the image to ensure the longest side does not exceed self.max_image_size.
        Returns the base64 encoded data URL string.
        """
        try:
            with Image.open(image_path) as img:
                # Calculate new size maintaining aspect ratio
                width, height = img.size
                if max(width, height) > self.max_image_size:
                    scale = self.max_image_size / max(width, height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffered = BytesIO()
                # Use JPEG for compression efficiency unless transparency is needed
                fmt = 'PNG' if img.mode == 'RGBA' else 'JPEG'
                img.save(buffered, format=fmt)
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                mime_type = f"image/{fmt.lower()}"
                
                return f"data:{mime_type};base64,{img_str}"
        except Exception as e:
            print(f"[ERROR] Failed to compress image {image_path}: {e}")
            # Fallback to original method if compression fails
            return local_image_to_data_url(image_path)

    def build_multimodal_user_message(
        self,
        text: str,
        image_paths: Optional[List[str]] = None
    ) -> Dict:
        content = []

        if image_paths:
            for path in image_paths:
                # Use the compression method here
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": self.compress_image(path)
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
            # Use the compression method here
            content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": self.compress_image(image_path)
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

    def extract_tool_call(self, text: str, as_json: bool = False):
        """
        Extract content from <tool_call> tag.
        """
        # Ensure TOOL_CALL_RE is defined in .utils or locally
        # If not, you might need: TOOL_CALL_RE = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
        match = TOOL_CALL_RE.search(text)
        if not match:
            return None
        
        tool_text = match.group(1).strip()
        
        if as_json:
            try:
                return json.loads(tool_text)
            except json.JSONDecodeError:
                return None
        else:
            return tool_text

    # ---------- tool mock ----------

    async def _handle_tool_call(self, tool_call, image_path, step, max_rounds) -> str:
        if step >= max_rounds - 2:
            return [False, "You have used up all the available uses of `image_zoom_and_ocr_tool`, please return you final response without use tool."]
        
        result_list = await self.tool.call(tool_call, image_path)
        
        if result_list[0] == False:
            result_list = await self.tool.call(tool_call, image_path)
            if result_list[0] == False: # Retry once
                return [False, "`image_zoom_and_ocr_tool` is wrong, you can try it again."]
        
        if len(result_list) == 2:
            return [True, result_list[1]]
        
        if len(result_list) == 3:
            return [True, result_list[1], f"{result_list[2]}"]

    # ---------- agent loop ----------

    async def run_agent(
        self,
        user_text: str,
        image_paths: Optional[List[str]] = None,
        max_rounds: int = 10,
        uid: int = 1
    ):  
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            self.build_multimodal_user_message(user_text, image_paths),
        ]
        
        # Note: We store original paths in output log, but send compressed to model
        output = {
            "id": uid,
            "predictions": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "<image>\n" + user_text}
            ],
            "images": [image_paths[0]] if image_paths else []
        }

        for step in range(max_rounds):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=1.0
                )
            except Exception as e:
                print(f"[ERROR] API Call failed at step {step}: {e}")
                # You might want to break or continue depending on failure strategy
                break

            content = resp.choices[0].message.content

            messages.append({"role": "assistant", "content": content})
            output["predictions"].append({"role": "assistant", "content": content})

            tool_call = self.extract_tool_call(content, as_json=True)
            
            # No tool call -> agent finished
            if tool_call is None:
                return output
            
            # Use original image path for tool execution (to preserve precision),
            # but the visual feedback to the model will be compressed inside build_multimodal_tool_message
            tool_response_list = await self._handle_tool_call(tool_call, image_paths[0], step, max_rounds)
            
            if tool_response_list[0] == False:
                messages.append(
                    self.build_multimodal_tool_message(text=tool_response_list[1])
                )
                output["predictions"].append({"role": "user", "content": "<tool_response>\n" + tool_response_list[1] + "\n</tool_response>"})

            if tool_response_list[0] == True and len(tool_response_list) == 2:
                messages.append(
                    self.build_multimodal_tool_message(image_path=tool_response_list[1])
                )
                output["predictions"].append({"role": "user", "content": "<tool_response>\n<image>\n</tool_response>"})
                output["images"].append(tool_response_list[1])

            if tool_response_list[0] == True and len(tool_response_list) == 3:
                messages.append(
                    self.build_multimodal_tool_message(image_path=tool_response_list[1], text=tool_response_list[2])
                )
                output["predictions"].append({"role": "user", "content": "<tool_response>\n<image>\n" + tool_response_list[2] + "\n</tool_response>"})
                output["images"].append(tool_response_list[1])

        print(f"[WARN] uid={uid} exceeded max_rounds, drop this sample")
        return None

if __name__ == "__main__":
    # Ensure ImageZoomOCRTool is imported or defined
    # from .tools import ImageZoomOCRTool 

    agent = ElementExtractor(
        base_url="http://35.220.164.252:3888/v1",
        api_key="sk-6TGzZJkJ5HfZKwnrS1A1pMb1lH5D7EDfSVC6USq24aN2JaaR",
        model_name="qwen3-vl-plus",
        tool=ImageZoomOCRTool(work_dir="/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/src/agents/workspace"),
        max_image_size=2048 # Limit longest edge to 2048px
    )
    
    file_path = "/mnt/shared-storage-user/mineru2-shared/madongsheng/dataset/codevision/codevision_rl_with_ids.json"

    # Read JSON file
    with open(file_path, "r", encoding="utf-8") as f:
        inputs = json.load(f)
    
    item = inputs[0]
    # Defensive check for content structure
    if "conversations" in item and len(item["conversations"]) > 0:
        content_text = item["conversations"][0].get("content", "")
        # Assuming format like "<image>\nQuery..."
        user_text = content_text[8:] if content_text.startswith("<image>") else content_text
    else:
        user_text = "Analyze this page."
        
    image_paths = [item["images"][0]]
    
    results = asyncio.run(agent.run_agent(user_text, image_paths))