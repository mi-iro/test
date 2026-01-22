import os
from openai import OpenAI
from typing import Callable
import time

class LLMCaller:
    def __init__(self, 
                 base_url: str, 
                 api_key: str, 
                 model_name: str, 
                 temperature: float = 0.0,
                 max_retries: int = 3):
        """
        初始化 LLM Caller
        :param base_url: 模型 API 地址 (如 OpenAI 官方或 vLLM/SGLang 地址)
        :param api_key: API Key
        :param model_name: 模型名称 (如 "gpt-4o", "qwen3-vl-plus")
        :param temperature: 温度系数，提取任务建议设为 0
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries

    def __call__(self, prompt: str) -> str:
        """
        使得实例可以直接被调用，符合 Callable[[str], str] 签名
        """
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    # 可以根据需要添加 max_tokens
                )
                content = response.choices[0].message.content
                return content if content else ""
            except Exception as e:
                print(f"[LLM Caller] Error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2) # 等待后重试
                else:
                    print(f"[LLM Caller] Failed to get response after {self.max_retries} attempts.")
                    return ""

# 工厂函数，方便快速创建
def create_llm_caller(base_url: str = "http://localhost:3888/v1", api_key: str = "sk-6TGzZJkJ5HfZKwnrS1A1pMb1lH5D7EDfSVC6USq24aN2JaaR", model_name: str = "qwen3-max") -> Callable[[str], str]:
    caller = LLMCaller(base_url, api_key, model_name)
    return caller