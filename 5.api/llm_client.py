"""
通用 LLM 封装。

- 统一从环境变量 DEEPSEEK_API_KEY 取 key（langchain-deepseek 内置行为，这里显式校验）
- 统一关闭思考模式，控制成本、稳定 JSON 输出
- 提供 fallback JSON 解析，容忍模型在 JSON 外套 markdown 代码块 / 加前后废话
"""

import json
import os
import re
import sys

from langchain_deepseek import ChatDeepSeek

MODEL_NAME = "deepseek-flash"  # DeepSeek-V4.1-Flash

# 思考模式关掉：本任务是抽取/分类，不需要长推理，开着又慢又费 token
_DISABLE_THINKING = {"thinking": {"type": "disabled"}}


def check_api_key() -> None:
    """确保 key 来自环境变量而不是硬编码。"""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit(
            "错误：未找到环境变量 DEEPSEEK_API_KEY。\n"
            "请设置后重试，或复制 .env.example 为 .env 并填入你的 Key。"
        )


def get_llm(*, json_mode: bool = False, temperature: float = 0.0) -> ChatDeepSeek:
    """返回一个配置好的 ChatDeepSeek 实例。

    json_mode=True 时会带 response_format={"type": "json_object"}，
    让 DeepSeek 走官方的 JSON Output 模式（前提是提示词里出现 "json" 字样）。
    """
    check_api_key()
    extra_body = dict(_DISABLE_THINKING)
    if json_mode:
        # 官方 JSON Output 开关；必须在 prompt 里明确要求输出 JSON
        extra_body["response_format"] = {"type": "json_object"}
    return ChatDeepSeek(
        model=MODEL_NAME,
        temperature=temperature,
        max_retries=2,
        extra_body=extra_body,
    )


def extract_json(text: str) -> dict:
    """从模型输出里抠出 JSON 对象，容忍 ```json 代码块和前后废话。

    这类容错是必须的：即使开了 JSON mode，模型偶尔也会包一层 markdown。
    历史上这里最容易出问题，所以单独做一层兜底 + 抛清晰错误。
    """
    if not isinstance(text, str):
        raise ValueError(f"期望字符串，实际拿到 {type(text)}")

    # 1) 直接尝试
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2) 剥掉 ```json ... ``` 代码块
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3) 暴力截取第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从模型输出中解析出 JSON：\n{text!r}")


def content_to_str(content) -> str:
    """把 LangChain 的 message.content 归一化成纯字符串。

    有些模型的 content 会是 list[dict]（分块结构），这里统一拼起来。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)
