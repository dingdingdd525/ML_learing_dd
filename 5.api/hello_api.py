"""
任务一：最小 API 调用。

向 DeepSeek 发一条消息，打印回答。
API Key 从环境变量 DEEPSEEK_API_KEY 读取，绝不写进代码。

运行：
    python hello_api.py
    python hello_api.py "你的问题"
"""

import os
import sys

from langchain_deepseek import ChatDeepSeek

# Key 必须在环境变量里，ChatDeepSeek 默认就是读 DEEPSEEK_API_KEY
if not os.environ.get("DEEPSEEK_API_KEY"):
    sys.exit("错误：未找到环境变量 DEEPSEEK_API_KEY，请先配置（见 README）。")

llm = ChatDeepSeek(
    model="deepseek-flash",   # DeepSeek-V4.1-Flash
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}},  # 关掉思考模式，省钱且响应快
)

question = sys.argv[1] if len(sys.argv) > 1 else "用一句话介绍你自己"

resp = llm.invoke(question)
print(resp.content)
