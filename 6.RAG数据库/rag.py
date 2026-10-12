"""
第四阶段 + 主程序：RAG 问答。

流程：
    用户问题 → 向量化 → 与所有文本块算余弦相似度 → 取 Top-K
             → 拼「问题 + 资料」提示词 → 调 DeepSeek → 输出回答 + 来源

最终回答格式：
    回答：……
    参考资料：
    - 文件名1
    - 文件名2

知识库没有相关信息时，模型应回答「资料中未提及」。

运行：
    python rag.py                 # 交互模式
    python rag.py --test          # 跑内置四道测试题并写入 results.md
    python rag.py "你的问题"       # 单条提问
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ============ 配置 ============
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
API_URL = "https://api.deepseek.com/chat/completions"
MODEL_NAME = "deepseek-flash"          # DeepSeek-V4.1-Flash

CHUNKS_FILE = "chunks.json"
VECTORS_FILE = "vectors.npy"
EMBED_MODEL = "BAAI/bge-small-zh"
RESULTS_FILE = "results.md"

TOP_K = 3
# 相似度低于该值认为检索不到有效资料，直接回答"资料中未提及"
SCORE_THRESHOLD = 0.35

# 作业要求的四道测试题
TEST_QUESTIONS = [
    "本次招新的材料提交截止时间是什么？",
    "API Key 应如何保存，哪些内容不能提交到 GitHub？",
    "没有独立显卡的同学能否完成 RAG 题？应采用什么方式？",
    "本次招新的面试地点和面试时间是什么？",
]


# ============ 1. 加载索引 ============
def load_index():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    vectors = np.load(VECTORS_FILE)
    model = SentenceTransformer(EMBED_MODEL)
    return chunks, vectors, model


# ============ 2. 检索 ============
def search(query, chunks, vectors, model, top_k=TOP_K):
    """返回 Top-K 相关块（含相似度）。向量已归一化，点积即余弦相似度。"""
    q_vec = model.encode(query, normalize_embeddings=True)
    scores = vectors @ q_vec
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        c = chunks[int(idx)]
        results.append({
            "id": int(idx),
            "score": float(scores[idx]),
            "source": c["source"],
            "heading": c["heading"],
            "text": c["text"],
        })
    return results


# ============ 3. 拼提示词 ============
def build_prompt(question, results):
    context = ""
    for r in results:
        context += f"\n[来源：{r['source']} | {r['heading']}]\n{r['text']}\n"

    return f"""你是一个严谨的问答助手。请严格根据下面提供的资料回答问题。

规则：
1. 只能使用资料中的信息作答，禁止使用你自己已有知识补充。
2. 如果资料中没有足够信息，必须回答「资料中未提及」，不要猜测。
3. 回答末尾必须单独列出本次用到的来源文件名。

资料：
{context}

问题：{question}

请按以下格式回答：
回答：<你的回答>

参考资料：
- <用到的文件名>
"""


# ============ 4. 调用 DeepSeek ============
def call_llm(prompt):
    if not API_KEY:
        raise RuntimeError("环境变量 DEEPSEEK_API_KEY 未设置")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个只依据给定资料作答的问答助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "stream": False,
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ============ 5. 一个完整的问答回合 ============
def answer_question(question, chunks, vectors, model):
    results = search(question, chunks, vectors, model)

    # 检索质量兜底：最高分都过低，说明资料库答不了
    if results[0]["score"] < SCORE_THRESHOLD:
        answer_text = "资料中未提及"
        sources = []
    else:
        prompt = build_prompt(question, results)
        try:
            answer_text = clean_answer(call_llm(prompt))
        except Exception as e:
            answer_text = f"[调用大模型失败] {e}"
        # 只把实际用到的来源列出来（模型回答里可能只引用了部分）
        sources = list(dict.fromkeys(r["source"] for r in results))

    return results, answer_text, sources


# ============ 6. 输出格式化 ============
def clean_answer(raw: str) -> str:
    """模型有时会把「回答：」「参考资料：」这些格式标记也一起输出，
    这里剥掉，避免最终输出出现「回答：回答：」的重复。
    """
    text = raw.strip()
    # 去掉开头的 "回答：" / "**回答**：" 等前缀
    text = re.sub(r"^\**\s*回答\s*\**\s*[:：]\s*", "", text)
    # 截掉模型自己写的「参考资料」段（来源由程序统一列出，避免重复）
    text = re.split(r"\n\**\s*参考资料\s*\**\s*[:：]", text)[0]
    return text.strip()


def format_output(answer_text, sources):
    lines = [f"回答：{answer_text}", "", "参考资料："]
    if sources:
        lines += [f"- {s}" for s in sources]
    else:
        lines.append("- （无）")
    return "\n".join(lines)


def print_retrieval(results):
    print("===== 检索结果 =====")
    for i, r in enumerate(results, 1):
        print(f"[{i}] 来源：{r['source']} | {r['heading']}  相似度：{r['score']:.4f}")
        print(f"    {r['text'][:100].replace(chr(10), ' ')}...")
    print("====================")


def save_to_md(question, results, answer_text, sources):
    lines = [
        f"\n## 问题：{question}\n",
        f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "### 检索结果（Top-3）\n",
    ]
    for i, r in enumerate(results, 1):
        lines.append(f"**[{i}] 来源：{r['source']} | {r['heading']}  相似度：{r['score']:.4f}**\n")
        lines.append(f"> {r['text']}\n")
    lines.append("### 最终回答\n")
    lines.append(format_output(answer_text, sources))
    lines.append("\n---\n")

    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============ 7. 主流程 ============
def main():
    parser = argparse.ArgumentParser(description="RAG 问答（基于本地知识库）")
    parser.add_argument("question", nargs="?", help="要问的问题；不填则进入交互模式")
    parser.add_argument("--test", action="store_true", help="跑内置四道测试题并写入 results.md")
    args = parser.parse_args()

    chunks, vectors, model = load_index()
    print(f"已加载 {len(chunks)} 个文本块，模型：{MODEL_NAME}\n")

    if args.test:
        # 每次跑测试覆盖旧记录
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write("# RAG 测试记录\n")
        for q in TEST_QUESTIONS:
            print(f"\n### 问题：{q}")
            results, answer_text, sources = answer_question(q, chunks, vectors, model)
            print_retrieval(results)
            print(format_output(answer_text, sources))
            save_to_md(q, results, answer_text, sources)
        print(f"\n已写入 {RESULTS_FILE}")
        return

    if args.question:
        results, answer_text, sources = answer_question(args.question, chunks, vectors, model)
        print_retrieval(results)
        print(format_output(answer_text, sources))
        return

    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write("# RAG 测试记录\n")

    while True:
        try:
            question = input("请输入问题（q 退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() == "q":
            break
        if not question:
            continue

        results, answer_text, sources = answer_question(question, chunks, vectors, model)
        print_retrieval(results)
        print(format_output(answer_text, sources))
        print()
        save_to_md(question, results, answer_text, sources)
        print(f"[已保存到 {RESULTS_FILE}]\n")


if __name__ == "__main__":
    main()
