"""
第三阶段：建立向量索引。

读取 chunks.json，用中文 Embedding 模型把每块的 text 转成向量，
归一化后存为 vectors.npy（配合 rag.py 里的余弦相似度检索）。

运行：
    python build_index.py
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "chunks.json"
VECTORS_FILE = "vectors.npy"
EMBED_MODEL = "BAAI/bge-small-zh"  # 中文语义匹配，约 95MB，CPU 可跑


def main() -> None:
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"共加载 {len(chunks)} 个文本块")

    model = SentenceTransformer(EMBED_MODEL)
    print(f"Embedding 模型：{EMBED_MODEL}")

    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    print("向量矩阵形状：", vectors.shape)

    np.save(VECTORS_FILE, vectors)
    print(f"向量已保存到 {VECTORS_FILE}")


if __name__ == "__main__":
    main()
