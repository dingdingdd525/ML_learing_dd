"""
第一阶段 + 第二阶段：读取知识库 → 切分文本块。

切分策略：
  1. 先按 Markdown 标题（#/##/###）切分；
  2. 记录的标题采用「标题路径」形式（如 `# 标题 > ## 小节`），
     这样每个块都自带上下文，检索时不会因为丢失父级标题而跑偏；
  3. 文件的一级标题（# 开头）不再单独成块——它只是文件标题，
     正文并入紧随其后的第一个小节。这样避免出现 content 为空的块；
  4. 若某个块长度超过 MAX_CHARS，按句子边界二次切分，
     相邻小块保留 OVERLAP 字符重叠，且每块都继承同一个标题路径。

每块结构：
  {
    "id": 全局自增,
    "source": 来源文件名,
    "heading": 标题路径（如 "# 招新题任务说明 > ## 基础题"）,
    "text": 标题路径 + 正文（用于向量化和展示）
  }

运行：
    python cut.py
"""

import json
import re
from pathlib import Path

# ---------------- 配置 ----------------
KB_DIR = "knowledge_base"
OUTPUT_FILE = "chunks.json"

# 长块二次切分的上限与重叠（单位：字符）
MAX_CHARS = 400
OVERLAP = 80

# 句子边界：中英文句号、问号、感叹号、分号
_SENT_END = re.compile(r"(?<=[。！？；!?;])")


def _split_long_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP):
    """把过长的正文按句子边界切成若干段，段间保留 overlap 字符重叠。

    返回纯文本段列表（不含标题）。
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    # 按句子拆开，再贪心地拼回到 max_chars 以内
    sentences = [s for s in _SENT_END.split(text) if s.strip()]
    segments, buf = [], ""
    for sent in sentences:
        if buf and len(buf) + len(sent) > max_chars:
            segments.append(buf.strip())
            # 用上一段结尾的 overlap 字符做重叠，保持语义连贯
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = tail + sent
        else:
            buf += sent
    if buf.strip():
        segments.append(buf.strip())
    return segments


def split_markdown(text: str, source: str):
    """按 Markdown 标题切分单个文档，返回块列表。"""
    lines = text.splitlines()
    chunks = []

    # 记录当前各级标题，用于拼出标题路径
    heading_stack = {}  # level -> 标题文本
    cur_heading_path = ""      # 当前块的标题路径
    cur_body_lines = []        # 当前块正文
    file_title = ""            # 文件的一级标题，不单独成块

    def flush():
        """把当前累积的正文落成块（标题为空但有正文时也保留）。"""
        body = "\n".join(cur_body_lines).strip()
        if not body and not cur_heading_path:
            return
        for seg in _split_long_text(body) if body else [cur_heading_path]:
            full = (cur_heading_path + "\n" + seg).strip() if seg != cur_heading_path else cur_heading_path
            chunks.append({
                "source": source,
                "heading": cur_heading_path,
                "text": full,
            })

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()

            # 遇到新标题：先把上一块落盘
            flush()
            cur_body_lines = []

            # 更新标题栈：当前级别写入，更深的级别丢弃
            heading_stack = {k: v for k, v in heading_stack.items() if k < level}
            heading_stack[level] = title

            # 一级标题 = 文件标题，暂时不作为独立块的标题
            if level == 1:
                file_title = title
                # 该块暂时不产生标题路径，正文会并入下一个小节
                cur_heading_path = ""
                continue

            # 用栈里所有标题拼成路径，形如 "# 文件标题 > ## 小节"
            parts = []
            for lv in sorted(heading_stack):
                prefix = "#" * lv
                parts.append(f"{prefix} {heading_stack[lv]}")
            cur_heading_path = " > ".join(parts)
        else:
            cur_body_lines.append(line)

    # 文档末尾最后一块
    flush()

    # 如果整篇文档没有任何 ## 级标题（只有 # 文件标题 + 正文），
    # 上面会把正文并到空 heading 的块里；此时回填文件标题
    if file_title:
        for c in chunks:
            if not c["heading"]:
                c["heading"] = f"# {file_title}"
                c["text"] = f"# {file_title}\n{c['text']}".strip()

    return chunks


def load_and_split(kb_dir: str = KB_DIR):
    all_chunks = []
    kb_path = Path(kb_dir)

    if not kb_path.exists():
        raise FileNotFoundError(f"找不到知识库目录：{kb_dir}")

    md_files = sorted(kb_path.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"{kb_dir} 下没有 .md 文件")

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        chunks = split_markdown(text, md_file.name)
        print(f"[加载] {md_file.name} -> 切出 {len(chunks)} 块")
        all_chunks.extend(chunks)

    return all_chunks


def validate(chunks):
    """自检：不允许出现空 content / 空标题路径。"""
    problems = []
    for c in chunks:
        if not c["text"].strip():
            problems.append(c)
        if not c["heading"].strip():
            problems.append(c)
    return problems


def main():
    chunks = load_and_split()

    for i, chunk in enumerate(chunks):
        chunk["id"] = i

    problems = validate(chunks)
    if problems:
        print(f"\n[自检失败] 有 {len(problems)} 个块内容为空：")
        for c in problems:
            print(f"  id={c['id']} source={c['source']} heading={c['heading']!r}")
        raise SystemExit(1)
    print("\n[自检通过] 无空块。")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"共切出 {len(chunks)} 块，已保存到 {OUTPUT_FILE}\n")

    print("===== 切分预览 =====")
    for chunk in chunks:
        preview = chunk["text"][:60].replace("\n", " ")
        print(f"[id={chunk['id']:>2}] {chunk['source']} | {chunk['heading']}")
        print(f"       {preview}...")


if __name__ == "__main__":
    main()
