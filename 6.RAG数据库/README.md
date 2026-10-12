
# 基于本地资料的 RAG 问答程序

一个最小的 RAG（Retrieval-Augmented Generation）命令行问答程序：先从本地知识库检索相关文本块，再交给大模型生成回答，并列出用到的来源文件。

- **知识库**：`knowledge_base/` 下 4 个 Markdown 文件（原始资料，未修改）
- **Embedding 模型**：`BAAI/bge-small-zh`（中文小模型，约 95MB，CPU 可跑）
- **大模型**：DeepSeek `deepseek-flash`（DeepSeek-V4.1-Flash），OpenAI 兼容接口
- **向量检索**：numpy 余弦相似度（无需 FAISS/Chroma）
- **知识库无答案时**：输出「资料中未提及」，不编造

---

## 目录结构

```
6.RAG数据库/
├── knowledge_base/          # 知识库原始文件（4 个 .md，不得修改）
│   ├── 01_overview.md
│   ├── 02_tasks.md
│   ├── 03_submission.md
│   └── 04_faq.md
├── cut.py                   # 阶段 1+2：读取 + 切分 → chunks.json
├── build_index.py           # 阶段 3：向量化 → vectors.npy
├── rag.py                   # 阶段 4+5：检索 + 调用大模型
├── chunks.json              # 切分结果（17 个文本块）
├── vectors.npy              # 向量索引（17 × 512）
├── results.md               # 四道测试题的检索结果与最终回答
├── 思考题作答.md             # 思考题（答案待填写）
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

---

## 环境配置

要求 Python 3.10 或更高（本项目在 3.11.9 上测试）。

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 配置 API Key（二选一）
# 方式 A：设置环境变量
#   Windows PowerShell:  $env:DEEPSEEK_API_KEY = "sk-xxxx"
#   Linux / macOS:       export DEEPSEEK_API_KEY=sk-xxxx
# 方式 B：复制 .env.example 为 .env 并填写
cp .env.example .env
```

> `.env` 已在 `.gitignore` 中，不会被提交。`.env.example` 只保留变量名。

**Embedding 模型下载**：首次运行会自动从 HuggingFace 下载 `BAAI/bge-small-zh`。
若网络受限，可先手动下载模型到本地，并修改 `cut.py` / `build_index.py` / `rag.py` 中的 `EMBED_MODEL` 为本地路径。
模型已缓存时，可加环境变量强制离线：`set HF_HUB_OFFLINE=1`。

---

## 运行步骤

按顺序执行，每步都能单独验证：

```bash
# 阶段 1+2：读取知识库并切分
python cut.py
#   → 输出每个文件切出多少块、切分自检结果、块预览
#   → 生成 chunks.json

# 阶段 3：建立向量索引
python build_index.py
#   → 生成 vectors.npy

# 阶段 4+5：问答
python rag.py                      # 交互模式（输入 q 退出）
python rag.py "本次招新的截止时间是什么？"   # 单条提问
python rag.py --test               # 跑内置四道测试题，写入 results.md
```

---

## 关键设置说明

### 文本切分（作业要求第 2 条）

切分逻辑在 `cut.py`，设置如下：

| 项 | 取值 | 说明 |
|---|---|---|
| 切分依据 | Markdown 标题 | 按 `#` / `##` / `###` 切分 |
| 标题路径 | 拼接父级标题 | 每块标题形如 `# 招新题任务说明 > ## 基础题`，保留上下文 |
| 最大块长 | 400 字符 | 超过则按句子边界二次切分 |
| 重叠长度 | 80 字符 | 二次切分时相邻块重叠，避免语义截断 |
| 来源保留 | 每块带 `source` 字段 | 始终知道文本来自哪个文件 |

**修复说明**：早期版本把文件的 `# 一级标题` 单独切成一块，导致该块 `content` 为空（只有标题行）、共出现 4 个空块。这些空块因为内容就是「招新说明」「任务说明」等高频词，检索相似度虚高，会挤占 Top-K 名额。现在一级标题不再单独成块，其正文并入首个二级小节，并由程序在 `cut.py` 中做自检（发现空块直接报错退出）。

### 检索设置

| 项 | 取值 |
|---|---|
| Top-K | 3 |
| 相似度 | 余弦相似度（向量已归一化，点积即余弦） |
| 兜底阈值 | 0.35（最高相似度低于此值时，直接回答「资料中未提及」） |

### 提示词要点

`rag.py` 的 `build_prompt()` 明确要求模型：仅根据提供的资料回答、无法作答时回答「资料中未提及」、末尾列出来源文件名。

---

## 四道测试题结果

完整记录（含检索到的文本块、相似度、最终回答）见 `results.md`。摘要：

| # | 问题 | Top-1 来源 | 相似度 | 最终回答 |
|---|---|---|---|---|
| 1 | 本次招新的材料提交截止时间是什么？ | 01_overview.md | 0.840 | 2026 年 10 月 18 日 23:59 |
| 2 | API Key 应如何保存，哪些内容不能提交到 GitHub？ | 03_submission.md | 0.881 | 用 `.env` 保存并在 `.gitignore` 忽略；真实 Key/密码/敏感信息不得提交 |
| 3 | 没有独立显卡的同学能否完成 RAG 题？应采用什么方式？ | 04_faq.md | 0.889 | 可以；用云端 API 生成回答 + CPU 上建小规模本地向量索引 |
| 4 | 本次招新的面试地点和面试时间是什么？ | 01_overview.md | 0.781 | **资料中未提及** |

第 4 题是设计上的关键验证：知识库里根本没有「面试」相关信息，Top-1 只是话题相近的「时间安排」块。程序靠提示词约束 + 相似度兜底保证输出「资料中未提及」，而不是让模型拿自己知识编一个面试时间来。

---

## 思考题

见 `思考题作答.md`（含检索与生成的分工、为何不整篇塞提示词、RAG 与微调的区别、检索不相关时的风险）。

---

## 常见问题

**Q：Embedding 模型下载失败？**
检查网络；或手动下载模型后把代码里的 `EMBED_MODEL` 改成本地路径。实际使用的模型名已记录在本 README 与代码中。

**Q：检索结果不相关？**
先看 `cut.py` 的预览输出，确认文本块长度是否合理（过短丢上下文、过长稀释语义）；再调整 `MAX_CHARS`、`OVERLAP` 或 `TOP_K`。

**Q：模型没有回答「资料中未提及」？**
检查 `rag.py` 的提示词是否明确要求「仅根据资料回答」，以及是否把无关文本块也传给了模型（可调低 `SCORE_THRESHOLD`）。

**Q：必须用 FAISS / Chroma / LangChain 吗？**
不必。本项目用 numpy 手写余弦相似度完成检索，完整走通「切分 → 向量化 → 检索 → 基于检索结果调 API」链路。
