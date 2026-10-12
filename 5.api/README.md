# 大模型 API 初体验：校园信息助手

用 LangGraph 搭建的一条最小工作流：先判断输入是否与校园相关，相关才做结构化抽取。

- **平台**：DeepSeek 开放平台（OpenAI 兼容接口）
- **模型**：`deepseek-flash`（DeepSeek-V4.1-Flash），base_url `https://api.deepseek.com`
- **框架**：LangGraph 1.2 + langchain-deepseek 1.1
- **Key 来源**：环境变量 `DEEPSEEK_API_KEY`（不写进代码、不进仓库）

---

## 1. 环境配置

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 配置 API Key（二选一）
# 方式 A：直接设置环境变量
#   Windows (PowerShell):  $env:DEEPSEEK_API_KEY = "sk-xxxx"
#   Windows (CMD):         set DEEPSEEK_API_KEY=sk-xxxx
#   Linux / macOS:         export DEEPSEEK_API_KEY=sk-xxxx
# 方式 B：复制 .env.example 为 .env 并填写
cp .env.example .env
```

> `.env` 已在 `.gitignore` 中，不会被提交。`.env.example` 只保留变量名。

---

## 2. 运行命令

### 任务一：最小 API 调用

```bash
python hello_api.py                    # 默认问一句
python hello_api.py "你的问题"          # 自定义提问
```

### 任务二：校园信息助手（LangGraph 工作流）

```bash
python campus_assistant.py                          # 交互模式（终端逐条输入）
python campus_assistant.py --demo                   # 跑内置三类样例
python campus_assistant.py "本周五前提交高数作业……"   # 单条
python campus_assistant.py "文本" --json             # 输出完整状态（含相关性得分）
```

终端输入空行或 `Ctrl+C` 退出交互模式。

### 任务三：提示词对比实验

```bash
python compare_prompts.py              # 默认每类输入每种提示词跑 3 次
python compare_prompts.py --runs 5     # 自定义重复次数
```

结果写入 `results.md`（含逐次原始输出 + 稳定性汇总 + 分析）。

---

## 3. 工作流说明（任务二）

```
            ┌──────────────────────────────┐
    START ──► relevance（相关性判断，无工具）│
            └──────────────┬───────────────┘
                           │ score >= 0.5 ?
              ┌────────────┴────────────┐
              │ 否                      │ 是
              ▼                         ▼
        ┌──────────┐            ┌──────────────────┐
        │ rejected │            │ extract          │
        │ 固定回复 │            │ JSON 结构化抽取  │
        └────┬─────┘            └────────┬─────────┘
             └──────────► END ◄──────────┘
```

- **relevance 节点**：同一个模型、不带工具，只做判定，输出 `{"score": 0.0~1.0}`。
- **条件边**：`score >= RELEVANCE_THRESHOLD`（`config.py`，默认 0.5）走 `extract`，否则走 `rejected`。
- **extract 节点**：提示词强约束输出三个固定字段的 JSON。
- **rejected 节点**：返回固定文案，不再调用推理。
- 程序另做一层**幻觉检测**：输出中出现原文没有的时间/地点词时打印警告；原文缺时间/地点时主动提示。

### 三类样例实测

| 输入类型 | 相关性得分 | 分支 |
|---|---|---|
| 课程/作业通知 | 0.98 | extract |
| 活动/讲座通知 | 0.98 | extract |
| 技术问题 | 0.72 | extract |
| （对照）菜谱 | 0.02 | rejected |
| （对照）股市行情 | 0.02 | rejected |

---

## 4. 文件说明

| 文件 | 作用 |
|---|---|
| `hello_api.py` | 任务一：最小 API 调用 |
| `campus_assistant.py` | 任务二：LangGraph 工作流主程序 |
| `llm_client.py` | 模型封装、JSON 容错解析 |
| `config.py` | 阈值等配置 |
| `compare_prompts.py` | 任务三：提示词对比实验脚本 |
| `results.md` | 任务三的实验结果与分析 |
| `requirements.txt` | 依赖清单 |
| `.env.example` | 环境变量模板（不含真实 Key） |
| `.gitignore` | 忽略 `.env` 等 |

---

## 5. 提示词设计思路

（详见 `results.md` 的分析部分，要点如下）

1. **角色收窄输出空间**：给模型一个明确身份（"校园信息抽取助手"），比"请总结"的开放指令更不容易跑偏。
2. **给 JSON 模板而不是描述**：直接给出字段名 + 类型的骨架让模型照抄，比自然语言描述格式可靠得多。
3. **约束写成硬规则**：禁止编造、空值填 `[]`、禁止输出多余文字——这三条分别堵住了幻觉、字段缺失、格式污染。
4. **提示词 + 程序双保险**：提示词能保证"格式"，但不能保证"事实"，所以额外加了时空词校验做兜底。

---

## 6. 观察到的模型局限

最典型的一次：**用简单提示词 "请总结以下通知" 处理技术问题（"Python 列表和元组有什么区别"）时，三次运行模型全部拒绝总结并反问用户**（"这看起来更像是一个技术问题，而不是通知，请提供完整内容"）。

说明简单提示词无法覆盖输入的体裁多样性，模型会自行改变行为，导致程序拿到散文而非结构化数据。改用含角色/格式/约束的提示词后可稳定输出 JSON。

但同时暴露了反向问题：强制套格式会让模型对**任何**输入都硬套模板（例如对完全无关的内容也生成 `"一句话总结"`）。所以"该不该处理这个输入"不能只靠抽取提示词，必须由上游相关性判断节点兜住。
