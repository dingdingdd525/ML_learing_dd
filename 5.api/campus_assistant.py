"""
任务二：校园信息助手（LangGraph 工作流）

图结构：

            ┌─────────────────────────────┐
    START ──►  relevance（相关性判断智能体）│
            └──────────────┬──────────────┘
                           │ score >= 阈值 ?
              ┌────────────┴────────────┐
              │ 否                      │ 是
              ▼                         ▼
        ┌──────────┐            ┌──────────────────┐
        │ rejected │            │ extract（抽取）  │
        │ 固定回复 │            │ JSON 规范化输出  │
        └────┬─────┘            └────────┬─────────┘
             │                           │
             └──────────► END ◄──────────┘

- relevance 节点：无工具，同一模型，判断文本是否与校园相关，输出 0~1 的小数
- 阈值（默认 0.5）在 config.py 里，可调
- extract 节点：用提示词强约束输出 JSON（一句话总结 / 关键信息 / 待办事项）
- rejected 节点：返回固定文案，不调用额外推理

运行：
    python campus_assistant.py                # 交互模式（终端输入）
    python campus_assistant.py --demo         # 跑内置三类样例
    python campus_assistant.py "你的文本"      # 单条
"""

import argparse
import json
import re
import sys

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from config import RELEVANCE_THRESHOLD
from llm_client import content_to_str, extract_json, get_llm


# --------------------------------------------------------------------------
# 1. 状态定义：LangGraph 里节点之间传递的数据
# --------------------------------------------------------------------------
class CampusState(TypedDict, total=False):
    text: str            # 用户输入
    relevance: float     # 相关性得分 0~1
    is_campus: bool      # 是否通过阈值
    result: dict         # 最终结构化结果
    source: str          # 由哪个分支产出（extract / rejected）


# --------------------------------------------------------------------------
# 2. 提示词
# --------------------------------------------------------------------------
RELEVANCE_PROMPT = """你是一个文本分类器，不是聊天助手。

判断下面这段文本是否与【校园生活】相关（校园生活包括：课程、作业、考试、讲座、社团、宿舍、食堂、图书馆、校园活动、学生在校内的学习问题等）。

只输出一个 JSON 对象，不要任何解释、不要 markdown 代码块，格式严格如下：
{{"score": 0.87}}

score 是 0 到 1 之间的小数，保留两位：
- 越接近 1，表示越确定与校园相关
- 越接近 0，表示越确定与校园无关
- 模棱两可时给中间值

待判断文本：
---
{text}
---
"""

EXTRACT_PROMPT = """你是一个校园信息抽取助手。请从下面的校园文本中抽取信息。

必须严格遵守：
1. 只输出一个 JSON 对象，不要输出任何解释文字，不要用 markdown 代码块包裹。
2. JSON 必须且只能包含这三个字段：
   - "summary": 一句话总结（字符串，不超过 50 字）
   - "key_info": 关键信息（字符串数组，每条是一个短句；没有则填 []）
   - "todos": 待办事项（字符串数组，每条是一个动作短句；没有则填 []）
3. 原文没有明确提到的时间、地点、人物，一律不要编造，直接忽略。
4. 不要重复原文中的整段话，只要要点。

输出格式示例：
{{"summary": "...", "key_info": ["...", "..."], "todos": ["..."]}}

待处理的校园文本：
---
{text}
---
"""

# 与校园无关时的固定回复
REJECT_MESSAGE = "该内容与校园背景无关，未做处理。"

# 用于幻觉检测：出现这些词说明输出里带了时空信息
_TIME_PAT = re.compile(r"(周[一二三四五六日天]|\d+\s*[点时]|今天|明天|后天|下周|本周|上午|下午|晚上)")
_PLACE_PAT = re.compile(r"(楼|馆|室|厅|报告厅|教室|宿舍|食堂|图书馆|操场|广场)")


def detect_hallucination(text: str, result: dict) -> list[str]:
    """检查输出里是否出现了原文没有的时间/地点信息（模型编造检测）。

    LLM 有很强的"把通知补完整"的先验，即使提示词禁止编造也可能漏网。
    这里做一层程序化兜底，发现疑似编造就提醒使用者人工核对。
    """
    warnings = []
    blob = result.get("summary", "") + "".join(result.get("key_info", [])) + "".join(result.get("todos", []))
    if _TIME_PAT.search(blob) and not _TIME_PAT.search(text):
        warnings.append("输出中包含时间信息，但原文未提及，疑似模型编造")
    if _PLACE_PAT.search(blob) and not _PLACE_PAT.search(text):
        warnings.append("输出中包含地点信息，但原文未提及，疑似模型编造")
    return warnings


# --------------------------------------------------------------------------
# 3. 节点函数
# --------------------------------------------------------------------------
def relevance_node(state: CampusState) -> dict:
    """判断是否校园相关，输出 0~1 概率。无工具，纯判定。"""
    llm = get_llm(json_mode=True, temperature=0)
    raw = content_to_str(llm.invoke(RELEVANCE_PROMPT.format(text=state["text"])).content)

    try:
        score = float(extract_json(raw).get("score"))
    except (ValueError, TypeError):
        # 解析失败时保守处理：当作低相关，走兜底分支
        score = 0.0
    score = max(0.0, min(1.0, score))

    return {"relevance": score, "is_campus": score >= RELEVANCE_THRESHOLD}


def extract_node(state: CampusState) -> dict:
    """把校园文本抽成固定格式 JSON。"""
    llm = get_llm(json_mode=True, temperature=0)
    raw = content_to_str(llm.invoke(EXTRACT_PROMPT.format(text=state["text"])).content)

    try:
        data = extract_json(raw)
    except ValueError as e:
        # 兜底：仍然返回固定结构，保证下游格式统一
        data = {"summary": "", "key_info": [], "todos": [], "error": str(e)}

    # 字段规范化，保证下游拿到的永远是三件套
    result = {
        "summary": str(data.get("summary", "")),
        "key_info": list(data.get("key_info", []) or []),
        "todos": list(data.get("todos", []) or []),
    }
    if "error" in data:
        result["error"] = data["error"]

    return {"result": result, "source": "extract"}


def rejected_node(state: CampusState) -> dict:
    """与校园无关：返回固定回复，不做多余推理。"""
    return {
        "result": {"summary": REJECT_MESSAGE, "key_info": [], "todos": []},
        "source": "rejected",
    }


def route_by_relevance(state: CampusState) -> str:
    """条件边：按阈值分流。"""
    return "extract" if state.get("is_campus") else "rejected"


# --------------------------------------------------------------------------
# 4. 构图
# --------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(CampusState)
    builder.add_node("relevance", relevance_node)
    builder.add_node("extract", extract_node)
    builder.add_node("rejected", rejected_node)

    builder.add_edge(START, "relevance")
    builder.add_conditional_edges(
        "relevance",
        route_by_relevance,
        {"extract": "extract", "rejected": "rejected"},
    )
    builder.add_edge("extract", END)
    builder.add_edge("rejected", END)
    return builder.compile()


# --------------------------------------------------------------------------
# 5. 结果展示
# --------------------------------------------------------------------------
def render(state: CampusState) -> str:
    if state.get("source") == "rejected":
        return f"[判定] 相关性 {state.get('relevance'):.2f} < 阈值 {RELEVANCE_THRESHOLD}\n{REJECT_MESSAGE}"

    r = state["result"]
    lines = [
        f"[判定] 相关性 {state.get('relevance'):.2f} >= 阈值 {RELEVANCE_THRESHOLD}",
        f"一句话总结：{r['summary']}",
        "关键信息：" + ("；".join(r["key_info"]) if r["key_info"] else "（无）"),
        "待办事项：" + ("；".join(r["todos"]) if r["todos"] else "（无）"),
    ]
    if "error" in r:
        lines.append(f"[警告] JSON 解析失败，已兜底：{r['error'][:120]}")

    # 幻觉兜底检查：原文没给的信息，模型不该编
    for w in detect_hallucination(state["text"], r):
        lines.append(f"[警告] {w}，请人工核对。")

    # 正向提示：原文确实缺少关键要素时，明确告知，而不是沉默留空
    if not _TIME_PAT.search(state["text"]):
        lines.append("[提示] 原文未提及具体时间。")
    if not _PLACE_PAT.search(state["text"]):
        lines.append("[提示] 原文未提及具体地点。")

    return "\n".join(lines)


DEMO_INPUTS = [
    "本周五前提交高数作业，作业包括第三章习题 1、3、5。请写在作业本上，下周一课上交。",
    "周六下午三点将在图书馆报告厅举办人工智能讲座，主讲人是计算机学院李教授，请感兴趣的同学提前十分钟入场。",
    "Python 中列表和元组有什么区别？什么时候应该使用元组？",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="校园信息助手（LangGraph）")
    parser.add_argument("text", nargs="?", help="要处理的文本；不填则进入交互模式")
    parser.add_argument("--demo", action="store_true", help="跑内置三类样例")
    parser.add_argument("--json", action="store_true", help="以 JSON 形式打印结果")
    args = parser.parse_args()

    graph = build_graph()

    def run_one(text: str) -> None:
        state = graph.invoke({"text": text})
        if args.json:
            print(json.dumps(state, ensure_ascii=False, indent=2))
        else:
            print(render(state))

    print(f"模型：deepseek-flash | 阈值：{RELEVANCE_THRESHOLD}\n")

    if args.demo:
        for i, text in enumerate(DEMO_INPUTS, 1):
            print(f"===== 样例 {i} =====")
            print(f"输入：{text}")
            run_one(text)
            print()
        return

    if args.text:
        run_one(args.text)
        return

    # 交互模式
    print("输入校园相关文本后回车，空行或 Ctrl+C 退出。\n")
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            break
        run_one(text)
        print()


if __name__ == "__main__":
    main()
