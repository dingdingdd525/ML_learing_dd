"""
任务三：提示词对比实验。

对同一段输入，分别用两种提示词调用模型，各重复 N 次，把结果写入 results.md：

  提示词 A（简单）：     "请总结以下通知：..."
  提示词 B（结构化）：   角色 + 输出格式 + 约束条件，强制 JSON

观察点：
  1. 两种输出的差别
  2. 哪种更稳定、格式更符合预期
  3. 原文缺时间/地点时，模型会不会编造

运行：
    python compare_prompts.py           # 默认每类输入各跑 3 次
    python compare_prompts.py --runs 5  # 自定义重复次数
    python compare_prompts.py --out results.md
"""

import argparse
import datetime

from llm_client import content_to_str, get_llm

# --------------------------------------------------------------------------
# 测试输入（三类，均取自作业要求）
# --------------------------------------------------------------------------
CASES = [
    {
        "id": "1",
        "kind": "课程/作业通知",
        "text": "本周五前提交高数作业，作业包括第三章习题 1、3、5。请写在作业本上，下周一课上交。",
    },
    {
        "id": "2",
        "kind": "活动/讲座通知",
        "text": "周六下午三点将在图书馆报告厅举办人工智能讲座，主讲人是计算机学院李教授，请感兴趣的同学提前十分钟入场。",
    },
    {
        "id": "3",
        "kind": "技术问题",
        "text": "Python 中列表和元组有什么区别？什么时候应该使用元组？",
    },
]

# 提示词 A：一句话，无格式约束
PROMPT_A = "请总结以下通知：\n{text}"

# 提示词 B：角色 + 输出格式 + 约束条件
PROMPT_B = """你是一个校园信息抽取助手。请从下面的文本中抽取信息。

严格按以下 JSON 格式输出，不要输出任何其他文字、不要用 markdown 代码块：
{
  "一句话总结": "不超过 50 字的总结",
  "关键信息": ["关键点1", "关键点2"],
  "待办事项": ["动作1", "动作2"]
}

约束：
1. 原文未明确提到的时间、地点、人物，一律不要编造。
2. 没有待办事项时，"待办事项" 填空数组 []。
3. 只输出 JSON 本身。

文本：
{text}
"""


def run_once(prompt: str, text: str) -> str:
    llm = get_llm(temperature=0)
    # 用 replace 而不是 str.format：提示词 B 里含 JSON 花括号，format 会误当占位符
    return content_to_str(llm.invoke(prompt.replace("{text}", text)).content).strip()


def looks_like_json(s: str) -> bool:
    """粗略判断输出是否可直接被 json.loads 解析（允许 ``` 包裹）。"""
    import json

    t = s.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    t = t.strip()

    # 剥掉可能的外层文字，取第一个 { 到最后一个 }
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b > a:
        t = t[a : b + 1]
    try:
        json.loads(t)
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="每类输入每种提示词重复次数")
    parser.add_argument("--out", default="results.md")
    args = parser.parse_args()

    lines = [
        "# 任务三：提示词对比实验结果",
        "",
        f"- 模型：`deepseek-flash`（DeepSeek-V4.1-Flash），temperature=0，思考模式关闭",
        f"- 每种提示词对每类输入重复 **{args.runs}** 次，用于观察稳定性",
        f"- 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 两种提示词",
        "",
        "**提示词 A（简单）**",
        "```text",
        PROMPT_A.replace("{text}", "<输入文本>"),
        "```",
        "",
        "**提示词 B（结构化：角色 + 输出格式 + 约束）**",
        "```text",
        PROMPT_B.replace("{text}", "<输入文本>"),
        "```",
        "",
        "---",
        "",
    ]

    summary_rows = []

    for case in CASES:
        lines.append(f"## 输入 {case['id']}：{case['kind']}")
        lines.append("")
        lines.append(f"> {case['text']}")
        lines.append("")

        # --- 提示词 A ---
        lines.append(f"### 提示词 A 输出（{args.runs} 次）")
        lines.append("")
        a_ok = 0
        for i in range(1, args.runs + 1):
            out = run_once(PROMPT_A, case["text"])
            flag = "✅ JSON" if looks_like_json(out) else "❌ 非JSON"
            a_ok += looks_like_json(out)
            lines.append(f"**第 {i} 次**（{flag}）：")
            lines.append("")
            lines.append("```")
            lines.append(out)
            lines.append("```")
            lines.append("")

        # --- 提示词 B ---
        lines.append(f"### 提示词 B 输出（{args.runs} 次）")
        lines.append("")
        b_ok = 0
        for i in range(1, args.runs + 1):
            out = run_once(PROMPT_B, case["text"])
            flag = "✅ JSON" if looks_like_json(out) else "❌ 非JSON"
            b_ok += looks_like_json(out)
            lines.append(f"**第 {i} 次**（{flag}）：")
            lines.append("")
            lines.append("```json" if flag == "✅ JSON" else "```")
            lines.append(out)
            lines.append("```")
            lines.append("")

        summary_rows.append((case, a_ok, b_ok))
        lines.append("---")
        lines.append("")

    # --- 汇总 ---
    lines.append("## 稳定性汇总")
    lines.append("")
    lines.append("| 输入 | 提示词 A 可解析为 JSON | 提示词 B 可解析为 JSON |")
    lines.append("|---|---|---|")
    for case, a_ok, b_ok in summary_rows:
        lines.append(
            f"| {case['id']}.{case['kind']} | {a_ok}/{args.runs} | {b_ok}/{args.runs} |"
        )
    lines.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"完成，已写入 {args.out}")
    for case, a_ok, b_ok in summary_rows:
        print(f"  输入{case['id']} ({case['kind']}): A={a_ok}/{args.runs} JSON, B={b_ok}/{args.runs} JSON")


if __name__ == "__main__":
    main()
