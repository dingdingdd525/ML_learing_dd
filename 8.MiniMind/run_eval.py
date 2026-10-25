# -*- coding: utf-8 -*-
"""
MiniMind-3 推理评测（指标采集版）
逻辑与官方 eval_llm.py 完全一致：apply_chat_template -> tokenizer -> model.generate -> decode，
额外记录 prompt tokens / 生成 tokens / 耗时 / 速度 / 设备 / 生成参数，并保存逐题输出。

用法：
    python run_eval.py --model ../minimind/minimind-3 --device xpu
    python run_eval.py --model ../minimind/minimind-3 --device cpu
"""
import argparse
import json
import platform
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

QUESTIONS = [
    ("Q1-知识解释", "请用不超过 100 字解释什么是机器学习。"),
    ("Q2-代码生成", "用 Python 编写一个判断质数的函数。"),
    ("Q3-数学推理", "一名学生连续三天复习，第一天 2 小时，第二天 3 小时，第三天比前两天总和少 1 小时。第三天复习了多久？"),
    ("Q4-文本改写", "将下面这句话改写得更正式一些：\n“老师你好，我今天有事，作业能不能晚点交？”"),
    ("Q5-创意写作", "请为“人工智能社团招新”活动写一句简短宣传语。"),
    ("Q6-多步推理", "笼子里有鸡和兔共 35 个头、94 只脚，问鸡和兔各多少只？请写出计算过程。"),
]


def sync(device):
    if device.startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./minimind-3")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--open_thinking", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    device = args.device
    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, trust_remote_code=True)
    model = model.half().eval().to(device)
    sync(device)
    load_time = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters())
    meta = {
        "model_path": args.model,
        "model_class": type(model).__name__,
        "model_dtype": str(next(model.parameters()).dtype),
        "params_total": n_params,
        "params_million": round(n_params / 1e6, 2),
        "device": device,
        "device_name": (torch.xpu.get_device_name(0) if device.startswith("xpu") else
                        torch.cuda.get_device_name(0) if device.startswith("cuda") else platform.processor()),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "os": platform.platform(),
        "vocab_size": model.config.vocab_size,
        "num_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "max_position_embeddings": model.config.max_position_embeddings,
        "load_seconds": round(load_time, 2),
        "gen_params": {
            "do_sample": True,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "open_thinking": bool(args.open_thinking),
            "seed": args.seed,
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"[load] {meta['model_class']} {meta['params_million']}M {meta['model_dtype']} "
          f"on {device} ({meta['device_name']}) in {load_time:.2f}s")

    results = []
    for i, (tag, question) in enumerate(QUESTIONS):
        torch.manual_seed(args.seed + i)
        conversation = [{"role": "user", "content": question}]

        # ---- 1) chat template ----
        prompt_text = tok.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
            open_thinking=bool(args.open_thinking),
        )
        # ---- 2) tokenizer ----
        inputs = tok(prompt_text, return_tensors="pt", truncation=True).to(device)
        n_prompt = int(inputs["input_ids"].shape[1])

        # ---- 3) 单步 forward 测 prefill 耗时（与 generate 相同的输入）----
        sync(device)
        tp = time.time()
        with torch.no_grad():
            model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        sync(device)
        prefill_s = time.time() - tp

        # ---- 4) generate ----
        sync(device)
        tg = time.time()
        with torch.no_grad():
            out = model.generate(
                inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens, do_sample=True,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                top_p=args.top_p, temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
            )
        sync(device)
        gen_s = time.time() - tg

        out_ids = out[0][n_prompt:]
        n_gen = int(out_ids.shape[0])
        answer = tok.decode(out_ids, skip_special_tokens=True)
        finished_by_eos = (n_gen < args.max_new_tokens) and (int(out_ids[-1]) == tok.eos_token_id)
        decode_s = max(gen_s - prefill_s, 1e-9)

        row = {
            "tag": tag,
            "question": question,
            "chat_template_text": prompt_text,
            "prompt_token_ids": inputs["input_ids"][0].tolist(),
            "prompt_tokens": n_prompt,
            "gen_tokens": n_gen,
            "answer": answer,
            "finished_by_eos": finished_by_eos,
            "generate_seconds": round(gen_s, 3),
            "prefill_seconds": round(prefill_s, 3),
            "speed_total_tok_s": round(n_gen / gen_s, 2),
            "speed_decode_tok_s": round(n_gen / decode_s, 2),
            "first_decode_tok_ms": round(prefill_s / max(n_prompt, 1) * 1000, 2),
        }
        results.append(row)
        print(f"[{tag}] prompt={n_prompt}tok gen={n_gen}tok eos={finished_by_eos} "
              f"{gen_s:.2f}s total={row['speed_total_tok_s']}tok/s decode={row['speed_decode_tok_s']}tok/s")

    payload = {"meta": meta, "results": results}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 纯文本可读版
    txt = args.out.replace(".json", ".txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(f"model={meta['model_class']} {meta['params_million']}M dtype={meta['model_dtype']} "
                f"device={device}({meta['device_name']})\n")
        f.write(f"torch={meta['torch']} transformers={meta['transformers']} python={meta['python']}\n")
        f.write(f"gen_params={json.dumps(meta['gen_params'], ensure_ascii=False)}\n")
        f.write("=" * 70 + "\n")
        for r in results:
            f.write(f"\n### {r['tag']}\n[问] {r['question']}\n[答]\n{r['answer']}\n")
            f.write(f"-- prompt={r['prompt_tokens']}tok gen={r['gen_tokens']}tok "
                    f"eos={r['finished_by_eos']} time={r['generate_seconds']}s "
                    f"total={r['speed_total_tok_s']}tok/s decode={r['speed_decode_tok_s']}tok/s\n")
    print(f"\n[done] saved -> {args.out} / {txt}")


if __name__ == "__main__":
    main()
