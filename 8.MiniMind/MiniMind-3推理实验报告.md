# MiniMind-3 本地推理实验报告

> 日期：2026-10-25 ｜ 机器：Intel Core Ultra 9 285H / Arc 140T 核显（16GB 共享）/ 32GB LPDDR5X
> 仓库：`D:\pytorch\机器学习（工作室题目）\minimind`（已 git 忽略，不进版本库）

---

## 一、结论摘要

| 项目 | 结果 |
|---|---|
| checkpoint | `minimind-3`，**63.91M** 参数（实测），fp16，权重 121.9 MiB |
| 能否加载推理 | ✅ 官方 `eval_llm.py` 一次跑通，无需改代码 |
| 运行设备 | Intel Arc 140T（`--device xpu`）+ CPU 对照 |
| 速度 | XPU 平均 **28.10 tok/s**；CPU 平均 **89.47 tok/s**（CPU 快 3.2 倍） |
| 6 题质量 | 合格 1 题（Q1）、代码对但解释胡说 1 题（Q2）、其余 4 题全错 |
| 最值得记的现象 | ① 64M 模型在**核显上反而更慢**；② 同一 prompt 因官方脚本随机播种，两次输出不同；③ `generate(**enc)` 会因 `token_type_ids` 报错 |

---

## 二、拉取仓库与 git 忽略

```bash
cd "D:/pytorch/机器学习（工作室题目）"
git clone --depth 1 https://github.com/jingyaogong/minimind.git minimind
```

`.gitignore` 新增（第 16 行）：

```gitignore
# ===== 外部克隆的学习仓库：只做本地实验，绝不推送 =====
minimind/
```

验证「推送不会带上它」：

```bash
$ git check-ignore -v minimind/
.gitignore:16:minimind/	minimind/

$ git status --short
 M .gitignore
?? 1.机器学习深度学习基础知识问答/
?? 7.LLaMA/
?? 8.MiniMind/          # minimind/ 不在列表里 → 已被忽略

$ git status --short --ignored | grep minimind
!! minimind/           # !! = 被忽略

# 最硬的证据：全量暂存演练里也看不到它
$ git add -A --dry-run | grep -i minimind
（无输出）
$ git add -A --dry-run | head -3
add '8.MiniMind/MiniMind-3推理实验报告.md'
add '8.MiniMind/eval_llm_official_cpu.log'
add '8.MiniMind/eval_llm_official_xpu.log'
```

> 说明：`.gitignore` 目前是**已修改未提交**状态。要让忽略规则也进远端，提交这一行即可：
> `git add .gitignore && git commit -m "chore: ignore minimind clone"`。

---

## 三、环境与 checkpoint

### 环境

| 项 | 值 |
|---|---|
| Python | 3.11.9（复用现成的 `D:\pytorch\xpu-env`） |
| torch | **2.11.0+xpu**（Intel 核显版） |
| transformers | **4.57.6** |
| tokenizers | 0.22.2 |
| 额外安装 | `transformers==4.57.6 accelerate sentencepiece safetensors jinja2 modelscope` |

> 没有照搬 `requirements.txt`：那份锁了 `numpy==1.26.4`，会打坏 xpu-env 里 torch 2.11 依赖的 numpy 2.4.6。推理只需要 transformers 这一条链，其余（datasets/trl/wandb/tiktoken…）与本实验无关。

### 下载

```bash
python -m modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
# 实际用 API 等价调用（该版本 modelscope 没有 __main__ 入口）
```

30 个文件 / 128MB，其中 `model.safetensors` 127,834,168 字节。

### 模型结构（读 `config.json` + 实测）

| 项 | 值 |
|---|---|
| `architectures` | **`Qwen3ForCausalLM`**（`model_type: qwen3`）→ 直接用 transformers 原生 Qwen3 实现加载，**不需要** `trust_remote_code` |
| 参数 | 63,912,192 ≈ 63.91M |
| 层数 / 隐藏维 | 8 层 / 768 |
| 注意力 | q_heads=8, kv_heads=4, head_dim=96（GQA） |
| FFN | intermediate_size=2432，SwiGLU |
| 词表 | 6400 |
| 上下文 | max_position_embeddings=32768，rope_theta=1e6 |
| 词嵌入 | `tie_word_embeddings=true`（输入输出共用） |
| dtype | float16（`eval_llm.py` 里 `model.half()`） |
| 特殊 token | `<|im_start|>`=1 (bos)、`<|im_end|>`=2 (eos)、`<|endoftext|>`=0 (pad) |

---

## 四、运行方式

```bash
# 1) 官方脚本，手动模式喂 5 道题（CPU / XPU 各跑一遍，日志留在 8.MiniMind/）
python eval_llm.py --load_from ./minimind-3 --device cpu --max_new_tokens 512
python eval_llm.py --load_from ./minimind-3 --device xpu --max_new_tokens 512

# 2) 自写评测脚本（逻辑与 eval_llm.py 完全一致，多记了 token 数与分项耗时）
python ../8.MiniMind/run_eval.py --model ./minimind-3 --device xpu --out ../8.MiniMind/results_xpu.json
python ../8.MiniMind/run_eval.py --model ./minimind-3 --device cpu --out ../8.MiniMind/results_cpu.json
```

### 生成参数（两轮一致）

| 参数 | 值 | 来源 |
|---|---|---|
| `do_sample` | True | eval_llm.py |
| `max_new_tokens` | 512 | 手动指定（脚本默认 8192，太长） |
| `temperature` | 0.85 | eval_llm.py 默认 |
| `top_p` | 0.95 | eval_llm.py 默认 |
| `repetition_penalty` | 1.0 | eval_llm.py 默认 |
| `top_k` | **50** | **脚本没传，落在 `GenerationConfig` 默认值上** |
| `pad_token_id` / `eos_token_id` | 0 / 2 | tokenizer 配置 |
| `open_thinking` | 0 | eval_llm.py 默认 |
| `seed` | 20261025 + i | 自写脚本固定；官方脚本每轮随机 |

---

## 五、实测指标

### XPU（Intel Arc 140T）

| 题 | prompt tok | 生成 tok | 耗时 s | 总速度 tok/s | 纯解码 tok/s | 正常 EOS 结束 |
|---|---|---|---|---|---|---|
| Q1 知识解释 | 29 | 35 | 3.55 | 9.86 | 21.53 | ✅ |
| Q2 代码生成 | 26 | 380 | 12.94 | 29.36 | 29.44 | ✅ |
| Q3 数学推理 | 59 | 385 | 17.87 | 21.54 | 24.22 | ✅ |
| Q4 文本改写 | 47 | 286 | 9.46 | 30.23 | 30.37 | ✅ |
| Q5 创意写作 | 37 | 372 | 11.11 | 33.48 | 33.64 | ✅ |
| Q6 多步推理 | 54 | 463 | 13.42 | 34.50 | 34.60 | ✅ |
| **合计** | 252 | **1921** | 68.36 | **28.10** | — | 6/6 |

加载耗时 1.94s。官方脚本同一批题的打点：7.08 / 34.80 / 25.37 / 40.82 / 34.47 tok/s（第 1 题受冷启动拖累，与上表一致）。

### CPU（Core Ultra 9 285H）

| 题 | prompt tok | 生成 tok | 耗时 s | 总速度 tok/s | 纯解码 tok/s |
|---|---|---|---|---|---|
| Q1 | 29 | 23 | 0.56 | 41.22 | 61.37 |
| Q2 | 26 | 206 | 2.46 | 83.87 | 85.53 |
| Q3 | 59 | 397 | 4.09 | 97.02 | 97.99 |
| Q4 | 47 | 103 | 1.14 | 90.32 | 93.25 |
| Q5 | 37 | 360 | 3.76 | 95.81 | 96.56 |
| Q6 | 54 | 398 | 4.62 | 86.19 | 86.85 |
| **合计** | 252 | **1487** | 16.62 | **89.47** | — |

**速度怎么看**：
- 「总速度」= 生成 token ÷ `generate()` 全程（含 prefill），所以短回答被 prefill 拖累得厉害（Q1 只 9.86，因为它只生成 35 个 token，前面 29 token 的 prefill 摊不掉）。
- 「纯解码」= 用同输入单独跑一次单步 forward 测出 prefill 时间后剔除。这才是稳态吞吐，XPU 21→35 的上升是核显 kernel 首次加载 + 频率爬升的冷启动效应。
- **CPU 比核显快约 3.2 倍**。64M 模型每步只算 768×8 层的小张量，解码是典型的访存密集型逐 token 循环，核显的优势（大矩阵并行）完全用不上，反而每次 kernel launch 都被碎片化开销吃掉。这与 xpu-env 里那条经验完全一致：*逐 token 自回归、小张量碎片算子 → CPU 反而快*。
- CPU 两轮生成 token 数不同（1487 vs 1921），因为 fp16 在两种后端上数值有微小差异 + 采样，种子固定也不保证逐 token 一致。所以「总速度」的对比只用同一题横向看，别拿总和硬比。

---

## 六、逐题输出与判分

### Q1 知识解释 ✅ 合格

> **问**：请用不超过 100 字解释什么是机器学习。
>
> **答**：
> # 机器学习概述
> 机器学习是人工智能的一个重要分支，致力于让计算机系统从数据中学习规律和模式，而无需明确编程。

35 token 自然收尾，正文约 50 字，卡在 100 字内。唯一毛病是自作多情加了个 Markdown 一级标题。

### Q2 代码生成 ⚠️ 代码对、解释胡说

> **答**（节选）：
> 要编写一个判断质数的函数，首先需要明确质数的定义。质数的定义是只能被1和它本身整除的自然数，**即2和该数的所有整数相加等于2的整数**。……
> ```python
> def is_prime(n):
>     if n <= 1: return False
>     if n <= 3: return True
>     if n % 2 == 0 or n % 3 == 0: return False
>     i = 5
>     while i * i <= n:
>         if n % i == 0 or n % (i + 2) == 0: return False
>         i += 6
>     return True
> ```

代码是标准 6k±1 试除法，**能直接用**（`is_prime(2)=True`、`is_prime(9)=False`、`is_prime(97)=True`）。但自然语言解释整段崩坏：把质数定义说成「2和该数的所有整数相加等于2」，后面又说「接下来实现了一个辅助函数 is_prime 用来调用 is_prime」——自指循环。典型的「会背代码模板，不会讲概念」。

> 官方脚本用另一随机种子跑同一题时，答案在 512 token 处被硬截断（末行是半句 `for i in range(2, int(n**0.5) + 1`），说明这模型对"写代码"这类题有很强的复读惯性，`max_new_tokens` 给小了会切在半句上。

### Q3 数学推理 ❌ 错（正确答案 4 小时）

> **答**（节选）：计算第一天：2小时 × 2小时 = 4小时 …… 所以，第三天复习的总时间是：4小时 ÷ 9小时 = 1小时 …… 因此，第三天复习的总时间大约是 **1.333 小时**。

把「比前两天总和少 1 小时」这个减法硬编成了乘除，全程自相矛盾（前面说 1 小时，结论又变 1.333）。2+3-1=4 这种一步减法的题都不会。

### Q4 文本改写 ❌ 错

> **答**（节选）：
> “老师！今天你作业很紧张，作业会不会做不到？”
> “老师！今天作业已经很顺利，作业可以做到两点，比如保证成绩。但是，作业还需要准备第二天的交付物……”

要求「改写得更正式」，输出的是语义全毁的乱码，还夹着一段 `<think>\n\n</think>` 标签泄漏到正文。它甚至开始点评自己的改写「强调了作业的作业时间的安排」。

### Q5 创意写作 ❌ 错（标签泄漏最严重）

> **答**（节选）：
> `<think>` 嗯，用户让我帮忙写一句简短的宣传语。首先，我得确定这个活动的核心内容是什么……
> `</think>`
> 挑战、激动人心、未来展望！

**思考链整段漏进正文**（372 个生成 token 里绝大部分是内心戏），最后的「宣传语」是把三个词用顿号串起来，不是一句话。注意这是 `open_thinking=0`（模板已给空 `think` 块）时发生的——模型自己又复读了一遍 `<think>` 脚手架。

### Q6 多步推理 ❌ 错（正确答案 鸡 23 / 兔 12）

> **答**（节选）：我们可以使用组合数学的方法……组合数公式为 C(n, k) = n! / (k! * (n-k)!) …… 所以，鸡和兔各有 **35** 只。

完全跑偏：把鸡兔同笼当成组合数问题，`C(35,94)` 这种非法组合也照写，中间出现「35! / 6771 = 42*7210 = 31530」这类纯编造算式。**注意它开头就选错了工具**——不是算错，是压根没识别出这是二元一次方程组。

### 小结

| 能力 | 表现 |
|---|---|
| 中文流畅度 | 好，句子通顺、格式（Markdown 标题/列表/代码块）学得很像 |
| 短知识问答 | 可以（Q1） |
| 代码生成 | **模板级正确**，但配套解释不可信 |
| 算术 / 多步推理 | 崩（Q3、Q6），且不会「先建模再算」 |
| 指令跟随（改写/字数/风格） | 崩（Q4） |
| 格式控制 | 差：`<think>` 标签会漏进正文（Q4、Q5） |

64M 参数、8 层、词表 6400 的模型，这些表现符合预期：**语言形式学得很好，符号推理几乎为零**。

---

## 七、输入是怎么变成输出的（chat template → tokenizer → generate）

完整链路可以用 `minimind/trace_pipeline.py` 复现（输出存 `8.MiniMind/trace.log`）。以 Q1 为例：

### ① chat template：list[dict] → 一个字符串（纯文本操作，不碰模型）

输入是消息列表：

```python
conversation = [{"role": "user", "content": "请用不超过 100 字解释什么是机器学习。"}]
```

`tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=False)` 用 `minimind-3/chat_template.jinja` 渲染：

```
<|im_start|>user
请用不超过 100 字解释什么是机器学习。<|im_end|>
<|im_start|>assistant
<think>

</think>

```

三个关键点：

1. **它不只是拼接**。角色和边界靠特殊 token `<|im_start|>` / `<|im_end|>` 表达，而不是 `User: ...\nAssistant:` 这种自然语言分隔符。模型在预训练/SFT 里就是按这套边界学的，换别的分隔符效果会掉。
2. **`add_generation_prompt=True`** 才补上最后那句 `<|im_start|>assistant\n`——这是在告诉模型「轮到你了，从这儿往后写」。为 `False` 时字符串到 `<|im_end|>\n` 就结束（上面 trace 里两种都打出来了）。
3. **`open_thinking` 是这套模板的自定义参数**：`=False` 补一个**空** think 块 `<think>\n\n</think>\n\n`（模型看到已经有思考块了，直接答）；`=True` 只补 `<think>\n`，让模型自己往下展开推理。这也是为什么 Q4/Q5 会「多余地」再吐一遍 `<think>`——它在模仿训练数据里的思考格式，但没被这条样本约束住。

### ② tokenizer：字符串 → token id（词表只有 6400，切得很碎）

```python
inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True)  # -> input_ids (1, 29)
```

实测 29 个 token：

| idx | id | 还原 | idx | id | 还原 |
|---|---|---|---|---|---|
| 0 | 1 | `<|im_start|>` | 15 | 302 | `。` |
| 1 | 832 | `us` | 16 | 2 | `<|im_end|>` |
| 2 | 311 | `er` | 17 | 234 | `\n` |
| 3 | 234 | `\n` | 18 | 1 | `<|im_start|>` |
| 4 | 960 | `请` | 19 | 1388 | `ass` |
| 5 | 368 | `用` | 20 | 570 | `ist` |
| 6 | 417 | `不` | 21 | 811 | `ant` |
| 7 | 4976 | `超过` | 22 | 234 | `\n` |
| 8 | 2277 | ` 100` | 23 | 25 | `<think>` |
| 9 | 256 | ` ` | 24 | 234 | `\n` |
| 10 | 702 | `字` | 25 | 234 | `\n` |
| 11 | 2093 | `解释` | 26 | 26 | `</think>` |
| 12 | 1403 | `什么` | 27 | 234 | `\n` |
| 13 | 357 | `是` | 28 | 234 | `\n` |
| 14 | 2683 | `机器学习` | | | |

值得注意的三个细节：

- **英文被切碎**：`user` → `us`+`er`，`assistant` → `ass`+`ist`+`ant`。词表才 6400（Qwen3 是 151k），英文子词根本没地方放，只能退化成更短的片段。所以同一段中英混排文本，这个模型的 token 数会明显偏高。
- **中文词能整块保留**：`机器学习` 一个 id 2683，`超过`/`解释`/`什么` 也都是一块。词表小但中文词覆盖得不错。
- **没有额外 BOS**：`add_bos_token=False`，而 `bos_token` 恰好就是 `<|im_start|>`(=1)——序列开头那个 1 是模板渲染出来的，不是 tokenizer 自动加的。要是这里理解错、手动再 prepend 一个 BOS，模型会看到连续两个 `<|im_start|>`，输出立刻劣化。
- `attention_mask` 全 1（单条、无 padding）。`truncation=True` 但没给 `max_length`，实际截断上限取 `model_max_length=131072`，对本实验无影响。

### ③ `model.generate()`：token id → 新 token id

```python
generated_ids = model.generate(
    inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
    max_new_tokens=512, do_sample=True,
    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    top_p=0.95, temperature=0.85, repetition_penalty=1)
```

注意：加载出来的类是 **`Qwen3ForCausalLM`**（`config.json` 的 `architectures` 就写着它），所以走的是 **transformers 原生 `GenerationMixin.generate`**，不是仓库 `model/model_minimind.py` 里那个手写的 `generate`。两者行为不同（手写版默认 `top_p=0.85, top_k=50`），别搞混。

内部就四步，循环到停：

1. **prefill**：把 29 个 id 一次性喂进 8 层 Transformer，拿到每个位置的 hidden state，`lm_head`（与词嵌入共享权重）投到 6400 维 logits，只取**最后一个位置** → `logits[:, -1, :]`。
2. **改 logits 再采样**：先 `/temperature`（0.85 让分布更尖），再按 `top_k=50`（脚本没传，落在 `GenerationConfig` 默认值上）和 `top_p=0.95`（核采样，累计概率超 0.95 的尾巴全砍成 -inf），最后 `softmax` + `multinomial` 抽一个 token。
3. **拼回去 + KV cache**：新 token 接到 `input_ids` 末尾，**下一步只算这 1 个新 token**，前面 29 个的 K/V 从 cache 里读。这就是为什么解码速度与上下文长度弱相关、而 prefill 是二次复杂度。
4. **终止**：抽到 `<|im_end|>`（id=2）或攒够 `max_new_tokens=512` 就停。返回 shape `(1, 29 + 新增数)`。

实测这次是 29 + 24 = 53 个 token（`do_sample=False` 的短跑），解码出：

```
机器学习是人工智能的一个分支，它使计算机能够在不进行明确编程的情况下从数据中学习并改进其
```

### ④ decode：回到人话

```python
response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
```

只切掉 prompt 部分（`generated_ids` 是「prompt + 新增」拼在一起的），`skip_special_tokens=True` 把结尾的 `<|im_end|>` 抹掉。速度指标就是在这之后算的：`新增 token 数 / (time.time() - st)`。

### 三个踩过的坑（写代码时直接用得上）

1. **`model.generate(**enc)` 会报错**：`enc` 里带 `token_type_ids`，Qwen3 不认，抛
   `ValueError: The following model_kwargs are not used by the model: ['token_type_ids']`。
   必须像 `eval_llm.py` 那样显式只传 `input_ids` 和 `attention_mask`。
2. **官方脚本不可复现**：每个问题前都 `setup_seed(random.randint(0, 31415926))`，同一 prompt 跑两次结果不同。要复现得自己固定种子（`run_eval.py` 里用 `torch.manual_seed(20261025+i)`）。
3. **`model.half()` 是无条件的**：CPU 上也是 fp16 跑。想换 bf16 得自己改，`--device` 只换设备不换精度。

---

## 八、产物清单

| 文件 | 内容 |
|---|---|
| `8.MiniMind/run_eval.py` | 指标采集版评测脚本（逻辑对齐 eval_llm.py） |
| `8.MiniMind/results_xpu.json` / `.txt` | XPU 6 题完整输出 + 全部指标（含每题的 chat template 原文与 token id 序列） |
| `8.MiniMind/results_cpu.json` / `.txt` | CPU 同一套 |
| `8.MiniMind/eval_llm_official_cpu.log` | 官方 `eval_llm.py` 手动模式 5 题原始日志（CPU） |
| `8.MiniMind/eval_llm_official_xpu.log` | 同上（XPU） |
| `8.MiniMind/trace.log` | chat template → token id → generate → decode 的逐步打印 |
| `8.MiniMind/questions.txt` | 喂给官方脚本的 5 道题 |
| `minimind/trace_pipeline.py` | 生成 trace 的脚本（在被忽略的克隆目录里） |
