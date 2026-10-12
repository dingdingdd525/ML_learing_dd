# Python 脚本语法扫盲（2.1~2.4 全部脚本）

> 本文档覆盖本目录下 9 个脚本用到的全部语法与库函数：`task2_1_linear.py`、`task2_2_mlp.py`、`exp1~exp5_*.py`、`struct_exp.py`、`noise_exp.py`。
> 按"通用骨架 → Python 基础 → NumPy → scikit-learn → PyTorch → matplotlib → 文件读写 → 常见坑"组织，每个函数给出语法、常用参数与脚本中的实际用法。

---

## 0. 一个脚本的通用骨架

所有脚本几乎都按这个顺序组织：

```
导入库 → 定义常量 → 生成/读取数据 → 划分数据集 → numpy 转 tensor → 定义模型
→ 定义训练函数 → 跑 N 次实验 → 统计 → 可视化 → 把结果写入 txt
```

对照 `exp1_lr0001.py` 看一遍即可建立整体印象，后续各节是每一块的语法拆解。

---

## 1. Python 基础语法

### 1.1 导入（import）

```python
import os                    # 标准库：路径操作
import numpy as np           # 数值计算，别名 np
import torch                 # PyTorch 主库
import torch.nn as nn        # 神经网络模块（nn.Linear 等），别名 nn
import matplotlib.pyplot as plt   # 画图，别名 plt
from sklearn.datasets import make_friedman1        # 从包中导入单个函数
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset   # 一次导入多个
```

- `import A as B`：给模块起别名，之后用 `B.xxx` 调用。
- `from A import x, y`：只导入 A 中的 x、y，之后直接写 `x()`。
- 约定：标准库 → 第三方库 → 自己的代码，之间空一行。

### 1.2 常量定义（全大写）

```python
EPOCHS = 10000      # 训练步数上限
PRINT_EVERY = 100   # 每多少步打印一次
PATIENCE = 1000     # 早停耐心值
N_RUNS = 5          # 每种配置独立训练几次
BASE_SEED = 42      # 随机种子起点
LR = 0.001          # 学习率
WEIGHT_DECAY = 1e-4 # L2 正则系数（1e-4 = 0.0001）
```

全大写是 Python 约定：表示"这是一个常量，运行中不改变"。

### 1.3 函数定义（def）

```python
def build_mlp():
    """MLP：输入10 -> 24 -> 输出1（单隐藏层，输出层无激活）"""   # docstring，函数说明
    return nn.Sequential(
        nn.Linear(10, 24),
        nn.ReLU(),
        nn.Linear(24, 1),
    )

def train_once(model):                    # 带参数
    ...
    return stop_epoch, best_step, best_val_loss, mse, r2   # 返回多个值（实为元组）

def count_params(model):
    return sum(p.numel() for p in model.parameters())      # 生成器表达式 + sum
```

- 函数用 `def 名字(参数):` 定义，缩进 4 空格是代码块。
- `"""..."""` 在函数开头叫 docstring（文档字符串），`help(函数)` 能看到。
- `return a, b, c` 返回元组；调用处 `stop, best, val, mse, r2 = train_once(model)` 解包。

### 1.4 for / while 循环

```python
for epoch in range(1, EPOCHS + 1):        # range(起点, 终点) 不含终点
for run in range(N_RUNS):                 # range(n) = 0,1,...,n-1
for xb, yb in train_loader:               # 遍历 DataLoader，每次得到一个 batch 的 (输入, 标签)
for i, r in enumerate(results, start=1):  # 同时拿序号和元素，从 1 开始编号
for ax, name, results, color in zip(axes, names, results_list, colors):  # 并行遍历多个列表
while step < EPOCHS:                      # while 循环
```

- `enumerate(可迭代对象, start=1)`：返回 `(序号, 元素)` 对。
- `zip(a, b, c)`：把多个可迭代对象按位置打包成元组序列，长度取最短的。

### 1.5 条件与跳出

```python
if val_loss < best_val_loss:
    ...
else:
    wait += 1
    if wait >= PATIENCE:
        stop_epoch = epoch
        break          # 跳出当前循环
```

### 1.6 列表（list）

```python
train_losses, val_losses = [], []                    # 空列表
train_losses.append(loss.item())                     # 尾部追加
stops, bests, mses, r2s = [], [], [], []             # 一次建多个
mses.append(mse)
r2s.append(r2)
r2_arr = np.array(r2s)                               # 列表 → numpy 数组
[r["test_mse"] for r in results]                     # 列表推导式：取出每个元素的 test_mse
min_len = min(len(r["val_losses"]) for r in results) # 生成器表达式（不建中间列表）
lines.append(f"...")                                 # 把文本行收集起来，最后 join 写入文件
report = "\n".join(lines)                            # 列表 → 一个字符串（用换行连接）
```

### 1.7 字典（dict）

```python
best_state = None
res = {
    "train_losses": train_losses,
    "stop_epoch": stop_epoch,
    "test_mse": mse,
    "test_r2": r2,
}                                  # 构造字典：键 → 值
res["test_r2"]                     # 取值
REF_R2 = {"基线": np.array([...]), "exp1": np.array([...])}   # 值可以是任意对象
for name, arr, c in zip(ref_names, ref_arrays, colors):      # 遍历
r["val_losses"][:min_len]          # 字典里的列表再切片
```

### 1.8 f-string（格式化字符串）——重点

```python
print(f"step={epoch:5d} | train_loss={loss.item():.6f}")
print(f"test_mse = {mse_arr.mean():.4f}±{mse_arr.std():.4f}")
print(f"{'run':<4}{'stop_step':>10}{'test_mse':>12}{'test_r2':>10}")   # 表头
print(f"{i + 1:<4}{stops[i]:>10}{mses[i]:>12.4f}{r2s[i]:>10.4f}")     # 表行
print(f"R² 提升 = {r2_arr.mean() - base_r2_arr.mean():+.4f}")
print(f"{'/'.join(f'{v:.4f}' for v in BASE_R2)}")    # f-string 里嵌套表达式
print(f"参数量：MLP={n_mlp}，约为线性的 {n_mlp / n_lin:.1f} 倍")
```

格式说明（`{值:格式}`）：

| 格式 | 含义 | 示例输出 |
|---|---|---|
| `:d` | 整数 | `42` |
| `:5d` | 整数，占 5 位（右对齐） | `   42` |
| `:.6f` / `:.4f` / `:.1f` | 保留 6/4/1 位小数 | `1.404100` |
| `:<4` | 左对齐，占 4 位 | `run ` |
| `:>10` | 右对齐，占 10 位 | `     1.40` |
| `:+.4f` | 带正负号，4 位小数 | `+0.1884` |
| `{'run':<4}` | 字符串字面量也可套格式（外层用单引号） | `run ` |

### 1.9 lambda（匿名函数）

```python
best = max(results, key=lambda r: r["test_r2"])   # 取 test_r2 最大的那一个结果
```

`lambda r: r["test_r2"]` 等价于 `def f(r): return r["test_r2"]`，传给 `key=` 作为比较依据。

### 1.10 with 语句（上下文管理器）

```python
with torch.no_grad():          # 块内不追踪梯度（评估时用）
    val_loss = loss_fn(model(X_val), y_val).item()

with open(report_path, "w", encoding="utf-8") as f:   # 打开文件，块结束自动关闭
    f.write(report + "\n")
```

`with ... as 变量:` 保证资源（文件、梯度记录）用完自动释放，无需手动 close。

### 1.11 其它小语法

```python
float("inf")              # 正无穷，用于"初始最佳损失"（任何有限数都比它小）
best_state = None         # 初始为"空"
stops.append(stop); bests.append(best_val)   # 分号可在一行写多条语句（不推荐多用）
# 单行注释
""" 多行字符串/文档注释 """
```

---

## 2. NumPy（np）

| 语法 | 作用 | 脚本示例 |
|---|---|---|
| `x.astype(np.float32)` | 转换数组元素类型为 float32 | `X = X.astype(np.float32)` |
| `y.reshape(-1, 1)` | 改变形状：`(1500,)` → `(1500, 1)`（-1 自动算） | `y = y.reshape(-1, 1).astype(np.float32)` |
| `np.array(列表)` | 列表转数组 | `r2_arr = np.array(r2s)` |
| `arr.mean()` / `arr.std()` / `arr.sum()` / `arr.min()` / `arr.max()` | 均值 / 标准差 / 和 / 最小 / 最大 | `r2_arr.mean()` |
| `np.mean(arr)` / `np.std(arr)` | 同上（函数式写法） | `float(np.mean((a - b) ** 2))` |
| `arr ** 2` | 逐元素平方 | `(y_test - y_pred) ** 2` |
| `arr > 0.80` | 逐元素比较，得布尔数组 | `(r2_arr > 0.80)` |
| `(r2_arr > 0.80).sum()` | 布尔数组求和 = 满足条件的个数 | `(r2_arr > 0.80).sum()` |
| `arr[:min_len]` | 切片取前 min_len 个 | `r["val_losses"][:min_len]` |
| `np.full(len(arr), i)` | 生成全为 i 的数组（长度 len(arr)） | 画图时给每个 run 定 x 坐标 |
| `np.random.uniform(-0.08, 0.08, len(arr))` | 生成 len(arr) 个均匀分布随机数（微调 x 坐标防重叠） | 散点横向错开 |
| `np.concatenate([a, b])` | 拼接数组 | `np.concatenate([lin_r2, mlp_r2])` |
| `np.mean([...], axis=0)` | 沿第 0 维求均值（对多条曲线逐点平均） | `mean_curve()` |

关键点：**numpy 数组是"数值矩阵"；PyTorch 张量是"带梯度的数值矩阵"**。两者可以互转（见 4.1），但方法名不同（numpy 用 `.astype`，tensor 用 `.float()`；tensor 有 `.numpy()`，numpy 没有）。

---

## 3. scikit-learn（sklearn）

### 3.1 make_friedman1 —— 生成回归数据

```python
X, y = make_friedman1(n_samples=1500, n_features=10, noise=1.0, random_state=42)
```

| 参数 | 含义 | 本例 |
|---|---|---|
| `n_samples` | 样本数 | 1500 |
| `n_features` | 特征数（必须是 5 的倍数，≥5；多余的是无关噪声特征） | 10 |
| `noise` | 叠加在目标上的高斯噪声标准差 | 1.0 |
| `random_state` | 随机种子，固定后每次生成的数据相同（可复现） | 42 |

- 返回 `X`：形状 `(1500, 10)` 的特征矩阵；`y`：形状 `(1500,)` 的目标值。
- 内部真实公式（这就是"非线性"的来源）：`y = 10·sin(π·x1·x2) + 20·(x3−0.5)² + 10·x4 + 5·x5 + noise·N(0,1)`。

### 3.2 train_test_split —— 数据划分

```python
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42)
```

| 参数 | 含义 |
|---|---|
| 第一个位置参数 | 特征数据 |
| 第二个位置参数 | 标签数据 |
| `test_size` | 测试/验证集比例（0~1 之间的小数）或个数 |
| `train_size` | 训练集比例（与 test_size 二选一即可） |
| `random_state` | 随机种子，固定则每次划分相同 |
| `shuffle` | 默认 True，先打乱再切 |
| `stratify` | 分类任务保持类别比例用（回归不用） |

- 返回顺序 = 传入顺序：`X_训练, X_测试, y_训练, y_测试`。
- **60/20/20 的两次切法**：先切出 20% 测试集，剩余 80% 再按 `test_size=0.25` 切——80% × 25% = 20% 验证集，剩下 60% 训练。两次都用同一个 `random_state=42`，保证所有脚本、所有实验共用完全相同的划分（可比性关键）。

### 3.3 StandardScaler —— 标准化

```python
scaler_x = StandardScaler().fit(X_train)     # 1. 只在训练集上 fit（学均值和标准差）
X_train = scaler_x.transform(X_train)        # 2. transform： (x - mean) / std
X_val = scaler_x.transform(X_val)
X_test = scaler_x.transform(X_test)          #    验证/测试用同一个 scaler（绝不能自己再 fit）

scaler_y = StandardScaler().fit(y_train)     # 目标也标准化
y_train = scaler_y.transform(y_train)
y_pred = scaler_y.inverse_transform(y_pred_std)   # 预测结果反标准化回原始尺度再算指标
```

| 方法 | 作用 |
|---|---|
| `fit(X)` | 只计算训练集的均值 μ 和标准差 σ（不改变数据） |
| `transform(X)` | 用已学到的 μ、σ 做标准化：`(X−μ)/σ` |
| `fit_transform(X)` | fit + transform 一步完成（仅训练集可用） |
| `inverse_transform(X)` | 反变换还原：`X·σ+μ` |

**为什么只 fit 训练集**：如果让验证/测试集参与计算 μ、σ，就等于"偷看了"测试集的信息（数据泄漏），测试成绩会偏乐观。这是固定规范。

---

## 4. PyTorch（torch / torch.nn / torch.optim）

### 4.1 numpy 与 tensor 互转

```python
torch.manual_seed(42)              # 固定全局随机种子 → 模型初始化可复现
X_train = torch.from_numpy(X_train)   # numpy 数组 → tensor（共享内存，不拷贝）
y_test_np = y_test.numpy()            # tensor → numpy（要求 CPU 且不需要梯度）
```

- `torch.manual_seed(seed)`：设置随机种子。放在创建模型之前，每次 `run` 换一个种子（42~46），得到 5 次不同但可复现的初始化。
- `torch.from_numpy(arr)`：numpy → tensor，**与 numpy 数组共享底层内存**（改一个另一个也变）。默认 `requires_grad=False`，dtype 跟随原数组。
- `tensor.numpy()`：tensor → numpy。**前提**：在 CPU 上、且 `requires_grad=False`（有梯度要先 `.detach()`）。脚本中都在 `torch.no_grad()` 块里调用，安全。
- 常见报错：对 numpy 数组调 `.numpy()`（如 `y_test.numpy()` 但 y_test 还是 numpy）→ `AttributeError: 'numpy.ndarray' object has no attribute 'numpy'`。这正是 exp2 修过的 bug：先 `y_test.astype(np.float32)` 而不是 `.numpy()`。

### 4.2 模型定义

```python
model = nn.Linear(10, 1)              # 线性层：输入10维 → 输出1维，无激活 = 线性回归
model = nn.Sequential(                # 顺序容器：按顺序执行里面的层
    nn.Linear(10, 24),                # 输入10 → 隐藏层24
    nn.ReLU(),                        # 激活函数
    nn.Linear(24, 1),                 # 隐藏层24 → 输出1（回归输出不加激活）
)
```

| 模块 | 语法 | 说明 |
|---|---|---|
| `nn.Linear(in, out, bias=True)` | 全连接层 `y = xWᵀ + b` | 权重形状 `(out, in)`，偏置 `(out,)`；参数量 `in×out + out` |
| `nn.Sequential(m1, m2, ...)` | 顺序容器 | 输入依次过每个模块；`state_dict()` 的键是 `'0.weight'`、`'2.weight'`（按层序号） |
| `nn.ReLU()` | ReLU 激活 `max(0, x)` | 负半轴为 0 |
| `nn.Tanh()` | tanh 激活，输出 (−1,1) | 对称、饱和 |
| `nn.LeakyReLU(negative_slope=0.1)` | 负半轴斜率 0.1 | 默认 0.01，避免死区 |
| `nn.MSELoss(reduction='mean')` | 均方误差损失 | `reduction` 可选 `'mean'`（默认，除以 N）/ `'sum'` / `'none'` |

参数量统计（struct_exp / task2_2 里）：

```python
def count_params(model):
    return sum(p.numel() for p in model.parameters())
# p.numel() = 该参数张量的元素总数（形状各维度相乘）
# model.parameters() 遍历所有可学习参数（权重+偏置）
```

### 4.3 优化器（Adam）

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `params` | 必填 | 要优化的参数（`model.parameters()`） |
| `lr` | 0.001 | 学习率（步长）；本例 lr=0.01 会失败，0.001 成功 |
| `betas` | (0.9, 0.999) | 一阶/二阶矩估计的指数衰减系数（一般不动） |
| `eps` | 1e-8 | 数值稳定小量（防除零） |
| `weight_decay` | 0 | **L2 正则系数**：每次更新额外把权重往 0 拉一点，防过拟合。本例统一用 1e-4 |
| `amsgrad` | False | 是否用 AMSGrad 变体（一般不动） |

注意：**`weight_decay` 就是 L2 正则**（等价于损失里加 `λ/2·‖w‖²`）。用户口中的"R2 正则化"在本实验中即以此实现。

### 4.4 训练循环五件套

```python
model.train()                    # 1. 切到训练模式（影响 BN/Dropout；本实验无影响但规范）
optimizer.zero_grad()            # 2. 清空上一步的梯度（不清会累积）
loss = loss_fn(model(X_train), y_train)   # 3. 前向：算预测 → 算损失
loss.backward()                  # 4. 反向传播：自动求梯度，存到各参数的 .grad
optimizer.step()                 # 5. 用梯度更新一次参数
```

| 方法 | 作用 | 不做的后果 |
|---|---|---|
| `optimizer.zero_grad()` | 把参数梯度清零 | 梯度跨步累积，更新方向错误 |
| `loss.backward()` | 从 loss 反向自动计算每个参数的梯度 | 没有梯度可更新 |
| `optimizer.step()` | 按优化器规则用 `.grad` 更新权重 | 训练不推进 |

替代写法（等价的"原地"清零）：

```python
optimizer.zero_grad(set_to_none=True)   # 比默认更省内存，等价于默认 zero_grad()
```

（手动清零的旧式写法 `for p in model.parameters(): p.grad = None` 也可，现在都封装进 zero_grad。）

### 4.5 评估模式与关闭梯度

```python
model.eval()                     # 切到评估模式
with torch.no_grad():            # 块内不建计算图、不存梯度
    val_loss = loss_fn(model(X_val), y_val).item()
    y_pred = model(X_test).numpy()
```

- `model.eval()`：评估/推理模式（影响 Dropout/BN 行为）。
- `torch.no_grad()`：上下文管理器。块内所有运算不追踪梯度——**省内存、加速、且能直接 `.numpy()`**。
- `loss.item()`：把 0 维 tensor 取成 Python 的 `float`（打印、记录用）。
- 早停用的验证损失、最终测试指标，都在 `no_grad` 下算——**验证集/测试集永远不参与训练**。

### 4.6 权重保存、恢复与深拷贝（早停核心）

```python
best_state = None
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_state = {k: v.clone() for k, v in model.state_dict().items()}  # 深拷贝
...
model.load_state_dict(best_state)   # 恢复最优权重
```

- `model.state_dict()`：返回有序字典 `OrderedDict`，键是参数名（如 `'0.weight'`、`'0.bias'`、`'2.weight'`），值是参数张量。
- `{k: v.clone() for k, v in ...}`：**字典推导式**——对每个 (键, 张量) 做 `clone()` 深拷贝。
- **为什么必须 clone**：`best_state = model.state_dict()` 直接赋值只是**引用**（浅拷贝），模型继续训练后字典里的张量也会跟着变，等于没保存。`clone()` 复制一份独立张量，之后模型再怎么变，`best_state` 都是当时的最优值。
- `model.load_state_dict(best_state)`：把保存的权重整体写回模型。

### 4.7 DataLoader 与 TensorDataset（小批量）

```python
from torch.utils.data import DataLoader, TensorDataset
train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=64, shuffle=True)
for xb, yb in train_loader:        # 每次取 64 个样本（xb 输入、yb 标签）
    loss = loss_fn(model(xb), yb)  # 在一个 batch 上前向
```

| 组件 | 作用 |
|---|---|
| `TensorDataset(*tensors)` | 把多个张量打包成"数据集"（按第 0 维对齐） |
| `DataLoader(dataset, batch_size=1, shuffle=False, ...)` | 按 batch 切分、可选打乱、按迭代器取用 |
| `batch_size` | 每个 batch 的样本数 |
| `shuffle=True` | 每个 epoch 前随机打乱顺序（引入随机性） |

小批量时"一步" = 处理一个 batch，所以外层用 `while step < EPOCHS` + 内层 `for xb, yb in train_loader`，每处理完一个 batch `step += 1`（见 exp4）。

### 4.8 张量基础运算

```python
pred = model(X_test)              # 前向：调用模型 = model.forward(X_test)
loss = loss_fn(pred, target)      # 计算损失（MSE）
(y_test - pred) ** 2              # 逐元素减、逐元素平方
((y_test - pred) ** 2).sum()      # 求和 → 张量
((y_test - pred) ** 2).mean()     # 求平均
tensor.item()                     # 0 维张量 → Python 数值
```

---

## 5. matplotlib（画图）

### 5.1 全局设置（中文支持）

```python
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]   # 指定中文字体
plt.rcParams["axes.unicode_minus"] = False                        # 负号正常显示
```

不设置这两行，图中中文会显示成方块、负号显示成乱码。`plt.rcParams` 是全局配置字典。

### 5.2 创建画布与子图

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))      # 1 行 2 列子图
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)  # 共享 y 轴
fig, ax = plt.subplots(figsize=(8, 4.5))               # 单图
```

- 返回 `fig`（整张画布）和 `axes`（子图对象，一维数组时用 `axes[0]`、`axes[1]` 取）。
- `figsize=(宽, 高)` 单位英寸；`sharey=True` 让子图共享 y 轴刻度。
- 也可以 `fig, ax = plt.subplots()` 后 `ax.plot(...)`。

### 5.3 常用绘图函数

```python
ax.plot(x, y, lw=0.8, alpha=0.6, color="tab:blue", label="train loss", ls="--")  # 折线
ax.scatter(x, y, s=10, alpha=0.6, color="tab:red", edgecolors="white",
           linewidths=0.5, zorder=3, label="...")       # 散点
ax.axhline(y, color="tab:red", ls="--", lw=1.2)          # 水平参考线
ax.axvline(x, color="red", ls="--", lw=1, label="best step")  # 垂直参考线
ax.errorbar(NOISES, r2_mu, yerr=r2_sd, fmt="o-", capsize=4, color=..., label=...)  # 误差棒折线
ax.bar(xs, mse_means, yerr=mse_stds, capsize=4, color="#4C72B0", alpha=0.85)       # 柱状图
```

| 常用参数 | 含义 |
|---|---|
| `lw` / `linewidth` | 线宽 |
| `ls` | 线型：`'-'` 实线、`'--'` 虚线、`':'` 点线、`'-.'` 点划线 |
| `alpha` | 透明度 0~1（多条曲线叠加时调低） |
| `color` | 颜色：`"tab:blue"` 等；`"r--"` 表示红色虚线 |
| `label` | 图例文字（需配合 `ax.legend()` 显示） |
| `s` | 散点大小 |
| `edgecolors` / `linewidths` | 散点描边颜色/粗细（白边让重叠点更清晰） |
| `zorder` | 图层顺序（越大越靠上） |
| `fmt` | 误差棒图的线型+标记（`'o-'` 圆点实线） |
| `yerr` | 误差棒数据（标准差） |
| `capsize` | 误差棒端帽大小 |

### 5.4 坐标轴、标题、图例

```python
ax.set_xlabel("step")
ax.set_ylabel("val loss (MSE)")
ax.set_title("训练 / 验证损失曲线（线性模型）")
ax.set_xticks([1, 2])                       # 自定义刻度位置
ax.set_xticklabels(["线性模型", "MLP"], fontsize=9)   # 刻度标签
ax.set_ylim(all_r2.min() - 0.01, all_r2.max() + 0.01) # 手动 y 范围
ax.legend()                                 # 显示图例（按 label）
ax.legend(fontsize=8)                       # 调图例字号
ax.grid(axis="y", alpha=0.3)                # 网格线（只画 y 方向）
```

### 5.5 保存、显示、关闭

```python
plt.tight_layout()                    # 自动调整子图间距，防止标题/标签重叠
plt.savefig(path, dpi=150)            # 保存为文件（dpi 越高越清晰）
plt.show()                            # 弹窗显示（交互环境用；跑批处理可注释掉）
plt.close(fig)                        # 关闭当前画布，释放内存（画多张图必须）
```

**画多张图的规范**（见 task2_2 / exp5）：每张图 `fig, ax = plt.subplots(...)` → 画 → `savefig` → `plt.close(fig)`，否则多张图会互相残留。

---

## 6. 文件读写（os + open）

```python
OUT_DIR = os.path.dirname(os.path.abspath(__file__))   # 脚本所在目录（绝对路径）
report_path = os.path.join(OUT_DIR, "exp1_lr0001_results.txt")  # 拼接路径

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report + "\n")
```

| 语法 | 作用 |
|---|---|
| `__file__` | 当前脚本文件路径 |
| `os.path.abspath(__file__)` | 转成绝对路径 |
| `os.path.dirname(...)` | 取所在目录 |
| `os.path.join(目录, 文件名)` | 跨平台拼接路径（自动处理 `\` 或 `/`） |
| `open(路径, "w", encoding="utf-8")` | 打开文件：`"w"` 覆盖写、`"r"` 读、`"a"` 追加；指定 UTF-8 防中文乱码 |
| `f.write(...)` | 写入内容 |

为什么用 `os.path.dirname(os.path.abspath(__file__))` 而不是写死路径：脚本不管从哪个目录运行（`python 脚本.py` 或双击），输出文件都落在脚本旁边，不会飘到当前工作目录。

---

## 7. 常见坑与注意事项（对照你的实验）

1. **numpy vs tensor 方法混用**：numpy 数组没有 `.numpy()`（要用 `.astype`/直接运算）；tensor 没有 `.astype`（用 `.float()`）。exp2 的报错就是这个。
2. **带梯度的 tensor 不能直接 `.numpy()`**：需在 `torch.no_grad()` 块里，或先 `.detach()`。
3. **保存最优权重必须深拷贝**：`{k: v.clone() for k, v in model.state_dict().items()}`，直接赋值是引用，权重会跟着变。
4. **标准化只 fit 训练集**：验证/测试集用同一个 scaler `transform`，否则数据泄漏。
5. **`reshape(-1, 1)` 的作用**：`nn.Linear` 要求输入 `(N, 特征数)` 二维；`(1500,)` 一维会报错，`-1` 让 numpy 自动算出 1500。
6. **`model.train()` / `model.eval()` 要成对出现**：训练步前 train()，验证/测试前 eval()。
7. **早停恢复权重**：训练结束后 `model.load_state_dict(best_state)`，否则用的是"最后一步"的权重，不是"最优"的权重。
8. **固定 `random_state`**：数据生成（42）、划分（42）、模型初始化（42~46）都固定，实验才可复现、可比。
9. **中文乱码**：脚本开头两行 `plt.rcParams` 必须写；txt 写入要 `encoding="utf-8"`。
10. **loss.item() vs loss**：打印/记录用 `.item()` 取 Python 数；直接打印 tensor 会带 `tensor(...)` 前缀。
11. **全批量 vs 小批量的"一步"含义不同**：全批量一步 = 全部训练数据算一次；小批量一步 = 一个 batch 算一次（exp4 用 while 控制总步数）。
12. **`best_val` 的尺度**：标准化实验里验证损失是标准化尺度（≈0.07），没标准化的实验是原始尺度（≈1.5），两者不能直接比大小（exp2/exp5 的结果文件里已注明）。

---

## 8. 一句话速查（按使用频率排序）

| 想做什么 | 写什么 |
|---|---|
| 生成数据 | `make_friedman1(n_samples, n_features, noise, random_state)` |
| 划分数据 | `train_test_split(X, y, test_size=0.2, random_state=42)` |
| 标准化 | `StandardScaler().fit(X_train)` → `.transform(...)` / `.inverse_transform(...)` |
| numpy→tensor | `torch.from_numpy(x)` |
| tensor→numpy | `x.numpy()`（须在 no_grad 内） |
| 定义线性层 | `nn.Linear(10, 1)` |
| 定义 MLP | `nn.Sequential(nn.Linear(10,24), nn.ReLU(), nn.Linear(24,1))` |
| 损失 | `loss_fn = nn.MSELoss()` |
| 优化器 | `torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)` |
| 训练一步 | `zero_grad()` → `loss.backward()` → `step()` |
| 评估 | `model.eval()` + `with torch.no_grad():` |
| 保存最优权重 | `{k: v.clone() for k, v in model.state_dict().items()}` |
| 恢复最优权重 | `model.load_state_dict(best_state)` |
| 小批量 | `DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)` |
| 画子图 | `fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))` |
| 画折线/散点 | `ax.plot(...)` / `ax.scatter(...)` |
| 画误差棒 | `ax.errorbar(x, y, yerr=std, fmt="o-", capsize=4)` |
| 保存图 | `plt.tight_layout()` → `plt.savefig(path, dpi=150)` → `plt.close(fig)` |
| 打印格式化 | `f"step={epoch:5d} | loss={v:.6f}"` |
| 写结果文件 | `with open(path, "w", encoding="utf-8") as f: f.write(report)` |
