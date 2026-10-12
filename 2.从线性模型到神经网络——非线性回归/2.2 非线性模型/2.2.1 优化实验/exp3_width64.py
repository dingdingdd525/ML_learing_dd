# -*- coding: utf-8 -*-
"""
优化实验 3/4：隐藏层宽度 24 → 64
网络：MLP Linear(10,64)-ReLU-Linear(64,1)（其余与 task2_2 相同）
统一修改：所有方法都加 L2 正则（Adam weight_decay=1e-4，防过拟合）
仅跑 MLP 5 次，只输出统计，不画图。
基线参考（task2_2_mlp.py，无 L2，宽度 24）：
    test_r2 = 0.7333/0.7340/0.8153/0.7344/0.7334，均值 0.7501±0.0326
"""

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

EPOCHS = 10000
PRINT_EVERY = 100
PATIENCE = 1000
N_RUNS = 5
BASE_SEED = 42
LR = 0.01              # 方法3：lr 不变，只改宽度
WEIGHT_DECAY = 1e-4    # L2 正则（R2正则化），所有方法统一；数值可调（想更明显可试 1e-3）

# 基线参考（task2_2_mlp.py 的结果，无 L2，宽度 24）
BASE_R2 = [0.7333, 0.7340, 0.8153, 0.7344, 0.7334]
BASE_MSE = [6.0872, 6.0698, 4.2156, 6.0615, 6.0848]

# ---------- 数据（与基线完全相同的划分） ----------
X, y = make_friedman1(n_samples=1500, n_features=10, noise=1.0, random_state=42)
X = X.astype(np.float32)
y = y.reshape(-1, 1).astype(np.float32)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42)

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)
y_test_np = y_test.numpy()

def build_mlp():
    """方法3：隐藏层宽度 24 → 64"""
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def train_once(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    best_step = 0
    wait = 0
    stop_epoch = EPOCHS

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_step = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                stop_epoch = epoch
                break

        if epoch % PRINT_EVERY == 0:
            print(f"    step={epoch:5d} | train_loss={loss.item():.6f} | val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()
    mse = float(np.mean((y_test_np - y_pred) ** 2))
    ss_res = float(np.sum((y_test_np - y_pred) ** 2))
    ss_tot = float(np.sum((y_test_np - y_test_np.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    return stop_epoch, best_step, best_val_loss, mse, r2

# ---------- 跑 5 次 ----------
print("实验3：隐藏层 64 + L2(weight_decay=1e-4)，MLP 训练 5 次\n")
stops, bests, mses, r2s = [], [], [], []
for run in range(N_RUNS):
    seed = BASE_SEED + run
    torch.manual_seed(seed)
    model = build_mlp()
    stop, best_step, best_val, mse, r2 = train_once(model)
    stops.append(stop); bests.append(best_val); mses.append(mse); r2s.append(r2)
    print(f"run{run + 1}: stop_step={stop}, best_step={best_step}, best_val={best_val:.4f}, "
          f"test_mse={mse:.4f}, test_r2={r2:.4f}")

r2_arr = np.array(r2s); mse_arr = np.array(mses)
stop_arr = np.array(stops); val_arr = np.array(bests)
n_params = count_params(build_mlp())

# ---------- 统计 + 写入结果文件 ----------
base_r2_arr = np.array(BASE_R2); base_mse_arr = np.array(BASE_MSE)
lines = []
lines.append("优化实验 3/4：隐藏层宽度 24 → 64")
lines.append("=" * 56)
lines.append("网络：MLP Linear(10,64)-ReLU-Linear(64,1)")
lines.append(f"参数量：{n_params}（基线宽度 24 时为 289，约为其 {n_params / 289:.1f} 倍）")
lines.append("训练：Adam(lr=0.01, weight_decay=1e-4)，每步验证，早停 patience=1000，10000 步上限，恢复最优权重")
lines.append("划分：训练60% / 验证20% / 测试20%（random_state=42，与基线相同）")
lines.append("")
lines.append("[基线参考（task2_2_mlp.py，无 L2，宽度 24）]")
lines.append(f"  test_r2 = {'/'.join(f'{v:.4f}' for v in BASE_R2)}，均值 {base_r2_arr.mean():.4f}±{base_r2_arr.std():.4f}")
lines.append(f"  test_mse = {'/'.join(f'{v:.4f}' for v in BASE_MSE)}，均值 {base_mse_arr.mean():.4f}±{base_mse_arr.std():.4f}")
lines.append("")
lines.append("[本次 5 次（宽度 64 + L2）]")
lines.append(f"{'run':<4}{'stop_step':>10}{'best_val':>12}{'test_mse':>12}{'test_r2':>10}")
for i in range(N_RUNS):
    lines.append(f"{i + 1:<4}{stops[i]:>10}{bests[i]:>12.4f}{mses[i]:>12.4f}{r2s[i]:>10.4f}")
lines.append("")
lines.append("[汇总（5 次均值 ± 标准差）]")
lines.append(f"test_mse = {mse_arr.mean():.4f}±{mse_arr.std():.4f}")
lines.append(f"test_r2  = {r2_arr.mean():.4f}±{r2_arr.std():.4f}")
lines.append(f"best_val = {val_arr.mean():.4f}±{val_arr.std():.4f}")
lines.append(f"stop_step= {stop_arr.mean():.1f}±{stop_arr.std():.1f}")
lines.append("")
lines.append("[与基线对比]")
lines.append(f"test_r2 提升 = {r2_arr.mean() - base_r2_arr.mean():+.4f}")
lines.append(f"test_mse 变化 = {mse_arr.mean() - base_mse_arr.mean():+.4f}")
lines.append(f"5 次中 test_r2>0.75 的次数：{(r2_arr > 0.75).sum()}/5（基线为 {(base_r2_arr > 0.75).sum()}/5）")
lines.append(f"5 次中 test_r2>0.80 的次数：{(r2_arr > 0.80).sum()}/5（基线为 {(base_r2_arr > 0.80).sum()}/5）")
lines.append("说明：R² 提升明显且失败次数减少，才视为该方法有效。")

report = "\n".join(lines)
print("\n" + report)

report_path = os.path.join(OUT_DIR, "exp3_width64_results.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report + "\n")
print(f"\n结果文件已保存：{report_path}")
