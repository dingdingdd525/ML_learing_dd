# -*- coding: utf-8 -*-
"""
延伸思考 1：网络结构实验
固定条件：Adam(lr=0.001, weight_decay=1e-4)，全批量，每训练一步在验证集评估（验证集不参与训练），
早停 patience=1000，10000 步上限，恢复验证集最优权重；数据划分与之前所有实验一致
（make_friedman1, n=1500, noise=1.0, random_state=42；60/20/20, random_state=42）。
每种结构独立训练 5 次（种子 42~46），报告均值 ± 标准差。
注意：激活函数实验只改隐藏层激活，输出层保持线性（回归输出须无界）。
"""
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE_SEED = 42
LR = 0.001
WEIGHT_DECAY = 1e-4
MAX_STEPS = 10000
PATIENCE = 1000
N_RUNS = 5
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 数据（固定） ----------
X, y = make_friedman1(n_samples=1500, n_features=10, noise=1.0, random_state=42)
X = X.astype(np.float32)
y = y.reshape(-1, 1).astype(np.float32)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.25, random_state=42)
X_tr_t, y_tr_t = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
X_va_t, y_va_t = torch.from_numpy(X_va), torch.from_numpy(y_va)
X_te_t, y_te_t = torch.from_numpy(X_te), torch.from_numpy(y_te)


def build_model(cfg):
    kind = cfg["kind"]
    if kind == "baseline":        # 10-24-1, ReLU（参照，与 exp1 相同）
        return nn.Sequential(nn.Linear(10, 24), nn.ReLU(), nn.Linear(24, 1))
    if kind == "depth_32_64_32":  # 10-32-64-32-1
        return nn.Sequential(nn.Linear(10, 32), nn.ReLU(),
                             nn.Linear(32, 64), nn.ReLU(),
                             nn.Linear(64, 32), nn.ReLU(),
                             nn.Linear(32, 1))
    if kind == "depth_64_64":     # 10-64-64-1
        return nn.Sequential(nn.Linear(10, 64), nn.ReLU(),
                             nn.Linear(64, 64), nn.ReLU(),
                             nn.Linear(64, 1))
    if kind == "width_32":        # 10-32-1
        return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
    if kind == "width_128":       # 10-128-1
        return nn.Sequential(nn.Linear(10, 128), nn.ReLU(), nn.Linear(128, 1))
    if kind == "act_tanh":        # 10-24-1, tanh
        return nn.Sequential(nn.Linear(10, 24), nn.Tanh(), nn.Linear(24, 1))
    if kind == "act_leaky":       # 10-24-1, LeakyReLU(0.1)
        return nn.Sequential(nn.Linear(10, 24), nn.LeakyReLU(0.1), nn.Linear(24, 1))
    raise ValueError(kind)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_and_eval(cfg, seed):
    torch.manual_seed(seed)
    model = build_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    patience_cnt = 0
    stop = MAX_STEPS
    for step in range(1, MAX_STEPS + 1):
        opt.zero_grad()
        loss = loss_fn(model(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t), y_va_t).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                stop = step
                break
    model.load_state_dict(best_state)
    with torch.no_grad():
        pred = model(X_te_t)
        test_mse = float(loss_fn(pred, y_te_t).item())
        ss_res = float(((y_te_t - pred) ** 2).sum())
        ss_tot = float(((y_te_t - y_te_t.mean()) ** 2).sum())
        test_r2 = 1.0 - ss_res / ss_tot
    return stop, best_val, test_mse, test_r2


CONFIGS = [
    {"name": "baseline_10_24_relu",  "kind": "baseline"},
    {"name": "depth_10_32_64_32_1",  "kind": "depth_32_64_32"},
    {"name": "depth_10_64_64_1",     "kind": "depth_64_64"},
    {"name": "width_10_32_1",        "kind": "width_32"},
    {"name": "width_10_128_1",       "kind": "width_128"},
    {"name": "act_tanh_10_24_1",     "kind": "act_tanh"},
    {"name": "act_leakyrelu_10_24_1", "kind": "act_leaky"},
]

SHORT = ["基线10-24\nReLU", "深度\n32-64-32", "深度\n64-64", "宽度\n32", "宽度\n128", "激活\ntanh", "激活\nLeakyReLU"]

results = {}
txt_path = os.path.join(OUT_DIR, "struct_exp_results.txt")
lines = []
lines.append("延伸思考 1：网络结构实验")
lines.append("=" * 62)
lines.append("数据：make_friedman1(n_samples=1500, n_features=10, noise=1.0, random_state=42)")
lines.append("划分：训练60% / 验证20% / 测试20%（random_state=42，与之前实验一致）")
lines.append("训练：Adam(lr=0.001, weight_decay=1e-4)，每步在验证集评估（不参与训练），早停 patience=1000，10000 步上限，恢复最优权重")
lines.append("实验：每种结构独立训练 5 次，仅模型初始化种子不同（42~46）")
lines.append("")

for cfg in CONFIGS:
    name = cfg["name"]
    n_params = count_params(build_model(cfg))
    rows = []
    for run in range(1, N_RUNS + 1):
        seed = BASE_SEED + run - 1
        stop, bv, tm, tr2 = train_and_eval(cfg, seed)
        rows.append((run, stop, bv, tm, tr2))
        print(f"[{name}] run {run}: stop={stop} best_val={bv:.4f} test_mse={tm:.4f} test_r2={tr2:.4f}", flush=True)
    results[name] = {"rows": rows, "params": n_params}

    lines.append(f"### {name}（参数量 {n_params}）")
    lines.append("run  stop_step  best_val    test_mse   test_r2")
    for run, stop, bv, tm, tr2 in rows:
        lines.append(f"{run:<6d}{stop:<10d}{bv:>10.4f}{tm:>11.4f}{tr2:>10.4f}")
    arr = np.array([r[1:] for r in rows])
    mu, sd = arr.mean(axis=0), arr.std(axis=0)
    lines.append(f"均值 {mu[0]:<10.1f}{mu[1]:>10.4f}{mu[2]:>11.4f}{mu[3]:>10.4f}")
    lines.append(f"std  {sd[0]:<10.1f}{sd[1]:>10.4f}{sd[2]:>11.4f}{sd[3]:>10.4f}")
    lines.append("")

lines.append("=" * 62)
lines.append("[汇总（5 次均值 ± 标准差）]")
lines.append("结构                  参数量     测试MSE            测试R2")
for i, cfg in enumerate(CONFIGS):
    name = cfg["name"]
    arr = np.array([r[1:] for r in results[name]["rows"]])
    mu, sd = arr.mean(axis=0), arr.std(axis=0)
    lines.append(f"{name:<22s}{results[name]['params']:<8d}{mu[2]:>8.4f}±{sd[2]:<.4f}   {mu[3]:>8.4f}±{sd[3]:<.4f}")
lines.append("")
lines.append("说明：所有结果均在同一数据划分上测试；best_val 为验证集最优 MSE（原始尺度）。")
lines.append("可视化文件：struct_exp_compare.png")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("results saved ->", txt_path, flush=True)

# ---------- 画图 ----------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
names = [cfg["name"] for cfg in CONFIGS]
xs = np.arange(len(CONFIGS))

r2_means, r2_stds, mse_means, mse_stds = [], [], [], []
for name in names:
    arr = np.array([r[1:] for r in results[name]["rows"]])
    mse_means.append(arr[:, 2].mean()); mse_stds.append(arr[:, 2].std())
    r2_means.append(arr[:, 3].mean()); r2_stds.append(arr[:, 3].std())

# 左图：R² 散点 + 均值线
ax = axes[0]
for i, name in enumerate(names):
    arr = np.array([r[1:] for r in results[name]["rows"]])
    ys = arr[:, 3]
    ax.scatter([i] * len(ys), ys, s=40, alpha=0.7, zorder=3)
ax.plot(xs, r2_means, "o-", color="tab:red", lw=1.5, label="均值", zorder=2)
for i, (m, s) in enumerate(zip(r2_means, r2_stds)):
    ax.text(i, m + 0.012, f"{m:.4f}", ha="center", fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels(SHORT)
ax.set_ylabel("测试 R²")
ax.set_ylim(0.80, 0.97)
ax.set_title("各结构 5 次测试 R²（散点为 5 次，红线为均值）")
ax.grid(axis="y", alpha=0.3)
ax.legend()

# 右图：MSE 均值条形 + 标准差误差棒
ax = axes[1]
ax.bar(xs, mse_means, yerr=mse_stds, capsize=4, color="#4C72B0", alpha=0.85)
for i, m in enumerate(mse_means):
    ax.text(i, m + 0.05, f"{m:.3f}", ha="center", fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels(SHORT)
ax.set_ylabel("测试 MSE")
ax.set_title("各结构测试 MSE（均值 ± 标准差）")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "struct_exp_compare.png"), dpi=150)
print("figure saved ->", os.path.join(OUT_DIR, "struct_exp_compare.png"), flush=True)
