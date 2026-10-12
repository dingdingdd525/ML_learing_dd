# -*- coding: utf-8 -*-
"""
延伸思考 2：研究噪声对模型的影响
数据：make_friedman1(n_samples=1500, n_features=10, noise=N, random_state=42)，N 取多个值从较小到较大。
划分：60/20/20（random_state=42，每个噪声值独立生成数据但划分方式固定）。
模型：线性（10-1）vs MLP（10-24-1, ReLU）——MLP 使用经优化实验确认的最佳训练配置。
训练：Adam(lr=0.001, weight_decay=1e-4)，全批量，每步在验证集评估（不参与训练），
早停 patience=1000，10000 步上限，恢复最优权重；每 (噪声, 模型) 组合独立训练 5 次（种子 42~46）。
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
NOISES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]


def make_data(noise):
    X, y = make_friedman1(n_samples=1500, n_features=10, noise=noise, random_state=42)
    X = X.astype(np.float32)
    y = y.reshape(-1, 1).astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.25, random_state=42)
    return (torch.from_numpy(X_tr), torch.from_numpy(y_tr),
            torch.from_numpy(X_va), torch.from_numpy(y_va),
            torch.from_numpy(X_te), torch.from_numpy(y_te))


def build_model(kind):
    if kind == "linear":
        return nn.Linear(10, 1)
    return nn.Sequential(nn.Linear(10, 24), nn.ReLU(), nn.Linear(24, 1))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_and_eval(kind, seed, data):
    X_tr_t, y_tr_t, X_va_t, y_va_t, X_te_t, y_te_t = data
    torch.manual_seed(seed)
    model = build_model(kind)
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


MODELS = ["linear", "mlp"]
MODEL_LABEL = {"linear": "线性模型", "mlp": "MLP(10-24-1)"}

results = {}  # (noise, kind) -> rows
txt_path = os.path.join(OUT_DIR, "noise_exp_results.txt")
lines = []
lines.append("延伸思考 2：研究噪声对模型的影响")
lines.append("=" * 62)
lines.append("数据：make_friedman1(n_samples=1500, n_features=10, noise=N, random_state=42)")
lines.append("划分：训练60% / 验证20% / 测试20%（random_state=42）")
lines.append("训练：Adam(lr=0.001, weight_decay=1e-4)，全批量，每步在验证集评估，早停 patience=1000，10000 步上限，恢复最优权重")
lines.append("实验：每个（噪声, 模型）组合独立训练 5 次（种子 42~46）")
lines.append(f"参数量：线性 = {count_params(build_model('linear'))}，MLP = {count_params(build_model('mlp'))}")
lines.append("")

for noise in NOISES:
    data = make_data(noise)
    for kind in MODELS:
        rows = []
        for run in range(1, N_RUNS + 1):
            seed = BASE_SEED + run - 1
            stop, bv, tm, tr2 = train_and_eval(kind, seed, data)
            rows.append((run, stop, bv, tm, tr2))
            print(f"[noise={noise} {kind}] run {run}: stop={stop} best_val={bv:.4f} test_mse={tm:.4f} test_r2={tr2:.4f}", flush=True)
        results[(noise, kind)] = rows

for noise in NOISES:
    lines.append(f"### noise = {noise}")
    for kind in MODELS:
        rows = results[(noise, kind)]
        lines.append(f"-- {MODEL_LABEL[kind]} --")
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
lines.append("noise  模型          测试MSE            测试R2")
for noise in NOISES:
    for kind in MODELS:
        arr = np.array([r[1:] for r in results[(noise, kind)]])
        mu, sd = arr.mean(axis=0), arr.std(axis=0)
        lines.append(f"{noise:<6.1f}  {MODEL_LABEL[kind]:<12s}{mu[2]:>8.4f}±{sd[2]:<.4f}   {mu[3]:>8.4f}±{sd[3]:<.4f}")
lines.append("")
lines.append("可视化文件：noise_compare.png")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("results saved ->", txt_path, flush=True)

# ---------- 画图 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
colors = {"linear": "tab:blue", "mlp": "tab:red"}
for kind in MODELS:
    r2_mu, r2_sd, mse_mu, mse_sd = [], [], [], []
    for noise in NOISES:
        arr = np.array([r[1:] for r in results[(noise, kind)]])
        r2_mu.append(arr[:, 3].mean()); r2_sd.append(arr[:, 3].std())
        mse_mu.append(arr[:, 2].mean()); mse_sd.append(arr[:, 2].std())
    axes[0].errorbar(NOISES, r2_mu, yerr=r2_sd, fmt="o-", color=colors[kind],
                     capsize=4, label=MODEL_LABEL[kind])
    axes[1].errorbar(NOISES, mse_mu, yerr=mse_sd, fmt="o-", color=colors[kind],
                     capsize=4, label=MODEL_LABEL[kind])

axes[0].set_xlabel("noise（噪声标准差）")
axes[0].set_ylabel("测试 R²")
axes[0].set_title("噪声对测试 R² 的影响（均值 ± 标准差）")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].set_xlabel("noise（噪声标准差）")
axes[1].set_ylabel("测试 MSE")
axes[1].set_title("噪声对测试 MSE 的影响（均值 ± 标准差）")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "noise_compare.png"), dpi=150)
print("figure saved ->", os.path.join(OUT_DIR, "noise_compare.png"), flush=True)
