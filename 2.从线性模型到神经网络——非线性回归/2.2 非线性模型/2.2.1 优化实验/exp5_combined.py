# -*- coding: utf-8 -*-
"""
优化实验 5/5：组合方法——lr=0.001 + 特征/目标标准化 + L2(weight_decay=1e-4)
网络：MLP Linear(10,24)-ReLU-Linear(24,1)
训练：Adam(lr=0.001, weight_decay=1e-4)，每步在验证集评估（不参与训练），
      早停 patience=1000，10000 步上限，每 100 步打印，恢复验证集最优权重。
划分：训练60% / 验证20% / 测试20%（random_state=42，与之前所有实验完全一致）
跑 5 次（种子 42~46），输出统计 + 完善可视化 + 结果记录文件。

对比参考（硬编码自已有结果文件，数据划分相同，均为 MLP 宽24）：
  基线 task2_2（无L2, lr=0.01, 全批量）：test_r2 = [0.7333, 0.7340, 0.8153, 0.7344, 0.7334]
  exp1 lr=0.001 + L2：test_r2 = [0.9409, 0.9416, 0.9400, 0.9361, 0.9337]
  exp2 标准化 + L2：test_r2 = [0.9127, 0.9293, 0.9387, 0.9210, 0.9372]

产物：exp5_combined_loss_curves.png（训练过程，5 次验证损失曲线）
      exp5_combined_test_predictions.png（测试集预测效果）
      exp5_combined_r2_compare.png（组合 vs 基线/单方法的 R² 对比）
      exp5_combined_results.txt（结果记录）
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

EPOCHS = 10000
PRINT_EVERY = 100
PATIENCE = 1000
N_RUNS = 5
BASE_SEED = 42
LR = 0.001            # 组合：lr=0.001
WEIGHT_DECAY = 1e-4   # 组合：L2 正则（R2正则化）

# 对比参考（硬编码自已有结果文件，注明来源）
REF_R2 = {
    "基线(task2_2)": np.array([0.7333, 0.7340, 0.8153, 0.7344, 0.7334]),
    "exp1 lr=0.001": np.array([0.9409, 0.9416, 0.9400, 0.9361, 0.9337]),
    "exp2 标准化": np.array([0.9127, 0.9293, 0.9387, 0.9210, 0.9372]),
}
REF_MSE = {
    "基线(task2_2)": np.array([6.0872, 6.0698, 4.2156, 6.0615, 6.0848]),
    "exp1 lr=0.001": np.array([1.3486, 1.3316, 1.3687, 1.4587, 1.5132]),
    "exp2 标准化": np.array([1.9914, 1.6133, 1.3992, 1.8025, 1.4341]),
}

# ---------- 数据：划分与基线一致 + 标准化（仅 fit 训练集） ----------
X, y = make_friedman1(n_samples=1500, n_features=10, noise=1.0, random_state=42)
X = X.astype(np.float32)
y = y.reshape(-1, 1).astype(np.float32)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42)

scaler_x = StandardScaler().fit(X_train)
X_train = scaler_x.transform(X_train)
X_val = scaler_x.transform(X_val)
X_test = scaler_x.transform(X_test)

scaler_y = StandardScaler().fit(y_train)
y_train = scaler_y.transform(y_train)
y_val = scaler_y.transform(y_val)

X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
X_val = torch.from_numpy(X_val.astype(np.float32))
y_val = torch.from_numpy(y_val.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test_np = y_test.astype(np.float32)   # 原始尺度测试目标，最终评估用

def build_mlp():
    return nn.Sequential(
        nn.Linear(10, 24),
        nn.ReLU(),
        nn.Linear(24, 1),
    )

def train_once(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    best_step = 0
    wait = 0
    stop_epoch = EPOCHS
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()   # 标准化尺度

        train_losses.append(loss.item())
        val_losses.append(val_loss)

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

    # 测试集评估：预测反标准化回原始尺度再算指标
    model.eval()
    with torch.no_grad():
        y_pred_std = model(X_test).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_std)
    mse = float(np.mean((y_test_np - y_pred) ** 2))
    ss_res = float(np.sum((y_test_np - y_pred) ** 2))
    ss_tot = float(np.sum((y_test_np - y_test_np.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "stop_epoch": stop_epoch,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "test_mse": mse,
        "test_r2": r2,
        "y_pred": y_pred,
    }

# ---------- 跑 5 次 ----------
print("实验5：组合（lr=0.001 + 标准化 + L2），MLP 训练 5 次\n")
results = []
for run in range(N_RUNS):
    seed = BASE_SEED + run
    torch.manual_seed(seed)
    model = build_mlp()
    res = train_once(model)
    results.append(res)
    print(f"run{run + 1}: stop_step={res['stop_epoch']}, best_step={res['best_step']}, "
          f"best_val={res['best_val_loss']:.4f}, test_mse={res['test_mse']:.4f}, "
          f"test_r2={res['test_r2']:.4f}")

mse_arr = np.array([r["test_mse"] for r in results])
r2_arr = np.array([r["test_r2"] for r in results])
stop_arr = np.array([r["stop_epoch"] for r in results])
val_arr = np.array([r["best_val_loss"] for r in results])

def fmt(arr):
    return f"{arr.mean():.4f}±{arr.std():.4f}"

def mean_curve(results):
    min_len = min(len(r["val_losses"]) for r in results)
    return np.mean([r["val_losses"][:min_len] for r in results], axis=0)

# ---------- 可视化 ----------
# 图1：5 次验证损失曲线（标准化尺度）+ 均值
fig, ax = plt.subplots(figsize=(8, 4.5))
for r in results:
    ax.plot(r["val_losses"], lw=0.6, alpha=0.4, color="tab:blue")
ax.plot(mean_curve(results), lw=2.0, color="tab:red", label="mean val loss")
ax.set_xlabel("step")
ax.set_ylabel("val loss (MSE，标准化尺度)")
ax.set_title("组合方法（lr=0.001+标准化+L2）：5 次验证损失曲线")
ax.legend()
plt.tight_layout()
p1 = os.path.join(OUT_DIR, "exp5_combined_loss_curves.png")
plt.savefig(p1, dpi=150)
plt.close(fig)

# 图2：测试集预测效果（取 5 次中 test R² 最高的一次）
best = max(results, key=lambda r: r["test_r2"])
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(y_test_np, best["y_pred"], s=10, alpha=0.6)
lo = min(y_test_np.min(), best["y_pred"].min())
hi = max(y_test_np.max(), best["y_pred"].max())
ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
ax.set_xlabel("真实值 y")
ax.set_ylabel("预测值 ŷ")
ax.set_title(f"组合方法（最优 run）：MSE={best['test_mse']:.3f}, R²={best['test_r2']:.3f}")
ax.legend()
plt.tight_layout()
p2 = os.path.join(OUT_DIR, "exp5_combined_test_predictions.png")
plt.savefig(p2, dpi=150)
plt.close(fig)

# 图3：R² 对比——组合 vs 基线/单独方法（每 run 一个点 + 均值线）
ref_names = list(REF_R2.keys()) + ["组合(本次)"]
ref_arrays = list(REF_R2.values()) + [r2_arr]
colors = ["tab:gray", "tab:green", "tab:purple", "tab:red"]
fig, ax = plt.subplots(figsize=(9, 5))
for i, (name, arr, c) in enumerate(zip(ref_names, ref_arrays, colors), start=1):
    xs = np.full(len(arr), i) + np.random.uniform(-0.08, 0.08, len(arr))
    ax.scatter(xs, arr, s=35, color=c, alpha=0.85, edgecolors="white",
               linewidths=0.5, zorder=3, label=f"{name}（均值 {arr.mean():.4f}）")
    ax.axhline(arr.mean(), color=c, ls="--", lw=1.2)
all_r2 = np.concatenate(ref_arrays)
ax.set_xticks(range(1, len(ref_names) + 1))
ax.set_xticklabels(ref_names, fontsize=9)
ax.set_ylim(all_r2.min() - 0.02, all_r2.max() + 0.01)
ax.set_ylabel("测试 R²")
ax.set_title("5 次独立训练的测试 R² 对比（点=每次运行，虚线=均值）")
ax.legend(fontsize=8)
plt.tight_layout()
p3 = os.path.join(OUT_DIR, "exp5_combined_r2_compare.png")
plt.savefig(p3, dpi=150)
plt.close(fig)

# ---------- 结果记录 ----------
lines = []
lines.append("优化实验 5/5：组合方法（lr=0.001 + 特征/目标标准化 + L2）")
lines.append("=" * 60)
lines.append("网络：MLP Linear(10,24)-ReLU-Linear(24,1)")
lines.append("训练：Adam(lr=0.001, weight_decay=1e-4)，每步在验证集评估，早停 patience=1000，10000 步上限，恢复最优权重")
lines.append("划分：训练60% / 验证20% / 测试20%（random_state=42，与之前实验一致）")
lines.append("说明：训练/早停损失为标准化尺度；test_mse / test_r2 已反标准化回原始尺度")
lines.append("")
lines.append("[本次 5 次（组合）]")
lines.append(f"{'run':<4}{'stop_step':>10}{'best_val':>12}{'test_mse':>12}{'test_r2':>10}")
for i, r in enumerate(results, start=1):
    lines.append(f"{i:<4}{r['stop_epoch']:>10}{r['best_val_loss']:>12.4f}{r['test_mse']:>12.4f}{r['test_r2']:>10.4f}")
lines.append("")
lines.append("[汇总（5 次均值 ± 标准差）]")
lines.append(f"test_mse = {fmt(mse_arr)}")
lines.append(f"test_r2  = {fmt(r2_arr)}")
lines.append(f"best_val = {fmt(val_arr)}（标准化尺度）")
lines.append(f"stop_step= {fmt(stop_arr)}")
lines.append("")
lines.append("[与其他实验对比（test_r2 均值±标准差，来源见各结果文件）]")
lines.append(f"基线 task2_2（无L2, lr=0.01, 全批量）：{fmt(REF_R2['基线(task2_2)'])}")
lines.append(f"exp1 lr=0.001 + L2            ：{fmt(REF_R2['exp1 lr=0.001'])}")
lines.append(f"exp2 标准化 + L2              ：{fmt(REF_R2['exp2 标准化'])}")
lines.append(f"exp5 组合（本次）             ：{fmt(r2_arr)}")
lines.append("")
lines.append("[对比结论（自动计算）]")
lines.append(f"1) 组合 vs 基线：R² 提升 {r2_arr.mean() - REF_R2['基线(task2_2)'].mean():+.4f}，MSE 降低 "
             f"{REF_MSE['基线(task2_2)'].mean() - mse_arr.mean():.4f}")
lines.append(f"2) 组合 vs exp1（lr 单独）：R² 变化 {r2_arr.mean() - REF_R2['exp1 lr=0.001'].mean():+.4f}")
lines.append(f"3) 组合 vs exp2（标准化单独）：R² 变化 {r2_arr.mean() - REF_R2['exp2 标准化'].mean():+.4f}")
lines.append(f"4) 5 次中 test_r2>0.80 的次数：{(r2_arr > 0.80).sum()}/5")
lines.append("")
lines.append("可视化文件：exp5_combined_loss_curves.png、exp5_combined_test_predictions.png、")
lines.append("            exp5_combined_r2_compare.png")

report = "\n".join(lines)
print("\n" + report)

report_path = os.path.join(OUT_DIR, "exp5_combined_results.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report + "\n")

print(f"\n结果文件已保存：{report_path}")
print(f"可视化已保存：\n{p1}\n{p2}\n{p3}")
