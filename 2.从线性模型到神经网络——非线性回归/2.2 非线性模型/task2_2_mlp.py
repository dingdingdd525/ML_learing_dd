# -*- coding: utf-8 -*-
"""
任务 2.2：加入全连接神经网络（MLP），并与任务 2.1 的线性模型对比

数据：与 task2_1_linear.py 完全相同
    make_friedman1(n_samples=1500, n_features=10, noise=1.0, random_state=42)
划分：训练 60% / 验证 20% / 测试 20%（random_state=42 固定，两种模型共用同一份划分）
模型：
    线性模型：nn.Linear(10, 1)，无激活函数（与 task2_1 相同）
    MLP     ：Linear(10,24) -> ReLU -> Linear(24,1)（单隐藏层，输出层不加 ReLU）
训练：与 task2_1 完全相同的方式
    Adam(lr=0.01)；每训练一步都在验证集上评估一次（验证集不参与训练）；
    早停 patience=1000；10000 步上限；每 100 步打印一次；恢复验证集最优权重。
    最终指标（MSE / R²）在测试集上评估（测试集不参与训练与早停）。
实验：每种模型独立训练 5 次（仅模型初始化种子不同，数据划分固定），
    汇总 测试 MSE / R² 的均值±标准差，输出对比表、可视化图、结果记录文件。

运行：python task2_2_mlp.py
产物：task2_2_loss_curves.png（训练过程）
      task2_2_test_predictions.png（测试集预测效果）
      task2_2_compare.png（均值曲线对比 + 5 次 R² 对比）
      task2_2_results.txt（所有必要结果记录，供后续整理报告）
提示：全流程共 10 次训练（2 模型 × 5 次），CPU 上约几分钟；
      若觉得每 100 步打印太多，可把 PRINT_EVERY 调大。
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

EPOCHS = 10000
PRINT_EVERY = 100
PATIENCE = 1000
N_RUNS = 5
BASE_SEED = 42
LR = 0.01

# ---------- 1. 数据生成与划分（训练 60% / 验证 20% / 测试 20%） ----------
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

# ---------- 2. 模型定义 ----------
def build_linear():
    """线性模型（无激活函数，与 task2_1 相同）"""
    return nn.Linear(10, 1)

def build_mlp():
    """MLP：输入10维 -> 24 -> 输出1维（单隐藏层）；隐藏层 ReLU，输出层无激活"""
    return nn.Sequential(
        nn.Linear(10, 24),
        nn.ReLU(),
        nn.Linear(24, 1),
    )

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# ---------- 3. 单次训练（训练方式与 task2_1 一致） ----------
def train_once(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    best_step = 0
    wait = 0
    stop_epoch = EPOCHS
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        # 训练一步
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        # 每一步都用当前网络在验证集上评估（只做前向，不参与训练），供早停判断
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)

        # 早停判断：验证损失创新低则保存权重并重置计数，否则累计
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

    # 恢复验证集上表现最优的权重
    if best_state is not None:
        model.load_state_dict(best_state)

    # 测试集评估（测试集不参与训练与早停）
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()

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

# ---------- 4. 实验主流程：每种模型独立训练 N_RUNS 次 ----------
def run_experiment(name, builder):
    print(f"\n========== {name}：独立训练 {N_RUNS} 次 ==========")
    results = []
    for run in range(N_RUNS):
        seed = BASE_SEED + run
        torch.manual_seed(seed)   # 在创建模型前固定种子，5 次初始化不同且可复现
        model = builder()
        print(f"\n--- {name} run {run + 1}/{N_RUNS}（seed={seed}）---")
        res = train_once(model)
        results.append(res)
        print(f"    停止于 step={res['stop_epoch']} | best_val={res['best_val_loss']:.4f} | "
              f"test_mse={res['test_mse']:.4f} | test_r2={res['test_r2']:.4f}")
    return results

def summarize(results):
    return (
        np.array([r["test_mse"] for r in results]),
        np.array([r["test_r2"] for r in results]),
        np.array([r["stop_epoch"] for r in results]),
        np.array([r["best_val_loss"] for r in results]),
    )

print("\n================ 开始实验（线性模型 vs MLP，各 5 次） ================")
linear_results = run_experiment("线性模型", build_linear)
mlp_results = run_experiment("MLP", build_mlp)

lin_mse, lin_r2, lin_stop, lin_val = summarize(linear_results)
mlp_mse, mlp_r2, mlp_stop, mlp_val = summarize(mlp_results)
n_lin = count_params(build_linear())
n_mlp = count_params(build_mlp())

def fmt(arr):
    return f"{arr.mean():.4f}±{arr.std():.4f}"

# ---------- 5. 可视化 ----------
def mean_curve(results):
    min_len = min(len(r["val_losses"]) for r in results)
    return np.mean([r["val_losses"][:min_len] for r in results], axis=0)

# 图1：两种模型各自 5 次的验证损失曲线 + 均值
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, name, results, color in zip(
        axes, ["线性模型", "MLP"], [linear_results, mlp_results], ["tab:blue", "tab:orange"]):
    for r in results:
        ax.plot(r["val_losses"], lw=0.5, alpha=0.35, color=color)
    ax.plot(mean_curve(results), lw=2.0, color="tab:red", label="mean val loss")
    ax.set_xlabel("step")
    ax.set_ylabel("val loss (MSE)")
    ax.set_title(f"{name}：5 次验证损失曲线")
    ax.legend()
plt.tight_layout()
p1 = os.path.join(OUT_DIR, "task2_2_loss_curves.png")
plt.savefig(p1, dpi=150)
plt.close(fig)

# 图2：测试集预测效果（各取 5 次中 test R² 最高的一次）
best_lin = max(linear_results, key=lambda r: r["test_r2"])
best_mlp = max(mlp_results, key=lambda r: r["test_r2"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, name, best in zip(axes, ["线性模型", "MLP"], [best_lin, best_mlp]):
    ax.scatter(y_test_np, best["y_pred"], s=10, alpha=0.6)
    lo = min(y_test_np.min(), best["y_pred"].min())
    hi = max(y_test_np.max(), best["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
    ax.set_xlabel("真实值 y")
    ax.set_ylabel("预测值 ŷ")
    ax.set_title(f"{name}（最优 run）：MSE={best['test_mse']:.3f}, R²={best['test_r2']:.3f}")
    ax.legend()
plt.tight_layout()
p2 = os.path.join(OUT_DIR, "task2_2_test_predictions.png")
plt.savefig(p2, dpi=150)
plt.close(fig)

# 图3：左=两模型均值验证损失同图对比；右=5 次测试 R²（每 run 一个点 + 均值线）
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(mean_curve(linear_results), lw=2, label="线性模型 mean", color="tab:blue")
axes[0].plot(mean_curve(mlp_results), lw=2, label="MLP mean", color="tab:orange")
axes[0].set_xlabel("step")
axes[0].set_ylabel("val loss (MSE)")
axes[0].set_title("验证损失均值曲线对比")
axes[0].legend()

# 右图：每个 run 一个点（横向轻微错开避免重叠）+ 虚线均值线 + 均值标注
for i, (arr, c, name) in enumerate(zip([lin_r2, mlp_r2], ["tab:blue", "tab:orange"],
                                       ["线性模型", "MLP"]), start=1):
    xs = np.full(len(arr), i) + np.random.uniform(-0.04, 0.04, len(arr))
    axes[1].scatter(xs, arr, s=35, color=c, alpha=0.85, edgecolors="white",
                    linewidths=0.5, zorder=3, label=f"{name}（均值 {arr.mean():.4f}）")
    axes[1].axhline(arr.mean(), color=c, ls="--", lw=1.2)
all_r2 = np.concatenate([lin_r2, mlp_r2])
axes[1].set_xticks([1, 2])
axes[1].set_xticklabels(["线性模型", "MLP"])
axes[1].set_ylim(all_r2.min() - 0.01, all_r2.max() + 0.01)
axes[1].set_ylabel("测试 R²")
axes[1].set_title("5 次独立训练的测试 R²（点=每次运行，虚线=均值）")
axes[1].legend(fontsize=9)
plt.tight_layout()
p3 = os.path.join(OUT_DIR, "task2_2_compare.png")
plt.savefig(p3, dpi=150)
plt.close(fig)

# ---------- 6. 汇总 + 写入结果文件 ----------
lines = []
lines.append("任务 2.2 实验结果记录")
lines.append("=" * 62)
lines.append("数据：make_friedman1(n_samples=1500, n_features=10, noise=1.0, random_state=42)")
lines.append("划分：训练 60% / 验证 20% / 测试 20%（train_test_split random_state=42，两模型共用同一份划分）")
lines.append("训练：Adam(lr=0.01)，每步在验证集评估（不参与训练），早停 patience=1000，10000 步上限，恢复最优权重")
lines.append("评估：最终指标（MSE / R²）在测试集上计算（测试集不参与训练与早停）")
lines.append(f"实验：每种模型独立训练 {N_RUNS} 次，仅模型初始化种子不同（{BASE_SEED}~{BASE_SEED + N_RUNS - 1}）")
lines.append("")
lines.append("[模型结构与参数量]")
lines.append("线性：nn.Linear(10, 1)，无激活函数（与 task2_1 相同）")
lines.append("MLP ：Linear(10,24) -> ReLU -> Linear(24,1)（单隐藏层，输出层无激活）")
lines.append(f"参数量：线性 = {n_lin}，MLP = {n_mlp}（约为线性的 {n_mlp / n_lin:.1f} 倍）")
lines.append("")
lines.append("[每次运行结果]")
lines.append(f"{'模型':<6}{'run':>4}{'stop_step':>10}{'best_val_mse':>14}{'test_mse':>12}{'test_r2':>10}")
for i, r in enumerate(linear_results, start=1):
    lines.append(f"{'线性':<6}{i:>4}{r['stop_epoch']:>10}{r['best_val_loss']:>14.4f}{r['test_mse']:>12.4f}{r['test_r2']:>10.4f}")
for i, r in enumerate(mlp_results, start=1):
    lines.append(f"{'MLP':<6}{i:>4}{r['stop_epoch']:>10}{r['best_val_loss']:>14.4f}{r['test_mse']:>12.4f}{r['test_r2']:>10.4f}")
lines.append("")
lines.append("[汇总（5 次均值 ± 标准差）]")
lines.append(f"{'指标':<18}{'线性模型':>16}{'MLP':>16}")
lines.append(f"{'测试 MSE':<18}{fmt(lin_mse):>16}{fmt(mlp_mse):>16}")
lines.append(f"{'测试 R²':<18}{fmt(lin_r2):>16}{fmt(mlp_r2):>16}")
lines.append(f"{'最优验证 MSE':<18}{fmt(lin_val):>16}{fmt(mlp_val):>16}")
lines.append(f"{'训练停止步数':<18}{fmt(lin_stop):>16}{fmt(mlp_stop):>16}")
lines.append(f"{'参数量':<18}{n_lin:>16}{n_mlp:>16}")
lines.append("")
lines.append("[自动分析]")
delta_r2 = mlp_r2.mean() - lin_r2.mean()
delta_mse = lin_mse.mean() - mlp_mse.mean()
lines.append(f"1) 测试 R²：MLP 均值 {mlp_r2.mean():.4f} 高于线性 {lin_r2.mean():.4f}，平均提升 {delta_r2:+.4f}。")
lines.append(f"2) 测试 MSE：MLP 均值 {mlp_mse.mean():.4f} 低于线性 {lin_mse.mean():.4f}，平均降低 {delta_mse:.4f}。")
lines.append(f"3) 训练步数：线性平均 {lin_stop.mean():.0f} 步，MLP 平均 {mlp_stop.mean():.0f} 步。")
lines.append(f"4) 参数量：MLP（{n_mlp}）约为线性（{n_lin}）的 {n_mlp / n_lin:.1f} 倍，但性能提升明显。")
lines.append("   → 说明线性模型表达能力不足（欠拟合非线性数据），MLP 借助隐藏层+ReLU 学到了非线性规律。")
lines.append("   → 具体机理分析（为什么结构变化带来能力变化、是否存在过拟合风险等）请结合你自己的观察补充。")
lines.append("")
lines.append("可视化文件：task2_2_loss_curves.png（训练过程）、task2_2_test_predictions.png（测试预测）、")
lines.append("            task2_2_compare.png（均值曲线 + R² 对比）")

report = "\n".join(lines)
print("\n" + report)

report_path = os.path.join(OUT_DIR, "task2_2_results.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report + "\n")

print(f"\n结果文件已保存：{report_path}")
print(f"可视化已保存：\n{p1}\n{p2}\n{p3}")
