# -*- coding: utf-8 -*-
"""
任务 2.1：线性模型（无隐藏层、无激活函数）
数据：make_friedman1(n_samples=1500, n_features=10, noise=1.0, random_state=42)
划分：训练 60% / 验证 20% / 测试 20%
模型：nn.Linear(10, 1) + Adam（无激活函数，本质是线性回归）
训练：共 10000 步，每 100 步打印一次；
      每一步都用当前网络在验证集上评估一次（验证集不参与训练），
      验证损失作为早停判断依据（patience=1000），保存并恢复最优权重。
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

torch.manual_seed(42)  # 固定模型初始化，便于复现

# 中文字体设置（若系统无对应字体可改为英文标签）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 1. 生成数据与划分 ----------
X, y = make_friedman1(n_samples=1500, n_features=10, noise=1.0, random_state=42)
X = X.astype(np.float32)
y = y.reshape(-1, 1).astype(np.float32)  # (1500, 1)

# 先切测试集 20%；剩余 80% 再按 25% 切出验证集（80% × 25% = 20%），余下为训练集 60%
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

# ---------- 2. 模型与优化器 ----------
model = nn.Linear(10, 1)  # 无激活函数
print("模型结构：", model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # lr 可自行调整
loss_fn = nn.MSELoss()

# ---------- 3. 训练 + 每步验证 + 早停 ----------
epochs = 10000
print_every = 100
patience = 1000

best_val_loss = float("inf")
best_state = None  # 验证损失最优时的模型权重
best_step = 0      # 验证损失最优时的步数
wait = 0           # 连续未改善的步数
stop_epoch = epochs

train_losses, val_losses = [], []

for epoch in range(1, epochs + 1):
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
        if wait >= patience:
            stop_epoch = epoch
            print(f"[早停] 第 {epoch} 步触发，最优验证损失 {best_val_loss:.4f}")
            break

    if epoch % print_every == 0:
        print(f"step={epoch:5d} | train_loss={loss.item():.6f} | val_loss={val_loss:.6f}")

# 恢复验证集上表现最优的权重
if best_state is not None:
    model.load_state_dict(best_state)

# ---------- 4. 测试集评估 ----------
model.eval()
with torch.no_grad():
    y_pred = model(X_test).numpy()

mse = float(np.mean((y_test.numpy() - y_pred) ** 2))
ss_res = float(np.sum((y_test.numpy() - y_pred) ** 2))
ss_tot = float(np.sum((y_test.numpy() - y_test.numpy().mean()) ** 2))
r2 = 1 - ss_res / ss_tot

print(f"\n训练步数：{stop_epoch} / {epochs}")
print(f"最优验证 MSE：{best_val_loss:.4f}（第 {best_step} 步）")
print(f"测试集 MSE：{mse:.4f} | R²：{r2:.4f}")

# ---------- 5. 可视化 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 左图：训练 / 验证损失曲线（标出验证损失最优的位置）
axes[0].plot(train_losses, label="train loss", lw=0.8)
axes[0].plot(val_losses, label="val loss", lw=0.8)
axes[0].axvline(best_step, color="red", ls="--", lw=1, label="best val step")
axes[0].set_xlabel("step")
axes[0].set_ylabel("MSE")
axes[0].set_title("训练 / 验证损失曲线（线性模型）")
axes[0].legend()

# 右图：测试集 预测值 vs 真实值
axes[1].scatter(y_test.numpy(), y_pred, s=10, alpha=0.6)
lo = min(y_test.numpy().min(), y_pred.min())
hi = max(y_test.numpy().max(), y_pred.max())
axes[1].plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
axes[1].set_xlabel("真实值 y")
axes[1].set_ylabel("预测值 ŷ")
axes[1].set_title(f"测试集预测效果（MSE={mse:.3f}, R²={r2:.3f}）")
axes[1].legend()

plt.tight_layout()
plt.savefig("task2_1_linear_result.png", dpi=150)
plt.show()
