# ============================================================
# GTSRB 交通标志分类 —— 完整训练脚本
# 设备：Intel Arc XPU
# 结构：Conv-BN-ReLU × 2 → Pool → Conv-BN-ReLU × 2 → Pool → GAP → FC
# 数据：官方 train 按 3:7 切 val:train；官方 test 作最终测试
# 早停：按 epoch，patience = 3
# ============================================================

import os
import time
import copy
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# ---------- 0. 设备与种子 ----------
from xpud import pick_device, seed_all

dev = pick_device()
seed_all(42, dev)
print(f"[device] {dev}")

# ---------- 1. 超参数统一管理 ----------
BATCH_SIZE   = 64
EPOCHS_MAX   = 50
PATIENCE     = 3
LR           = 1e-3
NUM_CLASSES  = 43
DATA_ROOT    = "./data"
IMG_SIZE     = 30

# ---------- 2. 统计训练集真实 mean / std ----------
# 统计时只用 Resize + ToTensor，不加任何增强、不加 Normalize
stat_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

stat_set = torchvision.datasets.GTSRB(
    root=DATA_ROOT, split="train", download=True, transform=stat_transform
)
stat_loader = DataLoader(stat_set, batch_size=256, shuffle=False, num_workers=0)

mean = torch.zeros(3)
sq_mean = torch.zeros(3)
n = 0
for imgs, _ in stat_loader:
    b = imgs.size(0)
    imgs = imgs.view(b, 3, -1)          # (B, 3, H*W)
    mean    += imgs.mean(dim=2).sum(0)
    sq_mean += (imgs ** 2).mean(dim=2).sum(0)
    n += b
mean    /= n
std      = (sq_mean / n - mean ** 2).sqrt()
MEAN = mean.tolist()
STD  = std.tolist()
print(f"[stat] mean = {MEAN}")
print(f"[stat] std  = {STD}")

# ---------- 3. 正式 transform ----------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ---------- 4. 数据集划分 ----------
# 官方 train 内部按 3:7 切 val : train
full_train = torchvision.datasets.GTSRB(
    root=DATA_ROOT, split="train", download=True, transform=train_transform
)
val_size   = int(len(full_train) * 0.3)
train_size = len(full_train) - val_size
trainset, valset = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

testset = torchvision.datasets.GTSRB(
    root=DATA_ROOT, split="test", download=True, transform=test_transform
)

print(f"[data] train={len(trainset)}  val={len(valset)}  test={len(testset)}")

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(valset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)
test_loader  = DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)

# ---------- 5. 模型 ----------
class TrafficCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # 块1：Conv-BN-ReLU × 2 + Pool
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # 块2：Conv-BN-ReLU × 2 + Pool
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # GAP + 分类
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.block1(x)      # 30×30 → 15×15×16
        x = self.block2(x)      # 15×15 → 7×7×32
        x = self.gap(x)         # 7×7 → 1×1×32
        x = self.flatten(x)     # → (B, 32)
        x = self.fc(x)          # → (B, 43)
        return x

model = TrafficCNN().to(dev)
print(f"[model] params = {sum(p.numel() for p in model.parameters())}")

# ---------- 6. 损失 / 优化器 ----------
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------- 7. 单 epoch 训练 / 评估 ----------
def run_epoch(model, loader, train=True):
    model.train(train)
    total_loss, total_correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)

            if train:
                optimizer.zero_grad()

            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                out  = model(xb)
                loss = loss_fn(out, yb)

            if train:
                loss.backward()
                optimizer.step()

            total_loss    += loss.item() * xb.size(0)
            total_correct += (out.argmax(1) == yb).sum().item()
            total         += xb.size(0)
    return total_loss / total, total_correct / total

# ---------- 8. 训练主循环 + 早停 ----------
train_losses, train_accs = [], []
val_losses,   val_accs   = [], []

best_val_loss   = float("inf")
best_state_dict = None
no_improve      = 0

for epoch in range(1, EPOCHS_MAX + 1):
    t0 = time.time()

    tr_loss, tr_acc = run_epoch(model, train_loader, train=True)
    va_loss, va_acc = run_epoch(model, val_loader,   train=False)

    train_losses.append(tr_loss); train_accs.append(tr_acc)
    val_losses.append(va_loss);   val_accs.append(va_acc)

    print(f"[Epoch {epoch:02d}] "
          f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
          f"val loss={va_loss:.4f} acc={va_acc:.4f} | "
          f"{time.time()-t0:.1f}s")

    # 早停判断
    if va_loss < best_val_loss - 1e-4:
        best_val_loss   = va_loss
        best_state_dict = copy.deepcopy(model.state_dict())
        no_improve      = 0
        print(f"           ↑ best val loss updated to {best_val_loss:.4f}")
    else:
        no_improve += 1
        print(f"           no improve {no_improve}/{PATIENCE}")
        if no_improve >= PATIENCE:
            print(f"[early stop] triggered at epoch {epoch}")
            break

# 载入最佳权重
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
    print(f"[best] val loss = {best_val_loss:.4f}")

# ---------- 9. 测试集评估（只跑一次） ----------
te_loss, te_acc = run_epoch(model, test_loader, train=False)
print(f"[test] loss={te_loss:.4f}  acc={te_acc:.4f}")

# ---------- 10. 可视化 ----------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="train")
plt.plot(val_losses,   label="val")
plt.xlabel("epoch"); plt.ylabel("loss")
plt.title("Loss curve"); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig("loss_curve.png", dpi=150); plt.show()

plt.figure(figsize=(8, 5))
plt.plot(train_accs, label="train")
plt.plot(val_accs,   label="val")
plt.xlabel("epoch"); plt.ylabel("accuracy")
plt.title("Accuracy curve"); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig("acc_curve.png", dpi=150); plt.show()