# ============================================================
# GTSRB 交通标志分类 V4
# 改动：绝对路径；训练完立刻保存权重；评估可独立跑；SKIP_TRAIN 开关
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

from xpud import pick_device, seed_all

# ---------------- 0. 路径与开关 ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "error_analysis")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

CKPT_LOSS = os.path.join(CKPT_DIR, "best_loss.pth")
CKPT_ACC  = os.path.join(CKPT_DIR, "best_acc.pth")

# 如果权重文件已存在，跳过训练，直接评估
SKIP_TRAIN = os.path.exists(CKPT_LOSS) and os.path.exists(CKPT_ACC)
print(f"[paths] OUT_DIR={OUT_DIR}")
print(f"[paths] CKPT_DIR={CKPT_DIR}")
print(f"[mode ] SKIP_TRAIN={SKIP_TRAIN}")

# ---------------- 1. 设备与种子 ----------------
dev = pick_device()
seed_all(42, dev)
print(f"[device] {dev}")

# ---------------- 2. 超参数 ----------------
BATCH_SIZE    = 64
EPOCHS_MAX    = 60
PATIENCE_LOSS = 3
PATIENCE_ACC  = 5
LR            = 1e-3
NUM_CLASSES   = 43
DATA_ROOT     = r"D:\pytorch\机器学习（工作室题目）\3.CNN\data"
IMG_SIZE      = 32

# ---------------- 3. 统计 mean/std ----------------
stat_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

stat_set_full = torchvision.datasets.GTSRB(
    root=DATA_ROOT, split="train", download=False, transform=stat_transform
)
val_size   = int(len(stat_set_full) * 0.3)
train_size = len(stat_set_full) - val_size
stat_train_subset, _ = random_split(
    stat_set_full, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
stat_loader = DataLoader(stat_train_subset, batch_size=256, shuffle=False, num_workers=0)

mean    = torch.zeros(3)
sq_mean = torch.zeros(3)
n = 0
for imgs, _ in stat_loader:
    b = imgs.size(0)
    imgs = imgs.view(b, 3, -1)
    mean    += imgs.mean(dim=2).sum(0)
    sq_mean += (imgs ** 2).mean(dim=2).sum(0)
    n += b
mean /= n
std   = (sq_mean / n - mean ** 2).sqrt()
MEAN = mean.tolist()
STD  = std.tolist()
print(f"[stat] mean = {MEAN}")
print(f"[stat] std  = {STD}")

# ---------------- 4. transform ----------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ---------------- 5. 数据集 ----------------
full_train = torchvision.datasets.GTSRB(
    root=DATA_ROOT, split="train", download=False, transform=train_transform
)
val_size   = int(len(full_train) * 0.3)
train_size = len(full_train) - val_size
trainset, valset = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

val_full = torchvision.datasets.GTSRB(
    root=DATA_ROOT, split="train", download=False, transform=eval_transform
)
valset_eval = torch.utils.data.Subset(val_full, valset.indices)

testset = torchvision.datasets.GTSRB(
    root=DATA_ROOT, split="test", download=False, transform=eval_transform
)

print(f"[data] train={len(trainset)}  val={len(valset_eval)}  test={len(testset)}")

train_loader = DataLoader(trainset,    batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(valset_eval, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(testset,     batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ---------------- 6. 模型 ----------------
class TrafficCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model = TrafficCNN().to(dev)
print(f"[model] params = {sum(p.numel() for p in model.parameters())}")

loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- 7. 单 epoch ----------------
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

# ---------------- 8. 训练 + 双早停 ----------------
train_losses, train_accs = [], []
val_losses,   val_accs   = [], []

if not SKIP_TRAIN:
    best_loss       = float("inf")
    best_loss_state = None
    loss_patience   = 0

    best_acc        = 0.0
    best_acc_state  = None
    acc_patience    = 0

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

        if va_loss < best_loss - 1e-4:
            best_loss = va_loss
            best_loss_state = copy.deepcopy(model.state_dict())
            loss_patience = 0
            print(f"           [loss] best updated -> {best_loss:.4f}")
        else:
            loss_patience += 1
            print(f"           [loss] no improve {loss_patience}/{PATIENCE_LOSS}")

        if va_acc > best_acc + 1e-4:
            best_acc = va_acc
            best_acc_state = copy.deepcopy(model.state_dict())
            acc_patience = 0
            print(f"           [acc ] best updated -> {best_acc:.4f}")
        else:
            acc_patience += 1
            print(f"           [acc ] no improve {acc_patience}/{PATIENCE_ACC}")

        if loss_patience >= PATIENCE_LOSS and acc_patience >= PATIENCE_ACC:
            print(f"[early stop] both triggered at epoch {epoch}")
            break

    # ---- 立刻保存权重，防止后面崩溃丢失 ----
    if best_loss_state is not None:
        torch.save(best_loss_state, CKPT_LOSS)
        print(f"[save] best_loss -> {CKPT_LOSS}")
    if best_acc_state is not None:
        torch.save(best_acc_state, CKPT_ACC)
        print(f"[save] best_acc  -> {CKPT_ACC}")

    # 画训练曲线（只在本次训练后画）
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train"); plt.plot(val_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss curve")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "loss_curve_v4.png"), dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="train"); plt.plot(val_accs, label="val")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy curve")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "acc_curve_v4.png"), dpi=150); plt.close()

else:
    print("[skip] 检测到已有权重文件，跳过训练，直接评估")

# ---------------- 9. 错误图收集/保存 ----------------
@torch.no_grad()
def collect_errors(model, loader, max_errors=200):
    model.eval()
    m = torch.tensor(MEAN).view(3, 1, 1)
    s = torch.tensor(STD).view(3, 1, 1)
    errors = []
    for xb, yb in loader:
        xb_dev = xb.to(dev, non_blocking=True)
        yb_dev = yb.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
            out = model(xb_dev)
        preds = out.argmax(1).cpu()
        mask  = preds != yb
        idxs  = mask.nonzero(as_tuple=True)[0]
        for i in idxs:
            img = xb[i] * s + m
            img = img.clamp(0, 1)
            errors.append((img, yb[i].item(), preds[i].item()))
            if len(errors) >= max_errors:
                return errors
    return errors


def save_grid(errors, save_path, cols=8, title=""):
    n = len(errors)
    if n == 0:
        print(f"[grid] no errors for {title}")
        return
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, (img, t, p) in enumerate(errors):
        ax = axes[i]
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(f"T:{t} P:{p}", fontsize=7)
        ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[grid] saved {n} errors -> {save_path}")


def dump_errors(errors, prefix):
    for k, (img, t, p) in enumerate(errors):
        plt.imsave(os.path.join(OUT_DIR, f"{prefix}_{k:03d}_T{t}_P{p}.png"),
                   img.permute(1, 2, 0).numpy())
    print(f"[dump] {len(errors)} single images saved with prefix {prefix}")


# ---------------- 10. 双权重评估 ----------------
def evaluate_state(state, name, collect_err=True):
    if state is None:
        print(f"[{name}] state is None, skip")
        return None
    model.load_state_dict(state)
    va_loss, va_acc = run_epoch(model, val_loader,  train=False)
    te_loss, te_acc = run_epoch(model, test_loader, train=False)
    print(f"[{name}] "
          f"val loss={va_loss:.4f} acc={va_acc:.4f} | "
          f"test loss={te_loss:.4f} acc={te_acc:.4f}")

    if collect_err:
        val_errors = collect_errors(model, val_loader)
        dump_errors(val_errors, f"{name}_val")
        save_grid(val_errors,
                  os.path.join(OUT_DIR, f"{name}_val_errors_grid.png"),
                  title=f"{name} Validation Errors")

        test_errors = collect_errors(model, test_loader)
        dump_errors(test_errors, f"{name}_test")
        save_grid(test_errors,
                  os.path.join(OUT_DIR, f"{name}_test_errors_grid.png"),
                  title=f"{name} Test Errors")

    return va_loss, va_acc, te_loss, te_acc


print("\n============= 双早停权重对比 =============")

# 从磁盘加载权重
best_loss_state = torch.load(CKPT_LOSS, map_location="cpu")
best_acc_state  = torch.load(CKPT_ACC,  map_location="cpu")

result_loss = evaluate_state(best_loss_state, "best_loss", collect_err=True)
result_acc  = evaluate_state(best_acc_state,  "best_acc",  collect_err=True)

print("=========================================")
print("结论：哪个 test acc 更高，哪个早停分支就更好。\n")
print("[done]")