# 数据处理与 CNN 语法手册（GTSRB 例题）

> 来源：工作室题目「3.CNN」对方回答整理。涵盖：数据集加载与预处理、`device` 选择、CNN 常用层与语法、训练/评估流程、常用方法与函数（数据增强、加噪声、学习率调度、优化器、损失、正则化、模型存取、可视化）。

---

## 一、数据处理（数据集加载与预处理）

### 1.1 原代码的问题

```python
trainset = torchvision.datasets.GTSRB(root="./data", split="train", download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
```

需要补的点：

1. **`DataLoader` 没有导入**，需加：
   ```python
   from torch.utils.data import DataLoader
   ```
2. **`transform` 是空的** —— GTSRB 图片尺寸不统一，必须加 `Resize`，否则后面卷积/全连接层对不上。
3. **训练集和测试集最好用不同 transform**（测试集不加随机增强）。
4. **`BATCH_SIZE` 要先定义**，例如 `BATCH_SIZE = 64`。
5. `split="train"` / `split="test"` / `split="val"` 都合法；`download=True` 首次下载、之后检查缓存；训练集 `shuffle=True`、测试集 `shuffle=False` 写法正确。

### 1.2 推荐写法：训练/测试用不同 transform

```python
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
```

- `Resize((32,32))`：统一尺寸。
- `ToTensor()`：PIL 图片转成 PyTorch 张量，并把像素从 0~255 缩放到 0~1。
- `Normalize`：标准化，让数据均值约 0、方差约 1，训练更稳。

> 注：上面 `test_transform` 第一行漏了右括号，正确应为 `transforms.Resize((32, 32)),`

### 1.3 完整数据加载代码

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

trainset = torchvision.datasets.GTSRB(root="./data", split="train", download=True, transform=train_transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.GTSRB(root="./data", split="test", download=True, transform=test_transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
```

---

## 二、CPU 还是 GPU？

**强烈建议用 GPU。**

- GTSRB 约 39,000 张训练图、43 类；CNN 参数量大，CPU 训练可能要几小时，GPU 可缩到几分钟~几十分钟。
- 代码里已写 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`，但**模型和数据都要搬到 GPU**：
  ```python
  model = SimpleCNN().to(device)

  for images, labels in trainloader:
      images, labels = images.to(device), labels.to(device)
      ...
  ```
- 没 GPU：CPU 也能跑，只是慢；可减小 batch size、减少 epoch、缩小模型。
- Colab 白嫖 GPU：代码执行程序 → 更改运行时类型 → 硬件加速器选 GPU。

---

## 三、CNN 相关 PyTorch 语法

### 3.1 卷积层 `Conv2d`

```python
nn.Conv2d(
    in_channels=3,      # 输入通道数
    out_channels=16,    # 输出通道数（卷积核个数）
    kernel_size=3,     # 卷积核大小，3 表示 3×3
    stride=1,           # 步长
    padding=1          # 填充
)
```
- 输入形状：`(batch, in_channels, H, W)`
- 输出形状：`(batch, out_channels, H_out, W_out)`

### 3.2 池化层

```python
nn.MaxPool2d(kernel_size=2, stride=2)      # 最大池化：输入 (B,C,H,W) → 输出 (B,C,H/2,W/2)，通道数不变
nn.AvgPool2d(kernel_size=2, stride=2)      # 平均池化
nn.AdaptiveAvgPool2d((1, 1))               # 全局平均池化：输出 (B,C,1,1)，展平即 (B,C)
```

### 3.3 激活函数

```python
nn.ReLU()   # 通常每个卷积层后面接一个 ReLU
```

### 3.4 展平

```python
nn.Flatten()   # 把 (B,C,H,W) 展平成 (B, C*H*W)，方便接全连接层
```

### 3.5 全连接层

```python
nn.Linear(in_features, out_features)   # 输入 (B, in_features) → 输出 (B, out_features)
```

### 3.6 Dropout

```python
nn.Dropout(p=0.5)   # 训练时随机丢弃 50% 神经元，防止过拟合
```

### 3.7 BatchNorm

```python
nn.BatchNorm2d(num_features=16)   # 对卷积输出做归一化，加速训练、提升稳定性
```

### 3.8 一个完整的 CNN 例子

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.flatten = nn.Flatten()
        # 输入 32×32，经过三次 pool 变成 4×4，通道 64
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 32→16
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 16→8
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # 8→4
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### 3.9 训练循环

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")
```

### 3.10 评估

```python
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
```

---

## 四、CNN 常用方法与函数（图像处理、加噪声等）

### 4.1 数据增强（transforms）

```python
transforms.Resize((32, 32))                                       # 缩放
transforms.CenterCrop(32)                                         # 中心裁剪
transforms.RandomCrop(32, padding=4)                              # 随机裁剪
transforms.RandomHorizontalFlip()                                 # 随机水平翻转
transforms.RandomVerticalFlip()                                   # 随机垂直翻转
transforms.RandomRotation(15)                                     # 随机旋转 ±15 度
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))          # 随机平移
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)  # 颜色抖动
transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))          # 高斯模糊
transforms.RandomErasing(p=0.5)                                   # 随机遮挡
transforms.ToTensor()                                             # 转张量
transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)                   # 标准化
```

### 4.2 手动加噪声（自己写）

```python
def add_gaussian_noise(img, mean=0., std=0.1):
    noise = torch.randn_like(img) * std + mean
    return img + noise

def add_salt_pepper_noise(img, prob=0.01):
    mask = torch.rand_like(img)
    img[mask < prob/2] = 0
    img[mask > 1 - prob/2] = 1
    return img
```

### 4.3 学习率调度器

```python
torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
```

### 4.4 优化器

```python
torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
torch.optim.Adam(model.parameters(), lr=0.001)
torch.optim.RMSprop(model.parameters(), lr=0.001)
```

### 4.5 损失函数

```python
nn.CrossEntropyLoss()   # 多分类标配
nn.MSELoss()            # 回归
nn.BCELoss()            # 二分类
```

### 4.6 正则化

```python
nn.Dropout(0.5)                       # 丢弃
weight_decay=1e-4                     # L2 正则（写在优化器里）
nn.BatchNorm2d(16)                    # 批归一化
```

### 4.7 保存和加载模型

```python
torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth"))
```

### 4.8 可视化

```python
import matplotlib.pyplot as plt

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.show()
```

---

## 五、速查总结

1. 数据加载语法大体对，但要补 `DataLoader` 导入、`Resize`、`Normalize`，训练/测试最好用不同 transform。
2. 用 GPU，模型和数据都要 `.to(device)`。
3. CNN 核心层：`Conv2d`、`MaxPool2d`、`ReLU`、`Flatten`、`Linear`、`Dropout`、`BatchNorm2d`；输入形状 `(B, C, H, W)`。
4. 常用方法：数据增强（旋转/裁剪/颜色抖动/模糊/遮挡）、加噪声、学习率调度、优化器、Dropout、BatchNorm。
