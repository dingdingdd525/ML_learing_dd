import torch
import torch.nn as nn

# 设置随机种子，保证每次运行结果一致
torch.manual_seed(42)

def run_experiment(batch_size=2, seq_len=4, embed_dim=8, num_heads=2, num_layers=2):
    print("=" * 60)
    print(f"实验参数：batch_size={batch_size}, seq_len={seq_len}, "
          f"embed_dim={embed_dim}, num_heads={num_heads}, num_layers={num_layers}")

    # 1. 构造输入 [batch_size, seq_len, embed_dim]
    x = torch.randn(batch_size, seq_len, embed_dim)

    # 2. 创建一个 TransformerEncoderLayer
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        batch_first=True,
        dim_feedforward=4 * embed_dim
    )

    # 3. 使用 TransformerEncoder 搭建 Encoder
    encoder = nn.TransformerEncoder(
        encoder_layer,
        num_layers=num_layers
    )

    # 4. 前向传播：把 x 送入 encoder
    out = encoder(x)

    # 5. 打印输入和输出的 shape
    print("Input shape :", x.shape)
    print("Output shape:", out.shape)
    print("输入输出 shape 是否相同:", x.shape == out.shape)

    # 打印模型结构和参数量
    print("\n模型结构：")
    print(encoder)
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"总参数量: {total_params}")
    print("=" * 60)

    return out


if __name__ == "__main__":
    # 原始参数
    run_experiment()

    # 修改 num_heads：2 -> 4
    run_experiment(num_heads=4)

    # 修改 num_layers：2 -> 4
    run_experiment(num_layers=4)