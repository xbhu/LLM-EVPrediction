"""
Use Case 2: EV Charging Demand Prediction
Flan-T5-base — Zero-Shot Regression（回归式，不做 fine-tune）

目的：验证 Flan-T5 的 Encoder 在没有 fine-tune 的情况下，
     直接接一个随机初始化的回归头，预测效果有多差。
     这是 fine-tune 之前的 baseline，帮助理解 fine-tune 的必要性。

Author: XB Hu / Smart Mobility Lab, Penn State
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoTokenizer, T5EncoderModel

# ─────────────────────────────────────────────────────────────────────────────
# 0. 配置
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = "/home/xzh5180/Research/llm-evprediction/datasets/dataset2_text_context.csv"
MODEL_NAME = "google/flan-t5-base"
N_EVAL     = 100       # 评估样本数
MAX_LENGTH = 128       # 输入文本最大 token 数
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Use Case 2: Flan-T5-base Zero-Shot Regression")
print("=" * 60)
print(f"  Device : {DEVICE}")
print(f"  Model  : {MODEL_NAME}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. 加载数据
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] 加载数据...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# 取最后 N_EVAL 行作为测试集
test_df = df.tail(N_EVAL).reset_index(drop=True)
print(f"    总数据量 : {len(df)} 行")
print(f"    测试样本 : {N_EVAL} 行")
print(f"    示例输入 : \"{test_df['context_text'].iloc[0][:80]}...\"")

# ─────────────────────────────────────────────────────────────────────────────
# 2. 加载模型和 Tokenizer
#
# 注意：这里用的是 T5EncoderModel，不是完整的 T5。
# 我们只需要 Encoder 部分来提取语义向量，不需要 Decoder。
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[2] 加载 Flan-T5 Encoder...")
print(f"    第一次运行会从 Hugging Face 下载模型（约 500MB）...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder   = T5EncoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
encoder.eval()  # 推理模式，关闭 dropout

print(f"    模型加载完成")
print(f"    Encoder 隐藏层维度: {encoder.config.d_model}")  # Flan-T5-base = 768

# ─────────────────────────────────────────────────────────────────────────────
# 3. 定义回归头
#
# 这是一个随机初始化的线性层：768维向量 → 1个数字
# zero-shot 的意思就是：这个回归头完全没有训练过
# 所以预测结果会非常差——这正是我们想展示的
# ─────────────────────────────────────────────────────────────────────────────
class RegressionHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

regression_head = RegressionHead(encoder.config.d_model).to(DEVICE)
# 注意：这里没有加载任何训练好的权重，W 和 b 是随机数

# ─────────────────────────────────────────────────────────────────────────────
# 4. 推理：文字 → 向量 → 预测数字
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[3] 推理中...")

predictions = []
actuals     = []

with torch.no_grad():  # 不计算梯度，节省显存
    for i, row in test_df.iterrows():
        # Step 1: 文字 → Token
        inputs = tokenizer(
            row["context_text"],
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length"
        ).to(DEVICE)

        # Step 2: Token → 向量序列（每个 token 一个 768 维向量）
        encoder_output = encoder(**inputs)
        hidden_states  = encoder_output.last_hidden_state  # shape: [1, seq_len, 768]

        # Step 3: 取 [CLS] 位置（第一个 token）的向量作为整句摘要
        cls_vector = hidden_states[:, 0, :]  # shape: [1, 768]

        # Step 4: 向量 → 数字（随机回归头）
        pred = regression_head(cls_vector).item()

        predictions.append(pred)
        actuals.append(row["next_day_demand"])

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{N_EVAL} 完成")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 评估
# ─────────────────────────────────────────────────────────────────────────────
predictions = np.array(predictions)
actuals     = np.array(actuals)

mae  = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-6))) * 100

# 基线：用均值预测
mae_mean = mean_absolute_error(actuals, np.full_like(actuals, actuals.mean()))

print("\n" + "=" * 60)
print("  结果")
print("=" * 60)
print(f"  Zero-Shot（随机回归头） → MAE: {mae:.1f} kWh  |  MAPE: {mape:.1f}%")
print(f"  基线（均值预测）        → MAE: {mae_mean:.1f} kWh")
print(f"\n  真实需求均值: {actuals.mean():.1f} kWh")
print(f"\n  结论：Zero-shot 回归头输出是随机数，远差于均值基线。")
print(f"        这证明了回归式模型必须 fine-tune 才能有实际意义。")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 6. 可视化
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Flan-T5-base Zero-Shot（随机回归头）— 预期结果很差", fontsize=12)

# 预测 vs 真实
ax = axes[0]
ax.plot(actuals,     label="真实值",   color="steelblue", lw=1.5)
ax.plot(predictions, label="预测值",   color="darkorange", lw=1, alpha=0.8, linestyle="--")
ax.set_title("预测 vs 真实（时间序列）")
ax.set_xlabel("样本序号")
ax.set_ylabel("需求 (kWh)")
ax.legend()
ax.grid(True, alpha=0.3)

# 散点图
ax = axes[1]
ax.scatter(actuals, predictions, alpha=0.4, s=15, color="steelblue")
lim = [min(actuals.min(), predictions.min()) * 0.9,
       max(actuals.max(), predictions.max()) * 1.1]
ax.plot(lim, lim, "r--", lw=1.5, label="理想预测线")
ax.set_xlabel("真实值 (kWh)")
ax.set_ylabel("预测值 (kWh)")
ax.set_title(f"散点图  (MAE: {mae:.0f} kWh)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = "/home/xzh5180/Research/llm-evprediction/outputs/usecase2_flant5_zeroshot/usecase2_zeroshot_results.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\n  图表已保存: {plot_path}")

print("\n✅ Zero-shot 运行完成")
print("   下一步：对回归头进行 fine-tune，观察 MAE 的变化。")
