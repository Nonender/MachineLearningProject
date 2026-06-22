# Meow：基于 Decoder-only Transformer 的金融时序预测

[GitHub](https://github.com/Nonender/MachineLearningProject)

---

## 一、任务描述

从高频限价订单簿（LOB）数据预测 A 股未来 12 分钟收益率。数据覆盖约 **300 只股票 × 7 个月 × 分钟级 tick**（2023.06–2023.12），训练集 6–11 月，测试集 12 月。

$$
\text{fret12} = \frac{P_{t+12} - P_t}{P_t}
$$

核心挑战：(1) 金融数据信噪比极低，可预测信号占比远低于 CV/NLP 任务；(2) 收益率分布非正态、存在异方差性；(3) 不同股票之间存在截面关联。

---

## 二、模型架构

模型为从零实现的 **Decoder-only Causal Transformer**，包含 RoPE 旋转位置编码、SwiGLU FFN、RMSNorm Pre-Norm、Causal Self-Attention 等组件。

```
x_raw: (B, T, 94)
    │
    ├── feat_gate ⊙ sigmoid(gate)          ← 可学习特征门控 (94 params)
    ├── feat_proj: Linear(94 → 256)
    ├── + stock_emb[stock_ids]             ← 股票身份嵌入
    ├── RMSNorm → Dropout
    ├── CrossStockAttention(256, 8 heads)   ← 跨股票注意力
    ├── DecoderBlock ×4 ──────────────────┐
    │   ├── RMSNorm (Pre-Norm)            │
    │   ├── CausalSelfAttn + RoPE         │
    │   │   └── local window = 256        │
    │   ├── + residual                    │
    │   ├── RMSNorm (Pre-Norm)            │
    │   ├── SwiGLU FFN (256 → 704)        │
    │   └── + residual                    │
    │   └─────────────────────────────────┘
    ├── RMSNorm
    └── head[horizon] + lin_skip ────────→ ŷ_h: (B, T)
         Linear(256,1) + Linear(94,1) per horizon (×4)
```

| 组件 | 参数量 |
|---|---|
| Feature Gate + Projection + Stock Embedding | ~36K |
| CrossStockAttention | ~0.4M |
| DecoderBlock ×4 | ~7.6M |
| 4 Horizon Heads + Skip Connections + Scale | ~1.4K |
| **总计** | **~6.2M** |

---

## 三、技术亮点

### 1. Cross-Stock Attention（跨股票注意力）

标准 Self-Attention 仅在单只股票的时间维度上操作，无法捕捉同行业、同因子暴露的股票间联动。通过将张量从 $(B,T,D)$ 转置为 $(T,B,D)$ 后做 Multi-Head Attention，模型能在每个时间步学习"哪些股票在同一方向运动"的截面关系。

### 2. Feature Gate（可学习特征门控）

94 维特征中天然存在冗余。每个特征 $i$ 对应可学习参数 $w_i$，前向时乘以 sigmoid 门控 $\sigma(w_i)$。训练时施加 L1 稀疏正则化 $5\times10^{-4} \cdot \sum_i \sigma(w_i)$，训练结束后可通过门控值观察模型学到了哪些关键因子。

### 3. 混合损失函数

$$\mathcal{L} = 3.0 \cdot \frac{\text{MSE}(\hat{y}, y)}{\text{Var}(y)} + 0.2 \cdot (-\text{Pearson}(\hat{y}, y)) + 5\times10^{-4} \cdot \|\sigma(\mathbf{w})\|_1$$

纯 MSE 会鼓励模型在不确定时预测均值（导致预测方差过小而相关性差），Correlation Loss 鼓励模型产出有区分度的排序预测——在金融中，排序正确比数值精确更重要。

---

## 四、特征工程

基于市场微观结构理论，手写 **94 维因子**：

| 类别 | 特征数 | 核心指标 |
|---|---|---|
| 价格与波动率 | 11 | 多周期收益率、波动率、波动率之比率 |
| 价差与深度 | 6 | 买卖价差、对数深度失衡、深度集中度 |
| 订单簿 | 7 | 多档位买卖失衡、订单簿斜率/曲率、EMA 平滑 |
| 交易流 | 8 | 主动买卖失衡、成交量、VWAP 偏离 |
| 截面 | 8 | 去均值 (`cx_*`) 和百分位排名 (`rank_*`) |
| 动量/反转 | 5 | 滞后收益、动量乘数、收益比率 |
| 非线性交互 | ~35 | 多项式、交叉积、三元组、tanh/sigmoid/log1p 变换、Rolling Z-score、时序排名、不对称分解 |
| 其他 | 14 | 时间编码、日内累计、市场质量、微观结构 |

预处理管线：Winsorize (P1/P99) → log1p (高偏度特征) → Z-score → 标签 vol20 归一化（消除异方差）。

---

## 五、实验结果

| 配置 | 值 |
|---|---|
| 训练集 | 2023.06.01 – 2023.11.30（~123 个交易日） |
| 测试集 | 2023.12.01 – 2023.12.29（~20 个交易日） |
| 训练轮次 | 8 epochs |
| 批次大小 | 128 stocks/batch |
| 优化器 | AdamW (lr=2e-4, wd=0.5) + Warmup + Cosine Annealing |
| 混合精度 | AMP (FP16) + 梯度裁剪 (max_norm=1.0) |
| 设备 | NVIDIA RTX 4060 |

**测试集指标：**

| 指标 | Meow | Ridge 线性基线 | 提升 |
|---|---|---|---|
| **Pearson r** | **0.0826** | ~0.02–0.03 | **~3×** |
| **R²** | **0.00527** | < 0 | 模型优于均值预测 |
| **MSE** | 2.24 × 10⁻⁵ | — | — |

> 金融时序预测中 Pearson r 天然低（随机噪声主导）。学术界共识：分钟级 r > 5% 即具有统计显著性。8.26% 的 r 和正的 R² 表明模型捕捉到了显著的预测信号，且非线性 Transformer + 深度特征工程带来的 3 倍提升是明确的。

---

## 六、预测效果可视化

<img src="runs/20260610_061014_bs_128/plots/pred_scatter.png" 
     alt="预测值 vs 真实值散点图" 
     style="width:400px; height:auto; display:block; margin:10px 0;">

> 六边形密度散点图。每个六边形代表一个密度区域（颜色越深样本越多）。红色虚线为 $y=x$（完美预测）。右上角标注 Pearson r = 0.0826，N ≈ 217 万样本。散点沿对角线分布的趋势表明模型具备稳定的正向预测能力。

---

## 七、工程实践

- **模块化设计**：模型 / 特征 / 数据加载 / 评估 / 可视化完全解耦，每个模块可独立测试与复用
- **自动化测试**：15 个测试覆盖模型前向传播、特征生成 pipeline、checkpoint 保存/加载轮转一致性、评估器数值正确性
- **流式预处理**：采用两遍流式算法（采样百分位数估计 + Welford 在线均值和方差），将预处理峰值内存从 ~5GB 降至 ~100MB
- **完备 CLI**：13 个命令行参数支持数据路径、训练配置、评估模式、可视化输出、断点续训、仅评估模式
- **超参数搜索**：`sweep.py` 支持多组配置自动串行训练，每组自动记录指标、checkpoint 和图表
