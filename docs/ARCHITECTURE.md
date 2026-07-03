# 模型架构与特征工程

## 模型架构

Meow 是一个 **Decoder-only Causal Transformer** 回归器，~3.5M 参数，从零基于 PyTorch 实现。

```
x_raw: (Batch, Time=256, Features=94)
    │
    ├── lin_skip ──────────────────────────────────────────┐
    │                                                      │
    ├── feat_gate ⊙ sigmoid(gate)                          │  门控特征选择
    │       ↓                                              │
    ├── feat_proj: Linear(94, 256)                         │  特征投影
    │       ↓                                              │
    ├── + stock_emb[stock_ids]                             │  股票身份嵌入
    │       ↓                                              │
    ├── RMSNorm(256) → Dropout                             │
    │       ↓                                              │
    ├── CrossStockAttention(256, 8 heads)                  │  跨股票注意力
    │       ↓                                              │
    ├── DecoderBlock × 4 ──────────────────────┐           │
    │   │  ├── RMSNorm (Pre-Norm)              │           │
    │   │  ├── CausalSelfAttn + RoPE           │           │
    │   │  │   └── local window = 256          │           │
    │   │  ├── + residual                      │           │
    │   │  ├── RMSNorm (Pre-Norm)              │           │
    │   │  ├── SwiGLU FFN (256 → 704)          │           │
    │   │  └── + residual                      │           │
    │   └──────────────────────────────────────┘           │
    │       ↓                                              │
    ├── RMSNorm(256)                                       │
    │       ↓                                              │
    └── head[horizon] + lin_skip ────────────────────────(+)──→  ŷ_h
         Linear(256,1) + Linear(94,1) per horizon (×4)
```

### 组件参数量

| 组件 | 说明 | 参数量 |
|---|---|---|
| `feat_gate` | 可学习 sigmoid 门控，L1 正则化软特征选择 | 94 |
| `feat_proj` | Linear(94 → 256) | ~24K |
| `stock_emb` | Linear(500 → 16) → Linear(16 → 256) | ~12K |
| `CrossStockAttention` | 跨股票多头注意力 | ~263K |
| `DecoderBlock` ×4 | Pre-Norm + CausalAttn(RoPE) + SwiGLU FFN | ~806K × 4 |
| `heads[horizon]` ×4 | Linear(256 → 1) | ~1K |
| `lin_skips[horizon]` ×4 | Linear(94 → 1) 跳跃连接 | ~380 |
| `log_scale` + norms | 全局缩放 + RMSNorm | ~2K |
| **总计** | | **~3.5M** |

### 关键设计

**Cross-Stock Attention**：将张量从 $(B,T,D)$ 转置为 $(T,B,D)$ 后做 MHA，在每个时间步对**不同股票之间**做注意力，捕捉截面联动。

**RoPE 旋转位置编码**：使注意力分数依赖相对位置，支持序列外推。

**Feature Gate**：每个特征对应可学习权重 $w_i$，前向乘 $\sigma(w_i)$，L1 正则化鼓励稀疏。

**混合损失函数**：$\mathcal{L} = 3.0 \cdot \text{MSE} / \text{Var}(y) + 0.2 \cdot (-\text{Pearson}) + 5\times10^{-4} \cdot \|\sigma(\mathbf{w})\|_1$

### 训练策略

| 配置 | 值 | 说明 |
|---|---|---|
| 优化器 | AdamW (lr=2e-4, wd=0.5) | 强正则化 |
| 调度 | Warmup (10%) + Cosine Annealing | 退火至 1e-5 |
| 混合精度 | AMP (FP16) | 显存减半 |
| 梯度裁剪 | max_norm=1.0 | 防梯度爆炸 |
| 早停 | patience=3 | 验证集 Pearson r 监控 |
| Dropout | 0.2 | |

预处理管线：Winsorize (P1/P99) → log1p (高偏度特征) → Z-score → 标签 vol20 归一化消除异方差。

---

## 特征工程

基于市场微观结构理论，从 LOB 数据构建 **94 维因子**：

| 类别 | 特征数 | 核心指标 |
|---|---|---|
| 价格与波动率 | 11 | 多周期收益率、波动率、波动率比率 |
| 价差与深度 | 6 | 买卖价差、对数深度失衡、深度集中度 |
| 订单簿失衡 | 7 | 多档位买卖失衡、订单簿斜率/曲率、EMA 平滑 |
| 交易流 | 8 | 主动买卖失衡、成交量、VWAP 偏离 |
| 截面因子 | 8 | 去均值 (`cx_*`) 和百分位排名 (`rank_*`) |
| 动量/反转 | 5 | 滞后收益、动量乘数 |
| 市场质量 | 2 | 波动率调整价差、价格冲击 |
| 日内累计 | 4 | 累计收益、累计成交量、累计失衡 |
| 时间特征 | 2 | sin/cos 分钟级周期编码 |
| 非线性交互 | ~30 | 多项式、交叉积、tanh/sigmoid/log1p、Z-score、时序排名 |
| 微观结构 | 3 | bid-ask bounce、自相关、偏度 |
