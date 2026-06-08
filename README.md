# Meow 金融时序预测分析

[作业详情](https://docs.qq.com/doc/DTnhIcVFtaHhwVXVN?qqInfo=eyJtc2dJZCI6Ijc2Mzc0NDExNDc5ODczMDI1MjYiLCJtc2dUaW1lIjoiMTc3ODIzMDMzOCIsImNoYXRUeXBlIjoyLCJwZWVyVWlkIjoiMTA5MTYyMjgxMyIsInBlZXJOYW1lIjoiMjAyNuaYpS3mqKHlvI%2For4bliKvkuI7mnLrlmajlrabkuaAiLCJlbGVtSWQiOiI3NjM3NDQxMTQ3OTg3MzAyNTI1Iiwic2VuZGVyVWlkIjoidV9ZVWNBc1J4ZzhLNnkzR1h3T05nRVJ3Iiwic2VuZE5pY2tOYW1lIjoiIn0%3D&client=qqclient_online)

[github仓库](https://github.com/Nonender/MachineLearningProject)

## 模型定义

Meow 是一个 **decoder-only (causal) Transformer** 回归器，用于从日内 LOB（限价订单簿）tick 数据预测股票收益。其核心架构是一条**因果 Transformer 解码器链**，将时序特征映射到多 horizon 的前向收益预测。

## 快速开始

1. 配置环境

```bash
# 1.创建虚拟环境
python -m venv .venv
# 2.激活环境
source .venv/bin/activate
# 3.安装依赖
pip install -r requirements.txt
```

2. 运行`meow.py`

```bash
python meow.py
```

## 项目架构

```
meow/
├── meow.py           # MeowEngine — 训练循环, 评估, 推理
├── mdl.py            # MeowModel + SparseAttentionRegressor (模型定义)
├── feat.py           # MeowFeatureGenerator (94 维特征生成)
├── parameters.py     # 超参数集中管理
├── dl.py             # MeowDataLoader (HDF5 数据读取)
├── eval.py           # MeowEvaluator (Pearson, R², MSE)
├── log.py            # 日志工具
├── tradingcalendar.py # 交易日历
└── archive/          # HDF5 原始数据
```

### 架构定义

```
x_raw: (B, T, 94)                              # B 只股票, T 个时间步, 94 个特征
    │
    ├── lin_skip ──────────────────────────────────────┐
    │                                                  │
    ├── feat_gate ⊙ sigmoid(gate)                     |  门控特征选择
    │       ↓                                          │
    ├── feat_proj: Linear(94, 256)                     │  特征投影
    │       ↓                                          │
    ├── + stock_emb[stock_ids]: Embedding(500, 16)     │  股票身份嵌入
    │       ↓                                          │
    ├── RMSNorm(256)                                   │
    │       ↓                                          │
    ├── Dropout(0.1)                                   │
    │       ↓                                          │
    ├── CrossStockAttention(256, 8 heads)              │  跨股票注意力
    │       ↓                                          │
    ├── DecoderBlock × 3 ──────────────────┐           │
    │   │  ├── RMSNorm (Pre-Norm)          │           │
    │   │  ├── CausalSelfAttn (RoPE)       │           │
    │   │  │   └── local window=256        │           │
    │   │  ├── + residual (dropout)        │           │
    │   │  ├── RMSNorm (Pre-Norm)          │           │
    │   │  ├── SwiGLU FFN (256→704)        │           │
    │   │  └── + residual (dropout)        │           │
    │   └──────────────────────────────────┘           │
    │       ↓                                          │
    ├── RMSNorm(256)                                   │
    │       ↓                                          │
    └── head[horizon] * exp(log_scale) ────────────────(+)──→  ŷ_h: (B, T)
         └── Linear(256, 1) per horizon (×4)
```

### 关键组件定义

| 组件 | 定义 | 参数量 |
|---|---|---|
| `feat_gate` | $g_i = \sigma(w_i)$，每个特征的可学习 sigmoid 门 | 94 |
| `feat_proj` | $\text{Linear}(94 \to 256)$，带 bias | ~24K |
| `stock_emb` | $\text{Linear}(500 \to 16)$ → $\text{Linear}(16 \to 256)$，单 ID 嵌入 | ~12K |
| `CrossStockAttention` | 按时间步的 MHA：$(B,T,D) \to (T,B,D) \to \text{MHA} \to (B,T,D)$ | ~0.4M |
| `DecoderBlock` | Pre-Norm + CausalAttn(RoPE) + SwiGLU FFN | ~1.9M × 3 |
| `heads[horizon]` | $\text{Linear}(256 \to 1)$ 每 horizon | 256 × 4 |
| `lin_skips[horizon]` | $\text{Linear}(94 \to 1)$ 每 horizon（线性基线跳跃连接） | 94 × 4 |
| `log_scale` | 可学习全局尺度 $s = e^w$，初始 $w=0$ | 1 |
| **总计** | | **~6.2M** |


