# Meow — 基于 Decoder-only Transformer 的金融时序预测

Meow 是一个 **decoder-only  Transformer** 回归模型，用于从日内高频 LOB（限价订单簿）tick 数据预测股票收益率。模型从零基于 PyTorch 实现，包含 RoPE 旋转位置编码、SwiGLU 前馈网络、RMSNorm 归一化、Causal Self-Attention 等组件，并创新性地引入了跨股票注意力机制（Cross-Stock Attention）和可学习特征门控（Feature Gate）。

## 快速开始

```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 将 HDF5 数据文件放入 archive/ 目录

# 4. 训练 + 评估（默认 4 epochs）
python meow.py --data-dir archive/ --epochs 8 --plot
```

常用命令：

```bash
python meow.py --help                 # 查看所有 CLI 参数
python meow.py --epochs 12 --plot     # 自定义训练轮次并生成图表
python meow.py --eval-only --resume checkpoints/checkpoint_best.pt --plot  # 仅评估

python sweep.py                       # 超参数搜索（多组自动串行）
python sweep.py --single --epochs 10 --lr 3e-4 --batch-size 128 --name my_run

python tests/test_eval.py             # 运行测试
python tests/test_feat.py
python tests/test_mdl.py
```

## 项目结构

```
meow/
├── meow.py              # MeowEngine — 训练循环、评估、推理 + argparse CLI
├── mdl.py               # 模型定义（SparseAttentionRegressor + MeowModel）
├── feat.py              # MeowFeatureGenerator — 94 维特征工程
├── parameters.py        # 超参数集中管理（dataclass）
├── dl.py                # MeowDataLoader — HDF5 数据读取
├── eval.py              # MeowEvaluator — Pearson / R² / MSE 评估
├── viz.py               # 可视化工具（散点图、残差、累计收益、特征重要性）
├── sweep.py             # 超参数搜索（多组串行，自动记录结果）
├── log.py               # 日志工具
├── tradingcalendar.py   # 交易日历
├── resources/calendar   # 交易日列表
├── tests/
│   ├── test_mdl.py      # 模型前向传播 + checkpoint 测试
│   ├── test_feat.py     # 特征生成 pipeline 测试
│   └── test_eval.py     # 评估器正确性测试
└── archive/             # HDF5 原始数据（需自行下载）
```

## 模型架构

### 数据流

```
x_raw: (Batch, Time=256, Features=94)              # B 只股票, 最多 256 个时间步, 94 个特征
    │
    ├── lin_skip ──────────────────────────────────────────┐
    │                                                      │
    ├── feat_gate ⊙ sigmoid(gate)                          │  门控特征选择 (94 个可学习参数)
    │       ↓                                              │
    ├── feat_proj: Linear(94, 256)                         │  特征投影到模型维度
    │       ↓                                              │
    ├── + stock_emb[stock_ids]: Linear(500→16)→Linear(16→256) │  股票身份嵌入
    │       ↓                                              │
    ├── RMSNorm(256)                                       │
    │       ↓                                              │
    ├── Dropout                                            │
    │       ↓                                              │
    ├── CrossStockAttention(256, 8 heads)                  │  跨股票注意力
    │       ↓                                              │
    ├── DecoderBlock × 4 ──────────────────────┐           │
    │   │  ├── RMSNorm (Pre-Norm)              │           │
    │   │  ├── CausalSelfAttn + RoPE           │           │
    │   │  │   └── local window = 256          │           │
    │   │  ├── + residual (dropout)            │           │
    │   │  ├── RMSNorm (Pre-Norm)              │           │
    │   │  ├── SwiGLU FFN (256 → 704)          │           │
    │   │  └── + residual (dropout)            │           │
    │   └──────────────────────────────────────┘           │
    │       ↓                                              │
    ├── RMSNorm(256)                                       │
    │       ↓                                              │
    └── head[horizon] * exp(log_scale) ────────────────────(+)──→  ŷ_h: (B, Time)
         └── Linear(256, 1) per horizon (×4 个预测目标)
         └── lin_skip: Linear(94, 1) 线性跳跃连接 (×4)
```

### 组件与参数量

| 组件 | 说明 | 参数量 |
|---|---|---|
| `feat_gate` | 每个特征的可学习 sigmoid 门控 $g_i = \sigma(w_i)$，L1 正则化实现软特征选择 | 94 |
| `feat_proj` | Linear(94 → 256)，将原始特征投影到模型维度 | ~24K |
| `stock_emb` | Linear(500 → 16) → Linear(16 → 256)，将股票 ID 映射为嵌入向量 | ~12K |
| `CrossStockAttention` | 按时间步转置后做多头注意力：$(B,T,D) \to (T,B,D) \to MHA \to (B,T,D)$，实现跨股票信息交互 | ~0.4M |
| `DecoderBlock` ×4 | Pre-Norm + Causal Self-Attention (RoPE, local window=256) + SwiGLU FFN (256→704) | ~1.9M × 4 |
| `heads[horizon]` ×4 | Linear(256 → 1)，每个预测时间尺度独立输出头 | 256 × 4 |
| `lin_skips[horizon]` ×4 | Linear(94 → 1)，线性跳跃连接，为每个 horizon 提供线性基线 | 94 × 4 |
| `log_scale` | 可学习全局输出缩放 $s = e^w$，初始 $w=0$ | 1 |
| **总计** | | **~6.2M** |

### 关键设计选择

**Cross-Stock Attention（跨股票注意力）**。标准的 Causal Self-Attention 在每只股票自己的时间序列上做注意力，无法捕捉同行业、同因子暴露的股票之间的联动。Cross-Stock Attention 通过将张量从 $(B,T,D)$ 转置为 $(T,B,D)$，在每个时间步对**不同股票之间**做多头注意力——模型可以学到"当前时刻哪些股票在同一方向运动"。

**Feature Gate（特征门控）**。94 维特征中不可避免地存在冗余。每个特征 $i$ 对应一个可学习参数 $w_i$，前向时乘以 $\sigma(w_i)$，并通过 L1 正则化 $5\times10^{-4} \cdot \sum \sigma(w_i)$ 鼓励稀疏性。训练结束后可观察 gate 值了解哪些特征被模型认为更重要。

**RoPE 旋转位置编码**。金融时序中"5 分钟前的价格变化"对应的是相对位置关系，而非绝对位置索引。RoPE 将位置信息编码到 Q/K 向量的旋转中，使注意力分数天然依赖相对位置，且支持外推到比训练时更长的序列。

**多 Horizon 联合预测**。同时预测 1/6/12/24 分钟后的收益率，共用 Transformer 主干，仅输出头独立。多任务学习有助于学习更鲁棒的时序表示。

## 特征工程

基于市场微观结构理论，从原始 LOB 数据构建 **94 维因子特征**：

| 类别 | 特征数 | 示例 |
|---|---|---|
| 价格与波动率 | 11 | `ret1`, `ret5`, `ret10`, `ret30`, `vol5`, `vol10`, `vol20`, `ret1_vol20`, `vol_ratio_5_20` |
| 价差与深度 | 6 | `spread`, `spread4`, `depth_imb`, `depth_conc`, `depth_total` |
| 订单簿失衡 | 7 | `ob_imb0`, `ob_imb4`, `ob_imb9`, `ob_slope`, `ob_curvature`, EMA 平滑版 |
| 交易流 | 8 | `trade_imb`, `trade_count_imb`, `log_trade_volume`, `vwap_dev`, `volume_intensity` |
| 截面因子 | 8 | 去均值特征 (`cx_*`)、百分位排名 (`rank_*`)，分离个股信号与市场信号 |
| 动量/反转 | 5 | `lagret12`, `mom_5_30`, `ret5_ret1`, `ret30_ret5` |
| 市场质量 | 2 | `spread_scaled`, `price_impact` |
| 日内累计 | 4 | `cum_ret1`, `cum_volume`, `cum_imb`, `cum_ob_imb0` |
| 时间特征 | 2 | `sin_time`, `cos_time`（分钟级周期编码） |
| 非线性交互 | ~30 | 多项式项、特征交叉乘积、三元组交互、tanh/sigmoid/log1p 变换、滚动 Z-score、滚动时序排名、不对称分解、波动率之波动率等 |
| 微观结构 | 3 | `bid_ask_bounce`, `ret_autocorr`, `ret_skew_20` |

## 训练策略

| 配置项 | 值 | 说明 |
|---|---|---|
| 优化器 | AdamW | 解耦权重衰减，适合 Transformer |
| 学习率 | 2×10⁻⁴ | 初始学习率 |
| 权重衰减 | 0.5 | 强正则化，防止过拟合 |
| 批次大小 | 128–256 | 每批股票数 |
| 学习率调度 | Warmup (10%) + Cosine Annealing | 线性预热后余弦退火至 10⁻⁵ |
| 梯度裁剪 | max_norm=1.0 | 防止极端行情样本导致梯度爆炸 |
| 混合精度 | AMP (FP16) | 减少约一半显存占用 |
| 早停 | patience=3 | 验证集 Pearson r 不再提升时停止 |
| 随机种子 | 42 | 可复现性 |

### 损失函数

$$\mathcal{L} = 3.0 \cdot \mathcal{L}_{MSE}(\hat{y}, y) / \text{Var}(y) + 0.2 \cdot \mathcal{L}_{corr}(\hat{y}, y) + 5\times10^{-4} \cdot \sum_i \sigma(w_i)$$

- **MSE**：预测误差的主损失，按标签方差归一化消除量纲影响
- **Correlation Loss**：负 Pearson 相关系数，鼓励模型做有区分度的排序预测
- **Gate L1**：特征门控的稀疏正则化

### 预处理管线

```
原始特征 → Winsorize (截尾 P1/P99) → log1p (高偏度特征) → Z-score 标准化 → input_scale 缩放
标签     → 除以 vol20（波动率归一化，消除异方差）→ 训练 → 推理时乘回 vol20
```

## 实验结果

### 数据与设置

| 配置项 | 说明 |
|---|---|
| 股票数量 | ~300 只 A 股 |
| 时间范围 | 2023 年 6 月 – 2023 年 12 月 |
| 数据频率 | 分钟级 tick |
| 训练集 | 2023.06.01 – 2023.11.30（~123 个交易日） |
| 测试集 | 2023.12.01 – 2023.12.29（~20 个交易日） |
| 预测目标 | 12 分钟后价格变化率 $\text{fret12} = (P_{12} - P_0) / P_0$ |
| 训练轮次 | 8 epochs |
| 设备 | NVIDIA GeForce RTX 4060 (CUDA) |

### 测试集指标

| 指标 | Meow | Ridge 线性基线 | 提升 |
|---|---|---|---|
| **Pearson r** | **0.0826** | ~0.02–0.03 | **~3×** |
| **R²** | **0.00527** | < 0 | 模型优于均值预测 |
| **MSE** | 2.24 × 10⁻⁵ | — | — |

> **关于指标量级**：金融时序预测中 Pearson r 天然很低——股票价格变动由大量独立随机因素驱动，可预测信号占比极低。学术界共识：分钟级预测 r > 5% 即具有统计显著性和经济价值。8.26% 的相关系数意味着模型捕捉到了显著且稳定的预测信号；R² > 0 表明模型优于零预测基线（简单均值预测）。Ridge 线性模型只能达到 2–3%，Nonlinear Transformer + 深度特征工程的增益是明确的。

### 可视化输出

运行 `python meow.py --plot` 后在 `plots/` 目录生成四张图表：

| 图表 | 内容 |
|---|---|
| `pred_scatter.png` | 预测值 vs 真实值六边形密度散点图，标注 Pearson r 和样本量 |
| `residual_dist.png` | 残差（预测 − 真实）直方图，标注均值与标准差 |
| `cumret_quantile.png` | 按预测值分 5 组的累计收益率曲线——顶部组（买）应显著高于底部组（卖） |
| `gate_importance.png` | Feature Gate 学到的 Top 20 最重要特征及其门控值 |

## CLI 参数

```
python meow.py [OPTIONS]

数据:
  --data-dir DIR           HDF5 数据目录 (default: archive/)
  --cache-dir DIR          预处理缓存目录 (default: None)
  --checkpoint-dir DIR     Checkpoint 保存目录 (default: checkpoints/)

训练:
  --train-start DATE       训练开始日期 YYYYMMDD (default: 20230601)
  --train-end DATE         训练结束日期 (default: 20231130)
  --epochs N               训练轮次 (default: 4)
  --lr LR                  学习率 (default: 2e-4)
  --batch-size N           批次大小 (default: 256)
  --preprocessing-fit-days N  预处理拟合天数 (default: 20)
  --no-early-stopping      禁用早停
  --patience N             早停耐心轮次 (default: 3)
  --resume PATH            从 checkpoint 恢复训练
  --seed N                 随机种子 (default: 42)

评估:
  --eval-start DATE        评估开始日期 (default: 20231201)
  --eval-end DATE          评估结束日期 (default: 20231229)
  --eval-only              跳过训练，仅评估
  --plot                   生成评估图表
  --plot-dir DIR           图表保存目录 (default: plots/)

验证:
  --val-start DATE         验证开始日期 (default: train-end 后一天)
  --val-window N           验证天数 (default: 5)
```

## 依赖

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy, Pandas, SciPy, scikit-learn
- PyTables (HDF5 读取)
- Matplotlib (可选，用于 `--plot`)

完整列表见 [requirements.txt](requirements.txt)。

## 测试

```bash
python tests/test_eval.py    # 评估器正确性（5 tests）
python tests/test_feat.py    # 特征生成 pipeline（5 tests）
python tests/test_mdl.py     # 模型前向 + checkpoint（5 tests）
```