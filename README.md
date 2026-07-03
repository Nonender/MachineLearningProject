# Meow — 基于 Decoder-only Transformer 的金融时序预测

从高频 LOB（限价订单簿）数据预测股票短期收益率。模型为从零实现的 **Decoder-only Causal Transformer**（~3.5M 参数），包含 Cross-Stock Attention、RoPE、SwiGLU FFN 等组件。基于市场微观结构理论构建了 94 维特征，在 ~300 只 A 股分钟级数据上达到 **Pearson r = 0.0802**（Ridge 基线的 ~3 倍）。

## 快速开始

```bash
# 1. 环境
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 将 HDF5 数据放入 archive/ 目录

# 3. 训练 + 评估
python meow.py --data-dir archive/ --epochs 10 --plot
```

## 主要结果

| 指标 | Meow | Ridge 基线 |
|---|---|---|
| **Test Pearson r** | **0.0802** | ~0.02–0.03 |
| **Val Pearson r (best)** | **0.0966** | — |
| 特征数 | 94 | — |
| 参数量 | ~3.5M | — |

## 项目结构

```
meow/
├── meow.py               # 训练入口 + CLI
├── mdl.py                # 模型定义
├── feat.py               # 94 维特征工程
├── parameters.py         # 超参数管理
├── ablation.py           # 消融实验
├── data_efficiency.py    # 数据效率分析
├── sweep.py              # 超参数搜索
├── viz.py                # 可视化
├── tests/                # 15 个测试
├── docs/                 # 详细文档
└── tutorial/             # 教学材料
```

## 详细文档

| 文档 | 内容 |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 模型架构、特征工程、训练策略 |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | 实验结果、消融分析、数据效率 |
| [docs/CLI.md](docs/CLI.md) | CLI 完整参数与使用示例 |

## 常用命令

```bash
python meow.py --epochs 10 --plot              # 训练 + 评估 + 出图
python meow.py --eval-only --resume checkpoints/checkpoint_best.pt  # 仅评估
python ablation.py --epochs 8                   # 消融实验
python data_efficiency.py --epochs 8            # 数据效率分析
python sweep.py                                 # 超参数搜索
python tests/test_mdl.py                        # 测试
```
