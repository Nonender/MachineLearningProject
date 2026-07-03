# CLI 参考

## 训练与评估

```bash
# 基础训练 + 评估
python meow.py --data-dir archive/ --epochs 10 --plot

# 自定义超参数
python meow.py --epochs 12 --lr 1e-4 --batch-size 128

# 仅评估已有模型
python meow.py --eval-only --resume checkpoints/checkpoint_best.pt --plot

# 断点续训
python meow.py --resume checkpoints/checkpoint_latest.pt

# 禁用早停
python meow.py --no-early-stopping --epochs 20
```

## 完整参数

```
python meow.py [OPTIONS]

数据:
  --data-dir DIR           数据目录 (default: archive/)
  --checkpoint-dir DIR     Checkpoint 目录 (default: checkpoints/)

训练:
  --train-start DATE       训练开始 (default: 20230601)
  --train-end DATE         训练结束 (default: 20231130)
  --epochs N               训练轮次 (default: 8)
  --lr LR                  学习率 (default: 2e-4)
  --batch-size N           批次大小 (default: 256)
  --preprocessing-fit-days N  预处理拟合天数 (default: 20)
  --no-early-stopping      禁用早停
  --patience N             早停耐心 (default: 3)
  --resume PATH            恢复训练
  --seed N                 随机种子 (default: 42)

评估:
  --eval-start DATE        评估开始 (default: 20231201)
  --eval-end DATE          评估结束 (default: 20231229)
  --eval-only              仅评估
  --plot                   生成评估图表
  --plot-dir DIR           图表目录 (default: plots/)

验证:
  --val-start DATE         验证开始
  --val-window N           验证天数 (default: 5)
```

## 分析工具

```bash
python ablation.py                       # 消融实验
python ablation.py --epochs 8            # 自定义轮次
python ablation.py --list-groups         # 查看特征分组

python data_efficiency.py                # 数据效率分析
python data_efficiency.py --fraction 0.5 # 单比例测试

python sweep.py                          # 超参数搜索
python sweep.py --single --epochs 12 --lr 1e-4

python tests/test_eval.py                # 运行测试
python tests/test_feat.py
python tests/test_mdl.py
```
