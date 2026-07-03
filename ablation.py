#!/usr/bin/env python3
"""Feature-category ablation study for Meow.

Systematically removes one feature category at a time and measures the
impact on prediction performance. This answers the question: "Which data
source contributes the most value to the model?"

Usage:
    python ablation.py                          # run all ablation groups
    python ablation.py --group order_book       # run a single group
    python ablation.py --epochs 4 --help        # see all options
"""

import argparse
import datetime
import gc
import json
import os
import sys
import numpy as np

from log import log
from parameters import MODEL_CONFIG, TRAINING_CONFIG, PREPROCESSING_CONFIG

# ── Feature category definitions ───────────────────────────────────────
# Each category lists the feature NAMES that belong to it.
# When a category is ablated, these features are removed from the input
# BEFORE the model sees them.  The remaining 94 - len(category) features
# are kept.
#
# Features NOT listed in any category form the "base" set and are always
# included (raw returns, time encoding, cumulative stats, etc.).

ABLATION_GROUPS = {
    "order_book": {
        "label": "核心订单簿微观结构",
        "features": [
            "spread", "spread4", "spread_ema5",
            "depth_imb", "depth_conc", "depth_total",
            "ob_imb0", "ob_imb4", "ob_imb9",
            "ob_slope", "ob_curvature",
            "ob_imb0_ema10", "ob_imb0_ema30",
            "spread_scaled",
            "ob_imb_change", "spread_change", "depth_skew",
        ],
    },
    "volatility": {
        "label": "波动率族",
        "features": [
            "vol5", "vol10", "vol20",
            "ret1_vol20", "vol_ratio_5_20",
            "ret1_vol10", "spread_vol10",
            "vol_of_vol", "vol_ratio_10_20",
        ],
    },
}

# Shared training config for all ablation runs (keep short for speed)
ABLATION_SHARED = {
    "data_dir": "archive/",
    "train_start": 20230601,
    "train_end": 20231130,
    "eval_start": 20231201,
    "eval_end": 20231229,
    "val_window": 5,
    "patience": 3,
    "preprocessing_fit_days": 20,
    "seed": 42,
    "epochs": 8,
    "lr": 2e-4,
    "batch_size": 128,
}


# ── Helpers ────────────────────────────────────────────────────────────

def get_active_features(group_name):
    """Return the list of feature names to KEEP when ablating a group."""
    from feat import MeowFeatureGenerator
    all_features = MeowFeatureGenerator.feature_names()

    if group_name is None:
        return list(all_features)

    removed = set(ABLATION_GROUPS[group_name]["features"])
    return [f for f in all_features if f not in removed]


def print_feature_summary():
    """Print a table showing how many features are in each category."""
    from feat import MeowFeatureGenerator
    all_features = set(MeowFeatureGenerator.feature_names())

    accounted = set()
    n_overlap = 0
    lines = ["", "Feature category breakdown:"]
    lines.append(f"  {'Group':<25s} {'Count':>5s}  Description")
    lines.append(f"  {'-'*25} {'-'*5}  {'-'*40}")
    for name, info in ABLATION_GROUPS.items():
        fset = set(info["features"])
        n_overlap += len(fset & accounted)
        accounted |= fset
        lines.append(f"  {name:<25s} {len(fset):>5d}  {info['label']}")
    base = all_features - accounted
    lines.append(f"  {'base (always included)':<25s} {len(base):>5d}  基础价格/时间/累计特征")
    if n_overlap > 0:
        lines.append(f"  {'(overlap removed)':<25s} {n_overlap:>5d}")
    lines.append(f"  {'TOTAL':<25s} {len(all_features):>5d}")
    log.inf("\n".join(lines))


# ── Core ablation logic ─────────────────────────────────────────────────

def run_ablation_group(group_name, shared, output_dir="ablation_results"):
    """Train and evaluate the model WITHOUT the specified feature group.

    Returns a dict with metrics, or None on failure.
    """
    group_label = (ABLATION_GROUPS[group_name]["label"]
                   if group_name else "完整模型（基线）")
    active_features = get_active_features(group_name)
    n_active = len(active_features)

    # ── Setup output paths ──
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "baseline" if group_name is None else f"no_{group_name}"
    run_dir = os.path.join(output_dir, f"{ts}_{run_name}")
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "output.log")

    orig_stdout = sys.stdout
    sys.stdout = open(log_file, "w", buffering=1)
    sys.stderr = sys.stdout

    log.inf("=" * 60)
    log.inf(f"Ablation: {group_label}")
    log.inf(f"  Removed category: {group_name}")
    log.inf(f"  Active features:  {n_active} / 94")
    log.inf(f"  Started: {datetime.datetime.now()}")
    log.inf("=" * 60)

    metrics = {
        "group": group_name or "baseline",
        "label": group_label,
        "n_features": n_active,
        "run_dir": run_dir,
        "status": "error",
    }

    try:
        from feat import MeowFeatureGenerator
        from meow import MeowEngine

        # ── Configure model for reduced feature set ──
        orig_n_features = MODEL_CONFIG.n_features
        orig_epochs = TRAINING_CONFIG.n_epochs
        orig_lr = TRAINING_CONFIG.lr
        orig_bs = TRAINING_CONFIG.batch_size
        orig_fit_days = PREPROCESSING_CONFIG.preprocessing_fit_days

        MODEL_CONFIG.n_features = n_active
        TRAINING_CONFIG.n_epochs = shared["epochs"]
        TRAINING_CONFIG.lr = shared["lr"]
        TRAINING_CONFIG.batch_size = shared["batch_size"]
        PREPROCESSING_CONFIG.preprocessing_fit_days = shared["preprocessing_fit_days"]

        # ── Patch feature generator to output only active features ──
        original_gen_features = MeowFeatureGenerator.gen_features

        def filtered_gen_features(self, df):
            xdf, ydf = original_gen_features(self, df)
            return xdf[active_features], ydf

        MeowFeatureGenerator.gen_features = filtered_gen_features

        # Monkey-patch feature_names to match
        original_feature_names = MeowFeatureGenerator.feature_names

        @classmethod
        def filtered_feature_names(cls):
            return active_features

        MeowFeatureGenerator.feature_names = filtered_feature_names

        # ── Train ──
        engine = MeowEngine(
            h5dir=shared["data_dir"],
            cache_dir=None,
            checkpoint_dir=os.path.join(run_dir, "checkpoints"))

        engine.fit(
            start_date=shared["train_start"],
            end_date=shared["train_end"],
            val_date=engine.calendar.next(shared["train_end"])
                      or shared["train_end"],
            val_window=shared["val_window"],
            early_stopping_patience=shared["patience"])

        result = engine.eval(
            shared["eval_start"],
            shared["eval_end"],
            make_plots=True,
            plot_dir=os.path.join(run_dir, "plots"))

        if result:
            metrics["pearson_r"] = float(result[0])
            metrics["r2"] = float(result[1])
            metrics["mse"] = float(result[2])
        metrics["status"] = "completed"

    except Exception:
        import traceback
        tb = traceback.format_exc()
        log.red(tb)
        metrics["status"] = "error"
        metrics["error"] = tb

    finally:
        # ── Restore original config and patches ──
        MODEL_CONFIG.n_features = orig_n_features
        TRAINING_CONFIG.n_epochs = orig_epochs
        TRAINING_CONFIG.lr = orig_lr
        TRAINING_CONFIG.batch_size = orig_bs
        PREPROCESSING_CONFIG.preprocessing_fit_days = orig_fit_days

        MeowFeatureGenerator.gen_features = original_gen_features
        MeowFeatureGenerator.feature_names = original_feature_names

        metrics["finished_at"] = datetime.datetime.now().isoformat()
        metrics["hyperparams"] = {
            "epochs": shared["epochs"],
            "lr": shared["lr"],
            "batch_size": shared["batch_size"],
        }
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        sys.stdout.close()
        sys.stdout = orig_stdout
        sys.stderr = sys.__stderr__

    return metrics


def run_all_ablations(groups, shared, output_dir="ablation_results"):
    """Run baseline + all ablation groups, then generate report."""
    os.makedirs(output_dir, exist_ok=True)

    print_feature_summary()

    # Baseline first
    log.cyan("=" * 60)
    log.cyan("[BASELINE] 完整模型")
    log.cyan("=" * 60)
    baseline = run_ablation_group(None, shared, output_dir)
    if baseline is None or baseline["status"] != "completed":
        log.red("ERROR: Baseline failed, aborting.")
        return

    baseline_r = baseline["pearson_r"]
    log.green(f"Baseline Pearson r = {baseline_r:.4f}")

    # Run each ablation group
    results = [baseline]
    for group_name in groups:
        info = ABLATION_GROUPS[group_name]
        n_removed = len(info["features"])
        log.cyan(f"\n{'=' * 60}")
        log.cyan(f"[ABLATION] {info['label']} — removing {n_removed} features")
        log.cyan(f"{'=' * 60}")

        metrics = run_ablation_group(group_name, shared, output_dir)
        if metrics:
            results.append(metrics)
            r = metrics.get("pearson_r", 0.0)
            delta = r - baseline_r
            log.inf(f"Pearson r = {r:.4f}  (Δ = {delta:+.4f} vs baseline)")

        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ── Generate summary ──
    save_summary(results, output_dir)
    return results


def save_summary(results, output_dir):
    """Save results CSV and a Markdown report."""
    import csv

    # CSV
    csv_path = os.path.join(output_dir, "ablation_summary.csv")
    keys = ["group", "label", "n_features", "pearson_r", "r2", "mse",
            "status"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    log.green(f"Summary CSV: {csv_path}")

    # Markdown report
    md_path = os.path.join(output_dir, "ablation_report.md")
    _write_report(results, md_path)
    log.green(f"Report:      {md_path}")

    # Bar chart
    _plot_results(results, os.path.join(output_dir, "ablation_bar.png"))


def _write_report(results, path):
    """Generate a Markdown ablation report."""
    baseline = next((r for r in results if r["group"] == "baseline"), None)
    if baseline is None:
        return
    base_r = baseline["pearson_r"]

    lines = [
        "# 特征类别消融实验报告",
        "",
        f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 实验设置",
        "",
        "每组实验训练 4 epochs，除移除指定特征类别外，其余超参数完全一致。",
        "",
        "## 结果汇总",
        "",
        "| 实验组 | 特征数 | Pearson r | R² | vs Baseline Δr |",
        "|---|---:|---:|---:|---:|",
    ]

    for r in results:
        label = r["label"]
        nf = r["n_features"]
        pr = r.get("pearson_r", 0)
        r2 = r.get("r2", 0)
        delta = pr - base_r
        delta_str = f"{delta:+.4f}"
        lines.append(f"| {label} | {nf} | {pr:.4f} | {r2:.5f} | {delta_str} |")

    lines += [
        "",
        "## 分析",
        "",
        "### 影响最大的数据类别",
        "",
    ]

    # Sort by delta (most negative first = most important)
    sorted_results = sorted(
        [r for r in results if r["group"] != "baseline"],
        key=lambda r: r.get("pearson_r", 0))

    for i, r in enumerate(sorted_results):
        delta = r.get("pearson_r", 0) - base_r
        importance = abs(delta) / (abs(base_r) + 1e-8)
        direction = "下降" if delta < 0 else "提升"
        lines.append(
            f"{i+1}. **{r['label']}**：移除后 Pearson r {direction} {delta:+.4f}，"
            f"相对变化幅度约 {importance:.0%}")
        if delta > 0.001:
            lines.append("   > ⚠️ 移除后效果反而提升，说明该类特征可能引入噪声，建议精简。")

    lines += [
        "",
        "### 结论",
        "",
        "1. 贡献最大的数据类别值得重点维护——保障其数据质量，并探索更丰富的衍生特征。",
        "2. 移除后效果提升或不变的类别可能是冗余特征，考虑在后续版本中永久精简。",
        "3. 本报告可作为「数据价值评估」的直接证据，指导数据采购、存储和特征工程投入。",
        "",
        "> **教学提示**：消融实验是数据价值评估的核心工具。它量化地回答了「哪些数据值得收集/维护」——这正是大规模数据挖掘方向的核心问题之一。",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _setup_cjk_font():
    """Configure matplotlib to use a CJK-capable font for Chinese labels.

    Tries common CJK fonts in order of preference.  Falls back to DejaVu
    Sans (with missing-glyph warnings) if none are available.
    """
    try:
        import matplotlib
        import matplotlib.font_manager as fm
    except ImportError:
        return

    # Candidate CJK fonts (system-dependent)
    candidates = [
        "Noto Sans CJK SC", "Noto Sans SC", "Noto Sans CJK TC",
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "SimHei", "Microsoft YaHei",
        "AR PL UMing CN", "AR PL UKai CN",
        "Source Han Sans SC", "Source Han Sans CN",
    ]

    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            # Suppress missing-glyph warnings for fallback glyphs
            matplotlib.rcParams["axes.unicode_minus"] = False
            return

    # No CJK font found — use English labels to avoid tofu glyphs
    matplotlib.rcParams["font.family"] = "sans-serif"


def _plot_results(results, save_path):
    """Generate a horizontal bar chart comparing ablation results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    _setup_cjk_font()

    # Sort: baseline first, then by Pearson r descending
    sorted_r = sorted(results, key=lambda r: r.get("pearson_r", 0), reverse=True)
    labels = [r["label"] for r in sorted_r]
    values = [r.get("pearson_r", 0) for r in sorted_r]
    baseline_r = values[0] if values else 0

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2c7bb6" if v == baseline_r else
              "#d7191c" if v < baseline_r else "#abd9e9"
              for v in values]

    bars = ax.barh(range(len(labels)), values, color=colors,
                   edgecolor="white", height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Pearson r")
    ax.set_title("Ablation Study: Impact of Removing Feature Categories")

    # Annotate values
    for i, (v, l) in enumerate(zip(values, labels)):
        delta = v - baseline_r
        ax.text(v + 0.001, i,
                f"{v:.4f}  (Δ{delta:+.4f})", va="center", fontsize=8)

    ax.axvline(baseline_r, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.inf(f"Bar chart:   {save_path}")


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Feature-category ablation study for Meow")
    p.add_argument("--group", default=None,
                   choices=[None] + list(ABLATION_GROUPS.keys()),
                   help="Run a single ablation group (default: run all)")
    p.add_argument("--epochs", type=int, default=4,
                   help="Training epochs per run (default: 4)")
    p.add_argument("--batch-size", type=int, default=128,
                   help="Batch size (default: 128)")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate (default: 2e-4)")
    p.add_argument("--data-dir", default="archive/",
                   help="HDF5 data directory")
    p.add_argument("--output-dir", default="ablation_results",
                   help="Output directory for results")
    p.add_argument("--list-groups", action="store_true",
                   help="Print feature category breakdown and exit")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_groups:
        print_feature_summary()
        sys.exit(0)

    shared = dict(ABLATION_SHARED)
    shared["epochs"] = args.epochs
    shared["batch_size"] = args.batch_size
    shared["lr"] = args.lr
    shared["data_dir"] = args.data_dir

    if args.group:
        print_feature_summary()
        metrics = run_ablation_group(args.group, shared, args.output_dir)
        if metrics:
            log.green(f"Result: {metrics['label']}")
            log.inf(f"  Pearson r = {metrics.get('pearson_r', 'N/A')}")
            log.inf(f"  Status:    {metrics['status']}")
    else:
        run_all_ablations(
            list(ABLATION_GROUPS.keys()), shared, args.output_dir)
