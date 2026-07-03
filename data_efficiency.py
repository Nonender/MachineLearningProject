#!/usr/bin/env python3
"""Data efficiency analysis for Meow.

Trains the model on increasing fractions of the training set (25 %, 50 %,
75 %, 100 %) to measure how prediction performance scales with data volume.
This answers the question: "How much data do we actually need before
diminishing returns set in?"

Usage:
    python data_efficiency.py                      # run all fractions
    python data_efficiency.py --epochs 4           # custom epochs
    python data_efficiency.py --fraction 0.5       # run a single fraction
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

# ── Configuration ──────────────────────────────────────────────────────

# Fractions of training data to evaluate (ordered smallest → largest)
FRACTIONS = [0.25, 0.50, 0.75, 1.00]

SHARED_CONFIG = {
    "data_dir": "archive/",
    "train_start": 20230601,
    "train_end": 20231130,
    "eval_start": 20231201,
    "eval_end": 20231229,
    "val_window": 5,
    "patience": 3,
    "preprocessing_fit_days": 20,
    "seed": 42,
    "epochs": 4,
    "lr": 2e-4,
    "batch_size": 128,
}


# ── Core logic ─────────────────────────────────────────────────────────

def run_fraction(fraction, shared, output_dir="runs/data_efficiency"):
    """Train and evaluate using only `fraction` of training dates.

    Dates are taken from the *beginning* of the training range (oldest
    first), which is the only causally-valid split for time series.
    """
    from tradingcalendar import Calendar
    from meow import MeowEngine

    cal = Calendar()
    all_train_dates = cal.range(shared["train_start"], shared["train_end"])
    n_use = max(int(len(all_train_dates) * fraction), 1)
    train_dates = all_train_dates[:n_use]

    # Validation uses the next few dates *after* the truncated training set
    val_start_date = train_dates[-1]
    val_start = cal.next(val_start_date) or val_start_date

    n_days_label = f"{fraction:.0%}"
    run_name = f"data_{int(fraction * 100):03d}pct"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{ts}_{run_name}")
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "output.log")

    log.cyan("=" * 60)
    log.cyan(f"Data efficiency: {n_days_label} ({n_use} training days)")
    log.cyan(f"  Train: {train_dates[0]} – {train_dates[-1]}")
    log.cyan(f"  Val:   {val_start}")
    log.cyan(f"  Eval:  {shared['eval_start']} – {shared['eval_end']}")
    log.cyan("=" * 60)

    # ── Redirect stdout for MeowEngine output ──
    orig_stdout = sys.stdout
    sys.stdout = open(log_file, "w", buffering=1)
    sys.stderr = sys.stdout

    # ── Apply config ──
    orig_epochs = TRAINING_CONFIG.n_epochs
    orig_lr = TRAINING_CONFIG.lr
    orig_bs = TRAINING_CONFIG.batch_size
    orig_fit_days = PREPROCESSING_CONFIG.preprocessing_fit_days

    TRAINING_CONFIG.n_epochs = shared["epochs"]
    TRAINING_CONFIG.lr = shared["lr"]
    TRAINING_CONFIG.batch_size = shared["batch_size"]
    PREPROCESSING_CONFIG.preprocessing_fit_days = min(
        shared["preprocessing_fit_days"], n_use)

    metrics = {
        "fraction": fraction,
        "n_train_days": n_use,
        "run_dir": run_dir,
        "status": "error",
    }

    try:
        engine = MeowEngine(
            h5dir=shared["data_dir"],
            cache_dir=None,
            checkpoint_dir=os.path.join(run_dir, "checkpoints"))

        # Monkey-patch fit to use truncated date list, then train
        engine.fit = _make_truncated_fit(
            engine, train_dates, val_start, shared)
        engine.fit()

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
        TRAINING_CONFIG.n_epochs = orig_epochs
        TRAINING_CONFIG.lr = orig_lr
        TRAINING_CONFIG.batch_size = orig_bs
        PREPROCESSING_CONFIG.preprocessing_fit_days = orig_fit_days

        metrics["finished_at"] = datetime.datetime.now().isoformat()
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        sys.stdout.close()
        sys.stdout = orig_stdout
        sys.stderr = sys.__stderr__

    return metrics


def _make_truncated_fit(engine, train_dates, val_start, shared):
    """Return a replacement ``fit()`` method that uses a truncated date list.

    We patch the engine's fit method rather than modifying MeowEngine
    directly, so the core code remains untouched.
    """
    original_fit = engine.fit

    def truncated_fit(**kwargs):
        return original_fit(
            start_date=train_dates[0],
            end_date=train_dates[-1],
            val_date=val_start,
            val_window=shared["val_window"],
            early_stopping_patience=shared["patience"])

    return truncated_fit


def run_all_fractions(fractions, shared, output_dir="runs/data_efficiency"):
    """Run all data fractions and generate summary artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = []

    log.inf(f"Data efficiency sweep: {len(fractions)} fractions × "
            f"{shared['epochs']} epochs each")
    log.inf(f"Training date range: {shared['train_start']} – "
            f"{shared['train_end']}")
    log.inf("")

    for i, frac in enumerate(fractions):
        n_pct = f"{frac:.0%}"
        log.cyan(f"[{i+1}/{len(fractions)}] {n_pct} data")
        metrics = run_fraction(frac, shared, output_dir)
        all_metrics.append(metrics)

        r = metrics.get("pearson_r", float("nan"))
        status = metrics.get("status", "?")
        log.inf(f"  -> {status}  |  {n_pct} data → Pearson r = {r:.4f}")

        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    save_summary(all_metrics, output_dir)
    return all_metrics


# ── Output ─────────────────────────────────────────────────────────────

def save_summary(results, output_dir):
    """Write CSV summary, Markdown report, and learning curve plot."""
    import csv

    # CSV
    csv_path = os.path.join(output_dir, "efficiency_summary.csv")
    keys = ["fraction", "n_train_days", "pearson_r", "r2", "mse", "status"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    log.green(f"Summary CSV: {csv_path}")

    # Report
    md_path = os.path.join(output_dir, "efficiency_report.md")
    _write_report(results, md_path)
    log.green(f"Report:      {md_path}")

    # Plot
    _plot_curve(results, os.path.join(output_dir, "efficiency_curve.png"))


def _write_report(results, path):
    """Generate a Markdown report."""
    valid = [r for r in results if r["status"] == "completed"]
    if len(valid) < 2:
        return

    best = max(valid, key=lambda r: r["pearson_r"])
    best_r, best_frac = best["pearson_r"], best["fraction"]
    worst_r = min(r["pearson_r"] for r in valid)

    # Estimate saturation point: where adding 25 % more data improves r by < 5 %
    saturation_frac = 1.0
    for i in range(len(valid) - 1):
        r_curr = valid[i]["pearson_r"]
        r_next = valid[i + 1]["pearson_r"]
        gain_pct = (r_next - r_curr) / (abs(r_curr) + 1e-8)
        if gain_pct < 0.05 and valid[i]["fraction"] >= 0.5:
            saturation_frac = valid[i]["fraction"]
            break

    lines = [
        "# 数据效率分析报告",
        "",
        f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 结果",
        "",
        "| 数据比例 | 训练天数 | Pearson r | R² |",
        "|---|---:|---:|---:|",
    ]
    for r in valid:
        lines.append(
            f"| {r['fraction']:.0%} | {r['n_train_days']} | "
            f"{r['pearson_r']:.4f} | {r.get('r2', 0):.5f} |")

    lines += [
        "",
        "## 分析",
        "",
        f"- **最佳效果**：{best_frac:.0%} 数据量时 Pearson r = {best_r:.4f}",
        f"- **边际增益**：25% → 100% 数据仅提升 r 约 {best_r - worst_r:.4f}",
        f"- **数据饱和点**：约 {saturation_frac:.0%} 数据量后边际收益显著递减",
        "",
        "## 结论",
        "",
    ]

    if saturation_frac < 0.8:
        lines.append(
            f"模型在约 {saturation_frac:.0%} 数据量时已接近饱和。继续增加数据带来的提升有限，"
            "建议优先提升数据质量（更精准的特征、更低噪声的数据源）"
            "而非盲目扩大数据规模。")
    else:
        lines.append(
            "模型性能随数据量持续提升，尚未达到明显饱和点。"
            "这意味着更多数据可能进一步改善效果——"
            "如果有条件获取更长周期的历史数据，值得尝试。")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _plot_curve(results, save_path):
    """Plot data fraction vs Pearson r learning curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # CJK font
    candidates = [
        "Noto Sans CJK SC", "Noto Sans SC",
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "SimHei", "AR PL UMing CN",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            matplotlib.rcParams["axes.unicode_minus"] = False
            break

    valid = [r for r in results if r["status"] == "completed"]
    if len(valid) < 2:
        return

    fractions = [r["fraction"] * 100 for r in valid]
    r_values = [r["pearson_r"] for r in valid]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fractions, r_values, "o-", color="#2c7bb6", lw=2,
            markersize=8, markerfacecolor="white", markeredgewidth=1.5)
    ax.set_xlabel("Training Data (%)")
    ax.set_ylabel("Pearson r")
    ax.set_title("Data Efficiency: Model Performance vs Training Data Volume")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Annotate each point
    for x, y in zip(fractions, r_values):
        ax.annotate(f"{y:.4f}", (x, y),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=8, ha="center")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log.inf(f"Learning curve: {save_path}")


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Data efficiency analysis for Meow")
    p.add_argument("--fraction", type=float, default=None,
                   help="Run a single data fraction (e.g. 0.5)")
    p.add_argument("--epochs", type=int, default=4,
                   help="Training epochs per run (default: 4)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--data-dir", default="archive/")
    p.add_argument("--output-dir", default="runs/data_efficiency")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    shared = dict(SHARED_CONFIG)
    shared["epochs"] = args.epochs
    shared["batch_size"] = args.batch_size
    shared["lr"] = args.lr
    shared["data_dir"] = args.data_dir

    if args.fraction is not None:
        if args.fraction <= 0 or args.fraction > 1:
            log.red("Fraction must be in (0, 1]")
            sys.exit(1)
        metrics = run_fraction(args.fraction, shared, args.output_dir)
        if metrics:
            log.green(f"Result: {metrics['fraction']:.0%} data → "
                      f"Pearson r = {metrics.get('pearson_r', 'N/A')}")
    else:
        run_all_fractions(FRACTIONS, shared, args.output_dir)
