#!/usr/bin/env python3
"""Hyperparameter sweep runner for Meow.

Runs multiple training jobs sequentially with different hyperparameters.
Results (checkpoints, logs, plots, metrics) saved to runs/<run_name>/.

Usage:
    python sweep.py                              # run default sweep
    python sweep.py --single --epochs 10 --lr 1e-4   # single custom run
"""

import argparse
import datetime
import gc
import json
import os
import sys
import traceback
from dataclasses import asdict

import numpy as np

from parameters import TRAINING_CONFIG, PREPROCESSING_CONFIG

# ── Default sweep configs ──────────────────────────────────────────────
# Edit this list to define your sweep.
SWEEP_CONFIGS = [
    {"name": "epochs_12", "epochs": 12, "lr": 2e-4, "batch_size": 256},
    {"name": "lr_5e4",    "epochs": 8,  "lr": 5e-4, "batch_size": 256},
    {"name": "lr_1e4",    "epochs": 8,  "lr": 1e-4, "batch_size": 256},
    {"name": "bs_128",    "epochs": 8,  "lr": 2e-4, "batch_size": 128},
]

# Shared settings for all runs (override via CLI or edit here)
SHARED_CONFIG = {
    "data_dir": "archive/",
    "train_start": 20230601,
    "train_end": 20231130,
    "eval_start": 20231201,
    "eval_end": 20231229,
    "val_window": 5,
    "patience": 5,
    "preprocessing_fit_days": 20,
    "seed": 42,
}


def make_run_dir(base_dir, run_name):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{ts}_{run_name}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def run_single(config, shared, base_dir="runs"):
    """Run one training job with the given hyperparameters. Returns metrics dict."""
    run_dir = make_run_dir(base_dir, config["name"])
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    plot_dir = os.path.join(run_dir, "plots")
    log_file = os.path.join(run_dir, "output.log")

    # ── Apply config to global parameter objects ──
    orig_epochs = TRAINING_CONFIG.n_epochs
    orig_lr = TRAINING_CONFIG.lr
    orig_bs = TRAINING_CONFIG.batch_size
    orig_fit_days = PREPROCESSING_CONFIG.preprocessing_fit_days

    TRAINING_CONFIG.n_epochs = config.get("epochs", orig_epochs)
    TRAINING_CONFIG.lr = config.get("lr", orig_lr)
    TRAINING_CONFIG.batch_size = config.get("batch_size", orig_bs)
    PREPROCESSING_CONFIG.preprocessing_fit_days = config.get(
        "preprocessing_fit_days", orig_fit_days)

    # ── Redirect stdout/stderr to log file ──
    sys.stdout = open(log_file, "w", buffering=1)
    sys.stderr = sys.stdout

    print("=" * 60)
    print(f"Run: {config['name']}")
    print(f"Started: {datetime.datetime.now()}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"Output dir: {run_dir}")
    print("=" * 60)

    metrics = {"run_name": config["name"], "run_dir": run_dir, "status": "error"}

    try:
        from meow import MeowEngine

        engine = MeowEngine(
            h5dir=shared["data_dir"],
            cache_dir=None,
            checkpoint_dir=checkpoint_dir)

        engine.fit(
            start_date=shared["train_start"],
            end_date=shared["train_end"],
            val_date=engine.calendar.next(shared["train_end"]) or shared["train_end"],
            val_window=shared["val_window"],
            early_stopping_patience=shared["patience"])

        result = engine.eval(
            shared["eval_start"],
            shared["eval_end"],
            make_plots=True,
            plot_dir=plot_dir)

        if result:
            metrics["pearson_r"] = float(result[0])
            metrics["r2"] = float(result[1])
            metrics["mse"] = float(result[2])
        metrics["status"] = "completed"

    except Exception:
        print(traceback.format_exc())
        metrics["status"] = "error"
        metrics["error"] = traceback.format_exc()

    finally:
        # ── Restore original config values ──
        TRAINING_CONFIG.n_epochs = orig_epochs
        TRAINING_CONFIG.lr = orig_lr
        TRAINING_CONFIG.batch_size = orig_bs
        PREPROCESSING_CONFIG.preprocessing_fit_days = orig_fit_days

        # ── Save metrics ──
        metrics["finished_at"] = datetime.datetime.now().isoformat()
        metrics["hyperparams"] = config
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # ── Restore stdout ──
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    return metrics


def run_sweep(configs, shared, base_dir="runs"):
    """Run all configs sequentially. Saves a summary CSV."""
    os.makedirs(base_dir, exist_ok=True)
    summary_path = os.path.join(base_dir, "sweep_summary.csv")
    all_metrics = []

    print(f"Starting sweep: {len(configs)} runs")
    print(f"Results: {base_dir}/")
    print(f"Summary: {summary_path}")
    print("-" * 50)

    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {cfg['name']}  (epochs={cfg.get('epochs','?')}, "
              f"lr={cfg.get('lr','?')}, bs={cfg.get('batch_size','?')})")
        metrics = run_single(cfg, shared, base_dir)
        all_metrics.append(metrics)

        # Print quick result
        status = metrics.get("status", "?")
        r = metrics.get("pearson_r", "N/A")
        print(f"  -> {status}  |  Pearson r = {r}")

        gc.collect()
        if torch_available():
            import torch
            torch.cuda.empty_cache()

    # ── Write summary CSV ──
    write_summary(all_metrics, summary_path)
    print(f"\nDone. Summary: {summary_path}")

    return all_metrics


def torch_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def write_summary(all_metrics, path):
    """Write a CSV summary of all runs."""
    import csv
    keys = ["run_name", "status", "pearson_r", "r2", "mse",
            "epochs", "lr", "batch_size", "preprocessing_fit_days", "run_dir"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for m in all_metrics:
            row = dict(m)
            hp = m.get("hyperparams", {})
            row.update({k: hp.get(k, "") for k in keys if k in hp})
            w.writerow(row)
    print(f"Summary written to {path}")


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Hyperparameter sweep runner for Meow")
    p.add_argument("--single", action="store_true",
                   help="Run a single job instead of full sweep")
    p.add_argument("--name", default=None,
                   help="Run name (default: auto-generated)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--preprocessing-fit-days", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--data-dir", default="archive/")
    p.add_argument("--output-dir", default="runs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.single:
        name = args.name or f"custom_e{args.epochs}_lr{args.lr}"
        config = {
            "name": name,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "preprocessing_fit_days": args.preprocessing_fit_days,
        }
        SHARED_CONFIG["patience"] = args.patience
        SHARED_CONFIG["data_dir"] = args.data_dir
        run_single(config, SHARED_CONFIG, base_dir=args.output_dir)
    else:
        run_sweep(SWEEP_CONFIGS, SHARED_CONFIG, base_dir=args.output_dir)
