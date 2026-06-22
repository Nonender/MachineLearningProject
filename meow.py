import argparse
import gc
import os
import random
import numpy as np
import pandas as pd
import torch
from log import log
from dl import MeowDataLoader
from feat import MeowFeatureGenerator
from mdl import MeowModel
from eval import MeowEvaluator
from tradingcalendar import Calendar
from parameters import TRAINING_CONFIG, PREPROCESSING_CONFIG


class MeowEngine:
    def __init__(self, h5dir, cache_dir=None, checkpoint_dir=None):
        self.calendar = Calendar()
        self.h5dir = h5dir
        if not os.path.exists(h5dir):
            raise ValueError("Data directory not exists: {}".format(self.h5dir))
        if not os.path.isdir(h5dir):
            raise ValueError("Invalid data directory: {}".format(self.h5dir))
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.feat_gen = MeowFeatureGenerator(cache_dir=cache_dir)
        self.model = MeowModel(checkpoint_dir=checkpoint_dir)
        self.evaluator = MeowEvaluator(cache_dir=cache_dir)
        self.checkpoint_dir = checkpoint_dir

    def fit(self, start_date, end_date, val_date=None, val_window=5,
            early_stopping_patience=3):
        all_train_dates = self.calendar.range(start_date, end_date)
        train_dates = [d for d in all_train_dates if self.dloader.has_date(d)]

        if not train_dates:
            raise RuntimeError(
                "No training data found in {} for range [{}, {}]. "
                "Available dates: {}".format(
                    self.h5dir, start_date, end_date,
                    sorted(self.dloader.available_dates())[:10]))
        if len(train_dates) < len(all_train_dates):
            log.yellow("Using {}/{} dates with available data for training"
                       .format(len(train_dates), len(all_train_dates)))

        n_epochs = TRAINING_CONFIG.n_epochs

        n_fit = min(PREPROCESSING_CONFIG.preprocessing_fit_days, len(train_dates))
        fit_dates = train_dates[:n_fit]

        log.inf("Fitting preprocessing on first {} dates (streaming)..."
                .format(len(fit_dates)))
        fit_chunks = []
        for date in fit_dates:
            raw = self.dloader.load_date(date)
            xdf, ydf = self.feat_gen.gen_features(raw)
            fit_chunks.append((xdf, ydf))
            del raw
        self.model.fit_preprocessing(fit_chunks)
        del fit_chunks
        gc.collect()
        self.model.set_scheduler(steps_per_epoch=len(train_dates), n_epochs=n_epochs)

        # ---------- validation split ----------
        val_dates = []
        if val_date is not None:
            val_candidates = self.calendar.range(
                val_date, self.calendar.shift(val_date, val_window - 1))
            val_dates = [d for d in val_candidates if self.dloader.has_date(d)]
            if len(val_dates) < len(val_candidates):
                log.yellow("Using {}/{} dates with available data for validation"
                           .format(len(val_dates), len(val_candidates)))
            log.inf("Validation on {} dates: {} – {}".format(
                len(val_dates),
                val_dates[0] if val_dates else "N/A",
                val_dates[-1] if val_dates else "N/A"))

        log.inf("Training on {} dates for {} epochs...".format(
            len(train_dates), n_epochs))
        best_val_r = self.model.best_val_metric
        epochs_no_improve = 0

        for epoch in range(n_epochs):
            shuffled = list(train_dates)
            random.shuffle(shuffled)
            train_loss = 0.0
            n_batches = 0
            for date in shuffled:
                raw = self.dloader.load_date(date)
                xdf, ydf = self.feat_gen.gen_features(raw)
                loss = self.model.partial_fit(xdf, ydf)
                train_loss += loss
                n_batches += 1
                del raw, xdf, ydf
                gc.collect()

            avg_loss = train_loss / max(n_batches, 1)

            # ---- validation ----
            if val_dates:
                val_r = self._validate(val_dates)
                log.inf("Epoch {}/{} | train_loss={:.6f} | val_pearson_r={:.4f} | "
                        "best_r={:.4f}".format(
                            epoch + 1, n_epochs, avg_loss, val_r, best_val_r))
                if val_r > best_val_r:
                    best_val_r = val_r
                    self.model.best_val_metric = best_val_r
                    epochs_no_improve = 0
                    if self.checkpoint_dir:
                        self.model.save_checkpoint(
                            os.path.join(self.checkpoint_dir, "checkpoint_best.pt"))
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stopping_patience:
                        log.inf("Early stopping after {} epochs without improvement"
                                .format(early_stopping_patience))
                        self.model.load_checkpoint(
                            os.path.join(self.checkpoint_dir, "checkpoint_best.pt"))
                        break
            else:
                log.inf("Epoch {}/{} | train_loss={:.6f}".format(
                    epoch + 1, n_epochs, avg_loss))

            # Save latest checkpoint
            if self.checkpoint_dir:
                self.model.save_checkpoint(
                    os.path.join(self.checkpoint_dir, "checkpoint_latest.pt"))

        log.inf("Done fitting. Best val Pearson r = {:.4f}".format(best_val_r))

    def _validate(self, val_dates):
        """Compute Pearson r on validation set (single-horizon for speed)."""
        self.model.model.eval()
        all_preds, all_truths = [], []
        with torch.no_grad():
            for date in val_dates:
                raw = self.dloader.load_date(date)
                xdf, ydf = self.feat_gen.gen_features(raw)
                p = self.model.predict(xdf, denormalize=True)
                all_preds.append(p)
                all_truths.append(ydf["fret12"].to_numpy().ravel()[:len(p)])
                del raw, xdf, ydf
        self.model.model.train()
        preds = np.concatenate(all_preds)
        truths = np.concatenate(all_truths)
        mask = np.isfinite(preds) & np.isfinite(truths)
        if mask.sum() < 10:
            return 0.0
        return float(np.corrcoef(preds[mask], truths[mask])[0, 1])

    def predict(self, xdf, denormalize=True):
        return self.model.predict(xdf, denormalize=denormalize)

    def eval(self, start_date, end_date, make_plots=False, plot_dir="plots/"):
        log.inf("Running model evaluation...")
        all_dates = self.calendar.range(start_date, end_date)
        dates = [d for d in all_dates if self.dloader.has_date(d)]
        if not dates:
            log.red("No evaluation data found in {} for range [{}, {}]"
                    .format(self.h5dir, start_date, end_date))
            return None
        if len(dates) < len(all_dates):
            log.yellow("Using {}/{} dates with available data for evaluation"
                       .format(len(dates), len(all_dates)))

        all_dfs = []
        for date in dates:
            raw = self.dloader.load_date(date)
            xdf, ydf = self.feat_gen.gen_features(raw)

            p = self.predict(xdf, denormalize=True)

            eval_df = ydf.copy()
            eval_df[self.evaluator.prediction_col] = p
            all_dfs.append(eval_df)

        combined = pd.concat(all_dfs)
        result = self.evaluator.eval(combined)

        if make_plots:
            import viz
            gate_vals = self.model.model.feat_gate.sigmoid().detach().cpu().numpy()
            feat_names = MeowFeatureGenerator.feature_names()
            viz.make_all_plots(
                combined[self.evaluator.ycol].to_numpy(),
                combined[self.evaluator.prediction_col].to_numpy(),
                gate_values=gate_vals,
                feature_names=feat_names,
                output_dir=plot_dir)

        return result


def parse_args():
    p = argparse.ArgumentParser(
        description="Meow — Decoder-only Transformer for stock return prediction")
    p.add_argument("--data-dir", default="archive/",
                   help="Directory containing HDF5 daily data files")
    p.add_argument("--cache-dir", default=None,
                   help="Directory for caching preprocessed features")
    p.add_argument("--checkpoint-dir", default="checkpoints/",
                   help="Directory for model checkpoints")
    p.add_argument("--train-start", type=int, default=20230601,
                   help="Training start date (YYYYMMDD)")
    p.add_argument("--train-end", type=int, default=20231130,
                   help="Training end date (YYYYMMDD)")
    p.add_argument("--val-start", type=int, default=None,
                   help="Validation start date (YYYYMMDD). "
                        "Defaults to first N days after train-end.")
    p.add_argument("--val-window", type=int, default=5,
                   help="Number of validation days")
    p.add_argument("--eval-start", type=int, default=20231201,
                   help="Evaluation start date (YYYYMMDD)")
    p.add_argument("--eval-end", type=int, default=20231229,
                   help="Evaluation end date (YYYYMMDD)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override number of training epochs")
    p.add_argument("--lr", type=float, default=None,
                   help="Override learning rate")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch size")
    p.add_argument("--no-early-stopping", action="store_true",
                   help="Disable early stopping")
    p.add_argument("--patience", type=int, default=3,
                   help="Early stopping patience (epochs)")
    p.add_argument("--preprocessing-fit-days", type=int, default=None,
                   help="Days used for preprocessing fit (default: from config)")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint path")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training, run evaluation only")
    p.add_argument("--plot", action="store_true",
                   help="Generate evaluation plots")
    p.add_argument("--plot-dir", default="plots",
                   help="Directory for saving plots")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.epochs is not None:
        TRAINING_CONFIG.n_epochs = args.epochs
    if args.lr is not None:
        TRAINING_CONFIG.lr = args.lr
    if args.batch_size is not None:
        TRAINING_CONFIG.batch_size = args.batch_size
    if args.preprocessing_fit_days is not None:
        PREPROCESSING_CONFIG.preprocessing_fit_days = args.preprocessing_fit_days

    engine = MeowEngine(
        h5dir=args.data_dir,
        cache_dir=args.cache_dir,
        checkpoint_dir=args.checkpoint_dir)

    if args.resume:
        engine.model.load_checkpoint(args.resume)

    if not args.eval_only:
        val_start = args.val_start
        if val_start is None:
            val_start = engine.calendar.next(args.train_end) or args.train_end
        engine.fit(
            start_date=args.train_start,
            end_date=args.train_end,
            val_date=val_start,
            val_window=args.val_window,
            early_stopping_patience=999 if args.no_early_stopping else args.patience,
        )

    engine.eval(args.eval_start, args.eval_end,
                make_plots=args.plot, plot_dir=args.plot_dir)
