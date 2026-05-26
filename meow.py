import gc
import os
import numpy as np
import pandas as pd
from log import log
from dl import MeowDataLoader
from feat import MeowFeatureGenerator
from mdl import MeowModel
from eval import MeowEvaluator
from tradingcalendar import Calendar
from parameters import TRAINING_CONFIG, PREPROCESSING_CONFIG


class MeowEngine(object):
    def __init__(self, h5dir, cacheDir):
        self.calendar = Calendar()
        self.h5dir = h5dir
        if not os.path.exists(h5dir):
            raise ValueError("Data directory not exists: {}".format(self.h5dir))
        if not os.path.isdir(h5dir):
            raise ValueError("Invalid data directory: {}".format(self.h5dir))
        self.cacheDir = cacheDir # this is not used in sample code
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.featGenerator = MeowFeatureGenerator(cacheDir=cacheDir)
        self.model = MeowModel(cacheDir=cacheDir)
        self.evaluator = MeowEvaluator(cacheDir=cacheDir)

    def fit(self, startDate, endDate):
        import random
        dates = self.calendar.range(startDate, endDate)
        n_epochs = TRAINING_CONFIG.n_epochs

        # ---- Fit preprocessing on first N days (pooled) ----
        n_fit = min(PREPROCESSING_CONFIG.preprocessing_fit_days, len(dates))
        fit_dates = dates[:n_fit]
        train_dates = dates  # all dates used for training (preprocessing dates included)

        log.inf("Fitting preprocessing on first {} dates...".format(len(fit_dates)))
        xdfs, ydfs = [], []
        for date in fit_dates:
            rawData = self.dloader.loadDate(date)
            xdf, ydf = self.featGenerator.genFeatures(rawData)
            xdfs.append(xdf)
            ydfs.append(ydf)
        self.model.fit_preprocessing(pd.concat(xdfs), pd.concat(ydfs))
        # Free preprocessing data to save memory
        del xdfs, ydfs, xdf, ydf, rawData
        gc.collect()
        # Set up LR scheduler now that we know steps_per_epoch
        self.model.set_scheduler(
            steps_per_epoch=len(train_dates), n_epochs=n_epochs,
        )

        # ---- Training loop ----
        log.inf("Training on {} dates for {} epochs...".format(len(train_dates), n_epochs))
        for epoch in range(n_epochs):
            shuffled = list(train_dates)
            random.shuffle(shuffled)
            pcor_sum, r2_sum, mse_sum = 0.0, 0.0, 0.0
            for i, date in enumerate(shuffled):
                rawData = self.dloader.loadDate(date)
                xdf, ydf = self.featGenerator.genFeatures(rawData)
                pcor, r2, mse = self.model.partial_fit(xdf, ydf)
                pcor_sum += pcor
                r2_sum += r2
                mse_sum += mse
                del rawData, xdf, ydf
                if (i + 1) % 5 == 0:
                    gc.collect()
                if (i + 1) % 20 == 0:
                    n = min(i + 1, 20)
                    lr = self.model.optimizer.param_groups[0]["lr"]
                    log.inf("  Epoch {}/{}: {}/{} dates, lr={:.2e} | Pearson={:+.4f}  R²={:+.5f}  MSE={:.4f}".format(
                        epoch + 1, n_epochs, i + 1, len(train_dates), lr,
                        pcor_sum / n, r2_sum / n, mse_sum / n,
                    ))
                    pcor_sum = r2_sum = mse_sum = 0.0
        log.inf("Done fitting")

    def predict(self, xdf):
        return self.model.predict(xdf)

    def eval(self, startDate, endDate):
        log.inf("Running model evaluation...")
        dates = self.calendar.range(startDate, endDate)
        ys, preds = [], []
        for i, date in enumerate(dates):
            rawData = self.dloader.loadDate(date)
            xdf, ydf = self.featGenerator.genFeatures(rawData)
            ys.append(ydf.to_numpy().ravel())
            preds.append(self.predict(xdf))
        ydf = pd.DataFrame({
            "fret12": np.concatenate(ys),
            "forecast": np.concatenate(preds),
        })
        self.evaluator.eval(ydf)


if __name__ == "__main__":
    engine = MeowEngine(h5dir="archive/", cacheDir=None)
    engine.fit(20230601, 20231130)
    engine.eval(20231201, 20231229)
