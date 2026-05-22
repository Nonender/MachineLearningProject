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
        n_epochs = 10
        log.inf("Training on {} dates for {} epochs...".format(len(dates), n_epochs))
        for epoch in range(n_epochs):
            shuffled = list(dates)
            random.shuffle(shuffled)
            for i, date in enumerate(shuffled):
                rawData = self.dloader.loadDate(date)
                xdf, ydf = self.featGenerator.genFeatures(rawData)
                self.model.partial_fit(xdf, ydf)
                del rawData, xdf, ydf
                if (i + 1) % 5 == 0:
                    gc.collect()
                # Set up LR scheduler after processing first date
                if epoch == 0 and i == 0 and self.model.scheduler is None:
                    self.model.set_scheduler(
                        steps_per_epoch=len(dates), n_epochs=n_epochs,
                    )
                if (i + 1) % 20 == 0:
                    lr = self.model.optimizer.param_groups[0]["lr"]
                    log.inf("  Epoch {}/{}: {}/{} dates, lr={:.2e}".format(
                        epoch + 1, n_epochs, i + 1, len(dates), lr,
                    ))
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
