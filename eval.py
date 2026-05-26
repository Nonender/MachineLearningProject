import os
import numpy as np
import pandas as pd
from log import log


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Return (pearson_r, r2, mse) for two numpy arrays."""
    y_true = np.nan_to_num(np.asarray(y_true).ravel(), nan=0.0)
    y_pred = np.nan_to_num(np.asarray(y_pred).ravel(), nan=0.0)
    pcor = np.corrcoef(y_pred, y_true)[0, 1]
    ss_res = ((y_pred - y_true) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mse = ss_res / len(y_true)
    return float(pcor), float(r2), float(mse)


class MeowEvaluator(object):
    def __init__(self, cacheDir):
        self.cacheDir = cacheDir
        self.predictionCol = "forecast"
        self.ycol = "fret12"

    def eval(self, ydf):
        ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0)
        pcor, r2, mse = compute_metrics(
            ydf[self.ycol].to_numpy(), ydf[self.predictionCol].to_numpy())
        log.inf("Meow evaluation summary: Pearson correlation={:.4f}, R2={:.5f}, MSE={:.6f}".format(
            pcor, r2, mse))
