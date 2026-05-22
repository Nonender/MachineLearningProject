import gc
import numpy as np
import pandas as pd
from log import log


class MeowFeatureGenerator(object):
    @classmethod
    def featureNames(cls):
        return [
            # Price (4)
            "ret1", "ret5", "ret10", "ret30",
            # Spread & depth (2)
            "spread", "depth_imb",
            # Order book (5)
            "ob_imb0", "ob_imb4", "ob_imb9",
            "ob_slope", "ob_imb0_ema10",
            # Trade (5)
            "trade_imb", "trade_imb_ema5",
            "trade_count_imb", "log_trade_volume", "vwap_dev",
            # Cross-sectional (4)
            "cx_ret1", "cx_ret10",
            "cx_ob_imb0", "cx_trade_imb",
            # Momentum (2)
            "lagret12", "mom_5_30",
            # Time (2)
            "sin_time", "cos_time",
        ]

    def __init__(self, cacheDir):
        self.cacheDir = cacheDir
        self.ycol = "fret12"
        self.mcols = ["symbol", "date", "interval"]

    def genFeatures(self, df):
        n_feat = len(self.featureNames())
        log.inf("Generating {} features from raw data...".format(n_feat))
        eps = 1e-8

        df = df.sort_values(["symbol", "interval"])

        # ---- keep only needed raw cols ----
        needed_raw = [
            "bid0", "ask0", "bid4", "ask4",
            "bsize0", "asize0",
            "bsize0_4", "asize0_4", "bsize5_9", "asize5_9",
            "tradeBuyQty", "tradeSellQty", "tradeBuyTurnover", "tradeSellTurnover",
            "nTradeBuy", "nTradeSell",
        ]
        df = df[self.mcols + needed_raw + [self.ycol]]

        df["midpx"] = (df["bid0"] + df["ask0"]) / 2.0

        # ---- Price returns — use built-in groupby.diff/shift (no lambda) ----
        g = df.groupby("symbol")
        for h in [1, 5, 10, 30]:
            df[f"ret{h}"] = g["midpx"].diff(h) / (g["midpx"].shift(h) + eps)

        # ---- Spread & depth ----
        df["spread"] = (df["ask0"] - df["bid0"]) / (df["midpx"] + eps)
        df["depth_imb"] = np.log((df["asize0_4"] + eps) / (df["bsize0_4"] + eps))

        # ---- Order book imbalance ----
        df["ob_imb0"] = (df["asize0"] - df["bsize0"]) / (df["asize0"] + df["bsize0"] + eps)
        df["ob_imb4"] = (df["asize0_4"] - df["bsize0_4"]) / (df["asize0_4"] + df["bsize0_4"] + eps)
        df["ob_imb9"] = (df["asize5_9"] - df["bsize5_9"]) / (df["asize5_9"] + df["bsize5_9"] + eps)
        df["ob_slope"] = df["ob_imb0"] - df["ob_imb9"]

        # EMA — recreate groupby so it sees newly-added columns
        g = df.groupby("symbol")
        df["ob_imb0_ema10"] = g["ob_imb0"].transform(
            lambda x: x.ewm(halflife=10, min_periods=1).mean()
        )

        # ---- Trade features ----
        df["trade_imb"] = (df["tradeBuyQty"] - df["tradeSellQty"]) / (
            df["tradeBuyQty"] + df["tradeSellQty"] + eps
        )
        df["trade_imb_ema5"] = g["trade_imb"].transform(
            lambda x: x.ewm(halflife=5, min_periods=1).mean()
        )
        df["trade_count_imb"] = (df["nTradeBuy"] - df["nTradeSell"]) / (
            df["nTradeBuy"] + df["nTradeSell"] + eps
        )
        df["log_trade_volume"] = np.log(df["tradeBuyQty"] + df["tradeSellQty"] + 1.0)
        buy_vwap = df["tradeBuyTurnover"] / (df["tradeBuyQty"] + eps)
        sell_vwap = df["tradeSellTurnover"] / (df["tradeSellQty"] + eps)
        df["vwap_dev"] = ((buy_vwap + sell_vwap) / 2.0 - df["midpx"]) / (df["midpx"] + eps)

        # ---- Cross-sectional ----
        for col in ["ret1", "ret10", "ob_imb0", "trade_imb"]:
            mu = df.groupby("interval")[col].transform("mean")
            df[f"cx_{col}"] = df[col] - mu

        # ---- Momentum / reversal ----
        g = df.groupby("symbol")
        df["bret12"] = g["midpx"].diff(12) / (g["midpx"].shift(12) + eps)
        cx_bret12 = df.groupby("interval")["bret12"].transform("mean")
        df["lagret12"] = df["bret12"] - cx_bret12
        df["mom_5_30"] = df["ret5"] * df["ret30"]

        # ---- Time-of-day ----
        minute_of_day = (df["interval"] / 60_000).astype(int) % 1440
        df["sin_time"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
        df["cos_time"] = np.cos(2 * np.pi * minute_of_day / 1440.0)

        # ---- Assemble output, drop intermediate columns ----
        feature_names = self.featureNames()
        keep_cols = self.mcols + feature_names + [self.ycol]
        xdf = df[keep_cols].set_index(self.mcols)
        ydf = xdf[[self.ycol]].copy()
        xdf = xdf[feature_names]
        xdf = xdf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        del df, buy_vwap, sell_vwap, cx_bret12, minute_of_day
        gc.collect()
        return xdf, ydf
