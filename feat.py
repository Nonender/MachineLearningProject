import gc
import numpy as np
import pandas as pd
from log import log
from parameters import PREPROCESSING_CONFIG


class MeowFeatureGenerator(object):
    @classmethod
    def featureNames(cls):
        return [
            # Price / volatility (11)
            "ret1", "ret5", "ret10", "ret30",
            "vol5", "vol10", "vol20", "ret1_vol20",
            "vol_ratio_5_20",  # NEW: volatility regime indicator
            "ret1_sign",       # NEW: directional bias
            # Spread & depth (6)
            "spread", "spread4", "spread_ema5",
            "depth_imb", "depth_conc",
            "depth_total",     # NEW: log total visible liquidity
            # Order book (7)
            "ob_imb0", "ob_imb4", "ob_imb9",
            "ob_slope", "ob_curvature",
            "ob_imb0_ema10", "ob_imb0_ema30",
            # Trade (8)
            "trade_imb", "trade_imb_ema5", "trade_imb_ema30",
            "trade_count_imb", "log_trade_volume",
            "vwap_dev", "vwap_dev_ema5",
            "volume_intensity",  # NEW: volume per unit vol
            # Cross-sectional (8)
            "cx_ret1", "cx_ret10",
            "cx_ob_imb0", "cx_trade_imb",
            "rank_ob_imb0", "rank_trade_imb",
            "cx_vol20",         # NEW: cross-sectional vol deviation
            "rank_vol20",       # NEW: percentile rank of vol20
            # Momentum / reversal (5)
            "lagret12", "mom_5_30",
            "ret5_ret1", "ret30_ret5", "rank_ret1",
            # Market quality (2)
            "spread_scaled",    # NEW: spread / vol20
            "price_impact",
            # Intraday cumulative (4)
            "cum_ret1", "cum_volume", "cum_imb", "cum_ob_imb0",
            # Time (2)
            "sin_time", "cos_time",
        ]

    def __init__(self, cacheDir):
        self.cacheDir = cacheDir
        self.ycol = "fret12"
        self.mcols = ["symbol", "date", "interval"]

    def genFeatures(self, df):
        n_feat = len(self.featureNames())
        # log.inf("Generating {} features from raw data...".format(n_feat))
        eps = PREPROCESSING_CONFIG.eps

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

        # ---- Price returns ----
        g = df.groupby("symbol")
        for h in [1, 5, 10, 30]:
            df[f"ret{h}"] = g["midpx"].diff(h) / (g["midpx"].shift(h) + eps)

        # ---- Volatility family ----
        g = df.groupby("symbol")
        df["vol5"] = g["ret1"].transform(
            lambda x: x.rolling(5, min_periods=3).std())
        df["vol10"] = g["ret1"].transform(
            lambda x: x.rolling(10, min_periods=5).std())
        df["vol20"] = g["ret1"].transform(
            lambda x: x.rolling(20, min_periods=5).std())
        df["ret1_vol20"] = df["ret1"] / (df["vol20"] + eps)
        # NEW: vol regime — ratio > 1 means vol is increasing
        df["vol_ratio_5_20"] = df["vol5"] / (df["vol20"] + eps)
        # NEW: direction of last return (-1, 0, +1)
        df["ret1_sign"] = np.sign(df["ret1"])

        # ---- Spread & depth ----
        df["spread"] = (df["ask0"] - df["bid0"]) / (df["midpx"] + eps)
        df["spread4"] = (df["ask4"] - df["bid4"]) / (df["midpx"] + eps)
        df["depth_imb"] = np.log((df["asize0_4"] + eps) / (df["bsize0_4"] + eps))
        df["depth_conc"] = (df["bsize0"] + df["asize0"]) / (
            df["bsize0_4"] + df["asize0_4"] + eps)
        # NEW: total visible liquidity available at best bid+ask
        df["depth_total"] = np.log(df["bsize0"] + df["asize0"] + 1.0)

        # ---- Order book imbalance ----
        df["ob_imb0"] = (df["asize0"] - df["bsize0"]) / (df["asize0"] + df["bsize0"] + eps)
        df["ob_imb4"] = (df["asize0_4"] - df["bsize0_4"]) / (df["asize0_4"] + df["bsize0_4"] + eps)
        df["ob_imb9"] = (df["asize5_9"] - df["bsize5_9"]) / (df["asize5_9"] + df["bsize5_9"] + eps)
        df["ob_slope"] = df["ob_imb0"] - df["ob_imb9"]
        df["ob_curvature"] = df["ob_imb0"] - 2 * df["ob_imb4"] + df["ob_imb9"]

        # ---- Trade features ----
        df["trade_imb"] = (df["tradeBuyQty"] - df["tradeSellQty"]) / (
            df["tradeBuyQty"] + df["tradeSellQty"] + eps)
        df["trade_count_imb"] = (df["nTradeBuy"] - df["nTradeSell"]) / (
            df["nTradeBuy"] + df["nTradeSell"] + eps)
        df["log_trade_volume"] = np.log(df["tradeBuyQty"] + df["tradeSellQty"] + 1.0)
        buy_vwap = df["tradeBuyTurnover"] / (df["tradeBuyQty"] + eps)
        sell_vwap = df["tradeSellTurnover"] / (df["tradeSellQty"] + eps)
        df["vwap_dev"] = ((buy_vwap + sell_vwap) / 2.0 - df["midpx"]) / (df["midpx"] + eps)
        # NEW: volume relative to recent volatility (normalised trading intensity)
        df["volume_intensity"] = df["log_trade_volume"] / (df["vol20"] + eps)

        # ---- EMA families (multi-scale smoothing, after base columns created) ----
        g = df.groupby("symbol")
        for col, hl in [("ob_imb0", 10), ("ob_imb0", 30),
                        ("trade_imb", 5), ("trade_imb", 30),
                        ("spread", 5),
                        ("vwap_dev", 5)]:
            df[f"{col}_ema{hl}"] = g[col].transform(
                lambda x, h=hl: x.ewm(halflife=h, min_periods=1).mean())

        # ---- Cross-sectional ----
        for col in ["ret1", "ret10", "ob_imb0", "trade_imb", "vol20"]:
            mu = df.groupby("interval")[col].transform("mean")
            df[f"cx_{col}"] = df[col] - mu

        # ---- Percentile ranks (robust cross-sectional signal) ----
        for col in ["ob_imb0", "trade_imb", "ret1", "vol20"]:
            df[f"rank_{col}"] = df.groupby("interval")[col].transform(
                lambda x: x.rank(pct=True))

        # ---- Momentum / reversal ----
        g = df.groupby("symbol")
        df["bret12"] = g["midpx"].diff(12) / (g["midpx"].shift(12) + eps)
        cx_bret12 = df.groupby("interval")["bret12"].transform("mean")
        df["lagret12"] = df["bret12"] - cx_bret12
        df["mom_5_30"] = df["ret5"] * df["ret30"]
        df["ret5_ret1"] = df["ret5"] / (np.abs(df["ret1"]) + eps)
        df["ret30_ret5"] = df["ret30"] / (np.abs(df["ret5"]) + eps)

        # ---- Market quality ----
        df["price_impact"] = np.abs(df["ret1"]) / (df["log_trade_volume"] + eps)
        # NEW: bid-ask spread scaled by volatility (cost per unit risk)
        df["spread_scaled"] = df["spread"] / (df["vol20"] + eps)

        # ---- Time-of-day ----
        minute_of_day = (df["interval"] / 60_000).astype(int) % 1440
        df["sin_time"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
        df["cos_time"] = np.cos(2 * np.pi * minute_of_day / 1440.0)

        # ---- Intraday cumulative (per symbol, backward-looking) ----
        g = df.groupby("symbol")
        df["cum_ret1"] = g["ret1"].cumsum()
        df["cum_volume"] = (g["tradeBuyQty"].cumsum()
                            + g["tradeSellQty"].cumsum())
        df["cum_imb"] = g["trade_imb"].expanding().mean().reset_index(level=0, drop=True)
        df["cum_ob_imb0"] = g["ob_imb0"].expanding().mean().reset_index(level=0, drop=True)

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
