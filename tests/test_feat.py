"""Tests for MeowFeatureGenerator — run with: python tests/test_feat.py"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feat import MeowFeatureGenerator


def make_mock_raw(n_symbols=3, n_ticks=40):
    np.random.seed(42)
    rows = []
    base_price = 10.0
    for si in range(n_symbols):
        price = base_price + si * 2.0
        for ti in range(n_ticks):
            shock = np.random.normal(0, 0.01)
            price = price * (1 + shock)
            mid = max(price, 0.01)
            spread = mid * np.random.uniform(0.0001, 0.002)
            interval = 20230601 * 1_000_000 + ti * 60_000
            rows.append({
                "symbol": f"STOCK_{si:03d}",
                "interval": interval,
                "bid0": mid - spread / 2,
                "ask0": mid + spread / 2,
                "bid4": mid - spread,
                "ask4": mid + spread,
                "bsize0": np.random.randint(100, 5000),
                "asize0": np.random.randint(100, 5000),
                "bsize0_4": np.random.randint(500, 20000),
                "asize0_4": np.random.randint(500, 20000),
                "bsize5_9": np.random.randint(200, 10000),
                "asize5_9": np.random.randint(200, 10000),
                "tradeBuyQty": np.random.randint(0, 2000),
                "tradeSellQty": np.random.randint(0, 2000),
                "tradeBuyTurnover": np.random.uniform(0, 50000),
                "tradeSellTurnover": np.random.uniform(0, 50000),
                "nTradeBuy": np.random.randint(0, 20),
                "nTradeSell": np.random.randint(0, 20),
                "date": 20230601,
            })
    df = pd.DataFrame(rows)
    df["midpx"] = (df["bid0"] + df["ask0"]) / 2.0
    return df


def test_feature_count():
    gen = MeowFeatureGenerator(cache_dir=None)
    assert len(gen.feature_names()) == 94


def test_gen_features_shape():
    gen = MeowFeatureGenerator(cache_dir=None)
    df = make_mock_raw(n_symbols=3, n_ticks=40)
    xdf, ydf = gen.gen_features(df)
    assert xdf.shape[1] == 94, f"Expected 94 features, got {xdf.shape[1]}"
    assert list(ydf.columns) == gen.target_horizons
    syms = xdf.index.get_level_values("symbol").unique()
    assert len(syms) == 3


def test_gen_features_no_inf():
    gen = MeowFeatureGenerator(cache_dir=None)
    df = make_mock_raw(n_symbols=2, n_ticks=30)
    xdf, ydf = gen.gen_features(df)
    assert not np.isinf(xdf.to_numpy()).any()
    assert not np.isinf(ydf.to_numpy()).any()


def test_gen_features_single_stock():
    gen = MeowFeatureGenerator(cache_dir=None)
    df = make_mock_raw(n_symbols=1, n_ticks=10)
    xdf, ydf = gen.gen_features(df)
    assert xdf.shape[0] == 10


def test_target_horizons_exist():
    gen = MeowFeatureGenerator(cache_dir=None)
    df = make_mock_raw(n_symbols=2, n_ticks=25)
    xdf, ydf = gen.gen_features(df)
    for h in gen.target_horizons:
        assert h in ydf.columns, f"Missing horizon {h}"
        valid_ratio = ydf[h].notna().mean()
        assert valid_ratio > 0.3, f"Horizon {h} only {valid_ratio:.1%} valid"


if __name__ == "__main__":
    for name, fn in list(locals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name}: PASSED")
    print("test_feat: ALL PASSED")
