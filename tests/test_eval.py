"""Tests for MeowEvaluator — run with: python tests/test_eval.py"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import MeowEvaluator


def test_perfect_prediction():
    e = MeowEvaluator()
    n = 100
    y = np.random.randn(n)
    ydf = pd.DataFrame({"fret12": y, "forecast": y})
    r, r2, mse = e.eval(ydf)
    assert abs(r - 1.0) < 1e-10
    assert abs(r2 - 1.0) < 1e-10
    assert abs(mse) < 1e-10


def test_anti_correlated():
    e = MeowEvaluator()
    n = 100
    y = np.random.randn(n)
    ydf = pd.DataFrame({"fret12": y, "forecast": -y})
    r, r2, mse = e.eval(ydf)
    assert abs(r - (-1.0)) < 1e-10
    assert r2 < 0


def test_constant_prediction():
    e = MeowEvaluator()
    n = 100
    y = np.random.randn(n)
    ydf = pd.DataFrame({"fret12": y, "forecast": np.full(n, y.mean())})
    r, r2, mse = e.eval(ydf)
    assert np.isnan(r) or abs(r) < 1e-10  # NaN when prediction has zero variance
    assert abs(r2) < 0.02  # ~1/n due to ddof=1 in var()


def test_output_types():
    e = MeowEvaluator()
    ydf = pd.DataFrame({
        "fret12": np.random.randn(50),
        "forecast": np.random.randn(50),
    })
    r, r2, mse = e.eval(ydf)
    assert isinstance(r, float)
    assert isinstance(r2, float)
    assert isinstance(mse, float)


def test_handles_inf():
    e = MeowEvaluator()
    y_true = np.array([1.0, 2.0, 3.0, np.inf, -np.inf, np.nan])
    y_pred = np.array([1.1, 1.9, 3.1, 0.0, 0.0, 0.0])
    ydf = pd.DataFrame({"fret12": y_true, "forecast": y_pred})
    r, r2, mse = e.eval(ydf)
    assert np.isfinite(r) and np.isfinite(r2) and np.isfinite(mse)


if __name__ == "__main__":
    for name, fn in list(locals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name}: PASSED")
    print("test_eval: ALL PASSED")
