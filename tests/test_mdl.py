"""Tests for model forward pass and checkpointing.

Run with: python tests/test_mdl.py
"""

import sys
import os
import tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameters import MODEL_CONFIG, PREPROCESSING_CONFIG
from mdl import SparseAttentionRegressor, MeowModel


def test_forward_shape_single_horizon():
    cfg = MODEL_CONFIG
    model = SparseAttentionRegressor(cfg)
    b, t, f = 4, 32, cfg.n_features
    x = torch.randn(b, t, f)
    stock_ids = torch.randint(0, 500, (b,))
    out = model(x, stock_ids, horizon="fret12")
    assert out.shape == (b, t), f"Expected (b,t), got {out.shape}"


def test_forward_shape_all_horizons():
    cfg = MODEL_CONFIG
    model = SparseAttentionRegressor(cfg)
    b, t, f = 4, 32, cfg.n_features
    x = torch.randn(b, t, f)
    stock_ids = torch.randint(0, 500, (b,))
    out = model(x, stock_ids)
    assert isinstance(out, dict)
    for h in cfg.target_horizons:
        assert h in out
        assert out[h].shape == (b, t)


def test_forward_with_padding():
    cfg = MODEL_CONFIG
    model = SparseAttentionRegressor(cfg)
    b, t, f = 4, 32, cfg.n_features
    x = torch.randn(b, t, f)
    stock_ids = torch.randint(0, 500, (b,))
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[:, -8:] = True
    out = model(x, stock_ids, padding_mask=mask, horizon="fret12")
    assert out.shape == (b, t)


def test_variable_length():
    cfg = MODEL_CONFIG
    model = SparseAttentionRegressor(cfg)
    b, t, f = 4, 64, cfg.n_features
    x = torch.zeros(b, t, f)
    lengths = [20, 35, 50, 64]
    for i, L in enumerate(lengths):
        x[i, :L] = torch.randn(L, f)
    mask = torch.ones(b, t, dtype=torch.bool)
    for i, L in enumerate(lengths):
        mask[i, :L] = False
    stock_ids = torch.randint(0, 500, (b,))
    out = model(x, stock_ids, padding_mask=mask)
    for h in cfg.target_horizons:
        assert out[h].shape == (b, t)


def test_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        PREPROCESSING_CONFIG.feat_mean = np.zeros((1, 94), dtype=np.float32)
        PREPROCESSING_CONFIG.feat_std = np.ones((1, 94), dtype=np.float32)
        PREPROCESSING_CONFIG.feat_p01 = np.full((94,), -1e6, dtype=np.float32)
        PREPROCESSING_CONFIG.feat_p99 = np.full((94,), 1e6, dtype=np.float32)
        PREPROCESSING_CONFIG.feat_log_mask = np.zeros(94, dtype=bool)
        PREPROCESSING_CONFIG.y_std = 1.0

        model1 = MeowModel()
        x = torch.randn(2, 16, 94).to(model1.device)
        stock_ids = torch.tensor([0, 1]).to(model1.device)
        model1.model.eval()
        with torch.no_grad():
            pred1 = model1.model(x, stock_ids, horizon="fret12").cpu().clone()

        ckpt_path = os.path.join(tmpdir, "test.pt")
        model1.save_checkpoint(ckpt_path)

        model2 = MeowModel()
        model2.load_checkpoint(ckpt_path)
        model2.model.eval()
        with torch.no_grad():
            pred2 = model2.model(x.to(model2.device),
                                 stock_ids.to(model2.device),
                                 horizon="fret12").cpu()

        assert torch.allclose(pred1, pred2, atol=1e-6), \
            "Predictions differ after round-trip"


if __name__ == "__main__":
    for name, fn in list(locals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name}: PASSED")
    print("test_mdl: ALL PASSED")
