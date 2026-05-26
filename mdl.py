import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from log import log
from parameters import (
    ModelConfig, MODEL_CONFIG, TRAINING_CONFIG, PREPROCESSING_CONFIG,
)
from eval import compute_metrics


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding  ("Attention Is All You Need")
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to input embeddings."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) — add positional encoding broadcast over batch."""
        return x + self.pe[:x.size(1)].unsqueeze(0)


# ---------------------------------------------------------------------------
# Core layers  (RMSNorm throughout — no BatchNorm / LayerNorm in blocks)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def _causal_mask(t: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular mask: position i can only attend to positions [0, i]."""
    return torch.triu(torch.ones((t, t), device=device, dtype=torch.bool), diagonal=1)


class SelfAttention(nn.Module):
    """Causal self-attention with sinusoidal PE and head-dim scaling.

    Causal mask prevents position t from attending to positions > t,
    avoiding look-ahead bias when predicting forward returns.
    """

    def __init__(self, *, model_dim: int, n_heads: int, dropout: float, context_len: int):
        super().__init__()
        if model_dim % n_heads != 0:
            raise ValueError("model_dim must be divisible by n_heads")
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads
        self.context_len = context_len

        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=True)
        self.out = nn.Linear(model_dim, model_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, d = x.shape
        if t > self.context_len:
            raise ValueError(f"seq len {t} exceeds context_len {self.context_len}")

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Causal + padding mask
        mask = _causal_mask(t, x.device).view(1, 1, t, t)
        if padding_mask is not None:
            mask = mask | padding_mask.view(b, 1, 1, t)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v
        return self.out(out.transpose(1, 2).contiguous().view(b, t, d))


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=True)
        self.W_up = nn.Linear(d_model, d_ff, bias=True)
        self.W_out = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.W_out(F.silu(self.W_gate(X)) * self.W_up(X))


class DecoderBlock(nn.Module):
    """Pre-norm Transformer block with RMSNorm + residual dropout (bidirectional)."""

    def __init__(self, *, model_dim: int, n_heads: int, ffn_dim: int, dropout: float, context_len: int):
        super().__init__()
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.attn = SelfAttention(
            model_dim=model_dim, n_heads=n_heads, dropout=dropout, context_len=context_len,
        )
        self.ffn = SwiGLUFFN(model_dim, ffn_dim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.resid_dropout(self.attn(self.norm1(x), padding_mask=padding_mask))
        x = x + self.resid_dropout(self.ffn(self.norm2(x)))
        return x


# ModelConfig is imported from parameters.py — see that file for all hyperparameter definitions.


# ---------------------------------------------------------------------------
# Regression Transformer
# ---------------------------------------------------------------------------

class RegressionTransformer(nn.Module):
    """Decoder-only Transformer for time-series regression.

    - Continuous features and discrete stock embedding projected to same dim.
    - RMSNorm throughout (no BatchNorm / LayerNorm in blocks).
    - sqrt(d_model) output scaling for stable logits.
    """

    def __init__(self, cfg: ModelConfig, max_stocks: int = 500):
        super().__init__()
        self.cfg = cfg

        # Feature projection  (continuous → model_dim)
        self.feat_proj = nn.Linear(cfg.n_features, cfg.model_dim, bias=True)

        # Stock embedding via Linear (one-hot → stock_emb_dim → model_dim)
        self.stock_emb = nn.Linear(max_stocks, cfg.stock_emb_dim, bias=False)
        self.stock_proj = nn.Linear(cfg.stock_emb_dim, cfg.model_dim, bias=False)

        self.feat_norm = RMSNorm(cfg.model_dim)
        self.pos_enc = PositionalEncoding(cfg.model_dim, cfg.context_len)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            DecoderBlock(
                model_dim=cfg.model_dim, n_heads=cfg.n_heads,
                ffn_dim=cfg.ffn_dim, dropout=cfg.dropout, context_len=cfg.context_len,
            )
            for _ in range(cfg.n_layers)
        ])

        self.norm_f = RMSNorm(cfg.model_dim)
        self.head = nn.Linear(cfg.model_dim, 1, bias=True)

    def forward(self, x: torch.Tensor, stock_ids: torch.Tensor,
                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        if t > self.cfg.context_len:
            raise ValueError(f"seq len {t} exceeds context_len {self.cfg.context_len}")

        # Continuous + discrete unified to same dimension
        x = self.feat_proj(x)                                    # (B, T, D)
        stock_onehot = F.one_hot(stock_ids, num_classes=self.stock_emb.in_features).float()
        s = self.stock_proj(self.stock_emb(stock_onehot))        # (B, D)
        x = x + s.unsqueeze(1)                                   # broadcast over T
        x = self.feat_norm(x)
        x = self.pos_enc(x)                                      # sinusoidal PE
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)

        x = self.norm_f(x)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# MeowModel — preprocessing + training + inference
# ---------------------------------------------------------------------------

class MeowModel(object):
    def __init__(self, cacheDir):
        self.cfg = MODEL_CONFIG
        self.tcfg = TRAINING_CONFIG
        self.pcfg = PREPROCESSING_CONFIG

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.inf("Using device: {}".format(self.device))

        self.model = RegressionTransformer(self.cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.tcfg.lr, weight_decay=self.tcfg.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None

        self.scheduler = None
        self._sym_to_id: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_x(self, batch_x: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline (in-place on copy-safe array).

        1. Clip to [p01, p99]
        2. log1p for positive long-tail features
        3. Z-score standardize
        4. Scale to avoid attention saturation
        """
        batch_x = np.clip(batch_x, self.pcfg.feat_p01, self.pcfg.feat_p99)
        if self.pcfg.feat_log_mask.any():
            batch_x[:, :, self.pcfg.feat_log_mask] = np.log1p(
                batch_x[:, :, self.pcfg.feat_log_mask])
        batch_x = (batch_x - self.pcfg.feat_mean) / self.pcfg.feat_std
        batch_x = batch_x * self.cfg.input_scale
        return batch_x

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _symbols(xdf):
        return xdf.index.get_level_values("symbol").unique()

    @staticmethod
    def _sequence(xdf_or_ydf, sym):
        return xdf_or_ydf.loc[sym].sort_index().to_numpy().astype(np.float32)

    @staticmethod
    def _clean(arr: np.ndarray) -> np.ndarray:
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _symbol_ids(self, syms) -> np.ndarray:
        ids = []
        for s in syms:
            if s not in self._sym_to_id:
                self._sym_to_id[s] = len(self._sym_to_id)
            ids.append(self._sym_to_id[s])
        return np.array(ids, dtype=np.int64)

    def _prepare_batch(self, xdf, ydf, syms):
        seqs_x = [self._clean(self._sequence(xdf, s)) for s in syms]
        seqs_y = [self._clean(self._sequence(ydf, s).ravel()) for s in syms]
        lengths = [len(s) for s in seqs_x]
        max_len = max(lengths)

        b, f = len(syms), seqs_x[0].shape[-1]
        batch_x = np.zeros((b, max_len, f), dtype=np.float32)
        batch_y = np.zeros((b, max_len), dtype=np.float32)
        mask = np.ones((b, max_len), dtype=bool)

        for i, (sx, sy, L) in enumerate(zip(seqs_x, seqs_y, lengths)):
            batch_x[i, :L] = sx
            batch_y[i, :L] = sy
            mask[i, :L] = False

        del seqs_x, seqs_y

        if self.pcfg.feat_mean is not None:
            batch_x = self._preprocess_x(batch_x)

        sym_ids = self._symbol_ids(syms)
        return (
            torch.from_numpy(batch_x).to(self.device),
            torch.from_numpy(batch_y).to(self.device),
            torch.from_numpy(mask).to(self.device),
            torch.from_numpy(sym_ids).to(self.device),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_preprocessing(self, xdf, ydf):
        """Fit all preprocessing parameters from pooled multi-date data.

        Must be called once before any partial_fit() calls.

        Normalisation strategy: target y is normalised by vol20 only (per-stock
        local volatility).  y_std is kept at 1.0 so that vol20-normalised
        returns are the direct training target.  This avoids scale explosion
        when vol20 is near zero for some stocks.
        """
        all_y = self._clean(ydf.to_numpy().ravel())
        all_x = self._clean(xdf.to_numpy())

        # Set vol20_idx and compute noise floor
        self.pcfg.vol20_idx = list(xdf.columns).index("vol20")
        vol20_vals = np.abs(all_x[:, self.pcfg.vol20_idx])
        self.pcfg.vol20_floor = max(
            float(np.percentile(vol20_vals, 5)), 1e-8)
        # Clip extreme low vol20 to prevent y explosion
        vol20_safe = np.clip(vol20_vals, self.pcfg.vol20_floor, None) + 1e-8
        # y_std = 1.0: vol20 normalisation alone is sufficient
        self.pcfg.y_std = 1.0
        # Log diagnostics
        vol_normed_std = float(np.std(all_y / vol20_safe))
        log.inf("vol20 index: {} | p05 floor: {:.6f} | std(return/vol20_clipped) = {:.4f}".format(
            self.pcfg.vol20_idx, self.pcfg.vol20_floor, vol_normed_std))

        # 1. Percentile clip points
        self.pcfg.feat_p01 = np.percentile(all_x, 1, axis=0).astype(np.float32)
        self.pcfg.feat_p99 = np.percentile(all_x, 99, axis=0).astype(np.float32)

        # 2. Identify positive long-tail features for log1p
        clipped = np.clip(all_x, self.pcfg.feat_p01, self.pcfg.feat_p99)
        feat_min = clipped.min(axis=0)
        mu = clipped.mean(axis=0)
        sigma = clipped.std(axis=0) + 1e-8
        skew = ((clipped - mu) ** 3).mean(axis=0) / (sigma ** 3)
        self.pcfg.feat_log_mask = (feat_min >= 0) & (skew > 1.5)

        # Apply log1p to positive long-tail features
        if self.pcfg.feat_log_mask.any():
            clipped[:, self.pcfg.feat_log_mask] = np.log1p(
                clipped[:, self.pcfg.feat_log_mask])

        # 3. Z-score statistics from transformed data
        self.pcfg.feat_mean = clipped.mean(axis=0, keepdims=True).astype(np.float32)
        raw_std = clipped.std(axis=0)
        std_floor = max(float(np.median(raw_std)) * 0.01, 1e-4)
        self.pcfg.feat_std = np.maximum(raw_std, std_floor).astype(np.float32)

        # Diagnostics
        feat_names = list(xdf.columns)
        log.inf(
            "Preprocessing fitted on {} rows | "
            "clip range: [{:.4f},{:.4f}] → [{:.4f},{:.4f}] | "
            "log1p features: {} | "
            "input_scale: {}".format(
                len(all_y),
                self.pcfg.feat_p01.min(), self.pcfg.feat_p01.max(),
                self.pcfg.feat_p99.min(), self.pcfg.feat_p99.max(),
                [feat_names[i] for i, m in enumerate(self.pcfg.feat_log_mask) if m],
                self.cfg.input_scale,
            ))
        actual_nf = xdf.shape[1]
        if actual_nf != self.cfg.n_features:
            raise RuntimeError(
                "Feature count mismatch: data has {} features but model expects {}. "
                "Update ModelConfig.n_features or MeowFeatureGenerator.featureNames().".format(
                    actual_nf, self.cfg.n_features))

        del all_x, clipped, vol20_vals, vol20_safe

    def partial_fit(self, xdf, ydf):
        """Train on one date's data.  fit_preprocessing() must be called first."""
        if self.pcfg.y_std is None:
            raise RuntimeError(
                "Preprocessing not fitted. Call fit_preprocessing() before partial_fit().")

        self.model.train()
        syms = self._symbols(xdf)

        # Normalise y by local vol (clip extreme lows to prevent explosion)
        if self.pcfg.vol20_idx is not None:
            vol20_arr = np.abs(self._clean(xdf["vol20"].to_numpy().ravel()))
            vol20_arr = np.clip(vol20_arr, self.pcfg.vol20_floor, None) + 1e-8
            ydf = ydf.copy()
            ydf.iloc[:, 0] = ydf.to_numpy().ravel() / vol20_arr

        all_preds, all_ys = [], []
        for start in range(0, len(syms), self.tcfg.batch_size):
            batch_syms = syms[start:start + self.tcfg.batch_size]
            x, y, mask, sym_ids = self._prepare_batch(xdf, ydf, batch_syms)

            y = y / self.pcfg.y_std

            with torch.amp.autocast('cuda'):
                pred = self.model(x, stock_ids=sym_ids, padding_mask=mask).squeeze(-1)
                valid = ~mask
                p_v, y_v = pred[valid], y[valid]

                # L1 regression loss
                loss_l1 = F.l1_loss(p_v, y_v)

                # Pairwise ranking loss (directly optimises Pearson)
                n_pairs = min(2000, p_v.numel() // 2)
                idx = torch.randint(0, p_v.numel(), (n_pairs * 2,), device=p_v.device)
                i, j = idx[:n_pairs], idx[n_pairs:]
                pred_diff = p_v[i] - p_v[j]
                target_diff = y_v[i] - y_v[j]
                loss_rank = F.relu(-torch.sign(target_diff) * pred_diff).mean()

                loss = loss_l1 + 0.1 * loss_rank

            # Collect pred/y on CPU before deletion
            all_preds.append(p_v.detach().float().cpu().numpy())
            all_ys.append(y_v.detach().float().cpu().numpy())

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip)
                self.optimizer.step()

            del x, y, mask, sym_ids, pred, valid, loss

        pcor, r2, mse = compute_metrics(
            np.concatenate(all_ys), np.concatenate(all_preds))
        del all_preds, all_ys

        if self.scheduler is not None:
            self.scheduler.step()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return pcor, r2, mse

    def set_scheduler(self, steps_per_epoch: int, n_epochs: int):
        t_max = steps_per_epoch * n_epochs
        warmup_steps = int(t_max * self.tcfg.warmup_ratio)
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=t_max - warmup_steps,
            eta_min=self.tcfg.lr_scheduler_eta_min)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
        log.inf("LR scheduler: warmup {} steps + CosineAnnealing, T_max={}".format(
            warmup_steps, t_max))

    def predict(self, xdf):
        self.model.eval()
        syms = self._symbols(xdf)
        all_preds = []

        with torch.no_grad():
            for start in range(0, len(syms), self.tcfg.batch_size):
                batch_syms = syms[start:start + self.tcfg.batch_size]
                seqs_x = [self._clean(self._sequence(xdf, s)) for s in batch_syms]
                lengths = [len(s) for s in seqs_x]
                max_len = max(lengths)

                b, f = len(batch_syms), seqs_x[0].shape[-1]
                batch_x = np.zeros((b, max_len, f), dtype=np.float32)
                mask = np.ones((b, max_len), dtype=bool)
                for i, (sx, L) in enumerate(zip(seqs_x, lengths)):
                    batch_x[i, :L] = sx
                    mask[i, :L] = False

                del seqs_x

                # Save raw vol20 before preprocessing (clipped to floor)
                vol20_raw = None
                if self.pcfg.vol20_idx is not None:
                    vol20_vals = np.abs(batch_x[:, :, self.pcfg.vol20_idx])
                    vol20_raw = np.clip(vol20_vals, self.pcfg.vol20_floor, None) + 1e-8

                if self.pcfg.feat_mean is not None:
                    batch_x = self._preprocess_x(batch_x)

                x = torch.from_numpy(batch_x).to(self.device)
                mask_t = torch.from_numpy(mask).to(self.device)
                sym_ids = torch.from_numpy(self._symbol_ids(batch_syms)).to(self.device)

                with torch.amp.autocast('cuda'):
                    p = self.model(x, stock_ids=sym_ids, padding_mask=mask_t).squeeze(-1)
                p = p.float().cpu().numpy()
                del x, mask_t, sym_ids

                # Denormalise: undo target scaling → undo vol normalisation
                if self.pcfg.y_std is not None:
                    p = p * self.pcfg.y_std
                if vol20_raw is not None:
                    p = p * vol20_raw

                for i, L in enumerate(lengths):
                    all_preds.append(p[i, :L])

        return np.concatenate(all_preds)
