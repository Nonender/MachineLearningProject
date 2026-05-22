import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from log import log


# ---------------------------------------------------------------------------
# RoPE (Rotary Positional Embedding)
# ---------------------------------------------------------------------------

def _causal_mask(t: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones((t, t), device=device, dtype=torch.bool), diagonal=1)


def precompute_rope_cache(
    head_dim: int, max_seq_len: int, *,
    base: float = 10000.0, device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires even head_dim")
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    )
    positions = torch.arange(max_seq_len, device=device, dtype=dtype)
    angles = torch.outer(positions, inv_freq)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("expected x with shape (B, H, T, Hd)")
    b, h, t, hd = x.shape
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    cos = cos[:t].unsqueeze(0).unsqueeze(0)
    sin = sin[:t].unsqueeze(0).unsqueeze(0)
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


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


class CausalSelfAttention(nn.Module):
    """Causal self-attention with RoPE, head-dim scaling, and dropout."""

    def __init__(self, *, model_dim: int, n_heads: int, dropout: float, context_len: int):
        super().__init__()
        if model_dim % n_heads != 0:
            raise ValueError("model_dim must be divisible by n_heads")
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires even head_dim")

        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=True)
        self.out = nn.Linear(model_dim, model_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

        cos, sin = precompute_rope_cache(
            self.head_dim, context_len,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _get_rope(self, t, device, dtype):
        cos, sin = self.rope_cos, self.rope_sin
        if cos.device != device:
            cos, sin = cos.to(device), sin.to(device)
        if cos.dtype != dtype:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos[:t], sin[:t]

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, d = x.shape
        if t > self.rope_cos.size(0):
            raise ValueError(f"seq len {t} exceeds context_len {self.rope_cos.size(0)}")

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._get_rope(t, x.device, x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # Head-dimension scaling constraint  (already canonical, kept explicit)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)

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
    """Pre-norm decoder block with RMSNorm + residual dropout."""

    def __init__(self, *, model_dim: int, n_heads: int, ffn_dim: int, dropout: float, context_len: int):
        super().__init__()
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.attn = CausalSelfAttention(
            model_dim=model_dim, n_heads=n_heads, dropout=dropout, context_len=context_len,
        )
        self.ffn = SwiGLUFFN(model_dim, ffn_dim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.resid_dropout(self.attn(self.norm1(x), padding_mask=padding_mask))
        x = x + self.resid_dropout(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

@dataclass()
class ModelConfig:
    n_features: int = 24
    context_len: int = 256
    n_layers: int = 8
    n_heads: int = 4
    model_dim: int = 256
    ffn_dim: int = 1024
    dropout: float = 0.1
    stock_emb_dim: int = 16
    input_scale: float = 0.5       # mild scaling to stabilise attention (not too aggressive)
    huber_delta: float = 1.0       # Huber loss threshold
    temperature: float = 1.0       # inference temperature


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

        # Stock embedding  (discrete → model_dim, same dimension as features)
        self.stock_emb = nn.Embedding(max_stocks, cfg.stock_emb_dim)
        self.stock_proj = nn.Linear(cfg.stock_emb_dim, cfg.model_dim, bias=False)

        self.feat_norm = RMSNorm(cfg.model_dim)
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
        s = self.stock_proj(self.stock_emb(stock_ids))           # (B, D)
        x = x + s.unsqueeze(1)                                   # broadcast over T
        x = self.feat_norm(x)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)

        x = self.norm_f(x)
        # sqrt(d_model) output scaling for stable logits
        return self.head(x) / math.sqrt(self.cfg.model_dim)


# ---------------------------------------------------------------------------
# MeowModel — preprocessing + training + inference
# ---------------------------------------------------------------------------

class MeowModel(object):
    def __init__(self, cacheDir):
        self.cfg = ModelConfig()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.inf("Using device: {}".format(self.device))

        self.model = RegressionTransformer(self.cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=5e-4, weight_decay=0.01,
        )
        self.batch_size = 128

        # Preprocessing state — fitted on first date
        self.y_std = None
        self.feat_p01 = None       # 1st percentile per feature
        self.feat_p99 = None       # 99th percentile per feature
        self.feat_log_mask = None  # bool mask: True → apply log1p
        self.feat_mean = None      # z-score mean
        self.feat_std = None       # z-score std
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
        batch_x = np.clip(batch_x, self.feat_p01, self.feat_p99)
        if self.feat_log_mask.any():
            batch_x[:, :, self.feat_log_mask] = np.log1p(
                batch_x[:, :, self.feat_log_mask])
        batch_x = (batch_x - self.feat_mean) / self.feat_std
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

        if self.feat_mean is not None:
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

    def partial_fit(self, xdf, ydf):
        """Train on one date's data."""
        if self.y_std is None:
            # ---- Fit preprocessing on first date ----
            all_y = self._clean(ydf.to_numpy().ravel())
            self.y_std = float(np.std(all_y)) or 1.0
            all_x = self._clean(xdf.to_numpy())

            # 1. Percentile clip points
            self.feat_p01 = np.percentile(all_x, 1, axis=0).astype(np.float32)
            self.feat_p99 = np.percentile(all_x, 99, axis=0).astype(np.float32)

            # 2. Identify positive long-tail features for log1p
            clipped = np.clip(all_x, self.feat_p01, self.feat_p99)
            feat_min = clipped.min(axis=0)
            # Skewness: use Fisher-Pearson coefficient
            mu = clipped.mean(axis=0)
            sigma = clipped.std(axis=0) + 1e-8
            skew = ((clipped - mu) ** 3).mean(axis=0) / (sigma ** 3)
            self.feat_log_mask = (feat_min >= 0) & (skew > 1.5)

            # Apply log1p to positive long-tail features
            if self.feat_log_mask.any():
                clipped[:, self.feat_log_mask] = np.log1p(clipped[:, self.feat_log_mask])

            # 3. Z-score statistics from transformed data
            self.feat_mean = clipped.mean(axis=0, keepdims=True).astype(np.float32)
            raw_std = clipped.std(axis=0)
            std_floor = max(float(np.median(raw_std)) * 0.01, 1e-4)
            self.feat_std = np.maximum(raw_std, std_floor).astype(np.float32)

            # Diagnostics
            feat_names = list(xdf.columns)
            log.inf(
                "Target std: {:.6f} | "
                "clip range: [{:.4f},{:.4f}] → [{:.4f},{:.4f}] | "
                "log1p features: {} | "
                "input_scale: {}".format(
                    self.y_std,
                    self.feat_p01.min(), self.feat_p01.max(),
                    self.feat_p99.min(), self.feat_p99.max(),
                    [feat_names[i] for i, m in enumerate(self.feat_log_mask) if m],
                    self.cfg.input_scale,
                ))
            actual_nf = xdf.shape[1]
            if actual_nf != self.cfg.n_features:
                raise RuntimeError(
                    "Feature count mismatch: data has {} features but model expects {}. "
                    "Update ModelConfig.n_features or MeowFeatureGenerator.featureNames().".format(
                        actual_nf, self.cfg.n_features))

            del all_x, clipped

        self.model.train()
        syms = self._symbols(xdf)

        for start in range(0, len(syms), self.batch_size):
            batch_syms = syms[start:start + self.batch_size]
            x, y, mask, sym_ids = self._prepare_batch(xdf, ydf, batch_syms)

            y = y / self.y_std

            pred = self.model(x, stock_ids=sym_ids, padding_mask=mask).squeeze(-1)
            valid = ~mask

            # Huber loss — robust to extreme values
            loss = F.huber_loss(
                pred[valid], y[valid], delta=self.cfg.huber_delta,
            )

            self.optimizer.zero_grad()
            loss.backward()
            # Global gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            del x, y, mask, sym_ids, pred, valid, loss

        if self.scheduler is not None:
            self.scheduler.step()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def set_scheduler(self, steps_per_epoch: int, n_epochs: int):
        t_max = steps_per_epoch * n_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=t_max, eta_min=1e-6,
        )
        log.inf("LR scheduler: CosineAnnealingLR with T_max={}".format(t_max))

    def predict(self, xdf):
        self.model.eval()
        syms = self._symbols(xdf)
        all_preds = []

        with torch.no_grad():
            for start in range(0, len(syms), self.batch_size):
                batch_syms = syms[start:start + self.batch_size]
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

                if self.feat_mean is not None:
                    batch_x = self._preprocess_x(batch_x)

                x = torch.from_numpy(batch_x).to(self.device)
                mask_t = torch.from_numpy(mask).to(self.device)
                sym_ids = torch.from_numpy(self._symbol_ids(batch_syms)).to(self.device)

                p = self.model(x, stock_ids=sym_ids, padding_mask=mask_t).squeeze(-1)
                p = p.cpu().numpy()
                del x, mask_t, sym_ids

                if self.y_std is not None:
                    p = p * self.y_std

                for i, L in enumerate(lengths):
                    all_preds.append(p[i, :L])

        return np.concatenate(all_preds)
