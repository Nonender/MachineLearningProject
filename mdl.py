import gc
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from log import log
from parameters import MODEL_CONFIG, TRAINING_CONFIG, PREPROCESSING_CONFIG

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

def _causal_mask(t: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones((t, t), device=device, dtype=torch.bool), diagonal=1)

def _precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0):
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(theta) / dim))
    angles = position * div_term.unsqueeze(0)
    return angles.cos(), angles.sin()

def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = torch.repeat_interleave(cos, 2, dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.repeat_interleave(sin, 2, dim=-1).unsqueeze(0).unsqueeze(0)
    x_half = x.shape[-1] // 2
    x1, x2 = x[..., :x_half], x[..., x_half:]
    x_rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + x_rot * sin

class SelfAttention(nn.Module):
    def __init__(self, *, model_dim: int, n_heads: int, dropout: float,
                 context_len: int, attn_window: int = 64):
        super().__init__()
        if model_dim % n_heads != 0:
            raise ValueError("model_dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads
        self.context_len = context_len
        self.attn_window = attn_window

        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=True)
        self.out = nn.Linear(model_dim, model_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

        rope_cos, rope_sin = _precompute_rope_freqs(self.head_dim, context_len)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def forward(self, x: torch.Tensor,
                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, d = x.shape
        if t > self.context_len:
            raise ValueError(f"seq len {t} exceeds context_len {self.context_len}")

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        q = _apply_rotary_emb(q, self.rope_cos[:t], self.rope_sin[:t])
        k = _apply_rotary_emb(k, self.rope_cos[:t], self.rope_sin[:t])

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        mask = _causal_mask(t, x.device)
        if self.attn_window > 0:
            row = torch.arange(t, device=x.device).unsqueeze(1)
            col = torch.arange(t, device=x.device).unsqueeze(0)
            mask = mask | ((row - col) > self.attn_window)
        mask = mask.view(1, 1, t, t)
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
        self.w_gate = nn.Linear(d_model, d_ff, bias=True)
        self.w_up = nn.Linear(d_model, d_ff, bias=True)
        self.w_out = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.silu(self.w_gate(x)) * self.w_up(x))


class DecoderBlock(nn.Module):
    def __init__(self, *, model_dim: int, n_heads: int, ffn_dim: int, dropout: float,
                 context_len: int, attn_window: int = 64):
        super().__init__()
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.attn = SelfAttention(
            model_dim=model_dim, n_heads=n_heads, dropout=dropout,
            context_len=context_len, attn_window=attn_window)
        self.ffn = SwiGLUFFN(model_dim, ffn_dim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.resid_dropout(self.attn(self.norm1(x), padding_mask=padding_mask))
        x = x + self.resid_dropout(self.ffn(self.norm2(x)))
        return x

class CrossStockAttention(nn.Module):

    def __init__(self, model_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(model_dim)
        self.attn = nn.MultiheadAttention(
            model_dim, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = x.transpose(0, 1)
        key_pad = None
        if padding_mask is not None:
            key_pad = padding_mask.transpose(0, 1)
        x, _ = self.attn(x, x, x, key_padding_mask=key_pad)
        x = x.transpose(0, 1)
        return residual + self.dropout(x)


class SparseAttentionRegressor(nn.Module):
    def __init__(self, cfg, max_stocks: int = 500):
        super().__init__()
        self.cfg = cfg

        self.feat_gate = nn.Parameter(torch.ones(cfg.n_features))

        self.feat_proj = nn.Linear(cfg.n_features, cfg.model_dim, bias=True)
        self.stock_emb = nn.Linear(max_stocks, cfg.stock_emb_dim, bias=False)
        self.stock_proj = nn.Linear(cfg.stock_emb_dim, cfg.model_dim, bias=False)
        self.feat_norm = RMSNorm(cfg.model_dim)
        self.drop = nn.Dropout(cfg.dropout)

        self.cross_stock = None
        if cfg.cross_stock_attn:
            self.cross_stock = CrossStockAttention(
                cfg.model_dim, cfg.n_heads, cfg.dropout)

        self.blocks = nn.ModuleList([
            DecoderBlock(
                model_dim=cfg.model_dim, n_heads=cfg.n_heads,
                ffn_dim=cfg.ffn_dim, dropout=cfg.dropout,
                context_len=cfg.context_len, attn_window=cfg.attn_window)
            for _ in range(cfg.n_layers)
        ])

        self.norm_f = RMSNorm(cfg.model_dim)
        self.heads = nn.ModuleDict({
            h: nn.Linear(cfg.model_dim, 1, bias=True)
            for h in cfg.target_horizons})
        self.lin_skips = nn.ModuleDict({
            h: nn.Linear(cfg.n_features, 1, bias=True)
            for h in cfg.target_horizons})
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x_raw: torch.Tensor, stock_ids: torch.Tensor,
                padding_mask: torch.Tensor | None = None,
                horizon: str | None = None) -> dict | torch.Tensor:
        b, t, _ = x_raw.shape

        gated = x_raw * self.feat_gate.sigmoid()

        x = self.feat_proj(gated)
        stock_onehot = F.one_hot(
            stock_ids, num_classes=self.stock_emb.in_features).float()
        s = self.stock_proj(self.stock_emb(stock_onehot))
        x = x + s.unsqueeze(1)
        x = self.feat_norm(x)
        x = self.drop(x)

        if self.cross_stock is not None:
            x = self.cross_stock(x, padding_mask)

        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)

        x = self.norm_f(x)
        scale = self.log_scale.exp()

        if horizon is not None:
            return self.heads[horizon](x).squeeze(-1) * scale + \
                self.lin_skips[horizon](gated).squeeze(-1)
        return {h: self.heads[h](x).squeeze(-1) * scale +
                self.lin_skips[h](gated).squeeze(-1)
                for h in self.heads}

class MeowModel:
    def __init__(self, checkpoint_dir: str | None = None):
        self.cfg = MODEL_CONFIG
        self.tcfg = TRAINING_CONFIG
        self.pcfg = PREPROCESSING_CONFIG

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SparseAttentionRegressor(self.cfg).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.tcfg.lr,
            weight_decay=self.tcfg.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None
        self.scheduler = None
        self._sym_to_id: dict[str, int] = {}
        self.checkpoint_dir = checkpoint_dir
        self.global_step = 0
        self.best_val_metric = -float("inf")

    def _preprocess_x(self, batch_x: np.ndarray) -> np.ndarray:
        batch_x = np.clip(batch_x, self.pcfg.feat_p01, self.pcfg.feat_p99)
        if self.pcfg.feat_log_mask.any():
            batch_x[:, :, self.pcfg.feat_log_mask] = np.log1p(
                batch_x[:, :, self.pcfg.feat_log_mask])
        batch_x = (batch_x - self.pcfg.feat_mean) / self.pcfg.feat_std
        batch_x = batch_x * self.cfg.input_scale
        return batch_x

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
        n_horizons = len(self.cfg.target_horizons)
        seqs_x = [self._clean(self._sequence(xdf, s)) for s in syms]
        seqs_y = [self._clean(self._sequence(ydf, s)) for s in syms]
        max_ctx = self.cfg.context_len
        seqs_x = [s[-max_ctx:] if len(s) > max_ctx else s for s in seqs_x]
        seqs_y = [s[-max_ctx:] if len(s) > max_ctx else s for s in seqs_y]
        lengths = [len(s) for s in seqs_x]
        max_len = max(lengths)

        b, f = len(syms), seqs_x[0].shape[-1]
        batch_x = np.zeros((b, max_len, f), dtype=np.float32)
        batch_y = np.zeros((b, max_len, n_horizons), dtype=np.float32)
        mask = np.ones((b, max_len), dtype=bool)

        for i, (sx, sy, L) in enumerate(zip(seqs_x, seqs_y, lengths)):
            batch_x[i, :L] = sx
            batch_y[i, :L] = sy
            mask[i, :L] = False

        del seqs_x, seqs_y

        if self.pcfg.feat_mean is not None:
            batch_x = self._preprocess_x(batch_x)

        sym_ids = self._symbol_ids(syms)
        x_t = torch.from_numpy(batch_x).to(self.device)
        return (
            x_t,
            torch.from_numpy(batch_y).to(self.device),
            torch.from_numpy(mask).to(self.device),
            torch.from_numpy(sym_ids).to(self.device),
        )

    def fit_preprocessing(self, fit_chunks):
        """Compute preprocessing statistics incrementally from chunks.

        Args:
            fit_chunks: List of (xdf, ydf) tuples. Each is processed one at a time
                        to keep memory bounded at O(1 day), avoiding OOM.
        """
        if not fit_chunks:
            raise ValueError("fit_chunks must not be empty")

        # Verify feature count against config
        first_xdf, first_ydf = fit_chunks[0]
        actual_nf = first_xdf.shape[1]
        if actual_nf != self.cfg.n_features:
            raise RuntimeError(
                "Feature count mismatch: data has {} features but model "
                "expects {}. Update ModelConfig.n_features or "
                "MeowFeatureGenerator.feature_names().".format(
                    actual_nf, self.cfg.n_features))

        feat_names = list(first_xdf.columns)
        self.pcfg.vol20_idx = feat_names.index("vol20")
        n_features = actual_nf

        # ── Pass 1: collect samples for percentiles + track per-feature min ──
        sample_rows = []       # every Nth row for percentile estimation
        sample_vol20 = []      # vol20 samples for vol20_floor
        feat_min = np.full(n_features, np.inf, dtype=np.float64)
        n_total = 0
        sample_every = 5       # sample 1 out of every N rows

        log.inf("Preprocessing pass 1/2: streaming {} chunks for percentiles..."
                .format(len(fit_chunks)))

        for xdf, ydf in fit_chunks:
            x_arr = self._clean(xdf.to_numpy())
            n_total += x_arr.shape[0]

            # Track min per feature
            feat_min = np.minimum(feat_min, x_arr.min(axis=0))

            # Sample rows for percentile estimation
            sample_idx = slice(0, x_arr.shape[0], sample_every)
            sample_rows.append(x_arr[sample_idx])
            sample_vol20.append(np.abs(x_arr[sample_idx, self.pcfg.vol20_idx]))

            del x_arr; gc.collect()

        # ── Compute percentiles from samples ──
        sample_all = np.concatenate(sample_rows, axis=0)
        sample_vol20 = np.concatenate(sample_vol20)
        del sample_rows; gc.collect()

        log.inf("  Sampled {} rows from {} total for percentile estimation"
                .format(sample_all.shape[0], n_total))

        self.pcfg.vol20_floor = max(
            float(np.percentile(sample_vol20, 5)), 1e-8)
        del sample_vol20

        self.pcfg.y_std = 1.0
        self.pcfg.feat_p01 = np.percentile(
            sample_all, 1, axis=0).astype(np.float32)
        self.pcfg.feat_p99 = np.percentile(
            sample_all, 99, axis=0).astype(np.float32)

        # Compute skewness from sample to determine log_mask
        clipped_sample = np.clip(
            sample_all, self.pcfg.feat_p01, self.pcfg.feat_p99)
        mu_sample = clipped_sample.mean(axis=0)
        sigma_sample = clipped_sample.std(axis=0) + 1e-8
        skew_sample = ((clipped_sample - mu_sample) ** 3).mean(axis=0) / (sigma_sample ** 3)
        self.pcfg.feat_log_mask = (feat_min >= 0) & (skew_sample > 1.5)

        del sample_all, clipped_sample, mu_sample, sigma_sample, skew_sample
        gc.collect()

        # ── Pass 2: clip, log1p, compute running mean & variance (Welford) ──
        log.inf("Preprocessing pass 2/2: streaming for mean/std after clipping...")
        n = 0
        mean = np.zeros(n_features, dtype=np.float64)
        M2 = np.zeros(n_features, dtype=np.float64)

        for xdf, ydf in fit_chunks:
            x_arr = self._clean(xdf.to_numpy())

            # Clip
            x_arr = np.clip(x_arr, self.pcfg.feat_p01, self.pcfg.feat_p99)

            # log1p for skewed non-negative features
            if self.pcfg.feat_log_mask.any():
                x_arr[:, self.pcfg.feat_log_mask] = np.log1p(
                    x_arr[:, self.pcfg.feat_log_mask])

            # Welford's online algorithm for mean & variance
            for row in x_arr:
                n += 1
                delta = row - mean
                mean += delta / n
                delta2 = row - mean
                M2 += delta * delta2

            del x_arr; gc.collect()

        self.pcfg.feat_mean = mean.reshape(1, -1).astype(np.float32)
        raw_std = np.sqrt(M2 / max(n - 1, 1))
        std_floor = max(float(np.median(raw_std)) * 0.01, 1e-4)
        self.pcfg.feat_std = np.maximum(raw_std, std_floor).astype(np.float32)

        # ── Compute vol20-related stats ──
        # Re-process first chunk for vol_normed_std (only need one day for this)
        first_x, first_y = fit_chunks[0]
        first_x_arr = self._clean(first_x.to_numpy())
        first_y_arr = self._clean(first_y["fret12"].to_numpy().ravel())
        vol20_vals = np.abs(first_x_arr[:, self.pcfg.vol20_idx])
        vol20_safe = np.clip(vol20_vals, self.pcfg.vol20_floor, None) + 1e-8
        vol_normed_std = float(np.std(first_y_arr / vol20_safe))
        del first_x_arr, first_y_arr, vol20_vals, vol20_safe

        log.inf("Preprocessing fitted on {} rows | "
                "clip range: [{:.4f}, {:.4f}] → [{:.4f}, {:.4f}] | "
                "log1p features: {} | input_scale: {}".format(
                    n_total,
                    self.pcfg.feat_p01.min(), self.pcfg.feat_p01.max(),
                    self.pcfg.feat_p99.min(), self.pcfg.feat_p99.max(),
                    [feat_names[i] for i, m in enumerate(self.pcfg.feat_log_mask) if m],
                    self.cfg.input_scale))
        log.inf("vol20 index: {} | p05 floor: {:.6f} | "
                "std(return/vol20_clipped) = {:.4f}".format(
                    self.pcfg.vol20_idx, self.pcfg.vol20_floor, vol_normed_std))

    def partial_fit(self, xdf, ydf):
        if self.pcfg.y_std is None:
            raise RuntimeError(
                "Preprocessing not fitted. Call fit_preprocessing() first.")

        self.model.train()
        syms = self._symbols(xdf)
        horizons = list(self.cfg.target_horizons)

        if self.pcfg.vol20_idx is not None:
            vol20_arr = np.abs(self._clean(xdf["vol20"].to_numpy().ravel()))
            vol20_arr = np.clip(vol20_arr, self.pcfg.vol20_floor, None) + 1e-8
            ydf = ydf.copy()
            for h in horizons:
                ydf[h] = ydf[h].to_numpy().ravel() / vol20_arr

        train_loss = 0.0
        n_batches = 0
        for start in range(0, len(syms), self.tcfg.batch_size):
            n_batches += 1
            batch_syms = syms[start:start + self.tcfg.batch_size]
            x, y, mask, sym_ids = self._prepare_batch(xdf, ydf, batch_syms)
            y = y / self.pcfg.y_std

            with torch.amp.autocast("cuda"):
                preds = self.model(
                    x, stock_ids=sym_ids, padding_mask=mask)
                valid = ~mask

                total_loss = 0.0
                for hi, h in enumerate(horizons):
                    p_v = preds[h][valid]
                    y_v = y[:, :, hi][valid]
                    y_var = y_v.var() + 1e-8
                    loss_mse = F.mse_loss(p_v, y_v) / y_var
                    p_c = p_v - p_v.mean()
                    y_c = y_v - y_v.mean()
                    cov = (p_c * y_c).mean()
                    loss_corr = -cov / ((p_v.std() + 1e-8) * (y_v.std() + 1e-8))
                    total_loss = total_loss + 3.0 * loss_mse + 0.2 * loss_corr

                gate_penalty = 5e-4 * self.model.feat_gate.sigmoid().sum()
                loss = total_loss + gate_penalty

            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.tcfg.grad_clip)
                self.optimizer.step()

            train_loss += float(loss.detach().cpu().item())
            del x, y, mask, sym_ids, preds, valid, loss

        if self.scheduler is not None:
            self.scheduler.step()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return train_loss / max(n_batches, 1)

    def set_scheduler(self, steps_per_epoch: int, n_epochs: int):
        t_max = steps_per_epoch * n_epochs
        warmup_steps = int(t_max * self.tcfg.warmup_ratio)
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=t_max - warmup_steps,
            eta_min=self.tcfg.lr_scheduler_eta_min)
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine],
            milestones=[warmup_steps])
        log.inf("LR scheduler: warmup {} steps + CosineAnnealing, "
                "T_max={}".format(warmup_steps, t_max))

    def save_checkpoint(self, path: str | None = None):
        """Save model, optimizer, scheduler, and preprocessing state."""
        if path is None:
            if self.checkpoint_dir is None:
                log.yellow("No checkpoint_dir set, skipping save")
                return
            path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "sym_to_id": self._sym_to_id,
            "global_step": self.global_step,
            "best_val_metric": self.best_val_metric,
            "pcfg": {k: v for k, v in vars(self.pcfg).items()
                     if not k.startswith("_")},
        }
        torch.save(state, path)
        log.inf("Checkpoint saved to {}".format(path))

    def load_checkpoint(self, path: str):
        """Load model, optimizer, scheduler, and preprocessing state."""
        if not os.path.exists(path):
            raise FileNotFoundError("Checkpoint not found: {}".format(path))
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler is not None and state.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self._sym_to_id = state.get("sym_to_id", {})
        self.global_step = state.get("global_step", 0)
        self.best_val_metric = state.get("best_val_metric", -float("inf"))
        for k, v in state.get("pcfg", {}).items():
            if hasattr(self.pcfg, k):
                setattr(self.pcfg, k, v)
        log.inf("Checkpoint loaded from {} (global_step={})".format(
            path, self.global_step))

    def predict(self, xdf, denormalize=True, horizon="fret12"):
        self.model.eval()
        syms = self._symbols(xdf)
        all_preds = []

        with torch.no_grad():
            for start in range(0, len(syms), self.tcfg.batch_size):
                batch_syms = syms[start:start + self.tcfg.batch_size]
                seqs_x = [self._clean(self._sequence(xdf, s))
                          for s in batch_syms]
                lengths = [len(s) for s in seqs_x]
                max_len = max(lengths)

                b, f = len(batch_syms), seqs_x[0].shape[-1]
                batch_x = np.zeros((b, max_len, f), dtype=np.float32)
                mask = np.ones((b, max_len), dtype=bool)
                for i, (sx, L) in enumerate(zip(seqs_x, lengths)):
                    batch_x[i, :L] = sx
                    mask[i, :L] = False
                del seqs_x

                vol20_raw = None
                if self.pcfg.vol20_idx is not None:
                    vol20_vals = np.abs(batch_x[:, :, self.pcfg.vol20_idx])
                    vol20_raw = np.clip(
                        vol20_vals, self.pcfg.vol20_floor, None) + 1e-8

                if self.pcfg.feat_mean is not None:
                    batch_x = self._preprocess_x(batch_x)

                x = torch.from_numpy(batch_x).to(self.device)
                mask_t = torch.from_numpy(mask).to(self.device)
                sym_ids = torch.from_numpy(
                    self._symbol_ids(batch_syms)).to(self.device)

                with torch.amp.autocast("cuda"):
                    p = self.model(
                        x, stock_ids=sym_ids, padding_mask=mask_t,
                        horizon=horizon)
                p = p.float().cpu().numpy()
                del x, mask_t, sym_ids

                if denormalize:
                    if self.pcfg.y_std is not None:
                        p = p * self.pcfg.y_std
                    if vol20_raw is not None:
                        p = p * vol20_raw

                for i, L in enumerate(lengths):
                    all_preds.append(p[i, :L])

        return np.concatenate(all_preds)