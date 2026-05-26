"""
parameters.py — Centralized hyperparameter management for the Meow project.

All hyperparameters MUST be defined here. Other files import from this module only.
Each field is annotated with whether it changes during training.

Convention:
  [FIXED]   — Immutable after construction; changing requires rebuilding the model
  [DYNAMIC] — Modified during training (by scheduler, data fitting, etc.)
  [FITTED]  — Learned from data on the first training date; NEVER set manually
"""

from dataclasses import dataclass, field


# ============================================================================
# Model architecture — all [FIXED] after model instantiation
# ============================================================================

@dataclass
class ModelConfig:
    """Transformer architecture hyperparameters.

    Changing any of these requires re-creating the RegressionTransformer.
    """

    # --- Input / output ---
    n_features: int = 52
    """[FIXED] Number of input features. Must match MeowFeatureGenerator.featureNames()."""

    context_len: int = 256
    """[FIXED] Maximum sequence length (time steps). Sequences longer than this raise an error."""

    # --- Transformer dimensions ---
    n_layers: int = 3
    """[FIXED] Number of DecoderBlock layers. More layers = more capacity, more memory."""

    n_heads: int = 16
    """[FIXED] Number of attention heads. model_dim must be divisible by n_heads."""

    model_dim: int = 512
    """[FIXED] Hidden dimension throughout the Transformer. head_dim = model_dim / n_heads."""

    ffn_dim: int = 1408
    """[FIXED] Hidden dimension of the SwiGLU feed-forward network."""

    # --- Regularisation ---
    dropout: float = 0.2
    """[FIXED] Dropout rate applied to attention weights and residual connections."""

    # --- Embeddings ---
    stock_emb_dim: int = 128
    """[FIXED] Dimension of the per-stock embedding lookup (projected to model_dim afterwards)."""

    # --- Scaling ---
    input_scale: float = 1.0
    """[FIXED] Multiplier applied to preprocessed features before feeding into the model."""

    output_scale: float = 12.0
    """[FIXED] tanh output clamp range. Set > 2× std(return/vol20) to avoid saturation."""

    temperature: float = 1.0
    """[FIXED] Inference temperature for prediction scaling (reserved for future use)."""

    # --- Loss ---
    huber_delta: float = 0.1
    """[FIXED] Huber loss threshold — quadratic below this, linear above."""


# ============================================================================
# Training configuration — some [DYNAMIC], most fixed per run
# ============================================================================

@dataclass
class TrainingConfig:
    """Training loop hyperparameters.

    Most can be changed between runs without rebuilding the model.
    """

    # --- Optimizer ---
    lr: float = 3e-4
    """[DYNAMIC] Initial learning rate for AdamW. Modified every step by the LR scheduler."""

    weight_decay: float = 0.1
    """[FIXED] Weight decay (L2 regularisation) for AdamW. Constant throughout training."""

    grad_clip: float = 1.0
    """[FIXED] Maximum gradient norm for clipping. Prevents exploding gradients."""

    # --- Batch ---
    batch_size: int = 256
    """[FIXED] Number of stocks per batch. Reduce if OOM."""

    # --- Schedule ---
    n_epochs: int = 4
    """[FIXED] Number of full passes over the date range."""

    warmup_ratio: float = 0.1
    """[FIXED] Fraction of total steps used for linear LR warmup (0.1 = 10%)."""

    lr_scheduler_t_max: int | None = None
    """[FIXED] T_max for CosineAnnealingLR. Computed as steps_per_epoch * n_epochs if None."""

    lr_scheduler_eta_min: float = 1e-5
    """[FIXED] Minimum learning rate the cosine scheduler decays to."""


# ============================================================================
# Preprocessing configuration — [FITTED] fields change at first training date
# ============================================================================

@dataclass
class PreprocessingConfig:
    """Feature preprocessing hyperparameters and fitted state.

    Fields marked [FITTED] are computed from data by fit_preprocessing().
    Do NOT set them manually — they will be overwritten.
    """

    # --- Fitting strategy ---
    preprocessing_fit_days: int = 10
    """[FIXED] Number of initial training dates pooled to fit preprocessing parameters."""

    # --- Numerical stability ---
    eps: float = 1e-8
    """[FIXED] Small constant for division safety in feature formulas (feat.py)."""

    # --- Percentile clipping ---
    feat_p01: float | None = None
    """[FITTED] 1st percentile per feature — computed from data. Values below are clipped."""

    feat_p99: float | None = None
    """[FITTED] 99th percentile per feature — computed from data. Values above are clipped."""

    # --- Log transform ---
    feat_log_mask: list[bool] | None = None
    """[FITTED] Boolean mask per feature: True → apply log1p transform (features with skew > 3)."""

    # --- Z-score normalisation ---
    feat_mean: float | None = None
    """[FITTED] Mean per feature, computed from data. Used for z-score normalisation."""

    feat_std: float | None = None
    """[FITTED] Standard deviation per feature, computed from data. Set to 1.0 if computed std is 0."""

    # --- Target normalisation ---
    y_std: float | None = None
    """[FITTED] Standard deviation of the target variable. Computed from first date's y data."""

    # --- Derived indices / noise filters (set at fit time) ---
    vol20_idx: int | None = None
    """[FITTED] Column index of the 'vol20' feature, detected from feature names during fit."""

    vol20_floor: float = 1e-8
    """[FITTED] 5th percentile of vol20. vol20 values below this are clipped to prevent y explosion."""

    max_stocks: int = 500
    """[FIXED] Maximum number of unique stock symbols for the embedding table."""


# ============================================================================
# Default singleton instances — import these, don't instantiate your own
# ============================================================================

MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
PREPROCESSING_CONFIG = PreprocessingConfig()
