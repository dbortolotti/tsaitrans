"""
model.py

Shared module for the prediction pipeline.

Key design change: stocks are treated as independent realizations of the same
data-generating process. Splits are across stocks, not time. The transformer
operates in univariate mode (n_stocks=1), processing one stock at a time.
All stocks in a split are pooled into one dataset.

Contains:
  - TimeSeriesDataset   : per-stock sliding windows (univariate)
  - FactorTransformer   : encoder-only transformer for time series
  - get_device          : auto-selects mps / cuda / cpu
  - make_splits         : stock-based train/val/test datasets
  - normalize_horizons  : config helper for horizon backward compat
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Config utility
# ---------------------------------------------------------------------------

def normalize_horizons(config: dict) -> list:
    """Convert legacy 'horizon' int to 'horizons' list for backward compat.

    If config has 'horizons' (list), return it sorted.
    If config has 'horizon' (int), return [1, 2, ..., horizon].
    Otherwise return [1].
    """
    if "horizons" in config:
        return sorted(config["horizons"])
    h = config.get("horizon", 1)
    return list(range(1, h + 1))


# ---------------------------------------------------------------------------
# Device utility
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """
    Sliding window dataset treating each stock as an independent realization.

    Given returns (T, N_total) and a list of stock indices, creates univariate
    sliding windows from each stock independently. All stocks in the split are
    pooled into one dataset.

    Each sample:
        x : (context_len, 1)       -- input window (single stock)
        y : (num_horizons, 1)      -- cumulative return targets at each horizon

    Targets are cumulative returns: y_h = sum(returns[t : t+h]) for each h
    in the horizons list.

    Normalization uses a single scalar mean/std computed from train stocks.
    """

    def __init__(
        self,
        returns: np.ndarray,        # (T, N_total)
        stock_indices: list,        # which stocks to include
        context_len: int = 60,
        horizons: list = None,      # e.g. [1, 2, 4, 8, 16]
        mean: float = None,
        std: float = None,
    ):
        if horizons is None:
            horizons = [1]
        self.horizons = sorted(horizons)
        self.num_horizons = len(self.horizons)
        max_horizon = max(self.horizons)

        data = returns[:, stock_indices]  # (T, n_split)

        # Compute or use provided normalization stats
        if mean is None or std is None:
            mean = float(data.mean())
            std = float(data.std()) + 1e-8

        self.mean = mean
        self.std = std

        data_norm = (data - mean) / std

        # Build sliding windows: one per stock per valid start position
        T, n_split = data_norm.shape
        window = context_len + max_horizon

        # Horizon indices for cumsum slicing (0-indexed)
        h_indices = np.array(self.horizons) - 1

        self.X = []
        self.Y = []
        for s in range(n_split):
            series = data_norm[:, s]  # (T,)
            for t in range(T - window + 1):
                self.X.append(series[t : t + context_len].reshape(-1, 1))
                # Cumulative returns at each horizon
                future = series[t + context_len : t + context_len + max_horizon]
                cum = np.cumsum(future)
                self.Y.append(cum[h_indices].reshape(-1, 1))

        self.X = np.array(self.X, dtype=np.float32)  # (n_samples, context_len, 1)
        self.Y = np.array(self.Y, dtype=np.float32)  # (n_samples, num_horizons, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])


def make_splits(
    returns: np.ndarray,
    stock_split: dict,
    context_len: int = 60,
    horizons: list = None,
):
    """
    Create train/val/test datasets from stock-based splits.

    stock_split keys used: 'transformer_train', 'transformer_val', 'test'.
    Normalization stats computed from transformer_train stocks only.

    Returns (train_ds, val_ds, test_ds, mean, std).
    """
    train_idx = stock_split["transformer_train"]
    val_idx = stock_split["transformer_val"]
    test_idx = stock_split["test"]

    # Compute stats from train stocks only
    train_data = returns[:, train_idx]
    mean = float(train_data.mean())
    std = float(train_data.std()) + 1e-8

    train_ds = TimeSeriesDataset(returns, train_idx, context_len, horizons=horizons, mean=mean, std=std)
    val_ds = TimeSeriesDataset(returns, val_idx, context_len, horizons=horizons, mean=mean, std=std)
    test_ds = TimeSeriesDataset(returns, test_idx, context_len, horizons=horizons, mean=mean, std=std)

    return train_ds, val_ds, test_ds, mean, std


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding from "Attention Is All You Need".
    Shape: (1, seq_len, d_model) — broadcast over batch.
    """

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Model
# ---------------------------------------------------------------------------

class FactorTransformer(nn.Module):
    """
    Encoder-only transformer for univariate time series forecasting.

    Input:  (batch, context_len, 1)
    Output:
        - deterministic: (batch, num_horizons, 1)
        - probabilistic: ((batch, num_horizons, 1), (batch, num_horizons, 1))  — (mu, log_sigma)

    Each output slot predicts the cumulative return to the corresponding horizon.

    Architecture:
        1. Linear input projection: 1 -> d_model
        2. Sinusoidal positional encoding
        3. N x TransformerEncoderLayer (self-attention + FFN + LayerNorm)
        4. Output head(s): d_model -> num_horizons (+ uncertainty head if probabilistic)
    """

    # TODO: Add optional EOD prediction head (mu_eod, log_var_eod)
    #       Target: y_eod = mid[close] - mid[t], requires time_to_close as input feature.

    def __init__(
        self,
        n_stocks: int = 1,         # kept for API compat, should be 1
        context_len: int = 60,
        horizons: list = None,     # e.g. [1, 2, 4, 8, 16]
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        probabilistic: bool = False,
    ):
        super().__init__()
        self.n_stocks = n_stocks
        self.context_len = context_len
        self.horizons = sorted(horizons) if horizons is not None else [1]
        self.num_horizons = len(self.horizons)
        self.d_model = d_model
        self.probabilistic = probabilistic

        # 1. Project each timestep's value into d_model space
        self.input_proj = nn.Linear(n_stocks, d_model)

        # 2. Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=context_len + 10, dropout=dropout
        )

        # 3. Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)

        # 4. Output head: mean prediction (one per horizon)
        self.output_head = nn.Linear(d_model, n_stocks * self.num_horizons)

        # 5. Uncertainty head (probabilistic mode only)
        if probabilistic:
            self.log_sigma_head = nn.Linear(d_model, n_stocks * self.num_horizons)

        self._init_weights()

    def _init_weights(self):
        """Xavier init for projection layers."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        if self.probabilistic:
            nn.init.xavier_uniform_(self.log_sigma_head.weight)
            # Init bias to small negative value so initial sigma ~ 0.5
            nn.init.constant_(self.log_sigma_head.bias, -0.5)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, context_len, n_stocks)

        Returns:
            deterministic: (batch, num_horizons, n_stocks)
            probabilistic: tuple of (mu, log_sigma), each (batch, num_horizons, n_stocks)
        """
        x = self.input_proj(x)  # (batch, context_len, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)  # (batch, context_len, d_model)
        x = x[:, -1, :]  # (batch, d_model)

        mu = self.output_head(x)  # (batch, n_stocks * num_horizons)
        mu = mu.view(-1, self.num_horizons, self.n_stocks)

        if self.probabilistic:
            log_sigma = self.log_sigma_head(x)
            log_sigma = log_sigma.view(-1, self.num_horizons, self.n_stocks)
            # Clamp for numerical stability
            log_sigma = torch.clamp(log_sigma, -4.0, 2.0)
            return mu, log_sigma

        return mu

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
