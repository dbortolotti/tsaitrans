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
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


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
        x : (context_len, 1)  -- input window (single stock)
        y : (horizon, 1)      -- target (single stock)

    Normalization uses a single scalar mean/std computed from train stocks.
    """

    def __init__(
        self,
        returns: np.ndarray,        # (T, N_total)
        stock_indices: list,        # which stocks to include
        context_len: int = 60,
        horizon: int = 1,
        mean: float = None,
        std: float = None,
    ):
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
        window = context_len + horizon

        self.X = []
        self.Y = []
        for s in range(n_split):
            series = data_norm[:, s : s + 1]  # (T, 1) — keep 2D
            for t in range(T - window + 1):
                self.X.append(series[t : t + context_len])
                self.Y.append(series[t + context_len : t + context_len + horizon])

        self.X = np.array(self.X, dtype=np.float32)  # (n_samples, context_len, 1)
        self.Y = np.array(self.Y, dtype=np.float32)  # (n_samples, horizon, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])


def make_splits(
    returns: np.ndarray,
    stock_split: dict,
    context_len: int = 60,
    horizon: int = 1,
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

    train_ds = TimeSeriesDataset(returns, train_idx, context_len, horizon, mean, std)
    val_ds = TimeSeriesDataset(returns, val_idx, context_len, horizon, mean, std)
    test_ds = TimeSeriesDataset(returns, test_idx, context_len, horizon, mean, std)

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
    Output: (batch, horizon, 1)

    Architecture:
        1. Linear input projection: 1 -> d_model
        2. Sinusoidal positional encoding
        3. N x TransformerEncoderLayer (self-attention + FFN + LayerNorm)
        4. Linear output head: d_model -> horizon
    """

    def __init__(
        self,
        n_stocks: int = 1,         # kept for API compat, should be 1
        context_len: int = 60,
        horizon: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_stocks = n_stocks
        self.context_len = context_len
        self.horizon = horizon
        self.d_model = d_model

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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 4. Output head: use only the last timestep's representation
        self.output_head = nn.Linear(d_model, n_stocks * horizon)

        self._init_weights()

    def _init_weights(self):
        """Xavier init for projection layers."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, context_len, n_stocks)
        returns: (batch, horizon, n_stocks)
        """
        x = self.input_proj(x)  # (batch, context_len, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)  # (batch, context_len, d_model)
        x = x[:, -1, :]  # (batch, d_model)
        x = self.output_head(x)  # (batch, n_stocks * horizon)
        x = x.view(-1, self.horizon, self.n_stocks)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
