"""
model.py

Shared module. Contains:
  - TimeSeriesDataset   : sliding window dataset from returns array
  - FactorTransformer   : encoder-only transformer for multivariate time series
  - get_device          : auto-selects mps / cuda / cpu
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
    Sliding window dataset over a (n_timesteps, n_stocks) returns array.

    Each sample:
        x : (context_len, n_stocks)  -- input window
        y : (horizon, n_stocks)      -- what to predict

    Train/val/test split is done by time (no shuffling across splits).
    Normalization uses train set mean/std only — no leakage into val/test.
    """

    def __init__(
        self,
        returns: np.ndarray,        # (n_timesteps, n_stocks)
        context_len: int = 60,
        horizon: int = 1,
        split: str = "train",       # "train" | "val" | "test"
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        mean: np.ndarray = None,    # if provided, use these stats (for val/test)
        std: np.ndarray = None,
    ):
        assert split in ("train", "val", "test")
        T, N = returns.shape

        # --- Time-based splits ---
        t_train_end = int(T * train_frac)
        t_val_end   = int(T * (train_frac + val_frac))

        if split == "train":
            data = returns[:t_train_end]
        elif split == "val":
            data = returns[t_train_end:t_val_end]
        else:
            data = returns[t_val_end:]

        # --- Normalise using train stats ---
        # If mean/std not passed in, compute from this slice (only correct for train)
        if mean is None or std is None:
            if split != "train":
                raise ValueError("Must pass mean/std for val and test splits to avoid leakage")
            mean = returns[:t_train_end].mean(axis=0, keepdims=True)  # (1, N)
            std  = returns[:t_train_end].std(axis=0, keepdims=True) + 1e-8

        self.mean = mean
        self.std  = std

        data_norm = (data - mean) / std  # normalised

        # --- Build sliding windows ---
        self.X = []
        self.Y = []
        window = context_len + horizon
        for t in range(len(data_norm) - window + 1):
            self.X.append(data_norm[t : t + context_len])
            self.Y.append(data_norm[t + context_len : t + context_len + horizon])

        self.X = np.array(self.X, dtype=np.float32)  # (n_samples, context_len, N)
        self.Y = np.array(self.Y, dtype=np.float32)  # (n_samples, horizon, N)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])


def make_splits(
    returns: np.ndarray,
    context_len: int = 60,
    horizon: int = 1,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
):
    """
    Convenience function: returns (train_ds, val_ds, test_ds, mean, std).
    Computes stats once on train, passes them to val and test.
    """
    T = returns.shape[0]
    t_train_end = int(T * train_frac)

    mean = returns[:t_train_end].mean(axis=0, keepdims=True)
    std  = returns[:t_train_end].std(axis=0, keepdims=True) + 1e-8

    train_ds = TimeSeriesDataset(returns, context_len, horizon, "train", train_frac, val_frac, mean, std)
    val_ds   = TimeSeriesDataset(returns, context_len, horizon, "val",   train_frac, val_frac, mean, std)
    test_ds  = TimeSeriesDataset(returns, context_len, horizon, "test",  train_frac, val_frac, mean, std)

    return train_ds, val_ds, test_ds, mean, std


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding from "Attention Is All You Need".
    Adds position information to the token embeddings.
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
        pe = pe.unsqueeze(0)                  # (1, max_len, d_model)
        self.register_buffer("pe", pe)        # not a parameter, but moves with .to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Model
# ---------------------------------------------------------------------------

class FactorTransformer(nn.Module):
    """
    Encoder-only transformer for multivariate time series forecasting.

    Input:  (batch, context_len, n_stocks)
    Output: (batch, horizon, n_stocks)

    Architecture:
        1. Linear input projection: n_stocks -> d_model
        2. Sinusoidal positional encoding
        3. N x TransformerEncoderLayer (self-attention + FFN + LayerNorm)
        4. Linear output head: d_model -> n_stocks * horizon
    """

    def __init__(
        self,
        n_stocks: int,
        context_len: int,
        horizon: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_stocks    = n_stocks
        self.context_len = context_len
        self.horizon     = horizon
        self.d_model     = d_model

        # 1. Project each timestep's cross-stock vector into d_model space
        self.input_proj = nn.Linear(n_stocks, d_model)

        # 2. Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=context_len + 10, dropout=dropout)

        # 3. Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,   # expects (batch, seq, features) — more intuitive
            norm_first=True,    # Pre-LN: more stable training than Post-LN
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
        # Project to embedding space
        x = self.input_proj(x)                     # (batch, context_len, d_model)

        # Add positional encoding
        x = self.pos_enc(x)                         # (batch, context_len, d_model)

        # Self-attention over the time dimension
        x = self.encoder(x)                         # (batch, context_len, d_model)

        # Use the last timestep as the summary representation
        x = x[:, -1, :]                             # (batch, d_model)

        # Project to output
        x = self.output_head(x)                     # (batch, n_stocks * horizon)
        x = x.view(-1, self.horizon, self.n_stocks) # (batch, horizon, n_stocks)

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
