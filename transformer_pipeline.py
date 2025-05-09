import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm


def adf_test(series: pd.Series, name: str) -> None:
    """Augmented Dickey-Fuller test로 정상성 여부 확인"""
    result = adfuller(series.dropna())
    print(f"[{name}] ADF 통계량 = {result[0]:.4f}, p-value = {result[1]:.4f}")
    for k, v in result[4].items():
        print(f"  임계값({k}) = {v:.4f}")
    print()


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, q: float) -> torch.Tensor:
    """퀀타일 손실 함수"""
    e = target - pred
    return torch.max(q * e, (q - 1) * e).mean()


def direction_penalty(delta_pred: torch.Tensor, delta_true: torch.Tensor) -> torch.Tensor:
    """예측 방향과 실제 방향이 다를 때 페널티"""
    return torch.relu(-delta_pred * delta_true).mean()


class TimeSeriesDataset(Dataset):
    """시계열 데이터를 시퀀스로 변환하는 PyTorch Dataset"""
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 60):
        # NaN이 들어있지 않은 X, y를 전제
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx: int):
        x = self.X[idx : idx + self.seq_len]
        y = self.y[idx + self.seq_len - 1]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


class PositionalEncoding(nn.Module):
    """Transformer용 위치 인코딩 레이어"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class TimeSeriesMultiTransformer(nn.Module):
    """단기/장기 예측을 위한 Transformer 모델"""
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head_short = nn.Linear(model_dim, 2)
        self.head_long = nn.Linear(model_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_fc(x)
        h = self.pos_encoder(h)
        h = h.permute(1, 0, 2)
        h = self.transformer(h)
        h = h[-1]
        out_s = self.head_short(h)
        out_l = self.head_long(h)
        return torch.cat([out_s, out_l], dim=1)