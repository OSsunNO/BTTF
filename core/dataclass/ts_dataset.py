# datasets/exchange_rate.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .base import DataSpec


class StandardScaler:
    """Simple sklearn-like scaler for numpy arrays."""
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        # x: [T, C]
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        return x * self.std + self.mean


class SequenceDataset(Dataset):
    """
    Generic (x,y) time-series dataset.

    data: [T, C]
    returns:
      x: [L, C]
      y: [P, C]
    """
    def __init__(self, data: np.ndarray, input_len: int, pred_len: int, target_col: Optional[int] = None):
        if data.ndim != 2:
            raise ValueError(f"data must be [T,C], got {data.shape}")
        if input_len <= 0 or pred_len <= 0:
            raise ValueError("input_len and pred_len must be positive")

        self.data = data.astype(np.float32)
        self.L = int(input_len)
        self.P = int(pred_len)

        self.T, self.C = self.data.shape
        self.n = self.T - (self.L + self.P) + 1
        if self.n <= 0:
            raise ValueError(
                f"Not enough length T={self.T} for L={self.L}, P={self.P}"
            )
        self.target_col = target_col  # ← 저장

    def __len__(self) -> int:
        return self.n

    # def __getitem__(self, idx: int):
    #     st = idx
    #     x = self.data[st : st + self.L]                 # [L,C]
    #     y = self.data[st + self.L : st + self.L + self.P]  # [P,C]
    #     return torch.from_numpy(x), torch.from_numpy(y)
    

    def __getitem__(self, idx):
        st = idx
        x = self.data[st : st + self.L]                        # [L, C]
        y_full = self.data[st + self.L : st + self.L + self.P] # [P, C]

        if self.target_col is not None:
            x = x[:, self.target_col : self.target_col + 1]      # [L, 1]  ← 추가
            y = y_full[:, self.target_col : self.target_col + 1]  # [P, 1]
        else:
            y = y_full  # [P, C]

        return torch.from_numpy(x), torch.from_numpy(y)



# @dataclass(frozen=True)
# class TSDatasetConfig:
#     """
#     path: CSV file path. The file is expected to have a time column + feature columns,
#           or just feature columns. We will drop non-numeric columns automatically.
#     """
#     path: str | Path
#     train_ratio: float = 0.7
#     val_ratio: float = 0.1
#     test_ratio: float = 0.2
#     # If you want strict reproducibility, keep shuffle=False for time series.
#     shuffle_train: bool = True
#     target_col: Optional[int] = None  # ← 추가. None이면 다변량(기존), int면 단변량


@dataclass(frozen=True)
class TSDatasetConfig:
    path: str | Path
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    shuffle_train: bool = True
    target_col: Optional[int] = None
    train_border: Optional[int] = None 
    val_border: Optional[int] = None   
    test_border: Optional[int] = None  

class TSDatasetBuilder:
    """
    Build DataLoaders for time series dataset.

    - loads csv
    - numeric columns only
    - splits by time: train/val/test
    - fits scaler on train ONLY
    - returns SequenceDataset loaders (x: [L,C], y:[P,C])
    """
    def __init__(self, cfg: TSDatasetConfig):
        self.cfg = cfg
        self.scaler: Optional[StandardScaler] = None

    def load_raw(self) -> np.ndarray:
        path = Path(self.cfg.path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        df = pd.read_csv(path)
        # keep numeric columns only (drops date/time strings)
        df_num = df.select_dtypes(include=[np.number])
        if df_num.shape[1] == 0:
            raise ValueError("No numeric columns found in CSV.")

        data = df_num.to_numpy()  # [T, C]
        return data


    def split(self, data: np.ndarray, input_len: int = 0):
        T = data.shape[0]

        if self.cfg.train_border is not None:
            tr_end = self.cfg.train_border
            va_end = self.cfg.val_border
        else:
            tr_end = int(T * self.cfg.train_ratio)
            va_end = tr_end + int(T * self.cfg.val_ratio)

        tr = data[:tr_end]
        va = data[max(0, tr_end - input_len):va_end]
        te_end = self.cfg.test_border if self.cfg.test_border is not None else T
        te = data[max(0, va_end - input_len):te_end]
        if len(te) == 0:
            raise ValueError("Test split is empty. Check split ratios.")
        return tr, va, te

    def build_loaders(
        self,
        input_len: int,
        pred_len: int,
        *,
        data_spec: DataSpec,
        scale: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        raw = self.load_raw()
        tr_raw, va_raw, te_raw = self.split(raw, input_len=input_len)

        if scale:
            self.scaler = StandardScaler().fit(tr_raw)
            tr = self.scaler.transform(tr_raw)
            va = self.scaler.transform(va_raw)
            te = self.scaler.transform(te_raw)
        else:
            self.scaler = None
            tr, va, te = tr_raw, va_raw, te_raw

        train_ds = SequenceDataset(tr, input_len, pred_len, target_col=self.cfg.target_col)
        val_ds = SequenceDataset(va, input_len, pred_len, target_col=self.cfg.target_col)
        test_ds = SequenceDataset(te, input_len, pred_len, target_col=self.cfg.target_col)

        train_loader = DataLoader(
            train_ds,
            batch_size=data_spec.batch_size,
            shuffle=self.cfg.shuffle_train,
            num_workers=data_spec.num_workers,
            pin_memory=data_spec.pin_memory,
            drop_last=data_spec.drop_last,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=data_spec.batch_size,
            shuffle=False,
            num_workers=data_spec.num_workers,
            pin_memory=data_spec.pin_memory,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=data_spec.batch_size,
            shuffle=False,
            num_workers=data_spec.num_workers,
            pin_memory=data_spec.pin_memory,
            drop_last=False,
        )
        return train_loader, val_loader, test_loader
