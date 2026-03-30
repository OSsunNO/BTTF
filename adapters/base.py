# adapters/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple, Literal

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except Exception as e:
    raise ImportError("adapters require torch. Please install PyTorch.") from e


# @dataclass(frozen=True)
# class FitConfig:
#     """
#     Generic training config used by adapters.
#     """
#     epochs: int = 10
#     lr: float = 1e-3
#     weight_decay: float = 0.0
#     grad_clip: Optional[float] = None
#     amp: bool = False  # mixed precision
#     log_every: int = 5

@dataclass(frozen=True)
class FitConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None
    amp: bool = False
    log_every: int = 1
    patience: int = 3  # ← 추가. 0이면 Early Stopping 비활성화 (기존 동작 유지)
    lr_scheduler: str = "none"  # ← 추가. "none" | "type1"

def mse_torch(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_hat - y) ** 2)


class ForecastModelAdapter(ABC):
    """
    Base interface that ALL models should comply with (via adapters).

    The trainer/pipeline should only call these methods, not the raw model.
    """

    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        *,
        fit_cfg: FitConfig,
        save_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Train the model. Returns a dict of training logs/metrics.
        """

    @abstractmethod
    @torch.no_grad()
    def predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict for a batch input x: [B, L, C] -> y_hat: [B, P, C] (or compatible).
        """

    @torch.no_grad()
    def predict_loader(
        self,
        loader: DataLoader,
        *,
        return_y: bool = True,
        device: Optional[torch.device] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on a DataLoader.
        Assumes each batch yields either:
          - (x, y)
          - (x, y, extra...)  -> extra is ignored by default adapter
        Returns numpy arrays for easy saving (.npy).
        """
        self.eval()

        preds = []
        gts = []

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
                y = batch[1] if (len(batch) >= 2 and return_y) else None
            else:
                raise ValueError("Batch must be tuple/list like (x,y,...)")

            if device is not None:
                x = x.to(device)
                if y is not None:
                    y = y.to(device)

            y_hat = self.predict_batch(x)
            preds.append(y_hat.detach().cpu().numpy())

            if return_y and y is not None:
                gts.append(y.detach().cpu().numpy())

        out: Dict[str, np.ndarray] = {"pred": np.concatenate(preds, axis=0)}
        if return_y and len(gts) > 0:
            out["true"] = np.concatenate(gts, axis=0)
        return out

    @abstractmethod
    def train(self) -> None:
        ...

    @abstractmethod
    def eval(self) -> None:
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @abstractmethod
    def load(self, path: Path, *, map_location: Optional[str | torch.device] = None) -> None:
        ...
