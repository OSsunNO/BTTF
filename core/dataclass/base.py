# datasets/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Protocol, Optional
from torch.utils.data import Dataset, DataLoader


@dataclass(frozen=True)
class DataSpec:
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False


class DatasetBuilder(Protocol):
    def build_loaders(
        self,
        input_len: int,
        pred_len: int,
        *,
        data_spec: DataSpec,
        scale: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        ...
