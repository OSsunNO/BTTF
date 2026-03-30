# core/dataset_wrappers.py
from __future__ import annotations

from typing import Tuple, Union, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_tensor_f32(x: ArrayLike) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        # keep float32 for training stability/consistency
        return torch.from_numpy(x.astype(np.float32, copy=False))
    raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")


class AugmentedSequenceDataset(Dataset):
    """
    Stage2 dataset wrapper.

    Wrap a base (x, y) dataset and append per-sample PV segment to x.

    base_dataset[i] must return:
      x: [L, C]  (torch.Tensor)
      y: [P, C]  (torch.Tensor) or [P] (will be handled by trainer/adapter normalization)

    pv_seg_all:
      - shape: [N, S, C]
      - aligned with base_dataset indexing (idx)

    returns:
      x_aug: [L+S, C]
      y:     unchanged
    """

    def __init__(
        self,
        base_dataset: Dataset,
        pv_seg_all: ArrayLike,
        *,
        assert_len_match: bool = True,
        assert_channel_match: bool = True,
    ) -> None:
        self.base = base_dataset
        self.pv = _to_tensor_f32(pv_seg_all)  # [N,S,C]

        if self.pv.ndim != 3:
            raise ValueError(f"pv_seg_all must be [N,S,C], got {tuple(self.pv.shape)}")

        if assert_len_match and len(self.base) != self.pv.shape[0]:
            raise ValueError(
                f"Length mismatch: len(base)={len(self.base)} vs pv_seg_all.N={self.pv.shape[0]}. "
                "This usually means PV was generated/saved with shuffled order. "
                "PV saving must use shuffle=False and drop_last=False."
            )

        self._assert_channel_match = bool(assert_channel_match)
        if self._assert_channel_match:
            # try to check C once at init (cheap sanity check)
            x0, _ = self.base[0]
            if not isinstance(x0, torch.Tensor):
                raise TypeError("base_dataset must return torch.Tensor for x")
            if x0.ndim != 2:
                raise ValueError(f"base x must be [L,C], got {tuple(x0.shape)}")
            Cx = x0.shape[1]
            Cp = self.pv.shape[2]
            if Cx != Cp:
                raise ValueError(f"Channel mismatch: base C={Cx} vs pv C={Cp}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        pv_seg = self.pv[idx]  # [S,C]

        if not isinstance(x, torch.Tensor):
            raise TypeError("base_dataset must return torch.Tensor for x")
        if x.ndim != 2:
            raise ValueError(f"x must be [L,C], got {tuple(x.shape)}")
        if pv_seg.ndim != 2:
            raise ValueError(f"pv_seg must be [S,C], got {tuple(pv_seg.shape)}")

        if self._assert_channel_match and x.shape[1] != pv_seg.shape[1]:
            raise ValueError(f"Channel mismatch at idx={idx}: x.C={x.shape[1]} vs pv.C={pv_seg.shape[1]}")

        # concat on time axis: [L+S, C]
        x_aug = torch.cat([x, pv_seg], dim=0)
        return x_aug, y


def pvseg_from_segments(pv_segments: ArrayLike, seg_id: int) -> ArrayLike:
    """
    Utility: pick a single seg_id from pv_segments.

    pv_segments: [N, Nseg, S, C]
    returns:     [N, S, C]
    """
    if isinstance(pv_segments, torch.Tensor):
        if pv_segments.ndim != 4:
            raise ValueError(f"pv_segments must be [N,Nseg,S,C], got {tuple(pv_segments.shape)}")
        N, Nseg, S, C = pv_segments.shape
        if not (0 <= seg_id < Nseg):
            raise IndexError(f"seg_id out of range: {seg_id} not in [0,{Nseg})")
        return pv_segments[:, seg_id, :, :]

    if isinstance(pv_segments, np.ndarray):
        if pv_segments.ndim != 4:
            raise ValueError(f"pv_segments must be [N,Nseg,S,C], got {pv_segments.shape}")
        N, Nseg, S, C = pv_segments.shape
        if not (0 <= seg_id < Nseg):
            raise IndexError(f"seg_id out of range: {seg_id} not in [0,{Nseg})")
        return pv_segments[:, seg_id, :, :]

    raise TypeError(f"pv_segments must be np.ndarray or torch.Tensor, got {type(pv_segments)}")
