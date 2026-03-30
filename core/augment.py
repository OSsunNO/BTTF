# core/augment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch import Tensor
except Exception:  # torch 없을 수도 있으니
    torch = None
    Tensor = None


ArrayLike = Union[np.ndarray, "Tensor"]


def _is_torch(x: ArrayLike) -> bool:
    return (torch is not None) and isinstance(x, torch.Tensor)


def _assert_same_type(a: ArrayLike, b: ArrayLike) -> None:
    if _is_torch(a) != _is_torch(b):
        raise TypeError(f"type mismatch: x is {'torch' if _is_torch(a) else 'numpy'}, "
                        f"pv_seg is {'torch' if _is_torch(b) else 'numpy'}")


def _get_shape(x: ArrayLike) -> Tuple[int, ...]:
    return tuple(x.shape)


def _cat_time(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    # concat on time dimension = dim=1
    if _is_torch(x):
        return torch.cat([x, y], dim=1)
    return np.concatenate([x, y], axis=1)


def concat_time_augment(
    x: ArrayLike,
    pv_seg: ArrayLike,
    *,
    expect_x_ndim: int = 3,
    expect_seg_ndim: int = 3,
    check_device: bool = True,
) -> ArrayLike:
    """
    Make augmented input by concatenating PV segment to the time axis.

    Args:
      x:      [B, L, C] (numpy or torch)
      pv_seg: [B, S, C] (segment chosen from pv_segments)
    Returns:
      x_aug:  [B, L+S, C]

    Notes:
      - model-agnostic augmentation
      - enforces requirement: stage2 input length increases by S
    """
    _assert_same_type(x, pv_seg)

    sx = _get_shape(x)
    sp = _get_shape(pv_seg)

    if len(sx) != expect_x_ndim:
        raise ValueError(f"x must be {expect_x_ndim}D, got shape {sx}")
    if len(sp) != expect_seg_ndim:
        raise ValueError(f"pv_seg must be {expect_seg_ndim}D, got shape {sp}")

    Bx, L, Cx = sx
    Bp, S, Cp = sp

    if Bx != Bp:
        raise ValueError(f"batch mismatch: x.B={Bx}, pv_seg.B={Bp}")
    if Cx != Cp:
        raise ValueError(f"channel mismatch: x.C={Cx}, pv_seg.C={Cp}")
    if S <= 0:
        raise ValueError(f"segment length S must be positive, got {S}")

    if _is_torch(x) and check_device:
        if x.device != pv_seg.device:
            raise ValueError(f"device mismatch: x on {x.device}, pv_seg on {pv_seg.device}")
        if x.dtype != pv_seg.dtype:
            raise ValueError(f"dtype mismatch: x {x.dtype}, pv_seg {pv_seg.dtype}")

    return _cat_time(x, pv_seg)


def pick_segment(
    pv_segments: ArrayLike,
    seg_id: int,
) -> ArrayLike:
    """
    pv_segments: [B, Nseg, S, C]
    returns:     [B, S, C] for a single seg_id
    """
    if len(pv_segments.shape) != 4:
        raise ValueError(f"pv_segments must be 4D [B,Nseg,S,C], got {pv_segments.shape}")
    B, Nseg, S, C = pv_segments.shape
    if not (0 <= seg_id < Nseg):
        raise IndexError(f"seg_id out of range: {seg_id} not in [0,{Nseg})")
    return pv_segments[:, seg_id, :, :]
