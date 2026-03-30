# core/segment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class SegmentPolicy:
    """
    How to cut PV (predicted values) into segments.

    mode:
      - "ratio": segment_len = round(pred_len * ratio)
      - "div":   segment_len = pred_len // div
      - "fixed": segment_len = fixed_len
    """
    mode: Literal["ratio", "div", "fixed"] = "div"
    ratio: float = 1 / 3
    div: int = 3
    fixed_len: int = 32
    stride: int = 1

    def segment_len(self, pred_len: int) -> int:
        if pred_len <= 0:
            raise ValueError(f"pred_len must be positive, got {pred_len}")

        if self.mode == "ratio":
            s = int(round(pred_len * float(self.ratio)))
        elif self.mode == "div":
            if self.div <= 0:
                raise ValueError(f"div must be positive, got {self.div}")
            s = pred_len // int(self.div)
        elif self.mode == "fixed":
            s = int(self.fixed_len)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if s <= 0:
            raise ValueError(
                f"Computed segment_len={s} (pred_len={pred_len}, policy={self}). "
                "Choose a larger ratio/fixed_len or smaller div."
            )
        if s > pred_len:
            raise ValueError(
                f"segment_len={s} cannot exceed pred_len={pred_len}. policy={self}"
            )
        return s


def segmentize_pv(
    pv: np.ndarray,
    policy: SegmentPolicy,
) -> np.ndarray:
    """
    pv: [B, P, C] or [B, P] (univariate)
    returns:
      pv_segs: [B, Nseg, S, C]  (always 4D)
    where:
      S = segment_len
      Nseg = (P - S) // stride + 1
    """
    if pv.ndim == 2:
        pv = pv[:, :, None]  # [B,P,1]
    if pv.ndim != 3:
        raise ValueError(f"pv must be 2D or 3D, got shape {pv.shape}")

    B, P, C = pv.shape
    S = policy.segment_len(P)
    stride = int(policy.stride)
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    nseg = (P - S) // stride + 1
    if nseg <= 0:
        raise ValueError(f"nseg computed <= 0 (P={P}, S={S}, stride={stride})")

    out = np.empty((B, nseg, S, C), dtype=pv.dtype)
    for i in range(nseg):
        st = i * stride
        out[:, i, :, :] = pv[:, st:st + S, :]
    return out


def segmentize_pv_and_label(
    pv: np.ndarray,
    y: np.ndarray,
    policy: SegmentPolicy,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper: segmentize both pv and label in the same way.
    pv: [B,P,C] or [B,P]
    y:  [B,P,C] or [B,P]
    """
    pv_segs = segmentize_pv(pv, policy)
    y_segs = segmentize_pv(y, policy)
    if pv_segs.shape != y_segs.shape:
        raise ValueError(
            f"pv_segs and y_segs shape mismatch: {pv_segs.shape} vs {y_segs.shape}"
        )
    return pv_segs, y_segs
