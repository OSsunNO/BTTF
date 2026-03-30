# core/model_factory.py
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    name: str = "Linear"          # "Linear" | "DLinear" | others
    individual: bool = False      # for some linear baselines
    # DLinear 관련 옵션이 필요하면 여기에 추가 가능
    # e.g., kernel_size: int = 25


def _make_configs(input_len: int, pred_len: int, channels: int, *, spec: ModelSpec) -> Any:
    """
    Create a configs-like object expected by models/*.
    Uses SimpleNamespace to avoid defining a big configs dataclass.
    """
    return SimpleNamespace(
        seq_len=int(input_len),
        pred_len=int(pred_len),
        enc_in=int(channels),
        individual=bool(spec.individual),
        # 필요 시 모델들이 요구하는 필드들을 여기에 추가
    )


def build_model(spec: ModelSpec, input_len: int, pred_len: int, channels: int) -> nn.Module:
    name = spec.name.lower()

    if name == "linear":
        from models.Linear import Model
    elif name == "dlinear":
        from models.DLinear import Model
    else:
        raise ValueError(f"Unknown model name: {spec.name}")

    configs = _make_configs(input_len, pred_len, channels, spec=spec)
    return Model(configs)


def make_model_factory(spec: ModelSpec) -> Callable[[int, int, int], nn.Module]:
    """
    Returns a function model_factory(L, P, C) -> nn.Module
    that can be passed into stage2_train/select_k/aggregate.
    """
    def _factory(input_len: int, pred_len: int, channels: int) -> nn.Module:
        return build_model(spec, input_len, pred_len, channels)

    return _factory
