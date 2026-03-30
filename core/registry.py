
# core/registry.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

from torch import nn

from core.dataclass.dataset_configs import get_dataset_config
from core.dataclass.base import DataSpec
from core.dataclass.ts_dataset import TSDatasetBuilder, TSDatasetConfig

# models (현재 repo 기준)
from models.Linear import Model as LinearModel
from models.DLinear import Model as DLinearModel

# (참고) *_aug 모델들은 forward 시그니처가 enc-only(y_hat=model(x))와 다를 수 있어서
# EncOnlyAdapter로 바로 쓰기 어려울 수 있음. 필요해지면 별도 Adapter를 추가하는 게 안전.


# =========================================================
# Dataset registry
# =========================================================
_DATASET_ALIASES: Dict[str, str] = {
    # canonical -> itself
    "exchange_rate": "exchange_rate",
    "er": "exchange_rate",

    "etth1": "etth1",
    "etth2": "etth2",
    "ettm1": "ettm1",
    "ettm2": "ettm2",

    "electricity": "electricity",
    "traffic": "traffic",
    "weather": "weather",
    "illness": "illness",
    "national_illness": "illness",
}

_DEFAULT_DATASET_FILES: Dict[str, str] = {
    "exchange_rate": "exchange_rate.csv",
    "etth1": "ETTh1.csv",
    "etth2": "ETTh2.csv",
    "ettm1": "ETTm1.csv",
    "ettm2": "ETTm2.csv",
    "electricity": "electricity.csv",
    "traffic": "traffic.csv",
    "weather": "weather.csv",
    "illness": "national_illness.csv",
}


@dataclass(frozen=True)
class DatasetSpec:
    """
    Dataset configuration used by registry.

    name:
      - "exchange_rate"/"er"
      - "ETTh1"/"ETTh2"/"ETTm1"/"ETTm2"
      - "electricity" / "traffic" / "weather" / "illness"

    path:
      - optional. If None, we auto-resolve to repo_dataset_dir/<default file>.
    """
    name: str
    path: Optional[str | Path] = None

    # split ratios (time-based split)
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    use_default_config: bool = True   # ← 추가
    shuffle_train: bool = True

    # where to look for default csvs if path is None
    repo_dataset_dir: str | Path = "dataset"
    target_col: Optional[int] = None  # ← 추가. None이면 다변량(기존), int면 단변량


def _canonical_dataset_name(name: str) -> str:
    key = name.strip().lower()
    if key not in _DATASET_ALIASES:
        raise ValueError(
            f"Unknown dataset name: {name}. "
            f"Available={sorted(set(_DATASET_ALIASES.keys()))}"
        )
    return _DATASET_ALIASES[key]


def _resolve_dataset_path(spec: DatasetSpec) -> Path:
    if spec.path is not None:
        return Path(spec.path)

    canonical = _canonical_dataset_name(spec.name)
    if canonical not in _DEFAULT_DATASET_FILES:
        raise ValueError(
            f"No default csv registered for dataset={spec.name} (canonical={canonical}). "
            f"Either pass DatasetSpec(path=...) or add it to _DEFAULT_DATASET_FILES."
        )
    base_dir = Path(spec.repo_dataset_dir)
    return base_dir / _DEFAULT_DATASET_FILES[canonical]


def build_dataset_builder(spec: DatasetSpec) -> TSDatasetBuilder:
    """
    NOTE:
    - We reuse TSDatasetBuilder as a generic CSV time-series builder:
      it selects numeric columns, does time-split, and (optionally) scales using train only.
    """
    canonical = _canonical_dataset_name(spec.name)
    csv_path = _resolve_dataset_path(spec)
    

    if spec.use_default_config:
        ds_cfg = get_dataset_config(canonical)
        tr, va, te = ds_cfg.train_ratio, ds_cfg.val_ratio, ds_cfg.test_ratio
        train_border = ds_cfg.train_border
        val_border   = ds_cfg.val_border
        test_border  = ds_cfg.test_border  
    else:
        tr, va, te = spec.train_ratio, spec.val_ratio, spec.test_ratio
        train_border = None
        val_border   = None
        test_border  = None

    cfg = TSDatasetConfig(
        path=csv_path,
        train_ratio=tr,
        val_ratio=va,
        test_ratio=te,
        shuffle_train=bool(spec.shuffle_train),
        target_col=spec.target_col,
        train_border=train_border,
        val_border=val_border,
        test_border=test_border,
    )
    return TSDatasetBuilder(cfg)


def infer_num_channels(
    builder: TSDatasetBuilder,
    *,
    input_len: int,
    pred_len: int,
    data_spec: DataSpec,
    scale: bool = True,
) -> int:
    """
    Safely infer C from the dataset by grabbing the first sample of train dataset.
    """
    train_loader, _, _ = builder.build_loaders(
        input_len=input_len,
        pred_len=pred_len,
        data_spec=data_spec,
        scale=scale,
    )
    x0, _ = train_loader.dataset[0]  # x: [L,C]
    if x0.ndim != 2:
        raise ValueError(f"Expected x0 [L,C], got {tuple(x0.shape)}")
    return int(x0.shape[1])


# =========================================================
# Model registry
# =========================================================
@dataclass(frozen=True)
class ModelSpec:
    """
    Model config used by registry.

    name:
      - "linear"
      - "dlinear"
    individual:
      - per-channel heads (used by Linear/DLinear in your code)
    extra:
      - placeholder for future options (e.g., patch_len, d_model, etc.)
    """
    name: str
    individual: bool = False
    extra: Optional[Dict[str, Any]] = None


_MODEL_REGISTRY: Dict[str, Any] = {
    "linear": LinearModel,
    "dlinear": DLinearModel,
    # enable only if you add proper adapter for *_aug
    # "linear_aug": LinearAugModel,
    # "dlinear_aug": DLinearAugModel,
}


def get_model_class(name: str):
    key = name.strip().lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available={sorted(list(_MODEL_REGISTRY.keys()))}"
        )
    return _MODEL_REGISTRY[key]


def build_model(
    spec: ModelSpec,
    *,
    input_len: int,
    pred_len: int,
    channels: int,
) -> nn.Module:
    """
    Build a model instance with a unified 'configs' interface.

    Your models expect:
      configs.seq_len, configs.pred_len, configs.enc_in, configs.individual
    """
    ModelCls = get_model_class(spec.name)

    extra = spec.extra or {}
    configs = SimpleNamespace(
        seq_len=int(input_len),
        pred_len=int(pred_len),
        enc_in=int(channels),
        individual=bool(spec.individual),
        **extra,
    )
    return ModelCls(configs)


def make_model_factory(spec: ModelSpec) -> Callable[[int, int, int], nn.Module]:
    """
    Factory function needed for stage2 where we need many models (per segment).

    Signature:
      model_factory(input_len, pred_len, channels) -> nn.Module
    """
    def _factory(L: int, P: int, C: int) -> nn.Module:
        return build_model(
            spec,
            input_len=int(L),
            pred_len=int(P),
            channels=int(C),
        )
    return _factory
