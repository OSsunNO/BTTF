# core/dataclass/dataset_configs.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# 수정
@dataclass(frozen=True)
class PredLenConfig:
    batch_size: int
    lr: float
    stride: int = 1    # ← 추가


@dataclass(frozen=True)
class DatasetRunConfig:
    """
    데이터셋 하나에 대한 권장 실험 설정 모음.
    run.py에서 --dataset 인자만 주면 이 설정이 자동으로 적용됨.

    pred_len별 batch_size/lr은 get_pred_len_config(pred_len)으로 조회.
    pred_len_configs에 등록되지 않은 pred_len은 default_batch_size / default_lr로 fallback.
    """
    # split
    train_ratio: float
    val_ratio: float
    test_ratio: float

    # 논문에서 주로 쓰는 pred_len 후보들
    pred_len_candidates: List[int] = field(default_factory=lambda: [96, 192, 336, 720])

    # 데이터셋 특성
    freq: str = "h"           # "h"(hourly), "t"(15min), "d"(daily) 등 - 메모용
    multivariate: bool = True

    # 학습 권장 default (pred_len_configs에 없을 때 fallback)
    default_input_len: int = 336
    default_batch_size: int = 32
    default_lr: float = 1e-3
    default_patience: int = 3

    # pred_len별 세부 설정 (없으면 default 사용)
    train_border: Optional[int] = None
    val_border: Optional[int] = None
    test_border: Optional[int] = None
    pred_len_configs: Dict[int, PredLenConfig] = field(default_factory=dict)

    def get_pred_len_config(self, pred_len: int) -> PredLenConfig:
        """
        pred_len에 맞는 PredLenConfig 반환.
        등록되지 않은 pred_len은 default값으로 생성해서 반환.
        """
        if pred_len in self.pred_len_configs:
            return self.pred_len_configs[pred_len]
        return PredLenConfig(
            batch_size=self.default_batch_size,
            lr=self.default_lr,
        )


# --------------------------------------------------------
# 데이터셋별 설정 등록
# --------------------------------------------------------
DATASET_CONFIGS: Dict[str, DatasetRunConfig] = {
    "exchange_rate": DatasetRunConfig(
        train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
        pred_len_candidates=[96, 192, 336, 720],
        freq="d", multivariate=True,
        default_input_len=336, default_batch_size=8, default_lr=5e-4, default_patience=3,
        pred_len_configs={
            96:  PredLenConfig(batch_size=8,  lr=5e-4, stride=1),
            192: PredLenConfig(batch_size=8,  lr=5e-4, stride=2),
            336: PredLenConfig(batch_size=32, lr=5e-4, stride=4),
            720: PredLenConfig(batch_size=32, lr=5e-4, stride=8),
        },
    ),
    "etth1": DatasetRunConfig(
        train_ratio=12 / 20, val_ratio=4 / 20, test_ratio=4 / 20,
        pred_len_candidates=[96, 192, 336, 720],
        freq="h", multivariate=True,
        default_input_len=336, default_batch_size=32, default_lr=5e-3, default_patience=3,
        train_border=12*30*24,          # 8640
        val_border=12*30*24 + 4*30*24,  # 11520
        test_border=12*30*24 + 4*30*24 + 4*30*24,       # 14400  
        pred_len_configs={
            96:  PredLenConfig(batch_size=32, lr=5e-3, stride=1),
            192: PredLenConfig(batch_size=32, lr=5e-3, stride=2),
            336: PredLenConfig(batch_size=32, lr=5e-3, stride=4),
            720: PredLenConfig(batch_size=32, lr=5e-3, stride=8),
        },
    ),
    # "etth2": DatasetRunConfig(
    #     train_ratio=12 / 20, val_ratio=4 / 20, test_ratio=4 / 20,
    #     pred_len_candidates=[96, 192, 336, 720],
    #     freq="h", multivariate=True,
    #     default_input_len=336, default_batch_size=32, default_lr=5e-3, default_patience=3,
    #     pred_len_configs={
    #         96:  PredLenConfig(batch_size=32, lr=5e-3),
    #         192: PredLenConfig(batch_size=32, lr=5e-3),
    #         336: PredLenConfig(batch_size=32, lr=5e-3),
    #         720: PredLenConfig(batch_size=32, lr=5e-3),
    #     },
    # ),
    # "ettm1": DatasetRunConfig(
    #     train_ratio=12 / 20, val_ratio=4 / 20, test_ratio=4 / 20,
    #     pred_len_candidates=[96, 192, 336, 720],
    #     freq="t", multivariate=True,
    #     default_input_len=336, default_batch_size=32, default_lr=1e-3, default_patience=3,
    #     pred_len_configs={
    #         96:  PredLenConfig(batch_size=32, lr=1e-3),
    #         192: PredLenConfig(batch_size=32, lr=1e-3),
    #         336: PredLenConfig(batch_size=32, lr=1e-2),
    #         720: PredLenConfig(batch_size=32, lr=1e-2),
    #     },
    # ),
    "ettm2": DatasetRunConfig(
        train_ratio=12 / 20, val_ratio=4 / 20, test_ratio=4 / 20,
        pred_len_candidates=[96, 192, 336, 720],
        freq="t", multivariate=True,
        default_input_len=336, default_batch_size=32, default_lr=1e-3, default_patience=3,
        train_border=12*30*24*4,           # 34560
        val_border=12*30*24*4 + 4*30*24*4,    
        test_border=12*30*24*4 + 4*30*24*4 + 4*30*24*4,         # 57600 
        pred_len_configs={
            96:  PredLenConfig(batch_size=32, lr=1e-3, stride=1),
            192: PredLenConfig(batch_size=32, lr=1e-3, stride=2),
            336: PredLenConfig(batch_size=32, lr=1e-2, stride=4),
            720: PredLenConfig(batch_size=32, lr=1e-2, stride=8),
        },
    ),
    # "weather": DatasetRunConfig(
    #     train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
    #     pred_len_candidates=[96, 192, 336, 720],
    #     freq="h", multivariate=True,
    #     default_input_len=336, default_batch_size=32, default_lr=1e-3, default_patience=3,
    #     pred_len_configs={
    #         96:  PredLenConfig(batch_size=32, lr=1e-3),
    #         192: PredLenConfig(batch_size=32, lr=1e-3),
    #         336: PredLenConfig(batch_size=32, lr=1e-3),
    #         720: PredLenConfig(batch_size=32, lr=1e-3),
    #     },
    # ),
    "illness": DatasetRunConfig(
        train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
        pred_len_candidates=[24, 36, 48, 60],
        freq="d", multivariate=True,
        default_input_len=104, default_batch_size=32, default_lr=1e-2, default_patience=3,
        pred_len_configs={
            24: PredLenConfig(batch_size=32, lr=1e-2, stride=1),
            36: PredLenConfig(batch_size=32, lr=1e-2, stride=1),
            48: PredLenConfig(batch_size=32, lr=1e-2, stride=1),
            60: PredLenConfig(batch_size=32, lr=1e-2, stride=1),
        },
    ),
}


def get_dataset_config(name: str) -> DatasetRunConfig:
    """registry alias를 고려해 소문자로 조회"""
    from core.registry import _canonical_dataset_name  # 순환 import 방지
    canonical = _canonical_dataset_name(name)
    if canonical not in DATASET_CONFIGS:
        raise ValueError(
            f"No DatasetRunConfig for '{canonical}'. "
            f"Available: {sorted(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[canonical]
    