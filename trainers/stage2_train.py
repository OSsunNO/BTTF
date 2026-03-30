# trainers/stage2_train.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Literal

import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from core.task import ExperimentPlan, StageSpec, RunSpec
from core.segment import SegmentPolicy, segmentize_pv
from core.augment import pick_segment  # keep if you already have it
from core.dataclass.base import DataSpec
from core.dataclass.ts_dataset import TSDatasetBuilder
from adapters.base import FitConfig
from adapters.enc_only import EncOnlyAdapter, EncOnlyAdapterConfig

from core.dataset_wrappers import AugmentedSequenceDataset

Split = Literal["train", "val", "test"]


# -------------------------
# io utils
# -------------------------
def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _save_manifest(plan: ExperimentPlan, run: RunSpec, stage2: StageSpec, payload: Dict[str, Any]) -> Path:
    """
    Save stage2 manifest under a deterministic location.
    Prefer plan-provided helpers if they exist; otherwise fall back.
    """
    # If your ExperimentPlan already has something like plan.stage2_manifest_path(...), use it.
    # Fallback:
    base = plan.metrics_dir() if hasattr(plan, "metrics_dir") else plan.root_dir
    # Some plans may have a private helper; fallback if not.
    seed_folder = plan._seed_folder(run) if hasattr(plan, "_seed_folder") else f"seed{getattr(run, 'seed', 0)}"
    manifest_path = Path(base) / seed_folder / "stage2" / stage2.tag / "manifest.json"
    _save_json(manifest_path, payload)
    return manifest_path


# -------------------------
# data/loader utils
# -------------------------
def _ordered_loader_from_dataset(ds: Dataset, data_spec: DataSpec) -> DataLoader:
    """
    Deterministic DataLoader for index alignment / evaluation:
      - shuffle=False
      - drop_last=False
    """
    return DataLoader(
        ds,
        batch_size=data_spec.batch_size,
        shuffle=False,
        num_workers=data_spec.num_workers,
        pin_memory=data_spec.pin_memory,
        drop_last=False,
    )


def _train_loader_from_dataset(ds: Dataset, data_spec: DataSpec) -> DataLoader:
    """
    Training DataLoader:
      - shuffle=True is allowed because ds itself binds pv_seg by index
      - drop_last depends on data_spec
    """
    return DataLoader(
        ds,
        batch_size=data_spec.batch_size,
        shuffle=True,
        num_workers=data_spec.num_workers,
        pin_memory=data_spec.pin_memory,
        drop_last=data_spec.drop_last,
    )


# -------------------------
# eval utils
# -------------------------
@torch.no_grad()
def _eval_mse(adapter: EncOnlyAdapter, loader: DataLoader) -> float:
    adapter.eval()
    total = 0.0
    count = 0

    for batch in loader:
        x, y = batch[0], batch[1]
        y_hat = adapter.predict_batch(x)  # adapter handles device

        if y.ndim == 2:
            y = y[:, :, None]
        y = y.to(y_hat.device)

        loss = torch.mean((y_hat - y) ** 2)
        total += float(loss.detach().cpu().item())
        count += 1

    return total / max(count, 1)


# -------------------------
# pv utils
# -------------------------
def _load_pv(plan: ExperimentPlan, run: RunSpec, stage1: StageSpec, split: Split) -> np.ndarray:
    pv_path = plan.pv_path(stage1, run, split=split)  # expected .npy
    if not pv_path.exists():
        raise FileNotFoundError(
            f"PV not found: {pv_path}. "
            "Run stage1 PV generation first."
        )

    pv = np.load(str(pv_path))  # [N,P,C] or [N,P]
    if pv.ndim == 2:
        pv = pv[:, :, None]
    if pv.ndim != 3:
        raise ValueError(f"PV must be 2D/3D, got {pv.shape}")

    return pv


def _assert_pv_alignment(split: Split, pv: np.ndarray, ds: Dataset) -> None:
    """
    We can check N-length alignment. True pairing correctness relies on PV being saved
    with shuffle=False, drop_last=False in pv_generate.py.
    """
    if pv.shape[0] != len(ds):
        raise ValueError(
            f"PV({split}) length mismatch: pv.N={pv.shape[0]} vs dataset.N={len(ds)}. "
            "This indicates PV was saved with a different dataset or different slicing."
        )


# -------------------------
# main trainer
# -------------------------
def train_stage2(
    *,
    plan: ExperimentPlan,
    run: RunSpec,
    stage1: StageSpec,                 # PV source tag (stage1)
    stage2: StageSpec,                 # stage2 training tag
    builder: TSDatasetBuilder,
    data_spec: DataSpec,
    policy: SegmentPolicy,
    fit_cfg: FitConfig,
    model_factory: Callable[[int, int, int], nn.Module],
    scale: bool = True,
    max_segments: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train stage2 segment-wise models.

    Key alignment rule (CRITICAL):
      - stage1 PV must be saved in dataset index order:
          shuffle=False, drop_last=False
        (we enforced this in stages/pv_generate.py)

    model_factory(input_len, pred_len, channels) -> nn.Module
      - input_len here is L + S (augmented)
      - pred_len is P
      - channels is C

    Saves per-segment:
      - ckpt:    plan.ckpt_stage2_path(stage2, run, seg_id)
      - metrics: plan.metrics_stage2_path(stage2, run, seg_id)

    Returns summary dict with per-segment val/test MSE.
    """
    task = plan.task
    L = task.input_len
    P = task.pred_len

    # # 1) Build base datasets (the order must be deterministic for PV alignment)
    # #    Even if builder uses shuffle_train=True internally, we only use .dataset indexing.
    # base_train_loader, base_val_loader, base_test_loader = builder.build_loaders(
    #     input_len=L,
    #     pred_len=P,
    #     data_spec=DataSpec(
    #         batch_size=data_spec.batch_size,
    #         num_workers=data_spec.num_workers,
    #         pin_memory=data_spec.pin_memory,
    #         drop_last=False,  # for alignment checks / full coverage
    #     ),
    #     scale=scale,
    # )

    
    # 1) Build base loaders with an ordered spec (force drop_last=False)
    ordered_spec = DataSpec(
        batch_size=data_spec.batch_size,
        num_workers=data_spec.num_workers,
        pin_memory=data_spec.pin_memory,
        drop_last=False,
    )

    base_train_loader, base_val_loader, base_test_loader = builder.build_loaders(
        input_len=L,
        pred_len=P,
        data_spec=ordered_spec,
        scale=scale,
    )


    base_train_ds = base_train_loader.dataset
    base_val_ds = base_val_loader.dataset
    base_test_ds = base_test_loader.dataset

    # infer channels from a sample
    x0, _ = base_train_ds[0]
    if not isinstance(x0, torch.Tensor) or x0.ndim != 2:
        raise ValueError(f"Expected base x [L,C] torch.Tensor, got {type(x0)} {getattr(x0, 'shape', None)}")
    C = x0.shape[1]

    # 2) Load PV saved by stage1 and segmentize
    pv_train = _load_pv(plan, run, stage1, "train")
    pv_val = _load_pv(plan, run, stage1, "val")
    pv_test = _load_pv(plan, run, stage1, "test")

    _assert_pv_alignment("train", pv_train, base_train_ds)
    _assert_pv_alignment("val", pv_val, base_val_ds)
    _assert_pv_alignment("test", pv_test, base_test_ds)

    # pv_segs: [N, Nseg, S, C]
    pv_train_segs = segmentize_pv(pv_train, policy)
    pv_val_segs = segmentize_pv(pv_val, policy)
    pv_test_segs = segmentize_pv(pv_test, policy)

    Nseg = pv_train_segs.shape[1]
    S = pv_train_segs.shape[2]
    L2 = L + S

    seg_count = Nseg if max_segments is None else min(Nseg, int(max_segments))

    # plan dirs
    plan.ensure_dirs(stage2=stage2, run=run)

    summary: Dict[str, Any] = {
        "task": asdict(task),
        "run": asdict(run),
        "stage1": asdict(stage1),
        "stage2": asdict(stage2),
        "policy": asdict(policy),
        "derived": {
            "channels": C,
            "segment_len": S,
            "input_len_stage2": L2,
            "num_segments_total": Nseg,
            "num_segments_trained": seg_count,
        },
        "segments_trained": [],
        "notes": [
            "Stage2 requires PV saved with shuffle=False and drop_last=False (dataset index order).",
            "Training loaders for stage2 segments can shuffle=True because pv_seg is bound per index in wrapper dataset."
        ],
    }

    # 3) Train each segment model
    for seg_id in range(seg_count):
        # per-sample pv segments: [N,S,C]
        tr_seg = pick_segment(pv_train_segs, seg_id)
        va_seg = pick_segment(pv_val_segs, seg_id)
        te_seg = pick_segment(pv_test_segs, seg_id)

        # Wrap datasets so each sample returns x_aug = concat(x, pv_seg[idx])
        tr_aug_ds = AugmentedSequenceDataset(base_train_ds, tr_seg)
        va_aug_ds = AugmentedSequenceDataset(base_val_ds, va_seg)
        te_aug_ds = AugmentedSequenceDataset(base_test_ds, te_seg)

        # Stage2 loaders
        tr_loader = _train_loader_from_dataset(tr_aug_ds, data_spec)
        va_loader = _ordered_loader_from_dataset(va_aug_ds, data_spec)
        te_loader = _ordered_loader_from_dataset(te_aug_ds, data_spec)

        # Build model + adapter
        model = model_factory(L2, P, C)
        adapter = EncOnlyAdapter(
            model=model,
            cfg=EncOnlyAdapterConfig(pred_len=P, device=run.device),
        )

        ckpt_path = plan.ckpt_stage2_path(stage2, run, seg_id)
        metrics_path = plan.metrics_stage2_path(stage2, run, seg_id)
        plan.ensure_dirs(stage2=stage2, run=run, seg_id=seg_id)

        logs = adapter.fit(
            train_loader=tr_loader,
            val_loader=va_loader,
            fit_cfg=fit_cfg,
            save_path=ckpt_path,
        )

        mse_val = _eval_mse(adapter, va_loader)
        mse_test = _eval_mse(adapter, te_loader)

        per_seg_payload = {
            "task": asdict(task),
            "run": asdict(run),
            "stage1": asdict(stage1),
            "stage2": asdict(stage2),
            "seg_id": seg_id,
            "derived": {"segment_len": S, "input_len_stage2": L2, "channels": C},
            "paths": {"ckpt": str(ckpt_path), "metrics": str(metrics_path)},
            "logs": logs,
            "metrics": {"mse_val": mse_val, "mse_test": mse_test},
        }
        _save_json(metrics_path, per_seg_payload)

        summary["segments_trained"].append(
            {"seg_id": seg_id, "mse_val": mse_val, "mse_test": mse_test, "ckpt": str(ckpt_path)}
        )

    # 4) Save stage2 manifest (summary)
    manifest_path = _save_manifest(plan, run, stage2, summary)
    summary["manifest_path"] = str(manifest_path)
    return summary
