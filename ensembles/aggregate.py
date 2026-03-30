# ensembles/aggregate.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Literal, Callable

import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from core.task import ExperimentPlan, StageSpec, RunSpec
from core.segment import SegmentPolicy, segmentize_pv
from core.dataclass.base import DataSpec
from core.dataclass.ts_dataset import TSDatasetBuilder
from adapters.enc_only import EncOnlyAdapter, EncOnlyAdapterConfig
from core.dataset_wrappers import AugmentedSequenceDataset

Split = Literal["train", "val", "test"]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_pv(plan: ExperimentPlan, run: RunSpec, stage1: StageSpec, split: Split) -> np.ndarray:
    pv_path = plan.pv_path(stage1, run, split=split)
    if not pv_path.exists():
        raise FileNotFoundError(f"PV not found: {pv_path}")
    pv = np.load(str(pv_path))
    if pv.ndim == 2:
        pv = pv[:, :, None]
    if pv.ndim != 3:
        raise ValueError(f"PV must be 2D/3D, got {pv.shape}")
    return pv


@torch.no_grad()
def _predict_all(adapter: EncOnlyAdapter, loader: DataLoader) -> np.ndarray:
    adapter.eval()
    outs: List[np.ndarray] = []
    for batch in loader:
        x = batch[0]
        y_hat = adapter.predict_batch(x)  # [B,P,C]
        outs.append(y_hat.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)  # [N,P,C]

@torch.no_grad()
def _collect_gt(loader: DataLoader) -> np.ndarray:
    """DataLoader에서 GT(y)만 모아 반환. shape: [N, P, C]"""
    gts: List[np.ndarray] = []
    for batch in loader:
        y = batch[1]
        if y.ndim == 2:
            y = y[:, :, None]
        gts.append(y.numpy())
    return np.concatenate(gts, axis=0)

def aggregate_topk_mean(
    *,
    plan: ExperimentPlan,
    run: RunSpec,
    stage1: StageSpec,
    stage2: StageSpec,
    builder: TSDatasetBuilder,
    data_spec: DataSpec,
    policy: SegmentPolicy,
    top_ids: Sequence[int],
    model_factory: Callable[[int, int, int], nn.Module],
    split: Split = "test",
    scale: bool = True,
    save_pred: bool = True,
) -> Dict[str, Any]:
    """
    Create ensemble prediction by averaging top_ids segment models on given split.
    """
    task = plan.task
    L = task.input_len
    P = task.pred_len

    train_loader, val_loader, test_loader = builder.build_loaders(
        input_len=L,
        pred_len=P,
        data_spec=DataSpec(
            batch_size=data_spec.batch_size,
            num_workers=data_spec.num_workers,
            pin_memory=data_spec.pin_memory,
            drop_last=False,
        ),
        scale=scale,
    )
    base_loader = {"train": train_loader, "val": val_loader, "test": test_loader}[split]
    base_ds = base_loader.dataset

    x0, _ = base_ds[0]
    C = x0.shape[1]

    pv = _load_pv(plan, run, stage1, split)
    if pv.shape[0] != len(base_ds):
        raise ValueError(f"PV({split}) N mismatch: pv={pv.shape[0]} vs dataset={len(base_ds)}")

    pv_segs = segmentize_pv(pv, policy)  # [N,Nseg,S,C]
    S = pv_segs.shape[2]
    L2 = L + S

    preds = []
    for seg_id in [int(x) for x in top_ids]:
        pv_seg = pv_segs[:, seg_id, :, :]  # [N,S,C]
        aug_ds = AugmentedSequenceDataset(base_ds, pv_seg)

        aug_loader = DataLoader(
            aug_ds,
            batch_size=data_spec.batch_size,
            shuffle=False,
            num_workers=data_spec.num_workers,
            pin_memory=data_spec.pin_memory,
            drop_last=False,
        )

        model = model_factory(L2, P, C)
        adapter = EncOnlyAdapter(model, EncOnlyAdapterConfig(pred_len=P, device=run.device))

        ckpt_path = plan.ckpt_stage2_path(stage2, run, seg_id)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Stage2 ckpt not found for seg_id={seg_id}: {ckpt_path}")
        adapter.load(ckpt_path)

        y_hat = _predict_all(adapter, aug_loader)  # [N,P,C]
        preds.append(y_hat)


    y_hat_ens = np.mean(np.stack(preds, axis=0), axis=0)  # [N,P,C]

    # ── GT 수집 및 앙상블 오차 계산 ──
    y_true = _collect_gt(base_loader)  # [N,P,C]
    mse = float(np.mean((y_hat_ens - y_true) ** 2))
    mae = float(np.mean(np.abs(y_hat_ens - y_true)))


    # save
    base = plan.metrics_dir() if hasattr(plan, "metrics_dir") else Path(getattr(plan, "root_dir", "."))
    seed_folder = plan._seed_folder(run) if hasattr(plan, "_seed_folder") else f"seed{getattr(run, 'seed', 0)}"
    out_dir = Path(base) / seed_folder / stage2.tag / "ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / f"ensemble_{split}_pred.npy"
    meta_path = out_dir / f"ensemble_{split}_meta.json"

    payload: Dict[str, Any] = {
        "task": asdict(task),
        "run": asdict(run),
        "stage1": asdict(stage1),
        "stage2": asdict(stage2),
        "policy": asdict(policy),
        "split": split,
        "top_ids": [int(x) for x in top_ids],
        "metrics": {"mse": mse, "mae": mae},
        "paths": {"pred": str(pred_path), "meta": str(meta_path)},
    }

    if save_pred:
        np.save(str(pred_path), y_hat_ens)
    _save_json(meta_path, payload)
    return payload
