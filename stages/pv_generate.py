# stages/pv_generate.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Literal

import json
import numpy as np

from torch.utils.data import DataLoader

from core.task import ExperimentPlan, StageSpec, RunSpec
from core.dataclass.base import DataSpec
from core.dataclass.ts_dataset import TSDatasetBuilder
from adapters.enc_only import EncOnlyAdapter

Split = Literal["train", "val", "test"]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _save_npy(path: Path, arr: np.ndarray) -> None:
    _ensure_parent(path)
    np.save(str(path), arr)


def _ordered_loader(loader: DataLoader, data_spec: DataSpec) -> DataLoader:
    """
    Rebuild a DataLoader that iterates the SAME dataset in deterministic index order.
    - shuffle=False
    - drop_last=False
    This is critical for PV alignment with dataset indices.
    """
    return DataLoader(
        loader.dataset,
        batch_size=data_spec.batch_size,
        shuffle=False,
        num_workers=data_spec.num_workers,
        pin_memory=data_spec.pin_memory,
        drop_last=False,
    )


def generate_pv(
    *,
    plan: ExperimentPlan,
    run: RunSpec,
    stage1: StageSpec,
    builder: TSDatasetBuilder,
    data_spec: DataSpec,
    adapter: EncOnlyAdapter,
    scale: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Load trained stage1 checkpoint, run inference on train/val/test,
    and save PV (predictions) as .npy under plan.pv_path(...).

    IMPORTANT:
      PV is generated in dataset-index order (shuffle=False) to guarantee
      alignment for stage2 augmentation.
    """
    task = plan.task

    # # 1) build loaders (whatever builder returns)
    # train_loader, val_loader, test_loader = builder.build_loaders(
    #     input_len=task.input_len,
    #     pred_len=task.pred_len,
    #     data_spec=data_spec,
    #     scale=scale,
    # )

    # # 2) force deterministic iteration order for PV saving
    # train_loader = _ordered_loader(train_loader, data_spec)
    # val_loader = _ordered_loader(val_loader, data_spec)
    # test_loader = _ordered_loader(test_loader, data_spec)
    
    # 1) build loaders with PV-dedicated spec (force drop_last=False)
    pv_spec = DataSpec(
        batch_size=data_spec.batch_size,
        num_workers=data_spec.num_workers,
        pin_memory=data_spec.pin_memory,
        drop_last=False,
    )

    train_loader, val_loader, test_loader = builder.build_loaders(
        input_len=task.input_len,
        pred_len=task.pred_len,
        data_spec=pv_spec,
        scale=scale,
    )

    # 2) force deterministic iteration order for PV saving
    train_loader = _ordered_loader(train_loader, pv_spec)
    val_loader   = _ordered_loader(val_loader, pv_spec)
    test_loader  = _ordered_loader(test_loader, pv_spec)


    # 3) load stage1 checkpoint
    ckpt_path = plan.ckpt_stage1_path(stage1, run)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage1 checkpoint not found: {ckpt_path}")
    adapter.load(ckpt_path)

    # 4) predict and save
    out_paths: Dict[str, str] = {}
    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        pv_path = plan.pv_path(stage1, run, split=split)
        if pv_path.exists() and not overwrite:
            out_paths[split] = str(pv_path)
            continue

        pred_dict = adapter.predict_loader(loader, return_y=False)  # {'pred': np.ndarray}
        pv = pred_dict["pred"]  # [N, P, C]
        _save_npy(pv_path, pv)
        out_paths[split] = str(pv_path)

    # 5) save manifest
    manifest_path = plan.pv_dir(stage1, run) / "manifest.json"
    payload: Dict[str, Any] = {
        "task": asdict(task),
        "run": asdict(run),
        "stage1": asdict(stage1),
        "paths": out_paths,
        "scale": scale,
        "note": "PV saved with shuffle=False, drop_last=False for index alignment",
    }
    _save_json(manifest_path, payload)
    return payload
