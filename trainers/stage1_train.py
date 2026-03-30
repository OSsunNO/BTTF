# trainers/stage1_train.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import json
import torch
from torch.utils.data import DataLoader

from core.task import ExperimentPlan, StageSpec, RunSpec
from core.dataclass.base import DataSpec
from core.dataclass.ts_dataset import TSDatasetBuilder
from adapters.base import FitConfig
from adapters.enc_only import EncOnlyAdapter


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _ordered_loader(loader: DataLoader, data_spec: DataSpec) -> DataLoader:
    """
    Deterministic loader for evaluation/reporting.
    - shuffle=False
    - drop_last=False
    Ensures stable metrics and PV-index alignment.
    """
    return DataLoader(
        loader.dataset,
        batch_size=data_spec.batch_size,
        shuffle=False,
        num_workers=data_spec.num_workers,
        pin_memory=data_spec.pin_memory,
        drop_last=False,
    )


@torch.no_grad()
def _eval_mse(adapter: EncOnlyAdapter, loader: DataLoader) -> float:
    """
    Evaluate MSE on a loader.
    Assumes loader yields (x, y) or (x, y, ...).
    """
    adapter.eval()
    total = 0.0
    count = 0

    for batch in loader:
        x, y = batch[0], batch[1]
        y_hat = adapter.predict_batch(x)  # adapter handles device

        # normalize y -> [B,P,C]
        if y.ndim == 2:
            y = y[:, :, None]
        y = y.to(y_hat.device)

        loss = torch.mean((y_hat - y) ** 2)
        total += float(loss.detach().cpu().item())
        count += 1

    return total / max(count, 1)


def train_stage1(
    *,
    plan: ExperimentPlan,
    run: RunSpec,
    stage1: StageSpec,
    builder: TSDatasetBuilder,
    data_spec: DataSpec,
    adapter: EncOnlyAdapter,
    fit_cfg: FitConfig,
    scale: bool = True,
) -> Dict[str, Any]:
    """
    Train stage1 model and save checkpoint + metrics using ExperimentPlan.

    Key rule:
      - training loader may be shuffled (whatever builder provides)
      - evaluation loaders are rebuilt as ordered loaders (shuffle=False, drop_last=False)
        for deterministic metrics and PV alignment.
    """

    task = plan.task

    # 1) Build loaders (train loader may be shuffled for better training)
    train_loader, val_loader, test_loader = builder.build_loaders(
        input_len=task.input_len,
        pred_len=task.pred_len,
        data_spec=data_spec,
        scale=scale,
    )

    # 2) Prepare save paths
    ckpt_path = plan.ckpt_stage1_path(stage1, run)
    metrics_path = plan.metrics_stage1_path(stage1, run)
    plan.ensure_dirs(stage1=stage1, run=run)

    # 3) Fit stage1 model
    logs = adapter.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        fit_cfg=fit_cfg,
        save_path=ckpt_path,
    )

    # 4) Deterministic evaluation (IMPORTANT)
    #    Use ordered loaders to avoid shuffle artifacts and ensure stable reporting.
    train_eval_loader = _ordered_loader(train_loader, data_spec)
    val_eval_loader = _ordered_loader(val_loader, data_spec)
    test_eval_loader = _ordered_loader(test_loader, data_spec)

    mse_train = _eval_mse(adapter, train_eval_loader)
    mse_val = _eval_mse(adapter, val_eval_loader)
    mse_test = _eval_mse(adapter, test_eval_loader)

    payload: Dict[str, Any] = {
        "task": asdict(task),
        "run": asdict(run),
        "stage1": asdict(stage1),
        "data_spec": asdict(data_spec),
        "fit_cfg": asdict(fit_cfg),
        "paths": {
            "ckpt": str(ckpt_path),
            "metrics": str(metrics_path),
        },
        "logs": logs,
        "metrics": {
            "mse_train": mse_train,
            "mse_val": mse_val,
            "mse_test": mse_test,
        },
        "scale": scale,
        "eval_note": "metrics computed with shuffle=False, drop_last=False loaders for determinism",
        "scaler": None if builder.scaler is None else {
            "mean": builder.scaler.mean.squeeze(0).tolist(),
            "std": builder.scaler.std.squeeze(0).tolist(),
        },
    }

    _save_json(metrics_path, payload)
    return payload
