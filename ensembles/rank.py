# ensembles/rank.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import numpy as np

from core.task import ExperimentPlan, StageSpec, RunSpec


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default_rank_dir(plan: ExperimentPlan, run: RunSpec, stage2: StageSpec) -> Path:
    base = plan.metrics_dir() if hasattr(plan, "metrics_dir") else Path(getattr(plan, "root_dir", "."))
    seed_folder = plan._seed_folder(run) if hasattr(plan, "_seed_folder") else f"seed{getattr(run, 'seed', 0)}"
    return Path(base) / seed_folder / "stage2" / stage2.tag / "ranking"


def rank_segments_by_val(
    *,
    plan: ExperimentPlan,
    run: RunSpec,
    stage2: StageSpec,
    max_scan: int = 10_000,
    strict_contiguous: bool = True,
    save: bool = True,
) -> Dict[str, Any]:
    """
    Rank stage2 segments by validation MSE (ascending).

    Reads:
      plan.metrics_stage2_path(stage2, run, seg_id) for seg_id=0.. until missing.

    strict_contiguous=True:
      - stop at first missing seg_id (expects seg_id 0..N-1)

    strict_contiguous=False:
      - scan up to max_scan, skipping missing files

    Returns:
      payload with 'sorted' list: [{seg_id, mse_val, mse_test, ckpt, metrics_path}, ...]
    """
    rows: List[Dict[str, Any]] = []

    if strict_contiguous:
        seg_id = 0
        while True:
            mpath = plan.metrics_stage2_path(stage2, run, seg_id)
            if not mpath.exists():
                break
            data = _load_json(mpath)
            mse_val = data.get("metrics", {}).get("mse_val", None)
            mse_test = data.get("metrics", {}).get("mse_test", None)
            if mse_val is None:
                raise KeyError(f"mse_val not found in {mpath}")

            rows.append({
                "seg_id": int(seg_id),
                "mse_val": float(mse_val),
                "mse_test": None if mse_test is None else float(mse_test),
                "ckpt": data.get("paths", {}).get("ckpt", str(plan.ckpt_stage2_path(stage2, run, seg_id))),
                "metrics_path": str(mpath),
            })
            seg_id += 1
    else:
        for seg_id in range(int(max_scan)):
            mpath = plan.metrics_stage2_path(stage2, run, seg_id)
            if not mpath.exists():
                continue
            data = _load_json(mpath)
            mse_val = data.get("metrics", {}).get("mse_val", None)
            mse_test = data.get("metrics", {}).get("mse_test", None)
            if mse_val is None:
                continue
            rows.append({
                "seg_id": int(seg_id),
                "mse_val": float(mse_val),
                "mse_test": None if mse_test is None else float(mse_test),
                "ckpt": data.get("paths", {}).get("ckpt", str(plan.ckpt_stage2_path(stage2, run, seg_id))),
                "metrics_path": str(mpath),
            })

    if len(rows) == 0:
        raise FileNotFoundError(
            f"No stage2 metrics found for stage2={stage2.tag}. "
            "Run train_stage2() first."
        )

    rows_sorted = sorted(rows, key=lambda r: r["mse_val"])

    payload: Dict[str, Any] = {
        "task": asdict(plan.task),
        "run": asdict(run),
        "stage2": asdict(stage2),
        "criterion": "mse_val_asc",
        "sorted": rows_sorted,
        "num_segments_found": len(rows_sorted),
    }

    if save:
        out_dir = _default_rank_dir(plan, run, stage2)
        out_dir.mkdir(parents=True, exist_ok=True)

        ranking_path = out_dir / "ranking.json"
        top_ids = [r["seg_id"] for r in rows_sorted]
        topk_path = out_dir / "ranked_ids.npy"
        _save_json(ranking_path, payload)
        np.save(str(topk_path), np.array(top_ids, dtype=np.int64))

        payload["paths"] = {
            "ranking_json": str(ranking_path),
            "ranked_ids_npy": str(topk_path),
        }

    return payload
