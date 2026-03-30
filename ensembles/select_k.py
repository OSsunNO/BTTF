# # ensembles/select_k.py
# from __future__ import annotations

# from dataclasses import asdict
# from pathlib import Path
# from typing import Any, Dict, List, Sequence, Literal, Callable, Optional

# import json
# import numpy as np
# import torch
# from torch import nn
# from torch.utils.data import DataLoader

# from core.task import ExperimentPlan, StageSpec, RunSpec
# from core.segment import SegmentPolicy, segmentize_pv
# from core.dataclass.base import DataSpec
# from core.dataclass.exchange_rate import ExchangeRateBuilder
# from adapters.enc_only import EncOnlyAdapter, EncOnlyAdapterConfig
# from core.dataset_wrappers import AugmentedSequenceDataset

# Split = Literal["train", "val", "test"]


# def _ensure_parent(path: Path) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)


# def _save_json(path: Path, payload: Dict[str, Any]) -> None:
#     _ensure_parent(path)
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False, indent=2)


# def _load_pv(plan: ExperimentPlan, run: RunSpec, stage1: StageSpec, split: Split) -> np.ndarray:
#     pv_path = plan.pv_path(stage1, run, split=split)
#     if not pv_path.exists():
#         raise FileNotFoundError(f"PV not found: {pv_path}")
#     pv = np.load(str(pv_path))
#     if pv.ndim == 2:
#         pv = pv[:, :, None]
#     if pv.ndim != 3:
#         raise ValueError(f"PV must be 2D/3D, got {pv.shape}")
#     return pv


# @torch.no_grad()
# def _predict_all(adapter: EncOnlyAdapter, loader: DataLoader) -> np.ndarray:
#     adapter.eval()
#     outs: List[np.ndarray] = []
#     for batch in loader:
#         x = batch[0]
#         y_hat = adapter.predict_batch(x)  # [B,P,C]
#         outs.append(y_hat.detach().cpu().numpy())
#     return np.concatenate(outs, axis=0)  # [N,P,C]


# def _pred_variance(preds: np.ndarray) -> float:
#     """
#     preds: [K,N,P,C]
#     Return mean variance across ensemble members (higher => more diversity).
#     """
#     return float(np.mean(np.var(preds, axis=0)))


# def _pred_abs_corr(preds: np.ndarray) -> float:
#     """
#     preds: [K,N,P,C]
#     Compute average |corr| between ensemble members on flattened predictions.
#     """
#     K = preds.shape[0]
#     if K <= 1:
#         return 1.0

#     flat = preds.reshape(K, -1)  # [K, N*P*C]
#     corr = np.corrcoef(flat)     # [K,K]
#     # take upper triangle (excluding diag)
#     iu = np.triu_indices(K, k=1)
#     vals = np.abs(corr[iu])
#     return float(np.mean(vals))


# def _normalize_minmax(arr: np.ndarray) -> np.ndarray:
#     mn = float(arr.min())
#     mx = float(arr.max())
#     if mx - mn < 1e-12:
#         return np.zeros_like(arr)
#     return (arr - mn) / (mx - mn)


# def select_best_k_varcorr(
#     *,
#     plan: ExperimentPlan,
#     run: RunSpec,
#     stage1: StageSpec,
#     stage2: StageSpec,
#     builder: ExchangeRateBuilder,
#     data_spec: DataSpec,
#     policy: SegmentPolicy,
#     ranked_seg_ids: Sequence[int],
#     model_factory: Callable[[int, int, int], nn.Module],
#     mode_split: Split = "test",     # ipynb처럼 test 기반 선택하려면 "test"
#     k_list: Optional[Sequence[int]] = None,  # 예: [1,2,3,...] 또는 [5,10,15,...]
#     alpha: float = 1.0,  # Var term weight
#     beta: float = 1.0,   # Corr term weight
#     save: bool = True,
# ) -> Dict[str, Any]:
#     """
#     Choose best_K using only prediction statistics (Var / |Corr|) on mode_split.

#     Procedure:
#       - follow val-based ranking order (ranked_seg_ids)
#       - compute preds for each seg_id on mode_split
#       - for each K in k_list, stack top-K preds and compute:
#           V(K) = variance(preds_topK)
#           R(K) = abs_corr(preds_topK)
#       - normalize V and R across K, then:
#           score(K) = alpha * (1 - V_norm) + beta * R_norm
#         (interpretation: want high variance(diversity) and low corr(redundancy))
#       - pick K that minimizes score

#     Note: This is prediction-only selection (no labels used).
#     """

#     # 1) base dataset for mode_split (deterministic order)
#     task = plan.task
#     L = task.input_len
#     P = task.pred_len

#     train_loader, val_loader, test_loader = builder.build_loaders(
#         input_len=L,
#         pred_len=P,
#         data_spec=DataSpec(
#             batch_size=data_spec.batch_size,
#             num_workers=data_spec.num_workers,
#             pin_memory=data_spec.pin_memory,
#             drop_last=False,
#         ),
#         scale=True,
#     )
#     base_loader = {"train": train_loader, "val": val_loader, "test": test_loader}[mode_split]
#     base_ds = base_loader.dataset

#     x0, _ = base_ds[0]
#     C = x0.shape[1]

#     # 2) PV segments for mode_split
#     pv = _load_pv(plan, run, stage1, mode_split)
#     if pv.shape[0] != len(base_ds):
#         raise ValueError(f"PV({mode_split}) N mismatch: pv={pv.shape[0]} vs dataset={len(base_ds)}")
#     pv_segs = segmentize_pv(pv, policy)  # [N,Nseg,S,C]
#     S = pv_segs.shape[2]
#     L2 = L + S

#     # 3) decide k_list
#     ranked_seg_ids = [int(x) for x in ranked_seg_ids]
#     maxK = len(ranked_seg_ids)

#     if k_list is None:
#         # default: 1..min(20,maxK)
#         Kmax = min(20, maxK)
#         k_list = list(range(1, Kmax + 1))
#     else:
#         k_list = [int(k) for k in k_list if 1 <= int(k) <= maxK]
#         if len(k_list) == 0:
#             raise ValueError("k_list is empty after filtering")

#     # 4) precompute per-seg predictions in ranking order up to max(k_list)
#     needK = max(k_list)
#     preds_per_seg: List[np.ndarray] = []

#     for j in range(needK):
#         seg_id = ranked_seg_ids[j]

#         # build augmented dataset for this seg_id
#         pv_seg = pv_segs[:, seg_id, :, :]  # [N,S,C]
#         aug_ds = AugmentedSequenceDataset(base_ds, pv_seg)

#         aug_loader = DataLoader(
#             aug_ds,
#             batch_size=data_spec.batch_size,
#             shuffle=False,
#             num_workers=data_spec.num_workers,
#             pin_memory=data_spec.pin_memory,
#             drop_last=False,
#         )

#         # load model ckpt
#         model = model_factory(L2, P, C)
#         adapter = EncOnlyAdapter(model, EncOnlyAdapterConfig(pred_len=P, device=run.device))
#         ckpt_path = plan.ckpt_stage2_path(stage2, run, seg_id)
#         if not ckpt_path.exists():
#             raise FileNotFoundError(f"Stage2 ckpt not found for seg_id={seg_id}: {ckpt_path}")
#         adapter.load(ckpt_path)

#         preds = _predict_all(adapter, aug_loader)  # [N,P,C]
#         preds_per_seg.append(preds)

#     # 5) compute V(K), R(K)
#     V_list = []
#     R_list = []
#     for K in k_list:
#         top_preds = np.stack(preds_per_seg[:K], axis=0)  # [K,N,P,C]
#         V_list.append(_pred_variance(top_preds))
#         R_list.append(_pred_abs_corr(top_preds))

#     V = np.array(V_list, dtype=np.float64)
#     R = np.array(R_list, dtype=np.float64)

#     Vn = _normalize_minmax(V)
#     Rn = _normalize_minmax(R)

#     # score: want large V (diversity) => (1 - Vn) small, and small R => Rn small
#     score = alpha * (1.0 - Vn) + beta * Rn

#     best_i = int(np.argmin(score))
#     best_k = int(k_list[best_i])
#     best_top_ids = ranked_seg_ids[:best_k]

#     payload: Dict[str, Any] = {
#         "task": asdict(task),
#         "run": asdict(run),
#         "stage1": asdict(stage1),
#         "stage2": asdict(stage2),
#         "policy": asdict(policy),
#         "mode_split": mode_split,
#         "ranked_seg_ids_used": ranked_seg_ids[:needK],
#         "k_list": list(k_list),
#         "stats": {
#             "V": V.tolist(),
#             "R": R.tolist(),
#             "V_norm": Vn.tolist(),
#             "R_norm": Rn.tolist(),
#             "score": score.tolist(),
#             "alpha": float(alpha),
#             "beta": float(beta),
#         },
#         "best": {
#             "best_k": best_k,
#             "best_top_ids": best_top_ids,
#         },
#     }

#     if save:
#         base = plan.metrics_dir() if hasattr(plan, "metrics_dir") else Path(getattr(plan, "root_dir", "."))
#         seed_folder = plan._seed_folder(run) if hasattr(plan, "_seed_folder") else f"seed{getattr(run, 'seed', 0)}"
#         out_dir = Path(base) / seed_folder / "stage2" / stage2.tag / "ensemble_select"
#         out_dir.mkdir(parents=True, exist_ok=True)

#         out_path = out_dir / f"bestk_varcorr_{mode_split}.json"
#         _save_json(out_path, payload)
#         payload["paths"] = {"bestk_json": str(out_path)}

#     return payload


# ensembles/select_k.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Literal, Callable, Optional

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


def _pred_variance(preds: np.ndarray) -> float:
    """
    preds: [K,N,P,C]
    Paper-style proxy V(K): prediction variance across ensemble members.
    """
    return float(np.mean(np.var(preds, axis=0)))


def _pred_abs_corr(preds: np.ndarray) -> float:
    """
    preds: [K,N,P,C]
    Paper-style proxy R(K): mean absolute correlation among members.
    """
    K = preds.shape[0]
    if K <= 1:
        return 1.0
    flat = preds.reshape(K, -1)  # [K, N*P*C]
    corr = np.corrcoef(flat)     # [K,K]
    iu = np.triu_indices(K, k=1)
    vals = np.abs(corr[iu])
    return float(np.mean(vals))


def _normalize_minmax(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < eps:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn + eps)


def _default_out_dir(plan: ExperimentPlan, run: RunSpec, stage2: StageSpec) -> Path:
    base = plan.metrics_dir() if hasattr(plan, "metrics_dir") else Path(getattr(plan, "root_dir", "."))
    seed_folder = plan._seed_folder(run) if hasattr(plan, "_seed_folder") else f"seed{getattr(run, 'seed', 0)}"
    return Path(base) / seed_folder / "stage2" / stage2.tag / "ensemble_select"


def select_best_k_varcorr(
    *,
    plan: ExperimentPlan,
    run: RunSpec,
    stage1: StageSpec,
    stage2: StageSpec,
    builder: TSDatasetBuilder,
    data_spec: DataSpec,
    policy: SegmentPolicy,
    ranked_seg_ids: Sequence[int],
    model_factory: Callable[[int, int, int], nn.Module],
    mode_split: Split = "test",                # ipynb처럼 test 기반이면 "test"
    k_list: Optional[Sequence[int]] = None,    # 직접 지정 (예: [5,10,15,...])
    step: Optional[int] = 5,                   # k_list가 None일 때만 사용 (M,2M,...) 스타일
    k_max: Optional[int] = None,               # None이면 len(ranked_seg_ids)까지
    alpha: float = 1.0,
    beta: float = 1.0,
    scale: bool = True,
    eps: float = 1e-12,
    save: bool = True,
) -> Dict[str, Any]:
    """
    Paper / ipynb style:
      - Rank segments by val (outside)
      - Choose K* by minimizing:
          score(K) = alpha * V_norm(K) + beta * R_norm(K)
        where V_norm, R_norm are min-max normalized across candidate Ks.
    """

    # 1) base dataset (deterministic order)
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
    base_loader = {"train": train_loader, "val": val_loader, "test": test_loader}[mode_split]
    base_ds = base_loader.dataset

    x0, _ = base_ds[0]
    C = x0.shape[1]

    # 2) PV segments for mode_split
    pv = _load_pv(plan, run, stage1, mode_split)
    if pv.shape[0] != len(base_ds):
        raise ValueError(f"PV({mode_split}) N mismatch: pv={pv.shape[0]} vs dataset={len(base_ds)}")
    pv_segs = segmentize_pv(pv, policy)  # [N,Nseg,S,C]
    S = pv_segs.shape[2]
    L2 = L + S

    ranked_seg_ids = [int(x) for x in ranked_seg_ids]
    maxK_total = len(ranked_seg_ids)
    if maxK_total <= 0:
        raise ValueError("ranked_seg_ids is empty")

    # 3) candidate K set
    if k_max is None:
        k_max = maxK_total
    k_max = int(min(int(k_max), maxK_total))

    if k_list is None:
        if step is None or int(step) <= 0:
            raise ValueError("step must be positive when k_list is None")
        step = int(step)

        # paper/ipynb style: M, 2M, 3M, ... and ALWAYS include k_max (remainder case)
        k_list = list(range(step, k_max + 1, step))
        if len(k_list) == 0:
            k_list = [k_max]
        elif k_list[-1] != k_max:
            k_list.append(k_max)

    else:
        k_list = [int(k) for k in k_list if 1 <= int(k) <= k_max]
        if len(k_list) == 0:
            raise ValueError("k_list is empty after filtering")

    needK = max(k_list)

    # 4) precompute per-seg predictions in ranking order up to needK
    preds_per_seg: List[np.ndarray] = []
    used_seg_ids: List[int] = []

    for j in range(needK):
        seg_id = ranked_seg_ids[j]

        ckpt_path = plan.ckpt_stage2_path(stage2, run, seg_id)
        if not ckpt_path.exists():
            # ipynb에서 skip하는 스타일을 원하면 여기서 continue로 바꿔도 됨.
            raise FileNotFoundError(f"Stage2 ckpt not found for seg_id={seg_id}: {ckpt_path}")

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
        adapter.load(ckpt_path)

        preds = _predict_all(adapter, aug_loader)  # [N,P,C]
        preds_per_seg.append(preds)
        used_seg_ids.append(seg_id)

    # 5) compute V(K), R(K) across candidate Ks
    V_list = []
    R_list = []
    for K in k_list:
        top_preds = np.stack(preds_per_seg[:K], axis=0)  # [K,N,P,C]
        V_list.append(_pred_variance(top_preds))
        R_list.append(_pred_abs_corr(top_preds))

    V = np.array(V_list, dtype=np.float64)
    R = np.array(R_list, dtype=np.float64)

    Vn = _normalize_minmax(V, eps=eps)
    Rn = _normalize_minmax(R, eps=eps)

    # ✅ paper/ipynb score (minimize both)
    score = alpha * Vn + beta * Rn

    best_i = int(np.argmin(score))
    best_k = int(k_list[best_i])
    best_top_ids = used_seg_ids[:best_k]

    payload: Dict[str, Any] = {
        "task": asdict(task),
        "run": asdict(run),
        "stage1": asdict(stage1),
        "stage2": asdict(stage2),
        "policy": asdict(policy),
        "mode_split": mode_split,
        "ranked_seg_ids_used": used_seg_ids,  # length=needK
        "k_list": list(k_list),
        "stats": {
            "V": V.tolist(),
            "R": R.tolist(),
            "V_norm": Vn.tolist(),
            "R_norm": Rn.tolist(),
            "score": score.tolist(),
            "alpha": float(alpha),
            "beta": float(beta),
        },
        "best": {
            "best_k": best_k,
            "best_top_ids": best_top_ids,
        },
    }

    if save:
        out_dir = _default_out_dir(plan, run, stage2)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"bestk_varcorr_{mode_split}.json"
        _save_json(out_path, payload)
        payload["paths"] = {"bestk_json": str(out_path)}

    return payload
