

# scripts/run.py
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from core.task import make_default_plan, StageSpec, RunSpec
from core.segment import SegmentPolicy
from core.dataclass.dataset_configs import get_dataset_config   # 상단 import에 추가


from core.dataclass.base import DataSpec
from adapters.base import FitConfig
from adapters.enc_only import EncOnlyAdapter, EncOnlyAdapterConfig

from trainers.stage1_train import train_stage1
from stages.pv_generate import generate_pv
from trainers.stage2_train import train_stage2

from ensembles.rank import rank_segments_by_val
from ensembles.select_k import select_best_k_varcorr
from ensembles.aggregate import aggregate_topk_mean

# ✅ registry 사용
from core.registry import (
    DatasetSpec,
    build_dataset_builder,
    infer_num_channels,
    ModelSpec,
    build_model,
    make_model_factory,
)

# seed 유틸이 이미 있다면 그걸 쓰고, 없으면 fallback
try:
    from core.utils.seed import set_seed  # 네가 이미 구현한 seed.py가 이 경로면 OK
except Exception:
    def set_seed(seed: int, deterministic: bool = False) -> None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # --- task ---
    p.add_argument("--root_dir", type=str, default="./outputs")
    p.add_argument("--dataset", type=str, default="ER")  # registry에서 "er"로 처리
    p.add_argument("--csv_path", type=str, default=None)
    p.add_argument("--input_len",  type=int, default=None)  # None → ds_run_cfg.default_input_len
    p.add_argument("--pred_len",   type=int, default=None)  # None → ds_run_cfg.pred_len_candidates[0]
    p.add_argument("--target_col", type=int, default=None)  # None=다변량, int=단변량

    # --- run ---
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--deterministic", action="store_true")

    # --- models ---
    p.add_argument("--stage1_model", type=str, default="Linear", choices=["Linear", "DLinear"])
    p.add_argument("--stage2_model", type=str, default="Linear", choices=["Linear", "DLinear"])
    p.add_argument("--individual", action="store_true")
    p.add_argument("--patience", type=int, default=None)  # None이면 dataset config 기본값 사용
    

    # --- train configs ---
    p.add_argument("--batch_size", type=int, default=None)  # None → plc.batch_size
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs1", type=int, default=10)
    p.add_argument("--epochs2", type=int, default=10)
    p.add_argument("--lr1", type=float, default=None)  # None → plc.lr
    p.add_argument("--lr2", type=float, default=None)  # None → plc.lr
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--amp", action="store_true")

    # --- segment policy ---
    p.add_argument("--seg_mode", type=str, default="div", choices=["div", "ratio", "fixed"])
    p.add_argument("--seg_div", type=int, default=3)
    p.add_argument("--seg_ratio", type=float, default=1 / 3)
    p.add_argument("--seg_fixed", type=int, default=32)
    p.add_argument("--seg_stride", type=int, default=1)
    p.add_argument("--max_segments", type=int, default=None)

    # --- ensemble select K ---
    p.add_argument("--k_step", type=int, default=5)
    p.add_argument("--k_max", type=int, default=None)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--mode_split", type=str, default="test", choices=["train", "val", "test"])

    # --- tags ---
    p.add_argument("--stage1_tag", type=str, default="stage1")
    p.add_argument("--stage2_tag", type=str, default="stage2")

    # --- misc ---
    p.add_argument("--scale", action="store_true")
    p.add_argument("--overwrite_pv", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # device resolve
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    set_seed(args.seed, deterministic=args.deterministic)

    ds_run_cfg = get_dataset_config(args.dataset)
    patience   = args.patience   if args.patience   is not None else ds_run_cfg.default_patience
    input_len  = args.input_len  if args.input_len  is not None else ds_run_cfg.default_input_len
    pred_len   = args.pred_len   if args.pred_len   is not None else ds_run_cfg.pred_len_candidates[0]
    plc        = ds_run_cfg.get_pred_len_config(pred_len)
    batch_size = args.batch_size if args.batch_size is not None else plc.batch_size
    lr         = plc.lr
    stride     = args.seg_stride if args.seg_stride != 0 else plc.stride  


    # plan / run
    plan = make_default_plan(
        root_dir=Path(args.root_dir),
        dataset=args.dataset,
        input_len=input_len,
        pred_len=pred_len,
    )
    run = RunSpec(seed=args.seed, device=device)

    stage1 = StageSpec(tag=args.stage1_tag)
    stage2 = StageSpec(tag=args.stage2_tag)

    data_spec = DataSpec(
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    fit1 = FitConfig(
        epochs=args.epochs1,
        lr=args.lr1 if args.lr1 is not None else lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        amp=args.amp,
        patience=patience,
        lr_scheduler="type1", 
    )
    fit2 = FitConfig(
        epochs=args.epochs2,
        lr=args.lr2 if args.lr2 is not None else lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        amp=args.amp,
        patience=patience,
        lr_scheduler="type1",
    )
    
    policy = SegmentPolicy(
        mode=args.seg_mode,
        div=args.seg_div,
        ratio=args.seg_ratio,
        fixed_len=args.seg_fixed,
        stride=stride,            
    )

    # =========================================================
    # ✅ Dataset builder via registry
    # =========================================================
    ds_spec = DatasetSpec(
        name=args.dataset,          # "ER" or "er" 모두 OK (registry에서 lower 처리)
        path=args.csv_path,
        shuffle_train=True,
        target_col=args.target_col,
    )
    builder = build_dataset_builder(ds_spec)

    # ✅ infer channels via registry helper
    C = infer_num_channels(
        builder,
        input_len=input_len,
        pred_len=pred_len,
        data_spec=data_spec,
        scale=args.scale,
    )

    # =========================================================
    # Stage 1
    # =========================================================
    stage1_spec = ModelSpec(name=args.stage1_model, individual=args.individual)
    model1 = build_model(
        stage1_spec,
        input_len=input_len,
        pred_len=pred_len,
        channels=C,
    )
    adapter1 = EncOnlyAdapter(
        model=model1,
        cfg=EncOnlyAdapterConfig(pred_len=pred_len, device=device),
    )

    train_stage1(
        plan=plan,
        run=run,
        stage1=stage1,
        builder=builder,
        data_spec=data_spec,
        adapter=adapter1,
        fit_cfg=fit1,
        scale=args.scale,
    )

    generate_pv(
        plan=plan,
        run=run,
        stage1=stage1,
        builder=builder,
        data_spec=data_spec,
        adapter=adapter1,
        scale=args.scale,
        overwrite=args.overwrite_pv,
    )

    # =========================================================
    # Stage 2
    # =========================================================
    stage2_spec = ModelSpec(name=args.stage2_model, individual=args.individual)
    model2_factory = make_model_factory(stage2_spec)

    train_stage2(
        plan=plan,
        run=run,
        stage1=stage1,
        stage2=stage2,
        builder=builder,
        data_spec=data_spec,
        policy=policy,
        fit_cfg=fit2,
        model_factory=model2_factory,
        scale=args.scale,
        max_segments=args.max_segments,
    )

    # =========================================================
    # Rank segments (val)
    # =========================================================
    rank_payload = rank_segments_by_val(
        plan=plan,
        run=run,
        stage2=stage2,
        save=False,
    )
    ranked_ids = [int(r["seg_id"]) for r in rank_payload["sorted"]]

    # =========================================================
    # Select best K (Var+Corr)
    # =========================================================
    bestk_payload = select_best_k_varcorr(
        plan=plan,
        run=run,
        stage1=stage1,
        stage2=stage2,
        builder=builder,
        data_spec=data_spec,
        policy=policy,
        ranked_seg_ids=ranked_ids,
        model_factory=model2_factory,
        mode_split=args.mode_split,
        step=args.k_step,
        k_max=args.k_max,
        alpha=args.alpha,
        beta=args.beta,
        scale=args.scale,
        save=False,
    )
    best_top_ids = bestk_payload["best"]["best_top_ids"]
    best_k = int(bestk_payload["best"]["best_k"])

    # =========================================================
    # Aggregate final prediction (top-K mean) on TEST
    # =========================================================
    agg_payload = aggregate_topk_mean(
        plan=plan,
        run=run,
        stage1=stage1,
        stage2=stage2,
        builder=builder,
        data_spec=data_spec,
        policy=policy,
        top_ids=best_top_ids,
        model_factory=model2_factory,
        split="test",
        scale=args.scale,
        save_pred=True,
    )

    # print(f"[DONE] dataset={args.dataset} L={input_len} P={pred_len}")
    # print(f"       stage1={args.stage1_model} stage2={args.stage2_model} best_k={best_k}")
    # print(f"       top_ids={best_top_ids}")
    # if "paths" in agg_payload:
    #     print(f"       outputs={agg_payload['paths']}")
    
    metrics = agg_payload.get("metrics", {})
    mse = metrics.get("mse", float("nan"))
    mae = metrics.get("mae", float("nan"))

    print(f"[DONE] dataset={args.dataset} L={input_len} P={pred_len}")
    print(f"       stage1={args.stage1_model} stage2={args.stage2_model} best_k={best_k}")
    print(f"       top_ids={best_top_ids}")
    print(f"       [ensemble test] MSE={mse:.6f}  MAE={mae:.6f}")   # ← 추가
    if "paths" in agg_payload:
        print(f"       outputs={agg_payload['paths']}")


if __name__ == "__main__":
    main()
