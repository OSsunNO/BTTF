from __future__ import annotations

import argparse
import subprocess
import time
from multiprocessing import Pool, Queue, Manager
from pathlib import Path
from typing import List, Tuple, Optional


ALL_EXPERIMENTS: List[Tuple[str, int, int]] = [
    # (dataset, pred_len, target_col)
    ("exchange_rate", 96,  7),
    ("exchange_rate", 192, 7),
    ("exchange_rate", 336, 7),
    ("exchange_rate", 720, 7),
    ("etth1",  96,  6),
    ("etth1",  192, 6),
    ("etth1",  336, 6),
    ("etth1",  720, 6),
    ("ettm2",  96,  6),
    ("ettm2",  192, 6),
    ("ettm2",  336, 6),
    ("ettm2",  720, 6),
    ("illness", 24, 6),
    ("illness", 36, 6),
    ("illness", 48, 6),
    ("illness", 60, 6),
]


COMMON_ARGS = [
    "--root_dir",    "./outputs/dlinear_1E1E",
    "--seed",        "0",
    "--epochs1",     "1",
    "--epochs2",     "1",
    "--num_workers", "0",
    "--seg_mode",    "div",
    "--seg_div",     "3",
    "--seg_stride",  "0",
    "--k_step",      "5",
    "--alpha",       "1.0",
    "--beta",        "1.0",
    "--mode_split",  "test",
    "--stage1_model", "DLinear",
    "--stage2_model", "DLinear",
    "--stage1_tag",  "stage1",
    "--stage2_tag",  "stage2",
    "--scale",
]

# ← 이 줄 추가
ROOT_DIR = Path(COMMON_ARGS[COMMON_ARGS.index("--root_dir") + 1])


def run_experiment(
    dataset: str,
    pred_len: int,
    target_col: int,
    gpu_id: int,
    log_dir: Path,
) -> Tuple[str, int, bool, float]:
    

    log_path = log_dir / f"{dataset}_P{pred_len}_gpu{gpu_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "scripts.run",
        "--dataset",    dataset,
        "--pred_len",   str(pred_len),
        "--target_col", str(target_col),
        "--device",     f"cuda:{gpu_id}",
        *COMMON_ARGS,
    ]

    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0
    success = proc.returncode == 0
    return dataset, pred_len, success, elapsed



def worker(gpu_id: int, job_queue: "Queue", result_list: list) -> None:
    
    log_dir = ROOT_DIR / "logs"

    while True:
        try:
            dataset, pred_len, target_col = job_queue.get_nowait()
        except Exception:
            break  # 큐가 비면 종료

        print(f"[GPU {gpu_id}] START  {dataset} P={pred_len}", flush=True)
        ds, pl, ok, elapsed = run_experiment(dataset, pred_len, target_col, gpu_id, log_dir)
        status = "✅ DONE" if ok else "❌ FAIL"
        print(f"[GPU {gpu_id}] {status} {ds} P={pl}  ({elapsed/60:.1f}min)  "
              f"log: {log_dir}/{ds}_P{pl}_gpu{gpu_id}.log", flush=True)

        result_list.append({
            "dataset": ds, "pred_len": pl,
            "gpu": gpu_id, "success": ok, "elapsed_min": round(elapsed / 60, 2),
            "log": str(log_dir / f"{ds}_P{pl}_gpu{gpu_id}.log"),
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus",       type=str, default="0,1,2,3",
                        help="사용할 GPU ID (쉼표 구분). 예: 0,1,2,3")
    parser.add_argument("--dataset",    type=str, default=None,
                        help="특정 데이터셋만 실행")
    parser.add_argument("--pred_len",   type=int, default=None,
                        help="특정 pred_len만 실행")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]

    
    experiments = ALL_EXPERIMENTS
    if args.dataset:
        experiments = [(d, p, c) for d, p, c in experiments if d == args.dataset]
    if args.pred_len:
        experiments = [(d, p, c) for d, p, c in experiments if p == args.pred_len]

    if not experiments:
        print("No experiments to run. Check --dataset / --pred_len.")
        return

    print(f"{'='*55}")
    print(f"  Total Experiments: {len(experiments)}  |  GPU {gpu_ids}  ({len(gpu_ids)} GPUs)")
    print(f"  Estimated Rounds: {-(-len(experiments) // len(gpu_ids))}")  # ceil division
    print(f"{'='*55}")
    for d, p, c in experiments:
        print(f"  {d:20s}  P={p:4d}  target_col={c}")
    print(f"{'='*55}\n")


    with Manager() as manager:
        job_queue: Queue = manager.Queue()
        result_list = manager.list()

        for exp in experiments:
            job_queue.put(exp)

    
        with Pool(processes=len(gpu_ids)) as pool:
            pool.starmap(
                worker,
                [(gpu_id, job_queue, result_list) for gpu_id in gpu_ids],
            )

        
        results = list(result_list)

    print(f"\n{'='*55}")
    print("  Experiment Results Summary")
    print(f"{'='*55}")
    total_ok   = sum(1 for r in results if r["success"])
    total_fail = len(results) - total_ok
    for r in sorted(results, key=lambda x: (x["dataset"], x["pred_len"])):
        mark = "✅" if r["success"] else "❌"
        print(f"  {mark} {r['dataset']:20s} P={r['pred_len']:4d}  "
              f"GPU={r['gpu']}  {r['elapsed_min']:.1f}min")
    print(f"{'='*55}")
    print(f"  Success: {total_ok}  Failure: {total_fail}")
    if total_fail > 0:
        print(f"  Failure Log: {ROOT_DIR / 'logs'}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
