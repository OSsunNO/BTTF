# BTTF

## Project Structure

```
BTTF/
├── scripts/
│   └── run.py              # Single experiment entry point
├── run_parallel.py         # Parallel experiment launcher (multi-GPU)
├── core/
│   ├── dataclass/
│   │   ├── dataset_configs.py   # Per-dataset hyperparameter configs
│   │   └── ts_dataset.py        # Data loading / split / DataLoader
│   ├── registry.py         # Dataset & model builders
│   ├── segment.py          # Segment policy
│   └── task.py             # Experiment path management
├── adapters/
│   ├── base.py             # FitConfig definition
│   └── enc_only.py         # Training loop (with LR scheduler)
├── trainers/
│   ├── stage1_train.py
│   └── stage2_train.py
├── stages/
│   └── pv_generate.py
├── ensembles/
│   ├── rank.py
│   ├── select_k.py
│   └── aggregate.py
├── models/
│   ├── Linear.py
│   └── DLinear.py
└── dataset/
    ├── ETTh1.csv
    ├── ETTm2.csv
    ├── exchange_rate.csv
    └── national_illness.csv
```

---

## Pipeline

```
Stage1 Train → PV Generate → Stage2 Train (per segment) → Rank → Select-K → Aggregate
```

1. **Stage1**: Train a first-stage forecasting model on the full time series
2. **PV Generate**: Run inference on train/val/test with the Stage1 model (shuffle=False, drop_last=False)
3. **Stage2**: Split PV into N segments and train a second-stage model per segment
4. **Rank**: Sort segments by val MSE
5. **Select-K**: Select the optimal K segments using Var+Corr criterion
6. **Aggregate**: Average predictions from top-K segments → output final MSE/MAE

---

## Running a Single Experiment

Run from the project root:

```bash
cd BTTF_New
python -m scripts.run \
    --dataset exchange_rate \
    --pred_len 96 \
    --target_col 7 \
    --device cuda:0 \
    --scale
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `ER` | Dataset name (`exchange_rate`, `etth1`, `ettm2`, `illness`) |
| `--pred_len` | dataset default | Forecast horizon |
| `--target_col` | `None` (multivariate) | Target column index for univariate (0-indexed, excluding date) |
| `--input_len` | dataset default (336) | Input sequence length |
| `--device` | `cuda` | `cuda`, `cuda:0`, `cpu` |
| `--seed` | `0` | Random seed |
| `--scale` | `False` | Apply StandardScaler (flag) |
| `--epochs1` | `10` | Max epochs for Stage1 |
| `--epochs2` | `10` | Max epochs for Stage2 |
| `--lr1` | dataset default | Stage1 learning rate (uses dataset_configs value if not set) |
| `--lr2` | dataset default | Stage2 learning rate |
| `--batch_size` | dataset default | Batch size (uses per-pred_len config if not set) |
| `--patience` | dataset default (3) | Early stopping patience |
| `--stage1_model` | `Linear` | `Linear` or `DLinear` |
| `--stage2_model` | `Linear` | `Linear` or `DLinear` |
| `--seg_mode` | `div` | Segment split mode (`div`, `ratio`, `fixed`) |
| `--seg_div` | `3` | Number of divisions for pred_len (when mode=div) |
| `--seg_stride` | `1` | Segment stride (`0` = auto from dataset_configs) |
| `--root_dir` | `./outputs` | Output root directory |
| `--stage1_tag` | `stage1` | Tag for stage1 output path |
| `--stage2_tag` | `stage2` | Tag for stage2 output path |
| `--overwrite_pv` | `False` | Force regenerate PV (flag) |

### target_col Index (0-indexed, excluding date column)

| Dataset | target_col |
|---|---|
| exchange_rate | `7` |
| etth1 | `6` |
| ettm2 | `6` |
| illness | `6` |

---

## Running All Experiments (Full Grid)

To run all datasets × all pred_len combinations, use `run_parallel.py`.
It distributes experiments across multiple GPUs dynamically.

```bash
cd BTTF_New

# Run all 16 experiments across GPUs 0,1,2,3 (default)
python run_parallel.py

# Use specific GPUs only
python run_parallel.py --gpus 0,1

# Filter by dataset only
python run_parallel.py --dataset exchange_rate

# Filter by dataset + pred_len
python run_parallel.py --dataset etth1 --pred_len 96
```

### Full experiment grid (defined in `run_parallel.py`)

| Dataset | pred_len candidates | target_col |
|---|---|---|
| exchange_rate | 96, 192, 336, 720 | 7 |
| etth1 | 96, 192, 336, 720 | 6 |
| ettm2 | 96, 192, 336, 720 | 6 |
| illness | 24, 36, 48, 60 | 6 |

**Total: 16 experiments**

### How parallel execution works

- Creates one worker process per GPU
- Workers pull experiments from a shared queue until it is empty
- As soon as a GPU finishes, it picks up the next experiment immediately
- Each experiment runs as a subprocess (`python -m scripts.run`)
- Logs saved to `outputs/logs/{dataset}_P{pred_len}_gpu{gpu_id}.log`

---

## Changing Models or Epochs (Full Grid)

To run the full grid with different models or epoch settings, edit `COMMON_ARGS` in `run_parallel.py`.
**Always change `--root_dir` or `--stage1_tag`/`--stage2_tag` at the same time to avoid overwriting existing results.**

### Option A — separate by `root_dir` (recommended for entirely different experiment sets)

```python
# run_parallel.py
COMMON_ARGS = [
    "--root_dir",     "./outputs_dlinear_ep30",  # ← new folder
    "--epochs1",      "30",
    "--epochs2",      "30",
    "--stage1_model", "DLinear",
    "--stage2_model", "DLinear",
    "--stage1_tag",   "stage1",
    "--stage2_tag",   "stage2",
    ...
]
```

### Option B — separate by `stage_tag` (for comparing models within same outputs folder)

```python
# run_parallel.py
COMMON_ARGS = [
    "--root_dir",     "./outputs",
    "--epochs1",      "30",
    "--epochs2",      "30",
    "--stage1_model", "DLinear",
    "--stage2_model", "DLinear",
    "--stage1_tag",   "stage1_dlinear_ep30",     # ← new tag
    "--stage2_tag",   "stage2_dlinear_ep30",
    ...
]
```

The resulting directory layout when using tags:

```
outputs/{dataset}/L{input_len}_P{pred_len}/seed{seed}/
├── stage1/                  ← Linear, epoch=1 (default run)
├── stage2/
├── stage1_dlinear_ep30/     ← DLinear, epoch=30
└── stage2_dlinear_ep30/
```

---

## Training Epochs & Early Stopping

Both Stage1 and Stage2 use the same epoch / early stopping settings.

| Use case | `--epochs1` / `--epochs2` | `--patience` | Behavior |
|---|---|---|---|
| Quick test | `1` | `3` | Runs exactly 1 epoch, no early stopping |
| Standard | `10` | `3` | Stops early if val loss does not improve for 3 epochs |
| Full training | `30`–`50` | `3` | Early stopping triggers well before the epoch limit |

Setting `--epochs1 30 --patience 3` means training runs **at most 30 epochs**, but stops as soon as val loss fails to improve for 3 consecutive epochs — in practice this is usually around epoch 5–10.

```python
# run_parallel.py — change epochs for full grid
COMMON_ARGS = [
    ...
    "--epochs1",  "30",
    "--epochs2",  "30",
    "--patience", "3",   # early stopping kicks in before epoch limit
    ...
]
```

---

## Dataset Configurations

### Data Split

| Dataset | Split method | Train end | Val end | Test end |
|---|---|---|---|---|
| exchange_rate | ratio | 70% | 80% | 100% |
| etth1 | fixed point | 8640 | 11520 | 14400 |
| ettm2 | fixed point | 34560 | 46080 | 57600 |
| illness | ratio | 70% | 80% | 100% |

Val and test windows are extended backward by `input_len` to allow window sliding from the start (border-based split, consistent with reference notebooks).

### Per-dataset Hyperparameters

**exchange_rate** — input_len=336, patience=3

| pred_len | batch_size | lr | stride |
|---|---|---|---|
| 96 | 8 | 5e-4 | 1 |
| 192 | 8 | 5e-4 | 2 |
| 336 | 32 | 5e-4 | 4 |
| 720 | 32 | 5e-4 | 8 |

**etth1** — input_len=336, patience=3

| pred_len | batch_size | lr | stride |
|---|---|---|---|
| 96 | 32 | 5e-3 | 1 |
| 192 | 32 | 5e-3 | 2 |
| 336 | 32 | 5e-3 | 4 |
| 720 | 32 | 5e-3 | 8 |

**ettm2** — input_len=336, patience=3

| pred_len | batch_size | lr | stride |
|---|---|---|---|
| 96 | 32 | 1e-3 | 1 |
| 192 | 32 | 1e-3 | 2 |
| 336 | 32 | 1e-2 | 4 |
| 720 | 32 | 1e-2 | 8 |

**illness** — input_len=104, patience=3

| pred_len | batch_size | lr | stride |
|---|---|---|---|
| 24 | 32 | 1e-2 | 1 |
| 36 | 32 | 1e-2 | 1 |
| 48 | 32 | 1e-2 | 1 |
| 60 | 32 | 1e-2 | 1 |


## LR Scheduler

`type1` scheduler is applied by default: learning rate is halved every epoch.

```
epoch 1: lr × 1.0
epoch 2: lr × 0.5
epoch 3: lr × 0.25
...
```

Combined with early stopping (patience=3), training typically stops within 3–5 epochs.

---

## ⚠️ Output Path Warning

The output path is determined only by: `dataset`, `input_len`, `pred_len`, `seed`, `stage1_tag`, `stage2_tag`.

**Changes to `--lr`, `--batch_size`, `--epochs`, `--stage1_model` etc. do NOT affect the path.**
Re-running with different settings will silently overwrite existing checkpoints and metrics.

Always change `--root_dir` or `--stage1_tag`/`--stage2_tag` when running a new experiment configuration.