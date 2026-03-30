# core/task.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Literal, Any


@dataclass(frozen=True)
class TaskSpec:
    """
    One experiment unit.

    dataset: dataset identifier (e.g., "ER")
    input_len: L (history length)
    pred_len:  P (forecast horizon)
    features:  "M" (multivariate) or "S" (univariate) - optional for extension
    """
    dataset: str
    input_len: int
    pred_len: int
    features: Literal["M", "S"] = "M"

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("dataset must be a non-empty string")
        if self.input_len <= 0:
            raise ValueError(f"input_len must be positive, got {self.input_len}")
        if self.pred_len <= 0:
            raise ValueError(f"pred_len must be positive, got {self.pred_len}")


@dataclass(frozen=True)
class RunSpec:
    """
    Run-time knobs that should NOT affect the model identity but affect execution.

    seed: random seed
    device: 'cpu' or 'cuda'
    precision: 'fp32' or 'fp16' (optional extension)
    """
    seed: int = 0
    device: str = "cuda"
    precision: Literal["fp32", "fp16"] = "fp32"


@dataclass(frozen=True)
class StageSpec:
    """
    Stage-specific knobs (for saving organization and reproducibility).

    tag: identifier for the training regime (e.g., "ES", "E1", "E10", "baseline")
    note: optional free-form string to keep around (not used in paths by default)
    """
    tag: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.tag or not isinstance(self.tag, str):
            raise ValueError("StageSpec.tag must be a non-empty string")


@dataclass(frozen=True)
class ExperimentPlan:
    """
    Defines how artifacts and checkpoints are organized on disk.

    root_dir/
      ckpt/{dataset}/L{L}_P{P}/stage1/{stage1_tag}/model.pth
      ckpt/{dataset}/L{L}_P{P}/stage2/{stage2_tag}/seg{seg_id}/model.pth

      artifacts/{dataset}/L{L}_P{P}/pv/{stage1_tag}/{split}.npy
      metrics/{dataset}/L{L}_P{P}/... (json, csv, etc.)
    """
    root_dir: Path
    task: TaskSpec

    # optional: isolate runs by seed (recommended)
    isolate_by_seed: bool = True
    # optional: include features in path to avoid collisions
    include_features_in_path: bool = False

    def _task_folder(self) -> Path:
        t = self.task
        feat = f"_{t.features}" if self.include_features_in_path else ""
        base = Path(t.dataset) / f"L{t.input_len}_P{t.pred_len}{feat}"
        return base

    def _seed_folder(self, run: RunSpec) -> Path:
        if self.isolate_by_seed:
            return Path(f"seed{run.seed}")
        return Path("")

    # --------- Checkpoints ---------
    def ckpt_stage1_dir(self, stage1: StageSpec, run: RunSpec) -> Path:
        return (self.root_dir / "ckpt" / self._task_folder() / self._seed_folder(run)
                / stage1.tag)

    def ckpt_stage2_dir(self, stage2: StageSpec, run: RunSpec, seg_id: int) -> Path:
        if seg_id < 0:
            raise ValueError(f"seg_id must be >= 0, got {seg_id}")
        return (self.root_dir / "ckpt" / self._task_folder() / self._seed_folder(run)
                 / stage2.tag / f"seg{seg_id:03d}")

    def ckpt_stage1_path(self, stage1: StageSpec, run: RunSpec) -> Path:
        return self.ckpt_stage1_dir(stage1, run) / "model.pth"

    def ckpt_stage2_path(self, stage2: StageSpec, run: RunSpec, seg_id: int) -> Path:
        return self.ckpt_stage2_dir(stage2, run, seg_id) / "model.pth"

    # --------- PV artifacts ---------
    def pv_dir(self, stage1: StageSpec, run: RunSpec) -> Path:
        return (self.root_dir / "artifacts" / self._task_folder() / self._seed_folder(run)
                / "pv" / stage1.tag)

    def pv_path(self, stage1: StageSpec, run: RunSpec, split: Literal["train", "val", "test"]) -> Path:
        return self.pv_dir(stage1, run) / f"{split}.npy"

    # --------- Metrics / results ---------
    def metrics_dir(self) -> Path:
        return self.root_dir / "metrics" / self._task_folder()

    def metrics_stage1_path(self, stage1: StageSpec, run: RunSpec) -> Path:
        # stage1 overall metrics (train/val/test)
        return self.metrics_dir() / self._seed_folder(run) / "stage1" / f"{stage1.tag}.json"

    def metrics_stage2_path(self, stage2: StageSpec, run: RunSpec, seg_id: int) -> Path:
        # stage2 per-segment metrics (val/test)
        return self.metrics_dir() / self._seed_folder(run) / "stage2" / stage2.tag / f"seg{seg_id:03d}.json"

    def ranking_path(self, stage2: StageSpec, run: RunSpec) -> Path:
        # top-k segment indices, etc.
        return self.metrics_dir() / self._seed_folder(run) / "rank" / f"{stage2.tag}.json"

    def ensemble_path(self, stage2: StageSpec, run: RunSpec) -> Path:
        return self.metrics_dir() / self._seed_folder(run) / "ensemble" / f"{stage2.tag}.json"

    # --------- utilities ---------
    def ensure_dirs(self, stage1: Optional[StageSpec] = None, stage2: Optional[StageSpec] = None,
                    run: Optional[RunSpec] = None, seg_id: Optional[int] = None) -> None:
        """
        Create relevant directories. Call this before saving.
        """
        self.root_dir.mkdir(parents=True, exist_ok=True)

        # Metrics root always helpful
        (self.metrics_dir()).mkdir(parents=True, exist_ok=True)

        if run is None:
            return

        if stage1 is not None:
            self.ckpt_stage1_dir(stage1, run).mkdir(parents=True, exist_ok=True)
            self.pv_dir(stage1, run).mkdir(parents=True, exist_ok=True)
            (self.metrics_dir() / self._seed_folder(run) / "stage1").mkdir(parents=True, exist_ok=True)

        if stage2 is not None:
            (self.metrics_dir() / self._seed_folder(run) / "stage2" / stage2.tag).mkdir(parents=True, exist_ok=True)
            # (self.metrics_dir() / self._seed_folder(run) / "rank").mkdir(parents=True, exist_ok=True)
            if seg_id is not None:
                self.ckpt_stage2_dir(stage2, run, seg_id).mkdir(parents=True, exist_ok=True)


def make_default_plan(
    root_dir: str | Path,
    dataset: str,
    input_len: int,
    pred_len: int,
    *,
    features: Literal["M", "S"] = "M",
    isolate_by_seed: bool = True,
    include_features_in_path: bool = False,
) -> ExperimentPlan:
    root_dir = Path(root_dir)
    task = TaskSpec(dataset=dataset, input_len=input_len, pred_len=pred_len, features=features)
    return ExperimentPlan(
        root_dir=root_dir,
        task=task,
        isolate_by_seed=isolate_by_seed,
        include_features_in_path=include_features_in_path,
    )
