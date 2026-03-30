# adapters/enc_only.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .base import ForecastModelAdapter, FitConfig, mse_torch


@dataclass
class EncOnlyAdapterConfig:
    """
    Config for encoder-only adapter.

    pred_len is not forced here because some models internally know it,
    but for safety you may set it and we will slice outputs if needed.
    """
    pred_len: Optional[int] = None
    device: str = "cuda"


class EncOnlyAdapter(ForecastModelAdapter):
    """
    Adapter for models that follow:
        y_hat = model(x)
    where:
        x: [B, L, C]
        y_hat: [B, P, C] (or [B, P] for univariate - we normalize to 3D if needed)
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: EncOnlyAdapterConfig,
        *,
        criterion=mse_torch,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.criterion = criterion

        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scaler: Optional[torch.cuda.amp.GradScaler] = None

    # ----------------- required by base -----------------
    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "cfg": {
                "pred_len": self.cfg.pred_len,
                "device_requested": str(self.cfg.device),
                "device_actual": str(self.device),
            },
        }
        if self._optimizer is not None:
            payload["optim_state"] = self._optimizer.state_dict()
        torch.save(payload, str(path))

    def load(self, path: Path, *, map_location: Optional[str | torch.device] = None) -> None:
        if map_location is None:
            map_location = self.device
        payload = torch.load(str(path), map_location=map_location)

        self.model.load_state_dict(payload["model_state"])
        # optimizer state is optional; only load if already created
        if self._optimizer is not None and "optim_state" in payload:
            self._optimizer.load_state_dict(payload["optim_state"])

    @torch.no_grad()
    def predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        x = x.to(self.device)

        y_hat = self.model(x)
        y_hat = self._normalize_pred(y_hat)

        # safety: slice to pred_len if configured
        if self.cfg.pred_len is not None:
            y_hat = y_hat[:, -self.cfg.pred_len:, :]
#                          ^ 콜론 꼭 필요



        return y_hat

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        *,
        fit_cfg: FitConfig,
        save_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        self._build_optimizer_if_needed(fit_cfg)
        logs: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
        }

        best_val = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None

        # for epoch in range(1, fit_cfg.epochs + 1):
        #     tr_loss = self._train_one_epoch(train_loader, fit_cfg)
        #     logs["train_loss"].append(tr_loss)

        #     va_loss = None
        #     if val_loader is not None:
        #         va_loss = self._eval_one_epoch(val_loader, fit_cfg)
        #         logs["val_loss"].append(va_loss)

        #         # keep best
        #         if va_loss < best_val:
        #             best_val = va_loss
        #             best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        #             # optional save each epoch (or best-only by external logic)
        #             if save_path is not None:
        #                 self.save(save_path)

        # 수정 후
        patience_counter = 0                                          # ← 추가

        for epoch in range(1, fit_cfg.epochs + 1):
            if fit_cfg.lr_scheduler == "type1" and self._optimizer is not None:
                new_lr = fit_cfg.lr * (0.5 ** (epoch - 1))
                for pg in self._optimizer.param_groups:
                    pg["lr"] = new_lr

            tr_loss = self._train_one_epoch(train_loader, fit_cfg)
            logs["train_loss"].append(tr_loss)

            va_loss = None
            if val_loader is not None:
                va_loss = self._eval_one_epoch(val_loader, fit_cfg)
                logs["val_loss"].append(va_loss)

                if va_loss < best_val:
                    best_val = va_loss
                    patience_counter = 0                              # ← 개선되면 리셋
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    if save_path is not None:
                        self.save(save_path)
                else:
                    patience_counter += 1                             # ← 개선 없으면 카운트
                    if fit_cfg.patience > 0 and patience_counter >= fit_cfg.patience:
                        logs["early_stopped_epoch"] = epoch          # ← 로그에 기록
                        break                                         # ← 조기 종료

        # restore best (if val available)
        if best_state is not None:
            self.model.load_state_dict(best_state)
            if save_path is not None:
                self.save(save_path)

        logs["best_val"] = best_val if val_loader is not None else None
        return logs

    # ----------------- internal helpers -----------------
    def _build_optimizer_if_needed(self, fit_cfg: FitConfig) -> None:
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=fit_cfg.lr,
                weight_decay=fit_cfg.weight_decay,
            )
        if fit_cfg.amp and torch.cuda.is_available():
            self._scaler = torch.cuda.amp.GradScaler()
        else:
            self._scaler = None

    def _normalize_pred(self, y_hat: torch.Tensor) -> torch.Tensor:
        # ensure [B, P, C]
        if y_hat.ndim == 2:
            # [B, P] -> [B, P, 1]
            y_hat = y_hat[:, :, None]
        if y_hat.ndim != 3:
            raise ValueError(f"model output must be 2D or 3D, got {tuple(y_hat.shape)}")
        return y_hat

    def _normalize_true(self, y: torch.Tensor) -> torch.Tensor:
        # ensure [B, P, C]
        if y.ndim == 2:
            y = y[:, :, None]
        if y.ndim != 3:
            raise ValueError(f"label y must be 2D or 3D, got {tuple(y.shape)}")
        return y

    def _train_one_epoch(self, loader: DataLoader, fit_cfg: FitConfig) -> float:
        assert self._optimizer is not None
        self.train()

        total = 0.0
        count = 0

        for it, batch in enumerate(loader, start=1):
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError("train batch must be (x, y, ...)")

            x, y = batch[0], batch[1]
            x = x.to(self.device)
            y = y.to(self.device)

            y = self._normalize_true(y)

            self._optimizer.zero_grad(set_to_none=True)

            if self._scaler is not None:
                with torch.cuda.amp.autocast():
                    y_hat = self._normalize_pred(self.model(x))
                    if self.cfg.pred_len is not None:
                        y_hat = y_hat[:, -self.cfg.pred_len :, :]
                    loss = self.criterion(y_hat, y)
                self._scaler.scale(loss).backward()
                if fit_cfg.grad_clip is not None:
                    self._scaler.unscale_(self._optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), fit_cfg.grad_clip)
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                y_hat = self._normalize_pred(self.model(x))
                if self.cfg.pred_len is not None:
                    y_hat = y_hat[:, -self.cfg.pred_len :, :]
                loss = self.criterion(y_hat, y)
                loss.backward()
                if fit_cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), fit_cfg.grad_clip)
                self._optimizer.step()

            total += float(loss.detach().cpu().item())
            count += 1

        return total / max(count, 1)

    @torch.no_grad()
    def _eval_one_epoch(self, loader: DataLoader, fit_cfg: FitConfig) -> float:
        self.eval()
        total = 0.0
        count = 0

        for batch in loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError("val batch must be (x, y, ...)")

            x, y = batch[0], batch[1]
            x = x.to(self.device)
            y = y.to(self.device)
            y = self._normalize_true(y)

            y_hat = self._normalize_pred(self.model(x))
            if self.cfg.pred_len is not None:
                y_hat = y_hat[:, -self.cfg.pred_len :, :]

            loss = self.criterion(y_hat, y)
            total += float(loss.detach().cpu().item())
            count += 1

        return total / max(count, 1)

