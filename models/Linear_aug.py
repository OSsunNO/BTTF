import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        # --- 새 옵션: attach_pv ---
        self.attach_pv = getattr(configs, "attach_pv", False)

        # 추가 시퀀스 길이 = pred_len // 3 (DLinear 개선 버전과 동일)
        self.extra_len = self.pred_len // 3

        # augmented seq_len = 기본 seq_len + extra_len (if attach_pv)
        self.seq_len_aug = self.seq_len + (self.extra_len if self.attach_pv else 0)

        self.individual = configs.individual

        # ----- Linear layer(s) -----
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len_aug, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len_aug, self.pred_len)


    def forward(self, x, ground_truth=None):
        B, L, C = x.size()

        # --------------------------------------------------
        # 1) 예측값 기반 증강: attach_pv = True & ground_truth 제공
        # --------------------------------------------------
        if self.attach_pv and (ground_truth is not None):
            # ground truth의 앞 extra_len 만큼만 사용 (DLinear 개선 버전과 동일)
            gt_part = ground_truth[:, :self.extra_len, :]  # [B, extra_len, C]

            # concat → [B, L + extra_len, C]
            x = torch.cat([x, gt_part], dim=1)

        # --------------------------------------------------
        # 2) Linear projection (기존 Linear.py 구조 동일)
        # --------------------------------------------------
        if self.individual:
            out = torch.zeros([B, self.pred_len, C], dtype=x.dtype).to(x.device)
            for i in range(C):
                out[:, :, i] = self.Linear[i](x[:, :, i])
            return out

        else:
            # shared linear: (B, C, L_aug) → Linear → (B, C, pred_len)
            out = self.Linear(x.permute(0, 2, 1))
            return out.permute(0, 2, 1)
