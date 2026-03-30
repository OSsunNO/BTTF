import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual

        # ----- Linear layer(s): 입력 길이는 항상 seq_len -----
        if self.individual:
            self.Linear = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len)
                for _ in range(self.channels)
            ])
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        """
        x: [B, seq_len, C]
        return: [B, pred_len, C]
        """
        B, L, C = x.size()

        if self.individual:
            out = torch.zeros(
                [B, self.pred_len, C],
                dtype=x.dtype,
                device=x.device
            )
            for i in range(C):
                # (B, seq_len) -> (B, pred_len)
                out[:, :, i] = self.Linear[i](x[:, :, i])
            return out
        else:
            # shared linear
            # (B, seq_len, C) -> (B, C, seq_len)
            out = self.Linear(x.permute(0, 2, 1))
            # (B, C, pred_len) -> (B, pred_len, C)
            return out.permute(0, 2, 1)
