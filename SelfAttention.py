import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.w_sa = nn.Parameter(torch.empty(1, hidden_dim, hidden_dim),requires_grad=False)
        self.b = nn.Parameter(torch.empty(1, 1, hidden_dim),requires_grad=True)

        # init
        nn.init.kaiming_normal_(self.w_sa)
        nn.init.kaiming_normal_(self.b)

    def forward(self, hiddens, mask):
        B = hiddens.size(0)
        w_sa = self.w_sa.expand(B, -1, -1)  # B, H, H
        b = self.b.expand(B, -1, -1)  # B, 1, H
        logits = torch.bmm(b, F.tanh(torch.bmm(hiddens, w_sa).permute(0, 2, 1)))  
        logits.masked_fill_(torch.logical_not(mask), -1e9)
        probs = torch.softmax(logits, -1)

        sa_hidden = torch.bmm(probs, hiddens)
        return sa_hidden