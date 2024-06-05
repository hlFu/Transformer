import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, head, d):
        super(Attention, self).__init__()

        self.head = head
        self.d = d

        self.q_w = nn.Linear(d, d)
        self.k_w = nn.Linear(d, d)
        self.v_w = nn.Linear(d, d)
        self.o_w = nn.Linear(d, d)

    def forward(self, q, k, v, mask):
        q = self.q_w(q)
        k = self.k_w(k)
        v = self.v_w(v)

        return self.multi_head_attention(q, k, v, mask)

    def multi_head_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask):
        t_q = q.shape[1]
        t_k = k.shape[1]
        t_v = v.shape[1]
        d_per_head = self.d // self.head

        q = q.view(-1, t_q, self.head, d_per_head)
        k = k.view(-1, t_k, self.head, d_per_head)
        v = v.view(-1, t_v, self.head, d_per_head)

        # shape (n, h, t, d_h)
        q = q.transpose(1, 2)
        # shape (n, h, d_h, t)
        k = k.permute(0, 2, 3, 1)
        # shape(n, h, t, d_h)
        v = v.transpose(1, 2)

        s = q @ k / math.sqrt(d_per_head)
        s = F.softmax(s.masked_fill(mask, -1000000), dim=3)
        o = s @ v
        o = o.transpose(1, 2).reshape(-1, t_q, self.d)

        return self.o_w(o)

