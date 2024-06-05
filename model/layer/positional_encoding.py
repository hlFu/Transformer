import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, max_len, d, device):
        super(PositionalEncoding, self).__init__()

        pos = torch.arange(0, max_len, device=device).unsqueeze(1)
        dimension_wise_denominator = torch.pow(torch.tensor([10000], device=device),
                                               torch.arange(0, d, 2, device=device) / d).unsqueeze(0)
        self.positional_encoding = torch.zeros(max_len, d, device=device)
        self.positional_encoding[:, 0::2] = torch.sin(pos / dimension_wise_denominator)
        self.positional_encoding[:, 1::2] = torch.cos(pos / dimension_wise_denominator)
        self.positional_encoding.requires_grad = False

    def forward(self, x):
        return x + self.positional_encoding.unsqueeze(0)[:, :x.shape[1], :]