import torch.nn as nn
from transformer.model.layer.attention import Attention
from transformer.model.layer.position_wise_ffn import PositionWiseFFN


class EncoderBlock(nn.Module):

    def __init__(self, head, d, ffn_hidden, dropout):
        super(EncoderBlock, self).__init__()

        self.attention = Attention(head, d)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d)
        self.position_wise_ffn = PositionWiseFFN(d, ffn_hidden, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x, mask):
        x = self.dropout1(self.attention(x, x, x, mask)) + x
        x = self.ln1(x)
        x = self.dropout2(self.position_wise_ffn(x)) + x
        return self.ln2(x)