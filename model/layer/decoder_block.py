import torch.nn as nn
from transformer.model.layer.attention import Attention
from transformer.model.layer.position_wise_ffn import PositionWiseFFN


class DecoderBlock(nn.Module):

    def __init__(self, head, d, ffn_hidden, dropout):
        super(DecoderBlock, self).__init__()

        self.decoder_attention = Attention(head, d)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d)
        self.decoder_encoder_attention = Attention(head, d)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d)
        self.position_wise_ffn = PositionWiseFFN(d, ffn_hidden, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.ln3 = nn.LayerNorm(d)

    def forward(self, x, enc_state, src_mask, trg_mask):
        x = self.dropout1(self.decoder_attention(x, x, x, trg_mask)) + x
        x = self.ln1(x)
        x = self.dropout2(self.decoder_encoder_attention(x, enc_state, enc_state, src_mask)) + x
        x = self.ln2(x)
        x = self.dropout3(self.position_wise_ffn(x)) + x
        return self.ln3(x)
