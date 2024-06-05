import torch
import torch.nn as nn
from transformer.model.layer.positional_encoding import PositionalEncoding
from transformer.model.layer.encoder_block import EncoderBlock
from transformer.model.layer.decoder_block import DecoderBlock


class Transformer(nn.Module):

    def __init__(self, head, d_model, sequence_max_len, transformer_block_num, src_pad_idx, trg_pad_idx, src_vocab_size, trg_vocab_size, ffn_hidden,
                 dropout, device):
        super(Transformer, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.device = device
        self.encoder = Encoder(head, d_model, sequence_max_len, transformer_block_num, src_vocab_size, ffn_hidden,
                               dropout, src_pad_idx, device)
        self.decoder = Decoder(head, d_model, sequence_max_len, transformer_block_num, trg_vocab_size, ffn_hidden,
                               dropout, trg_pad_idx, device)

    def forward(self, src, trg):
        src_mask = self.generate_src_mask(src, self.src_pad_idx)
        trg_mask = self.generate_trg_mask(trg)
        state = self.encoder(src, src_mask)
        return self.decoder(trg, state, src_mask, trg_mask)

    @staticmethod
    def generate_src_mask(x: torch.Tensor, pad_idx):
        # Mask the attention weights for dim = 3 so padding doesn't contribute to the attention output of other token
        return (x != pad_idx).unsqueeze(1).unsqueeze(2)

    def generate_trg_mask(self, x: torch.Tensor):
        return (torch.tril(torch.ones(x.shape[1], x.shape[1], device=self.device)) == 1).unsqueeze(0).unsqueeze(0)


class Encoder(nn.Module):

    def __init__(self, head, d, max_len, block_num, vocab_size, ffn_hidden, dropout, pad_idx, device):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(max_len, d, device)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList()
        for i in range(block_num):
            self.blocks.append(EncoderBlock(head, d, ffn_hidden, dropout))

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, mask)
        return x


class Decoder(nn.Module):

    def __init__(self, head, d, max_len, block_num, vocab_size, ffn_hidden, dropout, pad_idx, device):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(max_len, d, device)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList()
        for i in range(block_num):
            self.blocks.append(DecoderBlock(head, d, ffn_hidden, dropout))
        self.linear = nn.Linear(d, vocab_size)

    def forward(self, x, enc_state, src_mask, trg_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, enc_state, src_mask, trg_mask)
        return self.linear(x)
