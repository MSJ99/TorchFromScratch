import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_pad_mask(q, k, pad_idx=0):
    return (k != pad_idx).unsqueeze(dim=1).unsqueeze(dim=2)


def make_tgt_mask(tgt):
    seq_len = tgt.size(1)
    mask = torch.tril(torch.ones((seq_len, seq_len))).type(torch.bool).to(tgt.device)

    return mask.unsqueeze(dim=0).unsqueeze(dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(dim=1)
        div = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))

        pe[:,0::2] = torch.sin(pos * div)
        pe[:,1::2] = torch.cos(pos * div)

        self.register_buffer('pe', pe.unsqueeze(dim=0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]

        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x_q, x_k, x_v, mask=None):
        d_k = x_q.size(-1)

        attention = torch.matmul(x_q, x_k.permute(0, 1, 3, 2)) / math.sqrt(d_k)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)

        output = torch.matmul(F.softmax(attention, dim=-1), x_v)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, h=8):
        super().__init__()
        self.n_head = h
        self.d_k = d_model // h

        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_K = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)

        self.sdpa = ScaledDotProductAttention()

        self.linear_O = nn.Linear(d_model, d_model)


    def forward(self, x_q, x_k, x_v, mask=None):
        batch_size = x_q.size(0)

        L_Q = self.linear_Q(x_q).reshape(batch_size, -1, self.n_head, self.d_k).permute(0, 2, 1, 3)
        L_K = self.linear_K(x_k).reshape(batch_size, -1, self.n_head, self.d_k).permute(0, 2, 1, 3)
        L_V = self.linear_V(x_v).reshape(batch_size, -1, self.n_head, self.d_k).permute(0, 2, 1, 3)

        x = self.sdpa(L_Q, L_K, L_V, mask)

        x = self.linear_O(x.permute(0, 2, 1, 3).reshape(batch_size, -1, self.n_head * self.d_k))

        return x


class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )


    def forward(self, x):
        x = self.ffn(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, h)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)


    def forward(self, x, mask=None):
        x = self.norm1(self.mha(x, x, x, mask) + x)

        x = self.norm2(self.ffn(x) + x)

        return x


class Encoder(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, N=6):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, h, d_ff) for _ in range(N)
        ])


    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, h)
        self.norm1 = nn.LayerNorm(d_model)

        self.mha2 = MultiHeadAttention(d_model, h)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)


    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.norm1(self.mha1(x, x, x, tgt_mask) + x)
        x = self.norm2(self.mha2(x, enc_output, enc_output, src_mask) + x)
        x = self.norm3(self.ffn(x) + x)

        return x


class Decoder(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, N=6):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, h, d_ff) for _ in range(N)
        ])


    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, h=8, d_ff=2048, N=6):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model)

        self.enc = Encoder(d_model, h, d_ff, N)
        self.dec = Decoder(d_model, h, d_ff, N)

        self.fc = nn.Linear(d_model, tgt_vocab_size)


    def forward(self, src, tgt, src_pad_idx=0, tgt_pad_idx=0):
        src_mask = make_pad_mask(src, src, src_pad_idx)
        tgt_mask = make_pad_mask(tgt, tgt, tgt_pad_idx) & make_tgt_mask(tgt)

        x_enc = self.src_embedding(src)
        x_enc = self.pos_encoding(x_enc)
        enc_output = self.enc(x_enc, src_mask)

        x_dec = self.tgt_embedding(tgt)
        x_dec = self.pos_encoding(x_dec)
        dec_output = self.dec(x_dec, enc_output, src_mask, tgt_mask)

        x = self.fc(dec_output)

        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Transformer(src_vocab_size=10, tgt_vocab_size=10).to(device)
    src = torch.LongTensor([[1, 2, 3, 4, 5, 0, 0]]).to(device)
    tgt = torch.LongTensor([[1, 2, 3, 0]]).to(device)

    output = model(src, tgt)

    print(output.shape)