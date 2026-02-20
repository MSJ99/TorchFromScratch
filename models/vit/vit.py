import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d()
        self.cls_token = nn.Parameter(torch.zeros())
        self.pe = nn.Parameter(torch.zeros())


    def forward(self, x):
        B = x.shape[0]

        x = self.proj(x)
        x = x.flatten(2).permute(0, 2, 1)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pe
        
        return x

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(),
            nn.GELU(),
            nn.Linear(),
        )


    def forward(self, x):
        return self.mlp(x)


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm()
        self.msa = nn.MultiheadAttention()
        
        self.norm2 = nn.LayerNorm()
        self.mlp = MLP()


    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = residual + self.msa(x, x, x)[0]

        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)

        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer() for _ in range()
        ])
        self.norm = nn.LayerNorm()


    def forward(self, x):
        for layer in self.layers:
            x = layer()
        x = self.norm(x)

        return x

# TODO: CLS Token, PE
class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = PatchEmbedding()
        self.enc = Encoder()
        self.norm = nn.LayerNorm()
        self.head = nn.Linear()


    def forward(self, x):
        x = self.embedding(x)
        
        x = self.enc(x)
        
        x = self.norm(x)
        x = self.head(x)

        return x