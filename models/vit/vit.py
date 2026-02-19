import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()


class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()


class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear()
        self.layer2 = nn.Linear()
        self.layer3 = nn.Linear()
        self.gelu = nn.GELU()


    def forward(self, x):
        x = self.gelu(self.layer3(self.layer2(self.layer1(x))))

        return x

# TODO: y ?
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm()
        self.msa = MultiHeadSelfAttention()
        
        self.norm2 = nn.LayerNorm()
        self.mlp = MultiLayerPerceptron()


    def forward(self, x):
        residual1 = x
        x = self.norm1(x)
        x = residual1 + self.msa(x, x, x)

        residual2 = x
        x = self.norm2(x)
        x = residual2 + self.msa(x, x, x)

        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer() for _ in range()
        ])


    def forward(self, x):
        for layer in self.layers:
            x = layer()

        return x


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = PatchEmbedding()
        self.enc = Encoder()
        self.fc = nn.Linear()


    def forward(self, x):
        x = self.embedding(x)
        x = self.enc(x)
        x = self.fc(x)

        return x