from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_classes, input_shape=(1, 250, 90), hidden=(256, 128), dropout=0.2):
        super().__init__()
        c, t, f = input_shape
        layers: list[nn.Module] = [nn.Flatten()]
        in_dim = c * t * f
        for width in hidden:
            layers += [nn.Linear(in_dim, int(width)), nn.ReLU(), nn.Dropout(float(dropout))]
            in_dim = int(width)
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_shape=(1, 250, 90), channels=(16, 32, 64), blocks=None):
        super().__init__()
        c, _, _ = input_shape
        ch = [int(v) for v in channels]
        if blocks is not None:
            ch = ch[: int(blocks)]
        layers: list[nn.Module] = []
        in_ch = c
        for out_ch in ch:
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(), nn.MaxPool2d(2)]
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


class GRU(nn.Module):
    def __init__(self, num_classes, input_shape=(1, 250, 90), hidden_size=128, num_layers=1, dropout=0.0):
        super().__init__()
        c, _, f = input_shape
        self.input_size = c * f
        self.rnn = nn.GRU(
            self.input_size,
            int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.fc = nn.Linear(int(hidden_size), num_classes)

    def forward(self, x):
        b, c, t, f = x.shape
        z = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        out, _ = self.rnn(z)
        return self.fc(out[:, -1])


class LSTM(nn.Module):
    def __init__(self, num_classes, input_shape=(1, 250, 90), hidden_size=128, num_layers=1, dropout=0.0):
        super().__init__()
        c, _, f = input_shape
        self.input_size = c * f
        self.rnn = nn.LSTM(
            self.input_size,
            int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.fc = nn.Linear(int(hidden_size), num_classes)

    def forward(self, x):
        b, c, t, f = x.shape
        z = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        out, _ = self.rnn(z)
        return self.fc(out[:, -1])


class CNNGRU(nn.Module):
    def __init__(self, num_classes, input_shape=(1, 250, 90), channels=(16, 32), hidden_size=64, hidden=None):
        super().__init__()
        c, _, _ = input_shape
        hidden_size = int(hidden if hidden is not None else hidden_size)
        layers: list[nn.Module] = []
        in_ch = c
        for out_ch in [int(v) for v in channels]:
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(), nn.MaxPool2d((1, 2))]
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d((64, 16)))
        self.enc = nn.Sequential(*layers)
        self.gru = nn.GRU(in_ch * 16, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        z = self.enc(x).permute(0, 2, 1, 3).flatten(2)
        out, _ = self.gru(z)
        return self.fc(out[:, -1])


class TinyViT(nn.Module):
    def __init__(self, num_classes, input_shape=(1, 250, 90), patch_size=(10, 10), dim=128, depth=2, heads=4):
        super().__init__()
        c, t, f = input_shape
        pt, pf = tuple(patch_size)
        self.pad_t = (pt - t % pt) % pt
        self.pad_f = (pf - f % pf) % pf
        t2, f2 = t + self.pad_t, f + self.pad_f
        self.num_patches = (t2 // pt) * (f2 // pf)
        self.patch_embed = nn.Conv2d(c, int(dim), kernel_size=(pt, pf), stride=(pt, pf))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(dim)))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, int(dim)))
        layer = nn.TransformerEncoderLayer(int(dim), int(heads), dim_feedforward=int(dim) * 2, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=int(depth))
        self.norm = nn.LayerNorm(int(dim))
        self.fc = nn.Linear(int(dim), num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        if self.pad_f or self.pad_t:
            x = F.pad(x, (0, self.pad_f, 0, self.pad_t))
        z = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(z.size(0), -1, -1)
        z = torch.cat([cls, z], dim=1) + self.pos_embed
        z = self.enc(z)
        return self.fc(self.norm(z[:, 0]))


FAMILY_TO_CLASS = {
    "MLP": MLP,
    "CNN": SimpleCNN,
    "SimpleCNN": SimpleCNN,
    "GRU": GRU,
    "LSTM": LSTM,
    "CNN-GRU": CNNGRU,
    "CNNGRU": CNNGRU,
    "Transformer": TinyViT,
    "TinyViT": TinyViT,
}


def build_model(name, num_classes, input_shape=(1, 250, 90), **params):
    return FAMILY_TO_CLASS[name](num_classes, tuple(input_shape), **params)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
