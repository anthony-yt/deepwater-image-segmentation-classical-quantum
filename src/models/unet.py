"""U-Net y Attention U-Net en PyTorch puro.

Extraido de `notebooks/3_1_unet_baseline_training.ipynb` (task 2.2) sin cambios de
arquitectura, para que las tasks 4.2 y 5.11 entrenen exactamente el mismo modelo en vez de
mantener dos copias del codigo que se desincronizan.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_FEATURES = (64, 128, 256, 512)


class DoubleConv(nn.Module):
    """Dos convoluciones 3x3 con BatchNorm y ReLU."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention gate de Oktay et al. (2018): pondera el skip con la senal del decoder."""

    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.W_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
        return x * psi


class UNet(nn.Module):
    """U-Net con attention gates opcionales en los skips.

    `attention=True` es la variante Attention U-Net del Goal 2.3.
    """

    def __init__(self, in_ch: int = 3, n_classes: int = 2,
                 feats: tuple = DEFAULT_FEATURES, attention: bool = False):
        super().__init__()
        self.attention = attention
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        c = in_ch
        for f in feats:
            self.downs.append(DoubleConv(c, f))
            c = f
        self.bottleneck = DoubleConv(feats[-1], feats[-1] * 2)
        for f in reversed(feats):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            if attention:
                self.gates.append(AttentionGate(f, f, max(f // 2, 1)))
            self.ups.append(DoubleConv(f * 2, f))
        self.final = nn.Conv2d(feats[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            if self.attention:
                skip = self.gates[i // 2](x, skip)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)
        return self.final(x)


def build_unet(attention: bool = False, device: str = "cpu", **kwargs) -> UNet:
    model = UNet(attention=attention, **kwargs).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"{'Attention U-Net' if attention else 'U-Net'} - {n / 1e6:.2f}M parametros")
    return model
