"""
model.py — 2D Fourier Neural Operator for wave field prediction.

Architecture:
    Input (C_in, nx, nz) → Lift → [Fourier Layer x N] → Project → Output (C_out, nx, nz)

Each Fourier layer:
    x → FFT → truncate modes → linear in Fourier space → iFFT → + W·x → activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Spectral convolution: linear transform in truncated Fourier space."""

    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1  # number of Fourier modes in x
        self.modes2 = modes2  # number of Fourier modes in z

        scale = 1.0 / (in_ch * out_ch)
        # Complex weights for two sets of modes (positive and negative freqs in dim1)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        """Complex multiplication: (B, C_in, x, z) x (C_in, C_out, x, z) → (B, C_out, x, z)"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B = x.shape[0]

        # FFT
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            B, self.out_ch, x.size(2), x.size(3) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Positive frequencies in dim1
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        # Negative frequencies in dim1
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # iFFT
        x = torch.fft.irfft2(out_ft, s=(x.size(2), x.size(3)))
        return x


class FourierLayer(nn.Module):
    """Single Fourier layer: spectral conv + bypass conv + activation."""

    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.bypass = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width)

    def forward(self, x):
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator.

    Args:
        n_in_ch: number of input channels (fields + mask)
        n_out_ch: number of output channels (predicted fields)
        modes1: Fourier modes in x direction
        modes2: Fourier modes in z direction
        width: hidden channel dimension
        n_layers: number of Fourier layers
    """

    def __init__(self, n_in_ch, n_out_ch, modes1, modes2, width, n_layers):
        super().__init__()

        self.n_in_ch = n_in_ch
        self.n_out_ch = n_out_ch
        self.width = width

        # Lift: input channels → hidden width
        self.lift = nn.Conv2d(n_in_ch, width, 1)

        # Fourier layers
        self.layers = nn.ModuleList([
            FourierLayer(width, modes1, modes2) for _ in range(n_layers)
        ])

        # Project: hidden width → output channels
        self.project = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, n_out_ch, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C_in, nx, nz)
        Returns:
            (B, C_out, nx, nz)
        """
        x = self.lift(x)
        for layer in self.layers:
            x = layer(x)
        x = self.project(x)
        return x

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
