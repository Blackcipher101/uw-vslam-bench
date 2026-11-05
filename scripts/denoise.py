"""Noise-map-conditioned denoising (FFDNet) for lamp-corrected underwater images.

After the lightcone flat-field (dehaze.py) the noise is known per pixel: the camera's sensor noise
times the correction gain there. FFDNet (Zhang, Zuo, Zhang, TIP 2018) takes exactly such a noise
map as input and removes that much noise; it is a denoiser trained on additive noise, not a
generator, so it is not trained to make images look like anything. Weights: KAIR ffdnet_gray.pth
(MIT). The network is re-implemented here (layout matches the published state dict) and the
weights are loaded tensor-only.

Sensor noise is estimated with Immerkaer's operator (CVIU 1996) on low-gradient pixels of raw
frames, so edges do not inflate it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

FFDNET_WEIGHTS = Path(__file__).resolve().parent.parent / "third_party" / "kair_weights" / "ffdnet_gray.pth"
IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
FLAT_GRADIENT_PERCENTILE = 50   # pixels with gradient below this percentile count as flat


class FFDNet(nn.Module):
    """Grey FFDNet: pixel-unshuffle x2, concat noise map, 15 conv(+ReLU) layers, pixel-shuffle."""

    def __init__(self, nc: int = 64, nb: int = 15) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(4 + 1, nc, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(nb - 2):
            layers += [nn.Conv2d(nc, nc, 3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(nc, 4, 3, padding=1))
        self.model = nn.Sequential(*layers)
        self.down, self.up = nn.PixelUnshuffle(2), nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor, sigma_map: torch.Tensor) -> torch.Tensor:
        """x, sigma_map: (1,1,H,W) in [0,1] intensity units, H and W even."""
        s = nn.functional.avg_pool2d(sigma_map, 2)
        return self.up(self.model(torch.cat((self.down(x), s), 1)))


class Denoiser:
    def __init__(self, device: str = "cuda") -> None:
        self.dev = torch.device(device)
        self.net = FFDNet()
        self.net.load_state_dict(torch.load(FFDNET_WEIGHTS, map_location="cpu", weights_only=True), strict=True)
        self.net.eval().to(self.dev)

    def __call__(self, grey: np.ndarray, sigma_map: np.ndarray | float) -> np.ndarray:
        """grey uint8 HxW; sigma_map in 8-bit units (scalar or HxW). Returns uint8 HxW."""
        h, w = grey.shape
        sm = np.broadcast_to(np.asarray(sigma_map, np.float32), (h, w))
        ph, pw = h % 2, w % 2
        g = cv2.copyMakeBorder(grey, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
        sm = cv2.copyMakeBorder(np.ascontiguousarray(sm), 0, ph, 0, pw, cv2.BORDER_REPLICATE)
        x = torch.from_numpy(g).to(self.dev, torch.float32)[None, None] / 255.0
        s = torch.from_numpy(sm).to(self.dev, torch.float32)[None, None] / 255.0
        with torch.inference_mode():
            y = self.net(x, s)
        return (y[0, 0, :h, :w].clamp(0, 1) * 255.0).round().byte().cpu().numpy()


def estimate_sigma(grey: np.ndarray) -> float:
    """Immerkaer noise estimate (8-bit units) over the flatter half of the image."""
    g = grey.astype(np.float32)
    resp = np.abs(cv2.filter2D(g, -1, IMMERKAER))[1:-1, 1:-1]
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))[1:-1, 1:-1]
    flat = grad <= np.percentile(grad, FLAT_GRADIENT_PERCENTILE)
    return float(np.sqrt(np.pi / 2.0) / 6.0 * resp[flat].mean())
