"""Underwater image enhancement for odometry experiments.

Modes:
  none          image unchanged
  funie-native  FUnIE-GAN generator (Islam et al., RA-L 2020, MIT) run fully convolutionally at
                the input resolution (trained at 256x256, so this is outside its training scale)
  funie-gain    FUnIE-GAN at 256x256; the per-pixel, per-channel ratio enhanced/input is smoothed,
                upsampled and applied to the full-resolution image. Only a low-frequency colour and
                contrast correction reaches the image, so it cannot invent texture.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

FUNIE_DIR = Path(__file__).resolve().parent.parent / "third_party" / "FUnIE-GAN" / "PyTorch"
FUNIE_SIZE = 256
GAIN_BLUR_SIGMA_PX = 4.0   # at 256 px: keeps the gain map low-frequency
GAIN_RANGE = (0.2, 5.0)    # clamp ratios where the input is near black
MODES = ("none", "funie-native", "funie-gain")


# FUnIE-GAN generator, re-implemented from the paper/repo architecture (5-level UNet) so no code
# from the repo is executed; parameter names match the published state dict.
class _Down(nn.Module):
    def __init__(self, cin: int, cout: int, bn: bool = True) -> None:
        super().__init__()
        layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=False)]
        if bn:
            layers.append(nn.BatchNorm2d(cout, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _Up(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False),
                                   nn.BatchNorm2d(cout, momentum=0.8), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.model(x), skip), 1)


class FunieGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down1, self.down2 = _Down(3, 32, bn=False), _Down(32, 128)
        self.down3, self.down4, self.down5 = _Down(128, 256), _Down(256, 256), _Down(256, 256, bn=False)
        self.up1, self.up2, self.up3, self.up4 = _Up(256, 256), _Up(512, 256), _Up(512, 128), _Up(256, 32)
        self.final = nn.Sequential(nn.Upsample(scale_factor=2), nn.ZeroPad2d((1, 0, 1, 0)),
                                   nn.Conv2d(64, 3, 4, padding=1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        return self.final(self.up4(self.up3(self.up2(self.up1(d5, d4), d3), d2), d1))


@torch.no_grad()
def fold_batchnorm(net: nn.Module) -> None:
    """Fold every eval-mode BatchNorm into the (transposed) convolution before it, in place.

    Exact in inference: BN(y) = gamma * (y - mean) / sqrt(var + eps) + beta is a per-channel affine
    map of the convolution output. Removes BN kernels from the GPU path (and is faster)."""
    for mod in net.modules():
        if not isinstance(mod, nn.Sequential):
            continue
        layers = list(mod)
        for i in range(len(layers) - 1):
            conv, bn = layers[i], layers[i + 1]
            if not isinstance(bn, nn.BatchNorm2d) or not isinstance(conv, (nn.Conv2d, nn.ConvTranspose2d)):
                continue
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            shift = bn.bias - bn.running_mean * scale
            # Output channels are dim 0 for Conv2d weights and dim 1 for ConvTranspose2d weights.
            shape = (-1, 1, 1, 1) if isinstance(conv, nn.Conv2d) else (1, -1, 1, 1)
            conv.weight.mul_(scale.reshape(shape))
            bias = conv.bias if conv.bias is not None else torch.zeros_like(shift)
            conv.bias = nn.Parameter(bias * scale + shift)
            mod[i + 1] = nn.Identity()


class Enhancer:
    def __init__(self, mode: str, device: str = "cuda") -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode, self.dev = mode, torch.device(device)
        self.net = None
        if mode.startswith("funie"):
            self.net = FunieGenerator()
            # weights_only: tensors only, the file cannot run code when loaded
            state = torch.load(FUNIE_DIR / "models" / "funie_generator.pth", map_location="cpu", weights_only=True)
            self.net.load_state_dict(state, strict=True)
            self.net.eval()
            fold_batchnorm(self.net)
            self.net.to(self.dev)

    def _run(self, rgb: np.ndarray) -> np.ndarray:
        """rgb uint8 HxWx3 (H, W multiples of 32) -> enhanced float32 HxWx3 in [0, 1]."""
        x = torch.from_numpy(rgb).to(self.dev, torch.float32).permute(2, 0, 1)[None] / 127.5 - 1.0
        with torch.inference_mode():
            y = self.net(x)[0].permute(1, 2, 0)
        return ((y.clamp(-1, 1) + 1.0) / 2.0).cpu().numpy()

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 in, BGR uint8 out, same size."""
        if self.mode == "none":
            return bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if self.mode == "funie-native":
            ph, pw = (-h) % 32, (-w) % 32
            padded = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT)
            out = self._run(padded)[:h, :w]
            return cv2.cvtColor((out * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        small = cv2.resize(rgb, (FUNIE_SIZE, FUNIE_SIZE), interpolation=cv2.INTER_AREA)
        enh = self._run(small)
        gain = (enh + 1e-3) / (small.astype(np.float32) / 255.0 + 1e-3)
        gain = cv2.GaussianBlur(np.clip(gain, *GAIN_RANGE), (0, 0), GAIN_BLUR_SIGMA_PX)
        gain = cv2.resize(gain, (w, h), interpolation=cv2.INTER_LINEAR)
        out = np.clip(rgb.astype(np.float32) * gain, 0, 255).round().astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
