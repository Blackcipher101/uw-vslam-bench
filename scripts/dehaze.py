"""Physics-based underwater haze removal from stereo depth (Sea-thru-style image formation).

Model per colour channel c (Akkaynak & Treibitz, CVPR 2019), z = range from the camera [m]:
    I_c = J_c * exp(-beta_D_c z) + B_inf_c * (1 - exp(-beta_B_c z))
The second term is backscatter: light scattered into the camera by the water, independent of the
scene. With z known from the stereo pair, B_c(z) is estimated and subtracted.

For tracking, temporal consistency matters more than looks, so the water parameters are fitted
ONCE per sequence (from frames sampled across the bag) and then applied with the same constants to
every frame: the correction is a fixed function of depth and cannot flicker or invent texture.

Modes (depth-based, ambient light):
  backscatter  J = (I - B(z)) * g, with one global gain g per sequence (99th percentile -> 250)
  full         additionally multiplies by exp(beta_D z) (attenuation compensation), capped at
               ATTEN_GAIN_MAX, then the same global gain. Raises far contrast and far noise alike.
On lamp-lit footage (the vehicle or diver carries the light) the ambient model degenerates: the
darkest pixels are equally bright at every depth, the fit hits beta's bound and the correction
collapses to a global contrast stretch. For that case:
  lightcone    the lamp is rigid with the camera, so its backscatter glow V(x) and illumination
               falloff L(x) are fixed patterns in image coordinates. V = per-pixel 5th percentile,
               L = 95th - 5th percentile over frames sampled across the sequence, both heavily
               smoothed; J = (I - V) / L, rescaled once per sequence. A flat-field correction:
               no depth, no network, a fixed smooth per-pixel affine map.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import curve_fit

DEPTH_WIDTH_PX = 400        # working width for stereo depth (backscatter varies smoothly with z)
Z_MIN_M, Z_MAX_M = 0.5, 8.0  # depth range used for fitting and for pixels with no stereo depth
N_BINS = 12
DARK_PERCENTILE = 1.0        # darkest pixels per depth bin approximate pure backscatter
ATTEN_GAIN_MAX = 3.0
SGBM_BLOCK = 7


@dataclass
class Cam:
    model: str          # "radtan" | "equidistant"
    K: np.ndarray
    D: np.ndarray
    size: tuple[int, int]  # (w, h)


def load_rig(intrinsics: Path, imucam: Path, left: str, right: str) -> tuple[Cam, Cam, np.ndarray, np.ndarray]:
    """Kalibr files -> (left cam, right cam, R, t) with x_right = R x_left + t."""
    ci, ce = yaml.safe_load(intrinsics.read_text()), yaml.safe_load(imucam.read_text())

    def cam(n: str) -> Cam:
        d = ci[n]
        if d["camera_model"] != "pinhole" or d["distortion_model"] not in ("radtan", "equidistant"):
            raise SystemExit(f"{n}: unsupported camera {d['camera_model']}/{d['distortion_model']}")
        fx, fy, cx, cy = d["intrinsics"]
        return Cam(d["distortion_model"], np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]),
                   np.array(d["distortion_coeffs"], float), tuple(d["resolution"]))

    T = np.array(ce[right]["T_cam_imu"]) @ np.linalg.inv(np.array(ce[left]["T_cam_imu"]))
    return cam(left), cam(right), T[:3, :3].copy(), T[:3, 3].reshape(3, 1).copy()


class StereoDepth:
    """Per-pixel range (z) for both raw images of a stereo pair, at a coarse working resolution."""

    def __init__(self, left: Cam, right: Cam, R: np.ndarray, t: np.ndarray) -> None:
        if left.model != right.model:
            raise SystemExit("left and right cameras use different distortion models")
        self.fish = left.model == "equidistant"
        s = DEPTH_WIDTH_PX / left.size[0]
        self.size = (DEPTH_WIDTH_PX, int(round(left.size[1] * s)))
        self.raw_size = left.size
        Kl, Kr = left.K.copy(), right.K.copy()
        Kl[:2] *= s
        Kr[:2] *= s
        if self.fish:
            R1, R2, P1, P2, _ = cv2.fisheye.stereoRectify(Kl, left.D, Kr, right.D, self.size, R, t,
                                                          cv2.CALIB_ZERO_DISPARITY, newImageSize=self.size,
                                                          balance=0.0, fov_scale=1.0)
            init, undist = cv2.fisheye.initUndistortRectifyMap, cv2.fisheye.undistortPoints
        else:
            R1, R2, P1, P2, *_ = cv2.stereoRectify(Kl, left.D, Kr, right.D, self.size, R, t, alpha=0)
            init, undist = cv2.initUndistortRectifyMap, cv2.undistortPoints
        self.f = float(P1[0, 0])
        self.baseline = float(abs(P2[0, 3] / P2[0, 0]))
        if not self.baseline > 0:
            raise SystemExit("degenerate stereo baseline")
        self.maps = [init(Kl, left.D, R1, P1, self.size, cv2.CV_32FC1), init(Kr, right.D, R2, P2, self.size, cv2.CV_32FC1)]
        # raw pixel (working res) -> rectified pixel, to carry the rectified depth back to raw images
        gx, gy = np.meshgrid(np.arange(self.size[0], dtype=np.float32), np.arange(self.size[1], dtype=np.float32))
        pts = np.stack([gx.ravel(), gy.ravel()], 1).reshape(-1, 1, 2)
        self.back = []
        for K, D, Rr, P in ((Kl, left.D, R1, P1), (Kr, right.D, R2, P2)):
            q = undist(pts, K, D, R=Rr, P=P).reshape(self.size[1], self.size[0], 2)
            self.back.append((q[..., 0].copy(), q[..., 1].copy()))
        nd = int(np.ceil(self.f * self.baseline / Z_MIN_M / 16.0)) * 16
        self.sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=max(16, nd), blockSize=SGBM_BLOCK,
                                          P1=8 * SGBM_BLOCK ** 2, P2=32 * SGBM_BLOCK ** 2, uniquenessRatio=10,
                                          speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def _range(self, disp: np.ndarray) -> np.ndarray:
        z = np.full(disp.shape, Z_MAX_M, np.float32)  # no stereo depth: open water, backscatter saturated
        ok = disp > 0.5
        z[ok] = np.clip(self.f * self.baseline / disp[ok], Z_MIN_M, Z_MAX_M)
        return cv2.medianBlur(z, 5)

    def __call__(self, left_grey: np.ndarray, right_grey: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grey raw images (any resolution) -> range maps at raw resolution for left and right."""
        lw = cv2.resize(left_grey, self.size, interpolation=cv2.INTER_AREA)
        rw = cv2.resize(right_grey, self.size, interpolation=cv2.INTER_AREA)
        lr = cv2.remap(lw, *self.maps[0], cv2.INTER_LINEAR)
        rr = cv2.remap(rw, *self.maps[1], cv2.INTER_LINEAR)
        dl = self.sgbm.compute(lr, rr).astype(np.float32) / 16.0
        # right-referenced disparity via the mirror trick (SGBM only matches left-to-right)
        dr = cv2.flip(self.sgbm.compute(cv2.flip(rr, 1), cv2.flip(lr, 1)), 1).astype(np.float32) / 16.0
        out = []
        for d, (bx, by) in zip((dl, dr), self.back):
            z = cv2.remap(self._range(d), bx, by, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=Z_MAX_M)
            out.append(cv2.resize(cv2.GaussianBlur(z, (0, 0), 3), (left_grey.shape[1], left_grey.shape[0])))
        return out[0], out[1]


def _bs_curve(z, b_inf, beta):
    return b_inf * (1.0 - np.exp(-beta * z))


@dataclass
class WaterModel:
    """Fitted once per sequence; values in 8-bit intensity units per channel."""
    b_inf: np.ndarray = field(default_factory=lambda: np.zeros(3))
    beta_b: np.ndarray = field(default_factory=lambda: np.ones(3))
    beta_d: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gain: float = 1.0

    @staticmethod
    def fit(samples: list[tuple[np.ndarray, np.ndarray]], mode: str) -> "WaterModel":
        """samples: (bgr image, range map). Darkest pixels per depth bin -> backscatter curve."""
        zs = np.concatenate([z[::4, ::4].ravel() for _, z in samples])
        m = WaterModel()
        edges = np.linspace(Z_MIN_M, Z_MAX_M, N_BINS + 1)
        for c in range(3):
            vals = np.concatenate([im[::4, ::4, c].ravel() for im, _ in samples]).astype(np.float64)
            bz, bv = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                sel = (zs >= a) & (zs < b)
                if sel.sum() > 200:
                    bz.append(0.5 * (a + b))
                    bv.append(np.percentile(vals[sel], DARK_PERCENTILE))
            if len(bz) < 3:
                raise SystemExit(f"channel {c}: only {len(bz)} depth bins with data; cannot fit backscatter")
            (b_inf, beta), _ = curve_fit(_bs_curve, np.array(bz), np.array(bv), p0=(max(bv), 0.5),
                                         bounds=([0, 1e-3], [255, 10]), maxfev=10000)
            m.b_inf[c], m.beta_b[c] = b_inf, beta
            # attenuation ~ backscatter coefficient is a rough but common first guess
            m.beta_d[c] = beta if mode == "full" else 0.0
        direct = np.concatenate([m._direct(im, z)[::4, ::4].ravel() for im, z in samples])
        m.gain = 250.0 / max(np.percentile(direct, 99), 1.0)
        return m

    def _direct(self, bgr: np.ndarray, z: np.ndarray) -> np.ndarray:
        I = bgr.astype(np.float32)
        out = np.empty_like(I)
        for c in range(3):
            B = self.b_inf[c] * (1.0 - np.exp(-self.beta_b[c] * z))
            att = np.minimum(np.exp(self.beta_d[c] * z), ATTEN_GAIN_MAX)
            out[..., c] = np.maximum(I[..., c] - B, 0.0) * att
        return out

    def apply(self, bgr: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.clip(self._direct(bgr, z) * self.gain, 0, 255).astype(np.uint8)

    def describe(self) -> str:
        return (f"B_inf {np.round(self.b_inf, 1).tolist()} beta_B {np.round(self.beta_b, 3).tolist()} "
                f"beta_D {np.round(self.beta_d, 3).tolist()} gain {self.gain:.2f}")


LIGHTCONE_FIT_WIDTH_PX = 400     # fields are low-frequency; fit at reduced resolution
LIGHTCONE_BLUR_FRACTION = 0.06   # Gaussian sigma as a fraction of the image width
LIGHTCONE_TARGET = 200.0         # output level of the per-pixel 95th percentile
# Largest brightening applied anywhere. Where the lamp barely reaches, signal is below the sensor
# noise floor: amplifying more only amplifies noise, and a denoiser then paints structure there.
LIGHTCONE_MAX_GAIN = 8.0


@dataclass
class LightConeModel:
    veil: np.ndarray = None     # HxWx3 float32 at fit resolution
    spread: np.ndarray = None

    @staticmethod
    def fit(images: list[np.ndarray]) -> "LightConeModel":
        if len(images) < 20:
            raise SystemExit(f"lightcone fit needs >= 20 frames, got {len(images)}")
        h, w = images[0].shape[:2]
        size = (LIGHTCONE_FIT_WIDTH_PX, int(round(h * LIGHTCONE_FIT_WIDTH_PX / w)))
        stack = np.stack([cv2.resize(im, size, interpolation=cv2.INTER_AREA) for im in images]).astype(np.float32)
        sigma = LIGHTCONE_BLUR_FRACTION * size[0]
        m = LightConeModel()
        p5, p95 = np.percentile(stack, 5, axis=0), np.percentile(stack, 95, axis=0)
        m.veil = cv2.GaussianBlur(p5, (0, 0), sigma)
        m.spread = np.maximum(cv2.GaussianBlur(p95 - p5, (0, 0), sigma), 4.0)  # avoid dividing by ~0 in dead corners
        return m

    def apply(self, bgr: np.ndarray, max_gain: float | None = None) -> np.ndarray:
        """max_gain=None keeps the uncapped behaviour used by the lightcone-only trial."""
        h, w = bgr.shape[:2]
        V = cv2.resize(self.veil, (w, h), interpolation=cv2.INTER_LINEAR)
        g = cv2.resize(self.gain(max_gain), (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip((bgr.astype(np.float32) - V) * g, 0, 255).astype(np.uint8)

    def gain(self, max_gain: float | None = None) -> np.ndarray:
        """Per-pixel, per-channel brightening (fit resolution)."""
        g = LIGHTCONE_TARGET / self.spread
        return np.minimum(g, max_gain) if max_gain else g

    def describe(self) -> str:
        return (f"veil mean {self.veil.mean():.1f} (min {self.veil.min():.1f}, max {self.veil.max():.1f}); "
                f"spread mean {self.spread.mean():.1f} (min {self.spread.min():.1f}, max {self.spread.max():.1f})")
