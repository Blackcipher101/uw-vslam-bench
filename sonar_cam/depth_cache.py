"""Build a compact stereo depth cache: every --step-th stereo pair -> SGBM depth (rectified-left z, metres),
block-reduced 4x4 (valid if >= --min-valid of 16 pixels valid; median) at 1/8 native resolution, float16.

  python3 depth_cache.py --out out/depth_cache.npz [--step 5]
"""
from __future__ import annotations

import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sonar_cam.data import Bag, decode_compressed_image
from sonar_cam.stereo import StereoRig, stereo_pairs

BLOCK = 4
_rig = None
_bag = None


def block_reduce(z: np.ndarray, min_valid: int) -> np.ndarray:
    h, w = z.shape
    b = z[: h // BLOCK * BLOCK, : w // BLOCK * BLOCK].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK)
    b = b.transpose(0, 2, 1, 3).reshape(h // BLOCK, w // BLOCK, BLOCK * BLOCK)
    nvalid = np.isfinite(b).sum(-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks are masked by min_valid below
        med = np.nanmedian(b, axis=-1)
    med[nvalid < min_valid] = np.nan
    return med.astype(np.float16)


def _work(args):
    global _rig, _bag
    lr, rr, min_valid = args
    if _rig is None:
        _rig, _bag = StereoRig(), Bag()
    z, _ = _rig.depth(decode_compressed_image(_bag.blob(lr)), decode_compressed_image(_bag.blob(rr)))
    return block_reduce(z, min_valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--min-valid", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    pairs = stereo_pairs(Bag())[:: a.step]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        out = []
        for i, d in enumerate(ex.map(_work, [(p[0], p[1], a.min_valid) for p in pairs], chunksize=8)):
            out.append(d)
            if i % 100 == 0:
                print(f"{i}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    rig = StereoRig()
    np.savez_compressed(a.out, t=np.array([p[2] for p in pairs]), depth=np.stack(out), block=BLOCK,
                        f=rig.f, cx=rig.cx, cy=rig.cy, R1=rig.R1, baseline_m=rig.baseline_m, scale=rig.scale)
    print("saved", a.out, len(pairs))


if __name__ == "__main__":
    main()
