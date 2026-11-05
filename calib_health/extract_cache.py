"""Extract calibration-independent tracks for a stretch and cache them (small npz).

Usage: extract_cache.py ntnu <t0_s> <dur_s> | bus <t0_s> <dur_s>   (t0 = seconds since bag start)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import bagio
import tracks

CACHE = Path(__file__).parent / "cache"


def cache_path(rig: str, t0: float, dur: float, tag: str = "k6g6") -> Path:
    return CACHE / f"{rig}_{int(t0)}_{int(dur)}_{tag}.npz"


def get_stretch(rig: str, t0: float, dur: float, **kw) -> tracks.StretchData:
    p = cache_path(rig, t0, dur)
    if p.exists():
        return tracks.StretchData.load(p)
    CACHE.mkdir(exist_ok=True)
    if rig == "ntnu":
        fr = bagio.iter_pairs(bagio.NTNU_BAG, bagio.NTNU_LEFT, bagio.NTNU_RIGHT, t0, t0 + dur, compressed=False)
    else:
        fr = bagio.iter_pairs(bagio.BUS_BAG, bagio.BUS_LEFT, bagio.BUS_RIGHT, t0, t0 + dur, compressed=True)
    t = time.time()
    if rig == "bus":  # 1600x1200, 2 deg toed-in rig: wide raw disparity range, signed offset at infinity
        kw = {**dict(min_disp_px=-150.0, max_disp_px=450.0, max_abs_dy_px=40.0), **kw}
    sd = tracks.extract(fr, **kw)
    print(f"[extract] {rig} t0={t0} dur={dur}: {len(sd.t_key_s)} keyframes, {len(sd.frame)} matches, "
          f"{time.time() - t:.1f} s", flush=True)
    sd.save(p)
    return sd


if __name__ == "__main__":
    get_stretch(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
