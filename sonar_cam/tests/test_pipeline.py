"""Synthetic positive control: sonar returns from a fronto-parallel wall at z=2 m, stereo depth = 2 m everywhere.
The full window_pairs() path (ping range pick -> sonar points -> world -> image -> depth lookup) must give ~0 residual
with the right extrinsic and a clearly non-zero one when the extrinsic is perturbed."""
import numpy as np
import pytest

from sonar_cam.data import Ping
from sonar_cam.evalcore import Dataset, robust_stats
from sonar_cam.geometry import perturb


def make_dataset(tmp_path, moving=False):
    h, w = 150, 200
    np.savez(tmp_path / "c.npz", t=np.linspace(10.0, 30.0, 41), depth=np.full((41, h, w), 2.0, np.float16),
             block=4, f=500.0, cx=400.0, cy=300.0, R1=np.eye(3), baseline_m=0.23, scale=0.5)
    tt = np.linspace(0.0, 40.0, 81)
    x = 0.05 * (tt - 20.0) if moving else 0.0 * tt   # lateral motion would break the fronto-parallel assumption only via x
    rows = [f"{t} {xi} 0 0 0 0 0 1" for t, xi in zip(tt, x)]
    (tmp_path / "t.tum").write_text("\n".join(rows))
    pings = []
    for k, t in enumerate(np.arange(10.0, 30.0, 0.01)):
        head = -40.0 + 80.0 * ((k % 100) / 100.0)             # sweep +-40 deg about the optical axis
        r = 2.0 / np.cos(np.deg2rad(head))                    # wall z = r cos(head) = 2 m
        inten = np.zeros(250, np.uint8)
        inten[int(round(r / 0.024)) - 1] = 100
        pings.append(Ping(float(t), head, 6.0, inten))
    ds = Dataset(str(tmp_path / "c.npz"), str(tmp_path / "t.tum"), pings=pings)
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]])   # sonar x -> c0 z, sonar y -> c0 x
    return ds, T


def test_correct_extrinsic_gives_zero_residual(tmp_path):
    ds, T = make_dataset(tmp_path)
    p = ds.window_pairs(T, 0.0, window_s=1.0)
    s = robust_stats(p.res)
    assert s["n"] > 500
    assert abs(s["median"]) < 0.02 and s["mad"] < 0.03


def test_perturbed_extrinsic_is_detected(tmp_path):
    ds, T = make_dataset(tmp_path)
    p = ds.window_pairs(perturb(T, (0, 0, 0), (0, 0, 0.3)), 0.0, window_s=1.0)   # sonar 0.3 m closer along z
    assert abs(robust_stats(p.res)["median"]) > 0.2


def test_ping_pose_used_at_ping_time(tmp_path):
    """Camera slides laterally (x) while pings are collected: wall depth z is unchanged, old points must still
    re-project with ~0 residual (exercises per-ping pose interpolation and the world->image step)."""
    ds, T = make_dataset(tmp_path, moving=True)
    p = ds.window_pairs(T, 0.0, window_s=2.0)
    assert p.age_s.max() > 1.5          # points really are old
    assert abs(robust_stats(p.res)["median"]) < 0.02
