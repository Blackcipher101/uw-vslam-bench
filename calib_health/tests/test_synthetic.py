"""Synthetic stereo scene with known injected rotation error -> recovered by each method."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calib  # noqa: E402
import estimators as E  # noqa: E402
from calib import Cam, Rig  # noqa: E402
from tracks import StretchData  # noqa: E402

K = np.array([[457.0, 0, 357.0], [0, 457.0, 262.0], [0, 0, 1.0]])
D = np.array([0.0467, 0.1593, -0.2549, 0.4229])
CAM = Cam(K, D, True, (720, 540))
TRUE_RIG = Rig("synthetic", CAM, CAM, calib.exp_so3(np.deg2rad([0.02, -0.1, 0.03])), np.array([0.11, 0.0, 0.0]))


def make_stretch(n_pairs: int = 30, n_feat: int = 150, noise_px: float = 0.2, seed: int = 0) -> StretchData:
    rng = np.random.default_rng(seed)
    pl, pr, pl2, frame = [], [], [], []
    for k in range(n_pairs):
        # points in the left frame: ranges 1.2-6 m, inside the field of view
        z = rng.uniform(1.2, 6.0, n_feat)
        X = np.stack([rng.uniform(-0.6, 0.6, n_feat) * z, rng.uniform(-0.45, 0.45, n_feat) * z, z], axis=1)
        # camera moves between the two frames: X2 = R X + t (left frame k -> k+gap)
        R2 = calib.exp_so3(np.deg2rad(rng.normal(0, 1.0, 3)))
        t2 = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.05, 0.05), rng.uniform(0.05, 0.3)])
        X2 = X @ R2.T + t2
        # right-camera coordinates: x_l = R_lr x_r + t_lr  =>  x_r = R_lr^T (x_l - t_lr)
        Xr = (X - TRUE_RIG.t_lr) @ TRUE_RIG.R_lr
        vis = (X2[:, 2] > 0.3) & (np.abs(X2[:, 0] / X2[:, 2]) < 0.65) & (np.abs(X2[:, 1] / X2[:, 2]) < 0.5)

        def proj(P):
            return CAM.project(P[:, :2] / P[:, 2:3]) + rng.normal(0, noise_px, (len(P), 2))
        pl.append(proj(X)[vis]); pr.append(proj(Xr)[vis]); pl2.append(proj(X2)[vis])
        frame.append(np.full(vis.sum(), k))
    return StretchData(np.concatenate(frame), np.vstack(pl), np.vstack(pr), np.vstack(pl2),
                       np.arange(n_pairs) * 0.3, np.arange(n_pairs) * 0.3 + 0.3, (720, 540))


@pytest.fixture(scope="module")
def prep():
    return E.prepare(TRUE_RIG, make_stretch())


@pytest.mark.parametrize("inj", [[0.3, 0, 0], [0, 0, -0.3], [0.2, 0, 0.2], [0, 0, 0]])
def test_epipolar_recovers_x_and_z(prep, inj):
    w, *_ = E.epipolar_fit(prep, calib.inject(TRUE_RIG, np.array(inj)).R_lr, np.ones(len(prep.frame)), free_y=False)
    assert np.rad2deg(w[0]) == pytest.approx(inj[0], abs=0.03)
    assert np.rad2deg(w[2]) == pytest.approx(inj[2], abs=0.03)


@pytest.mark.parametrize("wy", [-0.3, 0.2, 0.5])
def test_static_epipolar_sees_y_via_xy_term_with_wide_fov(prep, wy):
    w, *_ = E.epipolar_fit(prep, calib.inject(TRUE_RIG, np.array([0, wy, 0])).R_lr, np.ones(len(prep.frame)), free_y=True)
    assert np.rad2deg(w[1]) == pytest.approx(wy, abs=0.08)


@pytest.mark.parametrize("wy", [-0.3, 0.0, 0.3])
def test_motion_intercept_recovers_y(prep, wy):
    rig = calib.inject(TRUE_RIG, np.array([0, wy, 0]))
    c, se, n, P = E.motion_intercept(prep, rig.R_lr, np.ones(len(prep.frame)))
    assert n > 500 and P > 10
    assert np.rad2deg(-c / TRUE_RIG.f_px) == pytest.approx(wy, abs=0.12)


def test_full_estimate_and_verdict(prep):
    ok = E.estimate(prep, calib.inject(TRUE_RIG, np.zeros(3)), n_boot=8)
    assert np.all(np.abs(ok.fused_deg) < 0.05)
    assert E.verdict(TRUE_RIG, ok.fused_deg) == "OK"
    bad = E.estimate(prep, calib.inject(TRUE_RIG, np.array([0.0, 0.5, 0.0])), n_boot=8)
    assert bad.fused_deg[1] == pytest.approx(0.5, abs=0.08)
    assert E.verdict(TRUE_RIG, bad.fused_deg) == "RECALIBRATE"


def test_inject_roundtrip():
    w = np.array([0.1, -0.3, 0.2])
    cand = calib.inject(TRUE_RIG, w)
    assert calib.rotation_error_deg(cand.R_lr, TRUE_RIG.R_lr) == pytest.approx(w, abs=1e-6)


def test_verdict_thresholds_tied_to_depth_error():
    # 0.1 deg about y on B=0.11 m: 4.8 % at 3 m, 12.7 % at 8 m  -> CHECK
    im = E.implied(TRUE_RIG, np.array([0.0, 0.1, 0.0]))
    assert im["err_3m"] == pytest.approx(0.0476, abs=0.003)
    assert im["err_8m"] == pytest.approx(0.127, abs=0.005)
    assert E.verdict(TRUE_RIG, np.array([0.0, 0.1, 0.0])) == "CHECK"
    assert E.verdict(TRUE_RIG, np.array([0.0, 0.3, 0.0])) == "RECALIBRATE"
    assert E.verdict(TRUE_RIG, np.array([0.0, 0.03, 0.0])) == "OK"
