import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import features as ft  # noqa: E402
import labels as lb  # noqa: E402
from metrics import SeqEval, alarm_stats  # noqa: E402

DT = 0.1


def _tum(t, xyz):
    q = np.tile([0.0, 0.0, 0.0, 1.0], (len(t), 1))
    return np.column_stack([t, xyz, q])


def _synthetic(burst=(30.0, 31.0), extra_mps=0.2, dur=60.0):
    """GT: 0.3 m/s along x (200 Hz). Estimate: same plus extra_mps along y during `burst`."""
    t = np.arange(int(dur / DT) + 1) / 10.0
    tg = np.arange(int(dur * 200) + 1) / 200.0
    gt = _tum(tg, np.column_stack([0.3 * tg, 0 * tg, 0 * tg]))
    y = extra_mps * np.clip(t - burst[0], 0, burst[1] - burst[0])
    est = _tum(t, np.column_stack([0.3 * t, y, 0 * t]))
    return t, est, gt


def test_label_goes_high_h_before_drift_onset():
    t, est, gt = _synthetic()
    d = lb.window_drift(t, est, gt, None)
    assert d.valid.sum() > 0.9 * len(t) - 20
    # window [s, s+2] overlaps the 1 s burst by o s -> drift 0.2*o/2 m/s; threshold 0.055 needs o >= 0.55
    thr, horizon = 0.055, 2.05
    labs = lb.make_labels(t, d, np.zeros(len(t)), horizon, thr)
    onset = t[labs.exceed][0]
    assert abs(onset - 28.6) < 1e-6
    first_pos = t[labs.y == 1][0]
    assert onset - horizon <= first_pos < onset - horizon + DT + 1e-9   # high at most one frame later than H before
    assert labs.y[t < first_pos - 1e-9].max() == 0                       # and not earlier
    last_exceed = t[labs.exceed][-1]
    assert abs(t[labs.y == 1][-1] - last_exceed) < 1e-6                  # drops once no exceeding window is ahead
    assert np.array_equal(lb.event_onsets(t, labs.onset, horizon), [onset])


def test_recovery_label_and_undefined_tail():
    t, est, gt = _synthetic(burst=(30.0, 30.0))
    rec = (t >= 45.0).astype(float)
    d = lb.window_drift(t, est, gt, None)
    labs = lb.make_labels(t, d, rec, 5.0, 0.055)
    assert labs.y[np.argmin(abs(t - 41.0))] == 1 and labs.y[np.argmin(abs(t - 39.0))] == 0
    assert labs.y[np.argmin(abs(t - 50.0))] == 0
    assert np.isnan(labs.y[-1])   # horizon runs past the end of the log and no positive: undefined


def test_gt_gap_invalidates_windows():
    t, est, gt = _synthetic()
    gt = gt[(gt[:, 0] < 10.0) | (gt[:, 0] > 14.0)]   # 4 s mocap dropout
    d = lb.window_drift(t, est, gt, None)
    inside = (t > 8.0) & (t < 14.0)
    assert not d.valid[inside].any()
    assert d.valid[(t > 20) & (t < 55)].all()


def _frame_log(n, rng):
    t = 1.7e9 + np.cumsum(rng.uniform(0.04, 0.12, n))
    df = pd.DataFrame({c: rng.integers(0, 100, n).astype(float) for c in ft.FRAME_LOG_COLS})
    df["t"] = t
    df["recoveries"] = np.cumsum(rng.random(n) < 0.02).astype(float)
    df["img_mean"] = rng.uniform(20, 80, n)
    df["img_std"] = rng.uniform(5, 30, n)
    return df


def test_features_are_causal():
    rng = np.random.default_rng(0)
    df = _frame_log(400, rng)
    full = ft.build_features(df)
    m = 200
    alt = df.copy()
    for c in ft.FRAME_LOG_COLS:
        if c != "t":
            alt.loc[m + 1:, c] = rng.uniform(-1e3, 1e3, len(df) - m - 1)
    alt.loc[m + 1:, "recoveries"] = alt.loc[m, "recoveries"] + 7
    f_alt = ft.build_features(alt)
    pd.testing.assert_frame_equal(full.iloc[:m + 1], f_alt.iloc[:m + 1])
    f_trunc = ft.build_features(df.iloc[:m + 1])
    pd.testing.assert_frame_equal(full.iloc[:m + 1], f_trunc)


def test_rolling_window_values():
    t = np.arange(20) * 0.5
    x = 2.0 * t + 1
    mean, mn, slope = ft._rolling(t, x, 3.0)   # window (t-3, t]: 6 samples
    i = 15
    assert np.isclose(slope[i], 2.0) and np.isclose(mn[i], x[i - 5]) and np.isclose(mean[i], x[i - 5:i + 1].mean())


def test_alarm_lead_time_and_miss():
    t = np.arange(0, 100, 0.1)
    y = np.zeros(len(t))
    score = np.zeros(len(t))
    events = np.array([40.0, 80.0])
    y[(t >= 35) & (t <= 40)] = 1
    y[(t >= 75) & (t <= 80)] = 1
    score[(t >= 37.0) & (t <= 38)] = 1       # alarm 3 s before the first event
    score[(t >= 10) & (t <= 10.5)] = 1       # one false alarm
    st = alarm_stats([SeqEval(t, score, y, events, 5.0)], 0.5)
    assert st["detected"] == 1 and st["missed"] == 1
    assert abs(st["median_lead_s"] - 3.0) < 0.11
    assert np.isclose(st["fa_per_min"], 1 / ((np.sum(y == 0) * 0.1) / 60))


def _est_tum(t, rng):
    pos = np.cumsum(rng.normal(0, 0.01, (len(t), 3)), axis=0)
    q = np.column_stack([rng.normal(0, 0.05, (len(t), 3)), np.ones(len(t))])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return np.column_stack([t, pos, q])


def test_motion_features_causal_and_values():
    rng = np.random.default_rng(1)
    df = _frame_log(400, rng)
    t = df["t"].to_numpy()
    est = _est_tum(t, rng)
    full = ft.build_features(df, est)
    assert any(c.startswith("motion_") for c in full.columns)
    m = 200
    est_alt = est.copy()
    est_alt[m + 1:, 1:4] += rng.uniform(-50, 50, (len(t) - m - 1, 3))
    est_alt[m + 1:, 4:8] = [0.5, 0.5, 0.5, 0.5]
    pd.testing.assert_frame_equal(full.iloc[:m + 1], ft.build_features(df, est_alt).iloc[:m + 1])
    pd.testing.assert_frame_equal(full.iloc[:m + 1], ft.build_features(df.iloc[:m + 1], est[:m + 1]))
    # only the first m+1 frame-log rows AND est poses
    pd.testing.assert_frame_equal(full.iloc[:m + 1], ft.build_features(df.iloc[:m + 1], est[:m + 1]))


def test_motion_values_constant_turn():
    t = np.arange(0, 20, 0.05)
    yaw = np.radians(30.0) * t                      # 30 deg/s about world z, 0.5 m/s along x
    q = np.column_stack([0 * t, 0 * t, np.sin(yaw / 2), np.cos(yaw / 2)])
    est = np.column_stack([t, 0.5 * t, 0 * t, 0 * t, q])
    mf = ft.motion_features(t, est)
    i = 200
    assert np.isclose(mf["motion_ang_rate_dps"][i], 30.0, atol=1e-6)
    assert np.isclose(mf["motion_yaw_rate_abs_dps"][i], 30.0, atol=1e-6)
    assert np.isclose(mf["motion_speed_mps"][i], 0.5, atol=1e-6)
    assert mf["motion_accel_mps2"][i] < 1e-6
