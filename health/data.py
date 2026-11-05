"""Run registry and per-run dataset construction."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import labels as lb
from features import build_features, load_frame_log


@dataclass
class RunSpec:
    split: str          # "ntnu23" (train / LOTO) or "ntnu25" (held-out)
    seq: str            # trajectory name, e.g. traj_03, fjord_2
    cfg: str            # estimator config tag, e.g. h05, nolcn
    framelog: Path
    est: Path           # left-camera TUM written with the frame log
    gt: Path            # mocap (ntnu23) or ReAqROVIO reference (ntnu25), TUM
    hand_eye: bool = True


@dataclass
class Run:
    spec: RunSpec
    t: np.ndarray
    feats: pd.DataFrame
    recoveries: np.ndarray
    drift: lb.Drift
    gt_duration_valid_s: float
    cache: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.spec.cfg}/{self.spec.seq}"


def load_manifest(path: Path) -> list[RunSpec]:
    return [RunSpec(r["split"], r["seq"], r["cfg"], Path(r["framelog"]), Path(r["est"]), Path(r["gt"]),
                    r.get("hand_eye", True)) for r in json.loads(Path(path).read_text())]


def build_run(spec: RunSpec) -> Run:
    df = load_frame_log(spec.framelog)
    t = df["t"].to_numpy(float)
    est = lb.ev.load_tum(spec.est)
    feats = build_features(df, est)
    gt = lb.ev.load_tum(spec.gt)
    X = lb.hand_eye_x(gt, est) if spec.hand_eye else None
    drift = lb.window_drift(t, est, gt, X)
    dt = float(np.median(np.diff(t)))
    return Run(spec, t, feats, df["recoveries"].to_numpy(float), drift, float(drift.valid.sum() * dt))
