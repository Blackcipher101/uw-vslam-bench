"""Train / evaluate the localization-health model. See README.md.

  python3 run_health.py --manifest manifest.json --out results/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import labels as lb
from data import Run, build_run, load_manifest
from metrics import SeqEval, alarm_stats, choose_threshold, frame_metrics

HORIZONS_S = (2.0, 5.0)
ABS_THR_MPS = 0.05          # physically meaningful: 5 cm/s of translational drift
REL_FACTOR = 3.0            # relative: 3x the median window drift of the training runs
TARGET_FA_PER_MIN = 1.0
PRIMARY = ("rel", 5.0)   # abs 5 cm/s sits at the drift noise floor (positive rate 0.7 at H=5), see README
SEED = 0
FEATURE_COLS: list[str] = []   # set in main from --feature-set


def fmat(r: Run) -> np.ndarray:
    return r.feats[FEATURE_COLS].to_numpy()


def log(*a) -> None:
    print(*a, flush=True)


def make_model(kind: str, param):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=param, class_weight="balanced", max_iter=500))
    if kind == "gbt":
        return HistGradientBoostingClassifier(max_depth=param, learning_rate=0.05, max_iter=150, min_samples_leaf=100,
                                              l2_regularization=1.0, random_state=SEED)
    raise ValueError(kind)


class _Constant:
    """Fallback when a training set has a single class: constant risk equal to that class."""
    def __init__(self, v: float):
        self.v = v

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.full(len(X), 1 - self.v), np.full(len(X), self.v)])


def fit_model(kind: str, param, X: np.ndarray, y: np.ndarray):
    if len(np.unique(y)) < 2:
        log(f"  WARNING: single-class training set (y={y[0] if len(y) else 'empty'}), constant risk")
        return _Constant(float(y[0]) if len(y) else 0.0)
    return make_model(kind, param).fit(X, y)


GRIDS = {"logreg": (0.01, 0.1, 1.0), "gbt": (3, 5)}
RULES = {  # name -> feature column (score = higher means riskier)
    "rule: vision_healthy==0": ("vision_healthy", -1.0),
    "rule: gated_frac_1s high": ("gated_frac_mean_1s", 1.0),
    "rule: tracked_1s low": ("tracked_mean_1s", -1.0),
}


def label_runs(runs: list[Run], scheme: str, horizon: float, thr: float) -> None:
    for r in runs:
        r.cache[(scheme, horizon)] = lb.make_labels(r.t, r.drift, r.recoveries, horizon, thr)


def xy(runs: list[Run], key) -> tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([fmat(r) for r in runs])
    y = np.concatenate([r.cache[key].y for r in runs])
    ok = np.isfinite(y)
    return X[ok], y[ok]


def seq_eval(r: Run, score: np.ndarray, key, horizon: float) -> SeqEval:
    labs = r.cache[key]
    return SeqEval(r.t, score, labs.y, lb.event_onsets(r.t, labs.onset, horizon), horizon)


def loto_scores(runs: list[Run], key, kind: str, param) -> dict[str, np.ndarray]:
    """Out-of-fold risk per run (leave one trajectory out; both configs of that trajectory are held out)."""
    out = {}
    for seq in sorted({r.spec.seq for r in runs}):
        tr = [r for r in runs if r.spec.seq != seq]
        te = [r for r in runs if r.spec.seq == seq]
        Xtr, ytr = xy(tr, key)
        m = fit_model(kind, param, Xtr, ytr)
        for r in te:
            out[r.name] = m.predict_proba(fmat(r))[:, 1]
    return out


def rule_score(r: Run, rule: str) -> np.ndarray:
    col, sign = RULES[rule]
    return sign * r.feats[col].to_numpy()


def row(model: str, setname: str, seqs: list[SeqEval], thr: float, extra: dict | None = None) -> dict:
    fm = frame_metrics(seqs)
    al = alarm_stats(seqs, thr)
    return {"model": model, "set": setname, **fm, **{k: al[k] for k in (
        "events", "detected", "missed", "event_recall", "median_lead_s", "fa_per_min", "frame_recall")}, **(extra or {})}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--feature-set", choices=("v0", "v1"), default="v1", help="v0 frame log only, v1 + motion")
    ap.add_argument("--plot-run", default="h05/traj_06")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    specs = load_manifest(a.manifest)
    if any("fjord_6" in str(s.seq) or "fjord_6" in str(s.framelog) for s in specs):
        raise SystemExit("fjord_6 must never be used")
    runs: list[Run] = []
    for s in specs:
        try:
            r = build_run(s)
        except Exception as e:  # report and continue: a failed run must be visible, not silent
            log(f"SKIP {s.split} {s.cfg}/{s.seq}: {type(e).__name__}: {e}")
            continue
        runs.append(r)
        log(f"loaded {s.split:6s} {r.name:18s} frames {len(r.t):5d} valid-drift windows {int(r.drift.valid.sum()):5d} "
            f"({r.gt_duration_valid_s:.0f} s) recoveries {int(r.recoveries[-1])}")
    global FEATURE_COLS
    FEATURE_COLS = [c for c in runs[0].feats.columns if a.feature_set == "v1" or not c.startswith("motion_")]
    log(f"feature set {a.feature_set}: {len(FEATURE_COLS)} features")
    train = [r for r in runs if r.spec.split == "ntnu23"]
    held = [r for r in runs if r.spec.split == "ntnu25"]
    cfgs = sorted({r.spec.cfg for r in train})
    if not train:
        raise SystemExit("no ntnu23 runs")

    ref_cfg = "h05" if "h05" in cfgs else cfgs[0]
    med = float(np.median(np.concatenate([r.drift.trans_mps[r.drift.valid] for r in train if r.spec.cfg == ref_cfg])))
    thresholds = {"abs": ABS_THR_MPS, "rel": REL_FACTOR * med}
    log(f"median window drift ({ref_cfg}, ntnu23) {med*100:.2f} cm/s -> rel threshold {thresholds['rel']*100:.2f} cm/s, "
        f"abs {ABS_THR_MPS*100:.0f} cm/s")

    results, ds_stats = [], []
    chosen: dict = {}
    for scheme, thr in thresholds.items():
        for H in HORIZONS_S:
            key = (scheme, H)
            label_runs(runs, scheme, H, thr)
            tag = f"{scheme}_H{H:g}"
            for setname, rs in (("ntnu23", train), ("ntnu25", held)):
                for cfg in sorted({r.spec.cfg for r in rs}):
                    sub = [r for r in rs if r.spec.cfg == cfg]
                    ys = np.concatenate([r.cache[key].y for r in sub])
                    n_ev = sum(len(lb.event_onsets(r.t, r.cache[key].onset, H)) for r in sub)
                    ds_stats.append({"label": tag, "set": setname, "cfg": cfg, "runs": len(sub),
                                     "frames": int(sum(len(r.t) for r in sub)), "labelled": int(np.isfinite(ys).sum()),
                                     "pos_rate": float(np.nanmean(ys)) if np.isfinite(ys).any() else float("nan"),
                                     "events": n_ev})
            log(f"\n=== label {tag}: threshold {thr*100:.2f} cm/s, horizon {H:g} s ===")
            # model selection by leave-one-trajectory-out
            oof: dict[str, dict[str, np.ndarray]] = {}
            for kind in ("logreg", "gbt"):
                best = None
                for p in GRIDS[kind]:
                    sc = loto_scores(train, key, kind, p)
                    fm = frame_metrics([seq_eval(r, sc[r.name], key, H) for r in train])
                    log(f"  select {kind} param={p}: LOTO pooled AUROC {fm['auroc']:.3f} AUPRC {fm['auprc']:.3f}")
                    if best is None or fm["auprc"] > best[0]:
                        best = (fm["auprc"], p, sc)
                chosen[(tag, kind)] = best[1]
                oof[kind] = best[2]
            for rule in RULES:
                oof[rule] = {r.name: rule_score(r, rule) for r in train}
            names = {"logreg": f"logreg C={chosen[(tag,'logreg')]}", "gbt": f"gbt depth={chosen[(tag,'gbt')]}"}
            for model, sc in oof.items():
                mname = names.get(model, model)
                seqs_all = [seq_eval(r, sc[r.name], key, H) for r in train]
                thr_a = choose_threshold(seqs_all, TARGET_FA_PER_MIN)
                for r, s in zip(train, seqs_all):
                    fm, al = frame_metrics([s]), alarm_stats([s], thr_a)
                    log(f"  fold {tag} {mname:24s} {r.name:18s} n {fm['n']:5d} pos {fm['pos_rate']:.2f} "
                        f"AUROC {fm['auroc']:.3f} AUPRC {fm['auprc']:.3f} events {al['events']} det {al['detected']}")
                for cfg in cfgs + ["all"]:
                    idx = [i for i, r in enumerate(train) if cfg == "all" or r.spec.cfg == cfg]
                    results.append({"label": tag, **row(mname, f"LOTO-{cfg}", [seqs_all[i] for i in idx], thr_a,
                                                      {"thr": thr_a})})
                if held:
                    # held-out: train on all ntnu23, threshold from the LOTO out-of-fold scores (no held-out data)
                    if model in ("logreg", "gbt"):
                        Xtr, ytr = xy(train, key)
                        m = fit_model(model, chosen[(tag, model)], Xtr, ytr)
                        hs = {r.name: m.predict_proba(fmat(r))[:, 1] for r in held}
                    else:
                        hs = {r.name: rule_score(r, model) for r in held}
                    hseq = [seq_eval(r, hs[r.name], key, H) for r in held]
                    for r, s in zip(held, hseq):
                        fm, al = frame_metrics([s]), alarm_stats([s], thr_a)
                        log(f"  held {tag} {mname:24s} {r.name:18s} n {fm['n']:5d} pos {fm['pos_rate']:.2f} "
                            f"AUROC {fm['auroc']:.3f} AUPRC {fm['auprc']:.3f} events {al['events']} det {al['detected']}")
                    results.append({"label": tag, **row(mname, "HELDOUT-all", hseq, thr_a, {"thr": thr_a})})
                    thr_o = choose_threshold(hseq, TARGET_FA_PER_MIN)
                    oracle = alarm_stats(hseq, thr_o)
                    results[-1].update({"oracle_event_recall": oracle["event_recall"],
                                        "oracle_median_lead_s": oracle["median_lead_s"],
                                        "oracle_fa_per_min": oracle["fa_per_min"]})
                    for r, s in zip(held, hseq):
                        results.append({"label": tag, **row(mname, f"HELDOUT-{r.spec.seq}", [s], thr_a, {"thr": thr_a})})
                if tag == f"{PRIMARY[0]}_H{PRIMARY[1]:g}" and model == "gbt":
                    np.savez(a.out / "plot_data.npz", **{n: v for n, v in sc.items()})
                    chosen["plot_thr"] = thr_a
    df = pd.DataFrame(results)
    df.to_csv(a.out / "metrics.csv", index=False)
    pd.DataFrame(ds_stats).to_csv(a.out / "dataset_stats.csv", index=False)
    (a.out / "chosen.json").write_text(json.dumps({"median_drift_mps": med, "thresholds_mps": thresholds,
                                                    "chosen": {str(k): v for k, v in chosen.items()}}, indent=1))
    _plot(a, train, chosen, thresholds)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        log("\n" + df[df.set.isin(["LOTO-all", "HELDOUT-all"])].to_string(index=False))
    log(f"done in {time.time()-t0:.0f} s")


def _plot(a, train: list[Run], chosen: dict, thresholds: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdat = a.out / "plot_data.npz"
    if not pdat.exists():
        return
    sc = np.load(pdat)
    r = next((x for x in train if x.name == a.plot_run), None)
    if r is None or r.name not in sc.files:
        return
    key = (PRIMARY[0], PRIMARY[1])
    labs = r.cache[key]
    t = r.t - r.t[0]
    fig, ax = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    ax[0].plot(t, 100 * r.drift.trans_mps, lw=0.7, label="drift rate (2 s window)")
    thr = thresholds[PRIMARY[0]]
    ax[0].axhline(100 * thr, color="r", ls="--", lw=0.8, label=f"label threshold {100 * thr:.1f} cm/s")
    for tr in t[labs.recovery]:
        ax[0].axvline(tr, color="k", lw=0.6, alpha=0.6)
    ax[0].set_ylabel("cm/s")
    ax[0].legend(loc="upper right", fontsize=7)
    ax[0].set_title(f"{r.name}: GBT out-of-fold risk, label {PRIMARY[0]} {100 * thr:.1f} cm/s, H={PRIMARY[1]:g} s "
                    "(black = gate recovery)", fontsize=9)
    ax[1].plot(t, sc[r.name], lw=0.7, color="C1", label="risk")
    ax[1].fill_between(t, 0, 1, where=labs.y == 1, color="r", alpha=0.12, label="y=1")
    ax[1].axhline(chosen["plot_thr"], color="k", ls=":", lw=0.8, label="alarm threshold (1 FA/min)")
    ax[1].set_xlabel("time since start (s)")
    ax[1].set_ylim(0, 1)
    ax[1].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out / "risk_plot.png", dpi=90)


if __name__ == "__main__":
    main()
