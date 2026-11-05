"""README figures for health/ -> ../docs/figures/.

Usage (from the repo root or health/):
  python3 health/make_figures.py [--results health/results/v0] [--manifest health/manifest.json]
  health_auc_bars.png    AUROC / AUPRC, LOTO vs held-out, from <results>/metrics.csv (label rel_H5, the pre-declared primary)
  health_risk_time.png   out-of-fold GBT risk vs drift over time for h05/traj_06; needs <results>/plot_data.npz, which
                         `run_health.py --feature-set v0 --out <results>` writes (git-ignored) and the ws1 logs named in the manifest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "docs" / "figures"
sys.path.insert(0, str(HERE))

GREY, BLUE, RED, GREEN = "#4d4d4d", "#0072B2", "#D55E00", "#009E73"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "figure.facecolor": "white",
                     "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False})
PRIMARY_LABEL = "rel_H5"
MODEL_LABELS = {"logreg C=1.0": "logistic\nregression", "gbt depth=5": "gradient-\nboosted trees",
                "rule: vision_healthy==0": "rule:\nvision_healthy\n== 0", "rule: gated_frac_1s high": "rule:\ngated fraction\nhigh",
                "rule: tracked_1s low": "rule:\ntracked features\nlow"}


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{p} {p.stat().st_size / 1024:.0f} KB")


def fig_bars(results: Path) -> None:
    df = pd.read_csv(results / "metrics.csv")
    df = df[(df.label == PRIMARY_LABEL) & df.set.isin(["LOTO-all", "HELDOUT-all"])]
    models = [m for m in MODEL_LABELS if m in set(df.model)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, name in zip(axes, ("auroc", "auprc"), ("AUROC", "AUPRC")):
        for j, (setname, col, lab) in enumerate((("LOTO-all", BLUE, "LOTO, NTNU 2023 (12 runs)"),
                                                 ("HELDOUT-all", RED, "held-out, NTNU 2025 (7 runs)"))):
            for i, m in enumerate(models):
                r = df[(df.model == m) & (df.set == setname)].iloc[0]
                x = i + (j - 0.5) * 0.38
                ax.bar(x, r[metric], width=0.36, color=col, label=lab if i == 0 else None)
                ax.text(x, r[metric] + 0.008, f"{r[metric]:.2f}", ha="center", va="bottom", fontsize=8)
                chance = 0.5 if metric == "auroc" else r["pos_rate"]
                ax.plot([x - 0.2, x + 0.2], [chance, chance], color=GREY, ls="--", lw=1.4, zorder=4,
                        label="chance" if (i == 0 and j == 0) else None)
        ax.set_xticks(range(len(models)), [MODEL_LABELS[m] for m in models], fontsize=8)
        ax.set_ylabel(name)
        ax.set_ylim(0.0, 0.85)
        ax.set_title(f"{name}" + (" (chance 0.5)" if metric == "auroc" else " (chance = positive rate: 0.336 LOTO, 0.272 held-out)"), fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("Drift warning from frame-log features is barely above chance (rel 7.58 cm/s, H = 5 s, features v0)", fontsize=11.5, y=1.02)
    fig.tight_layout()
    save(fig, "health_auc_bars.png")


def fig_risk(results: Path, manifest: Path, run_name: str = "h05/traj_06") -> None:
    pdat = results / "plot_data.npz"
    if not pdat.exists():
        print(f"health_risk_time.png skipped: {pdat} missing (run run_health.py first)")
        return
    import labels as lb  # noqa: F401  (health/labels.py)
    from data import build_run, load_manifest
    chosen = json.loads((results / "chosen.json").read_text())
    thr = chosen["thresholds_mps"]["rel"]
    spec = next(s for s in load_manifest(manifest) if f"{s.cfg}/{s.seq}" == run_name and s.split == "ntnu23")
    r = build_run(spec)
    labs = lb.make_labels(r.t, r.drift, r.recoveries, 5.0, thr)
    risk = np.load(pdat)[run_name]
    t = r.t - r.t[0]
    drift = 100 * r.drift.trans_mps.copy()
    drift[~r.drift.valid] = np.nan
    pos = labs.y == 1
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]})
    ax[0].plot(t, drift, lw=0.8, color=BLUE, label="drift rate vs GT (2 s window; gaps = no mocap)")
    ax[0].axhline(100 * thr, color=RED, ls="--", lw=1.2, label=f"failure threshold {100 * thr:.2f} cm/s")
    ax[0].set_ylabel("drift rate (cm/s)")
    ax[0].set_ylim(0, 15)
    ax[0].legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    ax[0].set_title("Out-of-fold risk does not single out the windows that precede drift (NTNU underwater 2023 traj_06, h05)", fontsize=10.5)
    ax[1].plot(t, risk, lw=0.8, color=GREY, label="GBT risk (leave-one-trajectory-out)")
    ax[1].fill_between(t, 0, 1, where=pos, color=RED, alpha=0.25, step="mid", label="label y = 1: drift exceeds within 5 s")
    ax[1].axhline(chosen["chosen"]["plot_thr"], color=GREEN, ls=":", lw=1.5, label="alarm threshold (1 false alarm / min, pooled)")
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("risk score")
    ax[1].set_xlabel("time since start of run (s)")
    ax[1].legend(loc="upper left", fontsize=8, frameon=False, ncol=1)
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "health_risk_time.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results" / "v0")
    ap.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    a = ap.parse_args()
    fig_bars(a.results)
    fig_risk(a.results, a.manifest)
