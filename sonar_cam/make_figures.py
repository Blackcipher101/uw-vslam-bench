"""README figures for sonar_cam -> ../docs/figures/ (needs out/ from the commands in README.md).

Usage: python3 make_figures.py
  sonar_geometry.png        scan plane vs camera FOV from the SVIn extrinsic (numbers from README / camchain)
  sonar_sweeps.png          window sweep (out/window_sweep_*.log) and time-offset sweep (out/baseline_w20_*.json)
  sonar_overlay_pair.png    two frames of out/overlay_svin (downscaled, palette PNG)
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
OUT = HERE / "out"
FIG = HERE.parent / "docs" / "figures"

GREY, BLUE, RED, GREEN = "#4d4d4d", "#0072B2", "#D55E00", "#009E73"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "figure.facecolor": "white",
                     "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False})

# bus_outside camchain cam0 (left, 1600x1200 radtan): fx, fy, cx, cy
FX, FY, CX, CY, W_PX, H_PX = 1167.87421447215, 1166.9553764645004, 771.763112280329, 589.9626482585169, 1600, 1200
# T_c0_So translation (m) in the camera frame (x right, y down, z forward), README "Rig / extrinsic"
SONAR_T = np.array([0.127, 0.227, 0.177])
MIN_RANGE_M = 0.5  # baseline.py --min-range used for every result in the README


def save_png(fig, path: Path, colors: int | None = None, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(io.BytesIO(buf.getvalue()))
    if colors:
        im = im.convert("RGB").quantize(colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
    im.save(path, optimize=True)
    print(f"{path} {path.stat().st_size / 1024:.0f} KB")


def fov_patch_at_plane() -> tuple[float, float, float, float]:
    """Camera-frame (x0, x1, y0, y1) of the image rectangle at the scan plane z = SONAR_T[2] (pinhole, no distortion)."""
    z = SONAR_T[2]
    return (-CX / FX * z, (W_PX - CX) / FX * z, -CY / FY * z, (H_PX - CY) / FY * z)


def fig_geometry() -> None:
    x0, x1, y0, y1 = fov_patch_at_plane()
    sx, sy, sz = SONAR_T
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    d = np.hypot(corners[:, 0] - sx, corners[:, 1] - sy)
    # nearest point of the rectangle to the sonar head (clamp), farthest = farthest corner
    nearest = np.hypot(np.clip(sx, x0, x1) - sx, np.clip(sy, y0, y1) - sy)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    # top-down: x right, z forward
    zmax = 1.1
    t = np.tan(np.arctan2(W_PX - CX, FX))
    tl = np.tan(np.arctan2(CX, FX))
    a.fill([0, zmax * t, -zmax * tl], [0, zmax, zmax], color=BLUE, alpha=0.12, lw=0)
    a.plot([0, zmax * t], [0, zmax], color=BLUE, lw=1)
    a.plot([0, -zmax * tl], [0, zmax], color=BLUE, lw=1)
    a.plot([0, 0], [0, zmax], color=GREY, ls=":", lw=1)
    a.plot([-0.75, 0.95], [sz, sz], color=RED, lw=2.5)
    a.plot([x0, x1], [sz, sz], color=GREEN, lw=6, solid_capstyle="butt", zorder=3)
    a.scatter([0], [0], color="k", zorder=4)
    a.scatter([sx], [sz], color=RED, marker="s", zorder=4)
    a.annotate("stereo camera (left, cam0)", (0, 0), (-0.55, 0.12), fontsize=9, arrowprops=dict(arrowstyle="-", color=GREY))
    a.annotate(f"sonar scan plane (edge-on), z = {sz:.3f} m,\nperpendicular to the optical axis;\nextends to the max range of 6 m",
               (0.55, sz), (0.12, 0.55), fontsize=9, color=RED, arrowprops=dict(arrowstyle="-", color=RED))
    a.annotate(f"only |x| < {max(abs(x0), x1):.2f} m of the\nplane is inside the image", (0.0, sz), (-0.7, 0.62),
               fontsize=9, color=GREEN, arrowprops=dict(arrowstyle="-", color=GREEN))
    a.text(sx + 0.02, sz - 0.05, "sonar head", fontsize=9, color=RED, va="top")
    a.text(0.02, 1.02, "camera FOV (h. half-angle 34 deg)", color=BLUE, fontsize=9)
    a.set_xlim(-0.8, 1.0)
    a.set_ylim(-0.1, zmax)
    a.set_aspect("equal")
    a.set_xlabel("camera x, right (m)")
    a.set_ylabel("camera z, forward (m)")
    a.set_title("Top-down: the scan plane sits just in front of the lens")
    # front view at the scan plane: camera x right, y down
    b.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color=GREEN, alpha=0.35, label="image footprint on the scan plane"))
    th = np.linspace(0, 2 * np.pi, 200)
    b.plot(sx + MIN_RANGE_M * np.cos(th), sy + MIN_RANGE_M * np.sin(th), color=RED, lw=1.5,
           label=f"range ring {MIN_RANGE_M} m (min range in baseline)")
    b.scatter([sx], [sy], color=RED, marker="s", zorder=4, label="sonar head")
    b.scatter([0], [0], color="k", zorder=4, label="optical axis")
    b.set_xlim(-0.65, 0.7)
    b.set_ylim(0.75, -0.55)
    b.set_aspect("equal")
    b.set_xlabel("camera x, right (m) at z = 0.177 m")
    b.set_ylabel("camera y, down (m)")
    b.set_title(f"Scan-plane view: the image sees ranges {nearest:.2f}-{d.max():.2f} m of the sonar", fontsize=10.5)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8, frameon=False, ncol=2)
    fig.suptitle("Why same-time sonar points are out of view: extrinsic T_c0_So (SVIn rig v1), bus_outside", fontsize=11.5, y=1.0)
    fig.tight_layout()
    save_png(fig, FIG / "sonar_geometry.png")


LOG_RE = re.compile(r"window\s+([\d.]+)s future [\d.]+: n=(\d+) median=(\S+) mad=(\S+) .*inlier_frac=(\S+) spearman")


def read_window_log(path: Path) -> dict[str, np.ndarray]:
    rows = []
    for line in path.read_text().splitlines():
        m = LOG_RE.search(line)
        if m and int(m.group(2)) > 0:
            rows.append([float(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))])
    a = np.array(rows)
    return dict(win=a[:, 0], n=a[:, 1], med=a[:, 2], mad=a[:, 3], inl=a[:, 4])


def fig_sweeps() -> None:
    gt, vio = read_window_log(OUT / "window_sweep_gt.log"), read_window_log(OUT / "window_sweep_vio.log")
    ts = {k: json.loads((OUT / f"baseline_w20_{k}.json").read_text())["time_sweep"] for k in ("gt", "vio")}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.9))
    for d, name, col, off in ((gt, "COLMAP GT poses", GREY, 0.97), (vio, "uw_ct_vio poses", BLUE, 1.03)):
        a.errorbar(d["win"] * off, d["med"], yerr=d["mad"], marker="o", color=col, capsize=3, label=name)
    a.axhline(0, color=GREEN, lw=1.5, label="perfect agreement")
    a.set_xscale("log")
    a.set_xticks([4.2, 10, 20, 40, 80], ["4.2", "10", "20", "40", "80"])
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlabel("accumulation window (s)")
    a.set_ylabel("median residual z_stereo - z_sonar (m), bar = MAD")
    a.set_title("Window: biased and widely spread", fontsize=10.5)
    a.legend(fontsize=8, frameon=False, loc="lower left")
    a.grid(alpha=0.25)
    for k, name, col in (("gt", "COLMAP GT poses", GREY), ("vio", "uw_ct_vio poses", BLUE)):
        dt = [r["dt_s"] for r in ts[k]]
        b.plot(dt, [r["median"] for r in ts[k]], marker="o", color=col, label=name)
        c.plot(dt, [100 * r["inlier_frac"] for r in ts[k]], marker="o", color=col, label=name)
    b.axhline(0, color=GREEN, lw=1.5)
    b.set_ylim(-2.2, 0.3)
    b.set_xlabel("sonar time offset (s), 20 s window")
    b.set_ylabel("median residual (m)")
    b.set_title("Time offset: flat, no optimum", fontsize=10.5)
    b.grid(alpha=0.25)
    c.set_ylim(0, 10)
    c.set_xlabel("sonar time offset (s), 20 s window")
    c.set_ylabel("inliers: |res| < 0.15 m + 5 % z (%)")
    c.set_title("Inlier fraction stays at chance level", fontsize=10.5)
    c.grid(alpha=0.25)
    fig.suptitle("Sonar vs stereo depth on AFRL bus_outside: no window or time offset makes the two agree", fontsize=11.5, y=1.02)
    fig.tight_layout()
    save_png(fig, FIG / "sonar_sweeps.png")


def fig_overlays() -> None:
    names = [("overlay_0100.06.png", "t = 100.1 s: 0 of 1924 sonar points in image"),
             ("best047_overlay_0132.56.png", "t = 132.6 s: 47 of 1908 in image (right edge)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, (n, t) in zip(axes, names):
        im = Image.open(OUT / "overlay_svin" / n).convert("RGB")
        ax.imshow(im.resize((440, 330), Image.Resampling.LANCZOS))
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.suptitle("Left camera with projected sonar points (colour = range), SVIn extrinsic, AFRL bus_outside",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    save_png(fig, FIG / "sonar_overlay_pair.png", colors=256)


if __name__ == "__main__":
    fig_geometry()
    fig_sweeps()
    fig_overlays()
