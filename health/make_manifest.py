"""Build the run manifest (JSON) for run_health.py from the ws1_sync layout.

  python3 make_manifest.py --out manifest.json [--ref25-glob 'results/.../{seq}.tum']

Frame logs: <sync>/framelog/fl_<cfg>_<seq>.csv; matching left-camera TUM: <sync>/framelog_tum/**/<seq>.tum whose
path contains the cfg tag (override with --tum-glob, placeholders {cfg} {seq}).
ntnu23: GT = <sync>/mocap/<seq>/<seq>_mocap.tum. ntnu25: GT = the ReAqROVIO reference (--ref25-glob, {seq}); without it the ntnu25 runs are skipped.
fjord_6 is excluded by name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEQ23 = [f"traj_0{i}" for i in range(1, 7)]
SEQ25 = ["mclab_1", "mclab_2"] + [f"fjord_{i}" for i in range(1, 6)]   # fjord_6 deliberately absent
CFG23 = ("h05", "nolcn")
CFG25 = ("h05",)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", type=Path, default=ROOT / "results/ws1_sync")
    ap.add_argument("--tum-glob", default="framelog_tum/fl_{cfg}/{seq}.tum")
    ap.add_argument("--ref25-glob", default=None, help="reference TUM for ntnu25, relative to --sync, with {seq}")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = []
    for split, seqs, cfgs in (("ntnu23", SEQ23, CFG23), ("ntnu25", SEQ25, CFG25)):
        for cfg in cfgs:
            for seq in seqs:
                fl = a.sync / "framelog" / f"fl_{cfg}_{seq}.csv"
                tums = [p for p in a.sync.glob(a.tum_glob.format(cfg=cfg, seq=seq)) if cfg in str(p.relative_to(a.sync))]
                gt = (a.sync / "mocap" / seq / f"{seq}_mocap.tum") if split == "ntnu23" else None
                if gt is None and a.ref25_glob:
                    g = sorted(a.sync.glob(a.ref25_glob.format(seq=seq)))
                    gt = g[0] if g else None
                if not (fl.exists() and len(tums) == 1 and gt is not None and gt.exists()):
                    print(f"missing {split} {cfg}/{seq}: framelog {fl.exists()} tum {len(tums)} gt {gt}")
                    continue
                out.append({"split": split, "seq": seq, "cfg": cfg, "framelog": str(fl), "est": str(tums[0]),
                            "gt": str(gt), "hand_eye": True})
    a.out.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} runs -> {a.out}")


if __name__ == "__main__":
    main()
