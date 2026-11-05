"""Re-score saved report outputs (no model): recompute faithfulness with faithfulness.check_report,
re-render report.md/html, and write the result back into vlm_outputs.json.

  python3 rescore.py --run run_bus_v1 --inputs run_bus [--traj s4_rep.tum]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import faithfulness
from ir_common import load_tum
from render import render_html, render_md


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--traj", default=None, help="TUM file for the plan view (default: path stored in coverage.json)")
    a = ap.parse_args(argv)
    run, src = Path(a.run), Path(a.inputs)
    findings = json.load(open(src / "findings.json"))["findings"]
    cov = json.load(open(src / "coverage.json"))
    if a.traj:
        cov["traj_file"] = a.traj
    v = json.load(open(run / "vlm_outputs.json"))
    v["faithfulness"] = faithfulness.check_report(v["summary"], findings, v["per_finding"])
    json.dump(v, open(run / "vlm_outputs.json", "w"), indent=1)
    track_xz = []
    if cov["traj_file"] != "none" and Path(cov["traj_file"]).exists():
        tum = load_tum(cov["traj_file"])
        m = (tum[:, 0] >= cov["t_start"]) & (tum[:, 0] <= cov["t_end"])
        track_xz = [(float(r[1]), float(r[3])) for r in tum[m][:: max(1, int(m.sum() // 400))]]
    f = {k: v["faithfulness"][k] for k in ("n_hallucinated", "summary_only_hallucinated", "assessment_hallucinated", "n_confirmed", "count_claim_errors")}
    (run / "report.md").write_text(render_md(v["meta"], cov, findings, v["per_finding"], v["summary"], v["faithfulness"]))
    (run / "report.html").write_text(render_html(v["meta"], cov, findings, v["per_finding"], v["summary"], v["faithfulness"], src, track_xz))
    print(run, f)


if __name__ == "__main__":
    main()
