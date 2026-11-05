import os
import json
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval as ev  # noqa: E402
import faithfulness as fa  # noqa: E402
import geometry as G  # noqa: E402
import ir_common as C  # noqa: E402

_DATA = Path(os.path.expanduser(os.environ.get("UWB_DATA", "~/svin_data")))  # dataset root (env UWB_DATA)
BUS_CC = _DATA / "bus_outside/camchain_bus_outside.yaml"
BUS_BAG = _DATA / "bus_outside/bus_outside.db3"


def test_select_keyframes_time_and_distance():
    t = list(np.arange(0, 10, 0.5))
    pos = [np.array([0.0, 0, 0])] * len(t)
    sel = C.select_keyframes(t, pos, 2.0, 4.0, 0.3)
    assert [t[i] for i in sel] == [0.0, 4.0, 8.0]  # stationary: forced by max_dt
    moving = [np.array([x, 0, 0]) for x in np.arange(len(t)) * 0.2]
    sel = C.select_keyframes(t, moving, 2.0, 4.0, 0.3)
    assert [t[i] for i in sel] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert C.select_keyframes(t, [None] * len(t), 2.0, 4.0, 0.3)[1] == 4  # unknown pose -> time only


def test_pair_stamps_unique_and_tolerance():
    p = C.pair_stamps([0.0, 0.1, 0.2], [0.001, 0.1, 0.35])
    assert p == [(0, 0), (1, 1)]


def test_interp_pose_midpoint_and_gap():
    tum = np.array([[0, 0, 0, 0, 0, 0, 0, 1.0], [1, 2, 0, 0, 0, 0, 0, 1.0], [10, 2, 0, 0, 0, 0, 0, 1.0]])
    T = C.interp_pose(tum, 0.5)
    assert np.allclose(T[:3, 3], [1, 0, 0]) and np.allclose(T[:3, :3], np.eye(3))
    assert C.interp_pose(tum, 5.0) is None  # 4 s from nearest sample
    assert C.interp_pose(tum, -0.4) is not None and C.interp_pose(tum, -0.6) is None


def test_stamp_from_cdr():
    data = b"\x00\x01\x00\x00" + struct.pack("<iI", 1494876479, 500_000_000) + b"rest"
    assert C._stamp_from_cdr(data) == pytest.approx(1494876479.5)


def test_load_tum_rejects_nan(tmp_path):
    p = tmp_path / "a.tum"
    p.write_text("0 0 0 0 0 0 0 1\n1 nan 0 0 0 0 0 1\n")
    assert len(C.load_tum(p)) == 1


@pytest.mark.skipif(not BUS_CC.exists(), reason="bus camchain not present")
def test_rectifier_baseline_and_synthetic_depth():
    cc = C.load_camchain(BUS_CC)
    rc = G.build_rectifier(cc, (1600, 1200))
    assert rc.baseline_m == pytest.approx(0.2318, abs=0.001)
    # right image = left shifted by a constant disparity => constant depth f*B/d
    rng = np.random.default_rng(0)
    base = (rng.random((1200, 1700)) * 255).astype(np.uint8)
    base = np.stack([__import__("cv2").GaussianBlur(base, (0, 0), 1.5)] * 3, -1)
    d = 40
    left, right = base[:, :1600], base[:, d:d + 1600]  # feature at x: left x, right x-d => disparity +d
    depth = G.sgbm_depth(left, right, rc, 0.5)
    expected = rc.P1[0, 0] * rc.baseline_m / d
    loc = G.locate_box([600, 500, 1000, 800], depth, rc.P1, 0.5)
    assert loc is not None and loc["depth_m"] == pytest.approx(expected, rel=0.05)
    assert loc["width_m"] == pytest.approx(400 * loc["depth_m"] / rc.P1[0, 0], rel=1e-6)


def test_locate_box_rejects_empty_depth():
    depth = np.full((600, 800), np.nan, np.float32)
    assert G.locate_box([100, 100, 300, 300], depth, np.diag([1000.0, 1000, 1]), 0.5) is None


def test_T_world_rect_identity_R1():
    T = np.eye(4)
    T[:3, 3] = [1, 2, 3]
    assert np.allclose(G.T_world_rect(T, np.eye(3)), T)


def _det(cls, t, xyz, score=0.5, box=(0, 0, 10, 10)):
    return {"cls": cls, "t": t, "score": score, "frame": f"kf{t}", "box": list(box), "xyz_world": xyz,
            "width_m": 0.1, "height_m": 0.2, "depth_m": 2.0}


def test_merge_by_3d_proximity_and_class():
    dets = [_det("fish", 0, [0, 0, 0]), _det("fish", 2, [0.5, 0, 0], 0.9), _det("fish", 4, [5, 0, 0]),
            _det("rope_cable", 2, [0.1, 0, 0])]
    f = G.merge_detections(dets, radius_m=1.0)
    assert len(f) == 3
    top = [x for x in f if x["cls"] == "fish" and x["n_obs"] == 2][0]
    assert top["score_max"] == 0.9 and top["best_frame"] == "kf2"
    assert [x["id"] for x in f] == ["F01", "F02", "F03"]


def test_merge_without_pose_uses_time_and_iou():
    a = _det("tire", 0, None, box=(0, 0, 100, 100))
    b = _det("tire", 2, None, box=(10, 10, 110, 110))
    c = _det("tire", 20, None, box=(10, 10, 110, 110))
    assert len(G.merge_detections([a, b, c])) == 2


FINDS = [{"id": "F01", "cls": "corrosion"}, {"id": "F02", "cls": "fish"}]


def test_faithfulness_clean_text():
    r = fa.check_text("Rust was found on the pipe [F01]. A fish was seen [F02].", FINDS)
    assert r["n_hallucinated"] == 0 and r["findings_not_mentioned"] == []


def test_faithfulness_counts_hallucinations():
    r = fa.check_text("A crack was found. Rust is visible [F09]. Fish noted [F01].", FINDS)
    assert len(r["unknown_ids"]) == 1
    assert [x["cls"] for x in r["absent_class_mentions"]] == ["crack"]
    assert len(r["id_class_mismatches"]) == 1
    assert r["n_hallucinated"] == 3


def test_faithfulness_negation_not_counted():
    assert fa.check_text("No cracks were detected.", FINDS)["n_hallucinated"] == 0


def test_image_level_prf_and_crop_eval():
    dets = {"a": [{"cls": "fish", "mode": "base", "score": 0.5}, {"cls": "tire", "mode": "base", "score": 0.1}],
            "b": [{"cls": "fish", "mode": "base", "score": 0.9}], "c": []}
    labels = {"a": ["fish"], "b": ["rope_cable"], "c": ["fish"]}
    r = ev.image_level(dets, labels, 0.15, "base")
    assert (r["fish"]["tp"], r["fish"]["fp"], r["fish"]["fn"]) == (1, 1, 1)
    assert r["tire"]["tp"] == 0 and r["tire"]["precision"] is None
    assert r["fish"]["precision"] == 0.5 and r["fish"]["recall"] == 0.5
    u = ev.image_level(dets, labels, 0.15, "base", unsure={"b": ["fish"]})
    assert (u["fish"]["tp"], u["fish"]["fp"], u["fish"]["fn"]) == (1, 0, 1)  # unsure (b, fish) dropped
    vlm = {"F01": {"shows_class": True}, "F02": {"shows_class": False}, "F03": {"shows_class": None}}
    c = ev.crop_eval(vlm, {"F01": True, "F02": True, "F03": False})
    assert c["vlm_verification_accuracy"] == 0.5 and c["vlm_unclear_or_unparsed"] == 1 and c["detector_crop_precision"] == pytest.approx(2 / 3)


@pytest.mark.skipif(not BUS_BAG.exists(), reason="bag absent")
def test_raw_messages_reads_real_db3_stamps():
    it = C.raw_messages(str(BUS_BAG), ["/slave1/image_raw/compressed"])
    topic, st, data, typ = next(it)
    assert typ == "sensor_msgs/msg/CompressedImage" and 1.4948e9 < st < 1.4949e9


def test_render_md_html_smoke(tmp_path):
    import cv2

    import render as R

    cv2.imwrite(str(tmp_path / "c.jpg"), np.zeros((50, 80, 3), np.uint8))
    f = {"id": "F01", "cls": "corrosion", "score_max": 0.4, "n_obs": 2, "frames": ["kf_0001"], "xyz_world": [1.0, 2.0, 3.0],
         "width_m": 0.3, "height_m": 0.2, "crop": "c.jpg"}
    meta = {"source": "s", "detector": "d", "vlm": "v", "thr": 0.15, "mode": "base"}
    cov = {"duration_s": 10, "n_keyframes": 5, "n_keyframes_with_pose": 5, "track_length_m": None, "traj_file": "none"}
    faith = fa.check_text("Rust [F01].", [f])
    vlm = {"F01": {"assessment": "pipe | rusty", "severity": "low"}}
    md = R.render_md(meta, cov, [f], vlm, "Summary [F01].", faith)
    assert "AI-GENERATED DRAFT" in md and "| F01 | corrosion |" in md and "pipe / rusty" in md
    h = R.render_html(meta, cov, [f], vlm, "Summary [F01].", faith, tmp_path, [(0, 0), (1, 1)])
    assert "data:image/jpeg;base64," in h and "<svg" in h


def test_faithfulness_flags_citing_rejected_finding():
    f = [{"id": "F01", "cls": "tire"}, {"id": "F02", "cls": "fish"}]
    r = fa.check_text("Key findings include [F01] tire and [F02] fish.", f, {"F01", "F02"})
    assert len(r["cites_rejected"]) == 2 and r["n_hallucinated"] >= 2
    ok = fa.check_text("[F01] is an unconfirmed tire candidate.", f, {"F01"})
    assert ok["cites_rejected"] == []
    # a class that only has rejected findings counts as absent
    assert fa.check_text("A fish was seen.", f, {"F02"})["absent_class_mentions"]
    assert fa.check_text("[F01] tire confirmed.", f, set())["n_hallucinated"] == 0


def test_crop_eval_skips_unsure_labels():
    vlm = {"F01": {"shows_class": True}, "F02": {"shows_class": True}, "F03": {"shows_class": False}}
    c = ev.crop_eval(vlm, {"F01": True, "F02": False, "F03": "unsure"})
    assert c["n_labelled_crops"] == 2 and c["precision_after_vlm_filter"] == 0.5 and c["vlm_recall_of_true_crops"] == 1.0


def test_check_report_scope_and_count_claim():
    f = [{"id": "F01", "cls": "tire"}, {"id": "F02", "cls": "fish"}, {"id": "F03", "cls": "window"}]
    pf = {"F01": {"shows_class": True, "assessment": "A worn tire."},
          "F02": {"shows_class": False, "assessment": "Too blurry to be a fish."},
          "F03": {"shows_class": True, "assessment": "Window frame looks corroded."}}
    r = fa.check_report("Draft with 3 confirmed findings: [F01] tire and [F03] window.", f, pf)
    assert r["count_claim_errors"] == [{"claimed": 3, "actual": 2}]
    assert r["cites_rejected"] == [] and len(r["absent_class_mentions"]) == 1  # F03 'corroded' (assessment)
    assert r["assessment_hallucinated"] == 1 and r["n_hallucinated"] == 2  # F02 rationale not checked
