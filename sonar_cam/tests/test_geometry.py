import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sonar_cam.data import (Trajectory, load_camchain, load_svin_rig, sonar_in_cam0)
from sonar_cam.geometry import (invert, perturb, ping_range, project, sonar_points, sonar_world_points, transform,
                                world_to_cam)

K = np.array([[1000.0, 0, 800], [0, 1000.0, 600], [0, 0, 1]])
D0 = np.zeros(4)
SIZE = (1600, 1200)


def synthetic_pose(yaw_deg=30.0, t=(1.0, -2.0, 0.5)):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("zyx", [yaw_deg, 5.0, -3.0], degrees=True).as_matrix()
    T[:3, 3] = t
    return T


def test_sonar_point_convention():
    p = sonar_points(np.array([2.0]), np.array([90.0]))
    np.testing.assert_allclose(p, [[0, 2.0, 0]], atol=1e-12)  # (r cos h, r sin h, 0), degrees -> rad


def test_round_trip_world_pixel():
    T_c0_So = np.eye(4)
    T_c0_So[:3, :3] = np.array([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]])  # So x -> c0 z, So y -> c0 x, So z -> c0 y
    T_c0_So[:3, 3] = [0.05, 0.1, 0.0]
    T_w_c_ping = synthetic_pose(10.0, (0.0, 0.0, 0.0))[None]
    r, h = np.array([2.0]), np.array([0.0])
    Pw = sonar_world_points(T_w_c_ping, T_c0_So, r, h)
    # camera moves 0.3 m along its own z and yaws 2 deg before the image is taken
    T_w_c_img = T_w_c_ping[0] @ np.array([[np.cos(0.035), 0, np.sin(0.035), 0], [0, 1, 0, 0],
                                          [-np.sin(0.035), 0, np.cos(0.035), 0.3], [0, 0, 0, 1]])
    Pc = world_to_cam(T_w_c_img, Pw)
    uv, ok = project(Pc, K, D0, SIZE)
    # independent computation of the same pixel via explicit matrices
    p_so = np.array([2.0, 0.0, 0.0, 1.0])
    p_c_img = invert(T_w_c_img) @ T_w_c_ping[0] @ T_c0_So @ p_so
    exp = K @ (p_c_img[:3] / p_c_img[2])
    np.testing.assert_allclose(uv[0], exp[:2], atol=1e-6)
    # unproject with the depth and map back to world: round trip
    ray = np.linalg.inv(K) @ np.array([uv[0, 0], uv[0, 1], 1.0])
    P_back = T_w_c_img[:3, :3] @ (ray * Pc[0, 2]) + T_w_c_img[:3, 3]
    np.testing.assert_allclose(P_back, Pw[0], atol=1e-9)


def test_behind_camera_not_projected():
    uv, ok = project(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]), K, D0, SIZE)
    assert list(ok) == [False, True]
    np.testing.assert_allclose(uv[1], [800, 600])


def test_distortion_model_applied():
    dist = np.array([-0.17, 0.12, 0.0, 0.0])
    P = np.array([[0.5, 0.3, 1.0]])
    uv_d, _ = project(P, K, dist, SIZE)
    uv_0, _ = project(P, K, D0, SIZE)
    assert np.linalg.norm(uv_d - uv_0) > 5.0


def test_perturb_is_camera_frame_left_multiplication():
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, 1.0]
    Tp = perturb(T, rotvec_rad=(0, 0, np.pi / 2), trans_m=(0.1, 0, 0))
    np.testing.assert_allclose(Tp[:3, 3], [0.1, 0, 1.0], atol=1e-12)
    np.testing.assert_allclose(perturb(T), T)


def test_ping_range_matches_svin_rule():
    inten = np.zeros(250, np.uint8)
    inten[99] = 50          # peak -> range = 100 * 6/250 = 2.4 m
    inten[200] = 120        # inside the last 100 bins -> ignored like SVIn
    r, peak = ping_range(inten, 6.0)
    assert peak == 50 and r == pytest.approx(2.4)
    inten[99] = 10          # not above the intensity threshold
    assert np.isnan(ping_range(inten, 6.0)[0])
    inten[:] = 0
    inten[149] = 60         # 150*0.024 = 3.6 m < 4.5 -> accepted
    assert ping_range(inten, 6.0)[0] == pytest.approx(3.6)


def test_trajectory_interpolation(tmp_path):
    q0 = Rotation.identity().as_quat()
    q1 = Rotation.from_euler("z", 90, degrees=True).as_quat()
    f = tmp_path / "t.tum"
    f.write_text(f"0 0 0 0 {' '.join(map(str, q0))}\n2 2 0 0 {' '.join(map(str, q1))}\n")
    tr = Trajectory(str(f))
    T = tr.poses(np.array([0.5]))[0]
    np.testing.assert_allclose(T[:3, 3], [0.5, 0, 0], atol=1e-12)
    np.testing.assert_allclose(np.rad2deg(Rotation.from_matrix(T[:3, :3]).as_rotvec()[2]), 22.5, atol=1e-9)
    assert list(tr.valid(np.array([-1.0, 1.0, 3.0]), max_gap_s=5.0)) == [False, True, False]


def test_svin_rig_v1_matches_camchain():
    cc, rig = load_camchain(), load_svin_rig()
    np.testing.assert_allclose(np.linalg.inv(cc["cam0"]["T_cam_imu"]), rig["T_SC0"], atol=1e-9)
    np.testing.assert_allclose(np.linalg.inv(cc["cam1"]["T_cam_imu"]), rig["T_SC1"], atol=1e-9)
    T = sonar_in_cam0(rig)
    np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)
    # sonar scan plane (sonar z axis) is the camera optical axis for this rig definition
    np.testing.assert_allclose(T[:3, 2], [0.0, 0.0, 1.0], atol=0.03)
