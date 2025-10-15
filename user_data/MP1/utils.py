import time
import numpy as np
import mujoco
import os
import matplotlib.pyplot as plt
import shutil
import cv2
import trimesh
import xml.etree.ElementTree as ET
import matplotlib.animation as animation
from datetime import datetime

def axis_angle_to_quat_wxyz(axis, angle):
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    s = np.sin(angle / 2.0)
    qw = np.cos(angle / 2.0)
    qx, qy, qz = axis * s
    return np.array([qw, qx, qy, qz])

def rot_mat_to_quat_wxyz(R, eps=1e-8):
    R = np.asarray(R, dtype=float)
    assert R.shape == (3,3)
    tr = np.trace(R)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0  # 4*qw
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    else:
        # choose the largest diagonal to keep S large
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(max(eps, 1.0 + R[0,0] - R[1,1] - R[2,2])) * 2.0
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(max(eps, 1.0 + R[1,1] - R[0,0] - R[2,2])) * 2.0
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(max(eps, 1.0 + R[2,2] - R[0,0] - R[1,1])) * 2.0
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S

    q = np.array([qw, qx, qy, qz])
    q /= np.linalg.norm(q)
    return q

def normalize_quat_wxyz(q):
    q = np.asarray(q, dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)

def generate_icp_traj(T, level="easy", seed=42):
    """
    Returns array of shape (T, 7): [x, y, z, qw, qx, qy, qz]
    Levels: "easy", "medium"
    """
    rng = np.random.default_rng(seed)
    traj = np.zeros((T, 7), dtype=float)

    # Base pose
    x0, y0, z0 = 0.2, 0.0, 0.04175

    if level == "easy":
        # Straight line in x; no rotation
        traj[:, 0] = np.linspace(x0, x0 + 0.2, T)  # x: 0.2 -> 0.4
        traj[:, 1] = np.linspace(y0, y0 + 0.1, T)  # x: 0.2 -> 0.4
        traj[:, 2] = z0
        traj[:, 3] = 1.0  # [qw,qx,qy,qz] = identity

    elif level == "medium":
        # Diagonal translation + slow rotation about z (0 -> 90°)
        theta = np.linspace(0, np.pi/3, T)
        traj[:, 0] = x0 + 0.05 * np.linspace(0, 1, T)
        traj[:, 1] = y0 + 0.18 * np.linspace(0, 1, T)
        traj[:, 2] = z0 + 0.02 * np.linspace(0, 1, T)
        for i, th in enumerate(theta):
            traj[i, 3:7] = axis_angle_to_quat_wxyz([0, 0, 1], th)

    else:
        raise ValueError(f"Unknown level: {level}")

    # Ensure quaternions normalized (numerical safety)
    for i in range(T):
        traj[i, 3:7] = normalize_quat_wxyz(traj[i, 3:7])

    return traj

def _quat_wxyz_to_R(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
    ], dtype=float)

def _transform_points(pts_local, R, t):
    return pts_local @ R.T + t

def draw_pointclouds_and_pose_in_viewer(
    viewer,
    points,                 # (N,3) array OR list of (Ni,3)
    pos_w,                  # (3,) world position of the object
    quat_wxyz,              # (4,) world quaternion of the object [qw,qx,qy,qz]
    points_in_world=True,   # set False if 'points' are in the object local frame
    clear=True,             # clear previous overlays in user_scn
    point_radius=0.005,
    axis_len=0.05
):
    """Fill viewer.user_scn with point spheres and pose axes. Does NOT call viewer.sync()."""
    if isinstance(points, np.ndarray):
        points = [points]

    if clear:
        viewer.user_scn.ngeom = 0

    pos_w = np.asarray(pos_w, dtype=float)
    quat_wxyz = np.asarray(quat_wxyz, dtype=float)
    R_w_from_obj = _quat_wxyz_to_R(quat_wxyz)

    # ---- draw points as spheres (fixed types/shapes) ----
    rgba_point = np.array([0.95, 0.2, 0.2, 1.0], dtype=np.float32)  # float32
    size_vec  = np.array([point_radius, 0.0, 0.0], dtype=np.float64) # float64 (3,)
    mat_I9    = np.eye(3, dtype=np.float64).reshape(-1)              # float64 (9,)

    for pc in points:  # ensure you made a list beforehand
        if pc is None or len(pc) == 0:
            continue
        pc = np.asarray(pc, dtype=np.float64)
        pc_w = pc if points_in_world else (pc @ R_w_from_obj.T + pos_w)

        for pt in pc_w:
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                break

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size_vec,                         # (3,) float64
                pt.astype(np.float64),            # (3,) float64
                mat_I9,                           # (9,) float64 rotation matrix
                rgba_point                        # (4,) float32
            )
            viewer.user_scn.ngeom += 1

    # ---- draw pose axes (X/Y/Z) as capsules/lines ----
    colors = [
        np.array([1, 0, 0, 1], dtype=np.float32),  # X
        np.array([0, 1, 0, 1], dtype=np.float32),  # Y
        np.array([0, 0, 1, 1], dtype=np.float32),  # Z
    ]
    axes_world = R_w_from_obj @ np.eye(3, dtype=np.float64)
    o = pos_w.astype(np.float64)
    width = 0.003

    for i in range(3):
        a = o
        b = (o + axis_len * axes_world[:, i]).astype(np.float64)

        # MuJoCo 3.x signature: (geom, type, width, from, to)
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        g.rgba = colors[i]  # set color separately on the geom
        viewer.user_scn.ngeom += 1