import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

def quat_conjugate(q):
    """Conjugate of quaternion [w,x,y,z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

def quat_mul(q1, q2):
    """Multiply two quaternions [w,x,y,z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

def joint_dofnum(model, jid):
    jtype = model.jnt_type[jid]
    if jtype == mujoco.mjtJoint.mjJNT_FREE:
        return 6
    elif jtype == mujoco.mjtJoint.mjJNT_BALL:
        return 3
    elif jtype in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
        return 1
    else:
        raise ValueError(f"Unknown joint type: {jtype}")
    
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

def rotate_quat_around_z90(q_wxyz):
    """Rotate quaternion q_wxyz (wxyz) by +90° about Z axis."""
    # Convert to xyzw for SciPy
    q_xyzw = np.roll(q_wxyz, -1)
    
    # Original rotation
    r = Rotation.from_quat(q_xyzw)
    
    # Extra 90° about Z
    r_z90 = Rotation.from_rotvec(np.pi/2 * np.array([0, 0, 1]))
    
    # Compose: r' = r_z90 * r
    r_new = r_z90 * r
    
    # Convert back to [w,x,y,z]
    q_new_xyzw = r_new.as_quat()
    q_new_wxyz = np.roll(q_new_xyzw, 1)
    return q_new_wxyz

def normalize_quat_wxyz(q):
    q = np.asarray(q, dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)

def same_points_unordered(A, B, tol=1e-8):

    # print(A[0:10])
    # print(B[0:10])
    if A.shape != B.shape:
        return False
    A_sorted = np.array(sorted(map(tuple, np.round(A, 8))))
    B_sorted = np.array(sorted(map(tuple, np.round(B, 8))))
    return np.allclose(A_sorted, B_sorted, atol=tol)

def generate_initial_object_pose(level="easy", seed=42):
    """
    Levels: "easy", "medium", "hard"
    """
    rng = np.random.default_rng(seed)
    init_pose = np.zeros((1, 7), dtype=float)

    # Base pose
    x0, y0, z0 = 0.2, 0.2, 0.04175

    if level == "easy":
        # Straight line in x; no rotation
        init_pose[0, 0] = x0
        init_pose[0, 1] = y0
        init_pose[0, 2] = z0
        init_pose[0, 3] = 1.0  # [qw,qx,qy,qz] = identity

    elif level == "medium":
        # Diagonal translation + slow rotation about z (0 -> 90°)
        theta = np.pi/2
        init_pose[0, 0] = x0
        init_pose[0, 1] = y0
        init_pose[0, 2] = z0
        init_pose[0, 3:7] = axis_angle_to_quat_wxyz([0, 0, 1], theta)

    elif level == "hard":
        # 3D path (circle in XY + bobbing in Z) + continuous rotation about arbitrary axis
        theta = np.pi/2
        init_pose[0, 0] = x0
        init_pose[0, 1] = y0
        init_pose[0, 2] = z0
        # Rotate around a tilted axis (mix of x,y,z)
        axis = np.array([0.6, 0.4, 0.7])
        init_pose[0, 3:7] = axis_angle_to_quat_wxyz(axis, theta)

    else:
        raise ValueError(f"Unknown level: {level}")

    # Ensure quaternions normalized (numerical safety)
    init_pose[0, 3:7] = normalize_quat_wxyz(init_pose[0, 3:7])

    return init_pose

def draw_pointclouds_and_normals_in_viewer(
    viewer,
    points,                 # (N,3) array of points in world frame
    normals=None,           # (N,3) array of unit normals in world frame (optional)
    clear=True,             # clear previous overlays in user_scn
    point_radius=0.002,
    normal_len=0.005,
):
    """Draw points as spheres and optional normals as line segments in the MuJoCo viewer.
       This operates directly in world coordinates and does NOT call viewer.sync().
    """
    if isinstance(points, np.ndarray):
        points = [points]

    if clear:
        viewer.user_scn.ngeom = 0

    # ---- draw points as spheres ----
    rgba_point = np.array([0.95, 0.2, 0.2, 1.0], dtype=np.float32)   # red
    size_vec  = np.array([point_radius, 0.0, 0.0], dtype=np.float64) # (3,)
    mat_I9    = np.eye(3, dtype=np.float64).reshape(-1)              # identity rot

    all_points_world = []
    for pc in points:
        if pc is None or len(pc) == 0:
            continue
        pc = np.asarray(pc, dtype=np.float64)
        all_points_world.append(pc)

        for pt in pc:
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size_vec,
                pt.astype(np.float64),
                mat_I9,
                rgba_point
            )
            viewer.user_scn.ngeom += 1

    if len(all_points_world) > 0:
        all_points_world = np.vstack(all_points_world)
    else:
        all_points_world = np.zeros((0, 3))

    # ---- draw normals as line segments ----
    if normals is not None and len(all_points_world) == len(normals):
        rgba_normal = np.array([0.1, 0.9, 0.1, 1.0], dtype=np.float32)  # green
        width = 0.002
        for pt, n in zip(all_points_world, normals):
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                break
            a = pt.astype(np.float64)
            b = (pt + normal_len * n).astype(np.float64)
            g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
            g.rgba = rgba_normal
            viewer.user_scn.ngeom += 1


def add_gaussian_outliers(points, num_outliers=30, scale=1.0, seed=42):
    """
    Add synthetic Gaussian outliers far from the bounding box of the input cloud.

    Args:
        points (ndarray): shape (N,3), original point cloud.
        num_outliers (int): how many outliers to add.
        scale (float): spread of outliers relative to cloud size.
        seed (int or None): random seed for reproducibility.

    Returns:
        new_points (ndarray): shape (N+num_outliers, 3)
        outliers (ndarray): shape (num_outliers, 3), the added outliers.
    """
    rng = np.random.default_rng(seed)

    # bounding box and extent
    min_pt = points.min(axis=0)
    max_pt = points.max(axis=0)
    center = 0.5 * (min_pt + max_pt)
    extent = max_pt - min_pt

    # Gaussian outliers scaled outside bounding box
    outliers = center + scale * (rng.standard_normal((num_outliers, 3)) * extent * 3.0)

    new_points = np.vstack([points, outliers])

    return new_points

# Vectorized version
def wrap_array_to_pi(arr):
    return (arr + np.pi) % (2 * np.pi) - np.pi

def solve_ik(model, data, body_name, target_pos, target_quat, joint_limit,
             joint_names=None, max_iter=4000, tol=1e-4, step_size=0.15):
    """
    Iterative IK solver using Jacobian pseudo-inverse, restricted to selected joints.

    Args:
        model, data : MuJoCo model and data
        body_name   : str, name of the end-effector body
        target_pos  : (3,) desired world position
        target_quat : (4,) desired world quaternion [w, x, y, z] 
        joint_limit : (Num_joints, 2) joint limits for those used in IK
        joint_names : list of str, joint names to use in IK (ignore free bodies etc.) 
        max_iter    : maximum iterations
        tol         : stopping tolerance
        step_size   : update scaling factor

    Returns:
        qpos (ndarray): joint configuration (copy of data.qpos after solving)
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    # If no joint list provided, include all joints
    if joint_names is None:
        joint_ids = list(range(model.njnt))
    else:
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joint_names]

    # Collect DOF indices for these joints
    dof_indices = []

    for jid in joint_ids:
        adr = model.jnt_dofadr[jid]
        nv = 1
        dof_indices.extend(range(adr, adr+nv))
    dof_indices = np.array(dof_indices, dtype=int)

    inital_qpos = data.qpos.copy()
    
    for it in range(max_iter):
        mujoco.mj_forward(model, data)

        # Current EE pose
        cur_pos = data.xpos[body_id].copy()
        cur_quat = data.xquat[body_id].copy()

        # Position error
        err_pos = target_pos - cur_pos

        # Orientation error (optional)
        if target_quat is not None:
            q_err = quat_mul(target_quat, quat_conjugate(cur_quat))
            err_ori = q_err[1:]  # vector part as orientation error
            err = np.concatenate([err_pos, err_ori])
        else:
            err = err_pos

        if np.linalg.norm(err) < tol:
            break

        # Jacobian
        Jp = np.zeros((3, model.nv))
        Jr = np.zeros((3, model.nv))

        mujoco.mj_jacBody(model, data, Jp, Jr, body_id)

        J = Jp if target_quat is None else np.vstack([Jp, Jr])

        # Restrict Jacobian to selected DOFs
        J_reduced = J[:, dof_indices]

        # Solve update step
        dq = step_size * (np.linalg.pinv(J_reduced) @ err)

        # Apply to selected DOFs
        data.qpos[dof_indices] += dq

        # Clamp to joint limits
        data.qpos[dof_indices] = np.clip(
            data.qpos[dof_indices],
            joint_limit[:, 0],
            joint_limit[:, 1]
        )

    final_qpos = data.qpos.copy()

    # rewrite the state back to the original state before the IK
    data.qpos = inital_qpos
    mujoco.mj_forward(model, data)
    
    return wrap_array_to_pi(final_qpos[:6])

def draw_antipodal_pairs_in_viewer(
    viewer,
    points,       # (N,3) array of points in world frame
    pairs,        # list of (i,j) index pairs
    clear=True,   # clear previous overlays in user_scn
    pair_width=0.003,
):
    """Draw only antipodal pairs as line segments in the MuJoCo viewer.

    Args:
        viewer: MuJoCo viewer object
        points (ndarray): (N,3) array of 3D points
        pairs (list of tuples): each (i,j) defines a pair of contact indices
        clear (bool): whether to clear previous overlays
        pair_width (float): visual thickness of the pair connector
    """
    if clear:
        viewer.user_scn.ngeom = 0

    points = np.asarray(points, dtype=np.float64)
    rgba_pair = np.array([0.2, 0.2, 0.95, 1.0], dtype=np.float32)  # blue

    for i, j in pairs:
        if i < 0 or j < 0 or i >= len(points) or j >= len(points):
            continue
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break

        a = points[i].astype(np.float64)
        b = points[j].astype(np.float64)

        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, pair_width, a, b)
        g.rgba = rgba_pair
        viewer.user_scn.ngeom += 1

def draw_grasp_origins_in_viewer(
    viewer,
    grasp_poses,   # list of dicts (with "R","t","pair") or SE(3) matrices
    points=None,   # (N,3) array, needed to draw antipodal pairs
    clear=True,
    axis_len=0.03,
    pair_width=0.003
):
    """
    Draw gripper origins as axis frames in the MuJoCo viewer,
    and optionally draw antipodal connection lines if pairs are available.

    Args:
        viewer: MuJoCo viewer object
        grasp_poses: list of dicts with {"R","t","pair"} or (4,4) SE3 numpy arrays
        points: (N,3) array of scene points (needed for pairs)
        clear (bool): clear previous overlays
        axis_len (float): length of axis arrows
        pair_width (float): line thickness for antipodal connector
    """
    if clear:
        viewer.user_scn.ngeom = 0

    colors_axis = [
        np.array([1, 0, 0, 1], dtype=np.float32),  # X red
        np.array([0, 1, 0, 1], dtype=np.float32),  # Y green
        np.array([0, 0, 1, 1], dtype=np.float32),  # Z blue
    ]
    rgba_pair = np.array([0.2, 0.2, 0.95, 1.0], dtype=np.float32)  # blue line

    for gpose in grasp_poses:
        # --- handle dict or SE(3) numpy array ---
        if isinstance(gpose, dict):
            R = np.asarray(gpose["R"], dtype=np.float64)
            t = np.asarray(gpose["t"], dtype=np.float64)
            pair = gpose.get("pair", None)
        else:  # assume (4,4) transform
            T = np.asarray(gpose, dtype=np.float64)
            R, t = T[:3, :3], T[:3, 3]
            pair = None

        # --- draw frame axes ---
        for k in range(3):
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                break
            start = t
            end = t + axis_len * R[:, k]
            g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.002, start, end)
            g.rgba = colors_axis[k]
            viewer.user_scn.ngeom += 1

        # --- draw antipodal pair connector (if available) ---
        if pair is not None and points is not None:
            i, j = map(int, pair)
            if 0 <= i < len(points) and 0 <= j < len(points):
                a = points[i].astype(np.float64)
                b = points[j].astype(np.float64)
                if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, pair_width, a, b)
                    g.rgba = rgba_pair
                    viewer.user_scn.ngeom += 1