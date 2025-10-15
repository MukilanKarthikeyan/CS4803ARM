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
from PointCloud import PointCloud
from utils import * 

# --- Load model/data ---
# Updated path to use the assets directory structure
model = mujoco.MjModel.from_xml_path("./scene.xml")  # Load the complete scene with robot and breadboard
data  = mujoco.MjData(model)

# (Optional) viewer - Updated for MuJoCo 3.3.5
import mujoco.viewer
viewer = mujoco.viewer.launch_passive(model, data)

# --- Helpers ---
def actuator_id(name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

def joint_id(name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

# Build a stable name->index mapping for actuators
# Updated for new robot: 6 arm joints + 1 gripper actuator
act_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
act_idx   = np.array([actuator_id(n) for n in act_names], dtype=int)

def set_qpos_by_joint_names(qpos_targets: dict):
    """
    Initialize pose by directly setting joint positions, then forward the model.
    Use joint names (not actuator names) for clarity.
    """
    for name, q in qpos_targets.items():
        jid = joint_id(name)
        dof = model.jnt_qposadr[jid]
        data.qpos[dof] = q
    mujoco.mj_forward(model, data)

# Extract actuator ctrl ranges for clamping (shape: [nu, 2])
ctrl_range = model.actuator_ctrlrange.copy()

# Print actuator information for debugging
print("Available actuators:")
for i in range(model.nu):
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  {i}: {actuator_name}")

print(f"\nControl ranges:")
for name in act_names:
    try:
        idx = actuator_id(name)
        lo, hi = ctrl_range[idx]
        print(f"  {name}: [{lo:.3f}, {hi:.3f}]")
    except:
        print(f"  {name}: NOT FOUND")
        
# --- Example: home, move, and operate gripper ---
# 1) Set a comfortable start pose (radians for revolute joints)
# Updated for new robot: 6 arm joints + gripper joints
home = {
    "joint1": 0.0,     # Base rotation
    "joint2": 0.0,     # Shoulder
    "joint3": 0.0,     # Elbow  
    "joint4": 0.0,     # Wrist roll
    "joint5": 0.0,     # Wrist pitch
    "joint6": 0.0,     # Wrist yaw
    "joint7": 0.0,     # Gripper (0 = closed, 0.035 = open)
}
# Set initial joint positions
set_qpos_by_joint_names(home)
data.ctrl[:] = 0.0
for t in range(200):
    mujoco.mj_step(model, data)
    if viewer is not None:
        viewer.sync()

    time.sleep(0.001)  # Small delay for smooth visualization

print("Move the object...")
object_selected = "banana"

# === 1. Find the qpos index for the object body ===
object_joint_name = f"{object_selected}_joint"
object_qpos_id = model.joint(name=object_joint_name)
object_geom_id = model.geom(name=f"{object_selected}_visual").id  # or object_collision
object_qpos_addr = object_qpos_id.qposadr[0] # start index in data.qpos
object_body_id = model.body(name=object_selected).id

# === 2. Define your trajectory ===
T = 50
diff_case = "easy"  # "easy", "medium"

object_traj = generate_icp_traj(T, diff_case)

# identify the path of each object's mesh file
tree = ET.parse("./scene.xml")
root = tree.getroot()

# Find all mesh file references
mesh_files = {}
for mesh in root.findall(".//mesh"):
    name = mesh.get("name")
    file = mesh.get("file")
    if name and file:
        mesh_files[name] = file

# number of point clouds
num_flow = 256

object_mesh_visual = trimesh.load(mesh_files[f"{object_selected}_visual_mesh"]) 
object_mesh_collision = trimesh.load(mesh_files[f"{object_selected}_collision_mesh"]) 

# Sample self.number_flow_point points on the surface
np.random.seed(42)  # fixed seed for reproducibility
surface_points_visual_mesh = object_mesh_visual.sample(num_flow)  # shape: (self.number_flow_point, 3)
surface_points_collision_mesh = object_mesh_collision.sample(num_flow)  # shape: (self.number_flow_point, 3)

positions = []
rotations_quat = []
rotations_mat = []
registered_surface_points_world_frame_trajectories_visual_mesh = []
random_surface_points_world_frame_trajectories_visual_mesh = []
registered_surface_points_world_frame_trajectories_collision_mesh = []
random_surface_points_world_frame_trajectories_collision_mesh = []

# Set random seed
np.random.seed(42)

# Main Loop
data.qpos[object_qpos_addr : object_qpos_addr + 7] = object_traj[0]
object_pos = data.xpos[object_body_id].copy()  # shape (3,)
object_rotmat = data.xmat[object_body_id].reshape(3, 3).copy()  # shape (3, 3)
registered_surface_points_world_frame_visual_mesh = surface_points_visual_mesh @ object_rotmat.T + object_pos  # registered surface points
pc = PointCloud(scene_points=registered_surface_points_world_frame_visual_mesh, model_points=surface_points_visual_mesh)

for t in range(T):
    # === 3. Playback loop ===
    data.qpos[object_qpos_addr : object_qpos_addr + 7] = object_traj[t]
    mujoco.mj_forward(model, data)

    # Position of object in world frame (x, y, z)
    object_pos = data.xpos[object_body_id].copy()  # shape (3,)

    # Orientation as quaternion (w, x, y, z)
    object_quat = data.xquat[object_body_id].copy()  # shape (4,)

    # Orientation as rotation matrix (3x3)
    object_rotmat = data.xmat[object_body_id].reshape(3, 3).copy()  # shape (3, 3)

    positions.append(object_pos)
    rotations_quat.append(object_quat)
    rotations_mat.append(object_rotmat)

    # Add position offset -> object flow points in the world frame
    registered_surface_points_world_frame_visual_mesh = surface_points_visual_mesh @ object_rotmat.T + object_pos  # registered surface points
    # registered_surface_points_world_frame_collision_mesh = object_pos + surface_points_collision_mesh  # registered surface points

    pc.sceneP = registered_surface_points_world_frame_visual_mesh
    # pc.modelP = pc.alignedP.copy()
    R_est, t_est, history, aligned = pc.icp(trim_fraction=0)

    # pc.modelP = aligned.copy()
    print("\n=== ICP ===")
    print("Iterations:", len(history))
    print("Final RMSE:", history[-1])
    print("Estimated R:\n", R_est)
    print("Ground truth R:\n", object_rotmat)
    print("Estimated t:", t_est)
    print("Ground truth t:", object_pos)

    # Optional: check how close (up to noise/outliers/sign ambiguities)
    R_err = R_est @ object_rotmat.T
    t_err = t_est - object_pos
    ang_err = np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)))
    print(f"\nRotation error (deg): {ang_err:.4f}")
    print("Translation error   :", np.linalg.norm(t_err))

    object_quat_est = rot_mat_to_quat_wxyz(R_est)
    
    if viewer is not None:
        estimated_point_cloud = surface_points_visual_mesh @ R_est.T + t_est
        # visualize the estimated pose with the estimated object point cloud
        draw_pointclouds_and_pose_in_viewer(viewer, estimated_point_cloud, t_est, object_quat_est)
        # visualize the estimated pose with the ground-truth object point cloud
        # draw_pointclouds_and_pose_in_viewer(viewer, registered_surface_points_world_frame_visual_mesh, t_est, object_quat_est)
        viewer.sync()

    time.sleep(1 / 100.0)  # ~100 FPS

print("Done.")

# Keep the viewer open
if viewer is not None:
    print("Simulation complete. Close the viewer window to exit.")
    try:
        while viewer.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Simulation interrupted.")
    finally:
        viewer.close()