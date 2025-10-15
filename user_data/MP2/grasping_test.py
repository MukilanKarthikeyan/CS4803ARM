import time
import numpy as np
import mujoco
import trimesh
import xml.etree.ElementTree as ET
from utils import * 
from Grasping import Grasping
from scipy.spatial.transform import Rotation

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

def set_targets_by_dict(targets: dict):
    """
    targets: dict like {"joint1": 0.0, "joint2": 1.0, ...}
    Writes into data.ctrl in actuator order, with safe clamping.
    """
    for name, value in targets.items():
        i = actuator_id(name)
        lo, hi = ctrl_range[i]
        data.ctrl[i] = np.clip(value, lo, hi)

def goto(targets, body_id, target_pos, target_quat, duration = 1.0, rate = 500.0):
    """
    Interpolate current actuator values -> 'targets' over 'duration' seconds.
    """
    steps = max(1, int(duration * rate))
    start = data.ctrl.copy()
    # Build a full vector goal in actuator order
    goal = start.copy()
    for i in range(targets.shape[0]):
        lo, hi = ctrl_range[i]
        goal[i] = np.clip(targets[i], lo, hi)

    for k in range(steps):  # interpolation to find the targets
        alpha = (k + 1) / steps
        data.ctrl[:] = (1 - alpha) * start + alpha * goal
        mujoco.mj_step(model, data)

        # Current EE pose
        cur_pos = data.xpos[body_id].copy()
        cur_quat = data.xquat[body_id].copy()

        # print("Current EE pose:")
        # print(f"  position   = {[f'{x:.4f}' for x in cur_pos]}")
        # print(f"  quaternion = {[f'{x:.4f}' for x in cur_quat]}")

        # print("Target EE pose:")
        # print(f"  position   = {[f'{x:.4f}' for x in target_pos]}")
        # print(f"  quaternion = {[f'{x:.4f}' for x in target_quat]}")

        if viewer is not None:
            viewer.sync()
        time.sleep(1.0 / rate)  # Control the update rate

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

def gripper(opening: float):
    """
    opening in [0.0, 1.0]: 0=closed, 1=fully open
    Controls the single gripper actuator (range 0 to 0.035m).
    """
    gripper_actuator = actuator_id("gripper")
    lo, hi = ctrl_range[gripper_actuator]   # Should be [0, 0.035]
    # Linear map from [0,1] to [lo,hi]
    target_pos = lo + opening * (hi - lo)
    set_targets_by_dict({"gripper": target_pos})

object_selected = "banana"

# === 2. Define the object initial pose ===
# Each row is [x, y, z, qw, qx, qy, qz]
diff_case = "easy"  # "easy", "medium", "hard"

init_pose = generate_initial_object_pose(diff_case)[0]

# === 1. Find the qpos index for the object body ===
object_joint_name = f"{object_selected}_joint"
object_qpos_id = model.joint(name=object_joint_name)
object_geom_id = model.geom(name=f"{object_selected}_visual").id  # or object_collision
object_qpos_addr = object_qpos_id.qposadr[0] # start index in data.qpos
object_body_id = model.body(name=object_selected).id

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
num_flow = 512

object_mesh_visual = trimesh.load(mesh_files[f"{object_selected}_visual_mesh"]) 

# Sample self.number_flow_point points on the surface
np.random.seed(42)  # fixed seed for reproducibility
surface_points_visual_mesh = object_mesh_visual.sample(num_flow)  # shape: (self.number_flow_point, 3)

# np.save(os.path.join(base_output_dir, "original_points_visual_mesh.npy"), surface_points_visual_mesh)

data.qpos[object_qpos_addr : object_qpos_addr + 7] = init_pose
mujoco.mj_forward(model, data)

# move the robot to the initial joint angles
# Initial robot pose
Init_pos = np.array([0.2, 0.2, 0.20])  # The working area is centered around (0.2, 0.2)
Init_quat = np.array([
    0,  # w
    0.9648696336811518,  # x
    -0.2627291202365383, # y
    0.0               # z
]) # (-Z axis)

# account for the robot end effector's axis
Init_quat = rotate_quat_around_z90(Init_quat)

# the name of 6 robot arm joints for IK
actuated_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

# name of the robot arm's end effecor name
ee_name = "link6"
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
qpos_sol = solve_ik(model, data,
                    body_name=ee_name,
                    target_pos=Init_pos,
                    target_quat=Init_quat,
                    joint_limit=ctrl_range[:len(actuated_joint_names)],
                    joint_names=actuated_joint_names)

print("Move arm to initial poses.")
gripper(1.0)
goto(qpos_sol, body_id, Init_pos, Init_quat, duration=1.5)

# Small delay for smooth transition to grasping
for t in range(200):
    mujoco.mj_step(model, data)
    if viewer is not None:
        viewer.sync()

    time.sleep(0.01)  

# Main function
print("Starting point cloud processing to generate grasp poses...")

# Position of object in world frame (x, y, z)
object_pos = data.xpos[object_body_id].copy()
object_quat = data.xquat[object_body_id].copy()
object_rotmat = data.xmat[object_body_id].reshape(3, 3).copy()

# Register flow points
registered_surface_points_world_frame_visual_mesh = (
    surface_points_visual_mesh @ object_rotmat.T + object_pos
)

registered_surface_points_world_frame_visual_mesh_w_outliers = add_gaussian_outliers(registered_surface_points_world_frame_visual_mesh)

# define the Grasping class
Grasping_class = Grasping(registered_surface_points_world_frame_visual_mesh_w_outliers)
filtered_points = Grasping_class.filter_outliers() 

# Check if the point clouds are cleaned.
print(same_points_unordered(registered_surface_points_world_frame_visual_mesh, filtered_points))

neighbors_idx = Grasping_class.nearest_neighbor(num_local_point=10)
normals = Grasping_class.estimate_normals(neighbors_idx=neighbors_idx)

antipodal_pairs = Grasping_class.find_antipodal_pairs(normals=normals)  # everyone can come up with different heuristic 
scores = Grasping_class.evaluate_grasp_score(normals, antipodal_pairs)

# sort indices by score descending (selecting the top 10)
selected_grasp = 10
top_idx = np.argsort(scores)[::-1][:selected_grasp]
top_pairs = [antipodal_pairs[i] for i in top_idx]
top_scores = scores[top_idx]

grasp_poses = Grasping_class.compute_grasp_poses(top_pairs)

# Visualize the grasp poses
for t in range(300):
    if viewer is not None:
        # Visualize the point cloud and their normals
        draw_pointclouds_and_normals_in_viewer(
            viewer,
            filtered_points,
            normals
        )
        # Visualize the generated grasp poses
        draw_grasp_origins_in_viewer(
            viewer,
            grasp_poses,
            registered_surface_points_world_frame_visual_mesh
        )
        viewer.sync()

        # Exit condition
        if not viewer.is_running():
            print("Viewer closed. Exiting simulation.")
            viewer.close()
            break
    time.sleep(0.01) 

# Begin the grasping evaluation
print(f"Testing the best grasp candidate.")
i = 0
best_grasp = grasp_poses[i]

"""Convert (3,3) rotation matrix to quaternion [w,x,y,z]."""
quat_xyzw = Rotation.from_matrix(best_grasp["R"]).as_quat() 
quat_wxyz = np.roll(quat_xyzw, 1)    # reorder to [w, x, y, z]

# account for the robot end effector's axis
target_pos = best_grasp["t"]
target_quat = rotate_quat_around_z90(quat_wxyz)

qpos_sol = solve_ik(model, data,
                    body_name=ee_name,
                    target_pos=target_pos,
                    target_quat=target_quat,
                    joint_limit=ctrl_range[:len(actuated_joint_names)],
                    joint_names=actuated_joint_names)

goto(qpos_sol, body_id, target_pos, target_quat, duration=1.5)

# Begin grasping
# Close gripper
print("Closing gripper...")
gripper(0.0)

for t in range(300):
    mujoco.mj_step(model, data)
    if viewer is not None:
        viewer.sync()

    time.sleep(0.01)  # ~100 FPS

# lifting
print("Lifting arm...")
qpos_sol = solve_ik(model, data,
                    body_name=ee_name,
                    target_pos=Init_pos,
                    target_quat=Init_quat,
                    joint_limit=ctrl_range[:len(actuated_joint_names)],
                    joint_names=actuated_joint_names)
goto(qpos_sol, body_id, Init_pos, Init_quat, duration=2.5)