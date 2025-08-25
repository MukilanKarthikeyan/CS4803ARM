import time
import numpy as np
import mujoco
import os

# --- Load model/data ---
# Updated path to use the assets directory structure
model = mujoco.MjModel.from_xml_path("../assets/scene.xml")  # Load the complete scene with robot and breadboard
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

def goto(targets: dict, duration: float = 1.0, rate: float = 500.0):
    """
    Interpolate current actuator values -> 'targets' over 'duration' seconds.
    """
    steps = max(1, int(duration * rate))
    start = data.ctrl.copy()
    # Build a full vector goal in actuator order
    goal = start.copy()
    for name, value in targets.items():
        i = actuator_id(name)
        lo, hi = ctrl_range[i]
        goal[i] = np.clip(value, lo, hi)

    for k in range(steps):
        alpha = (k + 1) / steps
        data.ctrl[:] = (1 - alpha) * start + alpha * goal
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        time.sleep(1.0 / rate)  # Control the update rate

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

print("Move arm with position actuators...")
# 2) Move arm with position actuators (these are target positions, not torques)
#    Updated joint ranges for new robot model
q_goal = {
    "joint1": 0.5,      # Base rotation: -2.618 to 2.618 rad
    "joint2": 1.0,      # Shoulder: 0 to 3.14 rad  
    "joint3": -1.0,     # Elbow: -2.697 to 0 rad
    "joint4": 0.3,      # Wrist roll: -1.832 to 1.832 rad
    "joint5": -0.2,     # Wrist pitch: -1.22 to 1.22 rad
    "joint6": 0.8,      # Wrist yaw: -3.14 to 3.14 rad
}
goto(q_goal, duration=1.5)

# 3) Open/close gripper.
#    Updated for new robot: single gripper actuator controls joint7
#    joint7 range: 0 (closed) to 0.035 (open)
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

# Open gripper
print("Opening gripper...")
gripper(1.0)
for t in range(300):
    mujoco.mj_step(model, data)
    if viewer is not None:
        viewer.sync()
    time.sleep(0.01)

# Close gripper
print("Closing gripper...")
gripper(0.0)
for t in range(300):
    mujoco.mj_step(model, data)
    if viewer is not None:
        viewer.sync()
    time.sleep(0.01)

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
