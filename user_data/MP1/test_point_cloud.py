# test_point_cloud.py
import numpy as np
import pytest
from PointCloud import PointCloud


# -----------------------
# Helpers
# -----------------------
def set_seed(seed=0):
    np.random.seed(seed)


def random_rotation(seed=None):
    """Generate a random proper rotation (det=+1)."""
    if seed is not None:
        set_seed(seed)
    A = np.random.randn(3, 3)
    Q, R = np.linalg.qr(A)
    # Make det +1
    if np.linalg.det(Q) < 0:
        Q[:, 2] *= -1
    return Q


def apply_transform(P, R, t):
    """Apply rigid transform to Nx3 points."""
    return (R @ P.T).T + t


def rms(a, b):
    return np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))


def rot(axis, angle_rad):
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([
            [c + x*x*(1-c),     x*y*(1-c) - z*s, x*z*(1-c) + y*s],
            [y*x*(1-c) + z*s,   c + y*y*(1-c),   y*z*(1-c) - x*s],
            [z*x*(1-c) - y*s,   z*y*(1-c) + x*s, c + z*z*(1-c)]
        ])
        return R

# -----------------------
# register() tests
# -----------------------
def test_register_identity():
    set_seed(1)
    P = np.random.randn(100, 3)
    pc = PointCloud(scene_points=P, model_points=P.copy())
    R, p, rmse = pc.register()
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-7)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-7)
    assert np.allclose(p, np.zeros(3), atol=1e-10)
    assert np.isclose(rmse, 0.0, atol=1e-10)


def test_register_pure_translation():
    set_seed(2)
    P = np.random.randn(200, 3)
    t = np.array([0.5, -1.2, 2.0])
    Q = P + t
    pc = PointCloud(scene_points=Q, model_points=P)
    R, p, rmse = pc.register()
    assert np.allclose(R, np.eye(3), atol=1e-7)
    assert np.allclose(p, t, atol=1e-7)
    assert np.isclose(rmse, 0.0, atol=1e-10)


def test_register_pure_rotation():
    set_seed(3)
    P = np.random.randn(150, 3)
    Rtrue = random_rotation(seed=3)
    Q = (Rtrue @ P.T).T
    pc = PointCloud(scene_points=Q, model_points=P)
    R, p, rmse = pc.register()
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-7)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-7)
    assert np.allclose(R, Rtrue, atol=1e-6)
    assert np.allclose(p, np.zeros(3), atol=1e-7)
    assert np.isclose(rmse, 0.0, atol=1e-10)


def test_register_noisy_points_small_error():
    set_seed(4)
    N = 300
    P = np.random.randn(N, 3)
    Rtrue = random_rotation(seed=4)
    ttrue = np.array([0.2, -0.3, 0.5])
    Qclean = apply_transform(P, Rtrue, ttrue)
    noise = 0.01 * np.random.randn(N, 3)
    Q = Qclean + noise
    pc = PointCloud(scene_points=Q, model_points=P)
    R, p, rmse = pc.register()
    # Should be close to ground truth; rmse ≈ noise level
    assert np.allclose(R, Rtrue, atol=5e-2)
    assert np.allclose(p, ttrue, atol=5e-2)
    assert rmse < 0.05


def test_register_handles_reflection_case():
    """
    Ensure D = diag(1,1,det(UV^T)) forces proper rotation (det=+1)
    even when data would otherwise suggest a reflection.
    """
    set_seed(5)
    P = np.random.randn(200, 3)
    # Create a reflection across x (det = -1) and a translation
    R_reflect = np.diag([-1, 1, 1])
    t = np.array([0.3, 0.1, -0.2])
    Q = apply_transform(P, R_reflect, t)
    pc = PointCloud(scene_points=Q, model_points=P)
    R, p, rmse = pc.register()
    # The algorithm enforces det +1; result won't exactly match reflection,
    # but must still be a proper rotation and minimize error.
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-7)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-7)
    # Error should be finite and reasonably small (best proper-rotation fit).
    assert np.isfinite(rmse)


def test_register_mismatched_shapes_raises():
    set_seed(6)
    P = np.random.randn(50, 3)
    Q = np.random.randn(51, 3)
    pc = PointCloud(scene_points=Q, model_points=P)
    with pytest.raises(AssertionError):
        pc.register()


# -----------------------
# nearest_neighbor() tests
# -----------------------
def test_nearest_neighbor_simple_layout():
    # Model points form a small 1D chain; scene has same points permuted
    model = np.array([[-1.0, 0.0, 0.0],
                      [ 0.0, 0.0, 0.0],
                      [ 1.0, 0.0, 0.0]])
    scene = np.array([[ 1.0, 0.0, 0.0],
                      [-1.0, 0.0, 0.0],
                      [ 0.0, 0.0, 0.0]])
    pc = PointCloud(scene_points=scene, model_points=model)
    idx, d2 = pc.nearest_neighbor(model)
    # For each model point, nearest in scene should be the identical coordinate
    # scene[1] matches model[0], scene[2] matches model[1], scene[0] matches model[2]
    assert np.all(idx == np.array([1, 2, 0]))
    assert np.allclose(d2, np.zeros(3))


def test_nearest_neighbor_with_offset():
    # Scene shifted by +t; nearest distances should equal ||t||^2 for identical configs
    t = np.array([0.4, -0.2, 0.1])
    model = np.array([[0, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=float)
    scene = model + t
    pc = PointCloud(scene_points=scene, model_points=model)
    idx, d2 = pc.nearest_neighbor(model)  # compare model to shifted scene
    assert np.allclose(d2, np.full(len(model), np.dot(t, t)))


# -----------------------
# icp() tests
# -----------------------
def test_icp_perfect_alignment_from_scratch():
    set_seed(7)
    R_true = rot(np.array([0.3, 0.7, 0.6]), np.deg2rad(25))
    t_true = np.array([0.4, -0.2, 0.3])
    P = np.random.randn(200, 3)

    scene = (R_true @ P.T).T + t_true

    pc = PointCloud(scene_points=scene, model_points=P)
    R, p, history, aligned = pc.icp(max_iterations=50, tolerance=1e-9)

    # Orthonormal and proper
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-7)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-7)

    # Close to truth
    print("Estimated R:\n", R)
    print("True R:\n", R_true)
    assert np.allclose(R, R_true, atol=1e-6)
    print("Estimated t:", p)
    print("True t:", t_true)
    assert np.allclose(p, t_true, atol=1e-6)
    # Final alignment quality
    assert rms(aligned, scene) < 1e-6
    # Monotonic non-increasing RMSE (numerically tolerant)
    assert all(history[i+1] <= history[i] + 1e-12 for i in range(len(history)-1))


def test_icp_with_noise_converges_close():
    set_seed(8)
    N = 500
    R_true = rot(np.array([0.1, 0.2, 0.4]), np.deg2rad(17))
    p_true = np.array([-0.1, 0.2, 0.05])
    model = np.random.randn(N, 3)

    scene_clean = (R_true @ model.T).T + p_true
    noise = 0.01 * np.random.randn(N, 3)
    scene = scene_clean + noise

    pc = PointCloud(scene_points=scene, model_points=model)
    R, p, history, aligned = pc.icp(max_iterations=100, tolerance=1e-7)

    print("Estimated R:\n", R)
    print("True R:\n", R_true)
    assert np.allclose(R, R_true, atol=5e-2)
    print("Estimated t:", p)
    print("True t:", p_true)
    assert np.allclose(p, p_true, atol=5e-2)
    assert rms(aligned, scene) < 0.05


def test_icp_with_outliers_and_trimming():
    set_seed(9)
    N_inliers = 400
    N_outliers = 100

    model_in = np.random.randn(N_inliers, 3)
    R_true = rot(np.array([0.2, 0.1, 0.7]), np.deg2rad(20))
    p_true = np.array([0.3, -0.2, 0.1])
    scene_in = apply_transform(model_in, R_true, p_true)

    # Add scene outliers far away
    scene_out = 10 + 5 * np.random.randn(N_outliers, 3)
    scene = np.vstack([scene_in, scene_out])

    # Start ICP with all model points (only inliers exist in model)
    pc = PointCloud(scene_points=scene, model_points=model_in)
    # Keep the closest 80% to reject the far outliers
    R, p, history, aligned = pc.icp(max_iterations=100, tolerance=1e-7, trim_fraction=0.2)

    print("Estimated R:\n", R)
    print("True R:\n", R_true)
    assert np.allclose(R, R_true, atol=8e-2)
    print("Estimated t:", p)
    print("True t:", p_true)
    assert np.allclose(p, p_true, atol=8e-2)
    # Alignment should be good on the inliers despite outliers
    assert rms(aligned, scene_in) < 0.1


def test_icp_respects_initial_guess_and_accumulates():
    set_seed(10)
    N = 300
    model = np.random.randn(N, 3)

    # True transform
    R_true = rot(np.array([0.3, 0.7, 0.6]), np.deg2rad(31))
    p_true = np.array([0.6, -0.1, 0.2])
    scene = apply_transform(model, R_true, p_true)

    # Give ICP a (nontrivial) initial guess
    R0 = rot(np.array([0.1, 0.5, 0.2]), np.deg2rad(25))
    p0 = np.array([-0.2, 0.1, 0.05])

    pc = PointCloud(scene_points=scene, model_points=model)
    R, p, history, aligned = pc.icp(R_init=R0, p_init=p0, max_iterations=100, tolerance=1e-8)

    # Should still converge to the true transform
    print("Estimated R:\n", R)
    print("True R:\n", R_true)
    assert np.allclose(R, R_true, atol=5e-2)
    print("Estimated t:", p)
    print("True t:", p_true)
    assert np.allclose(p, p_true, atol=5e-2)
    assert rms(aligned, scene) < 0.05


def test_icp_stops_on_tolerance():
    set_seed(12)
    N = 200
    model = np.random.randn(N, 3)
    R_true = random_rotation(seed=12)
    p_true = np.array([0.0, 0.0, 0.0])
    scene = apply_transform(model, R_true, p_true)

    pc = PointCloud(scene_points=scene, model_points=model)
    # Use relatively loose tolerance to ensure early stop
    R, p, history, aligned = pc.icp(max_iterations=100, tolerance=1e-4)

    assert len(history) < 100  # early stopping happened
    assert np.isfinite(history[-1])


# Optional: property-like checks for R from register() and icp()
@pytest.mark.parametrize("seed", [13, 14, 15, 16])
def test_R_is_orthonormal_and_proper_from_register(seed):
    set_seed(seed)
    N = 250
    P = np.random.randn(N, 3)
    Rtrue = random_rotation(seed=seed)
    ttrue = np.random.randn(3) * 0.3
    Q = apply_transform(P, Rtrue, ttrue)
    pc = PointCloud(scene_points=Q, model_points=P)
    R, p, rmse = pc.register()
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-7)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-7)


@pytest.mark.parametrize("seed", [17, 18, 19, 20])
def test_R_is_orthonormal_and_proper_from_icp(seed):
    set_seed(seed)
    N = 250
    P = np.random.randn(N, 3)
    Rtrue = random_rotation(seed=seed)
    ttrue = np.random.randn(3) * 0.2
    Q = apply_transform(P, Rtrue, ttrue)
    pc = PointCloud(scene_points=Q, model_points=P)
    R, p, history, aligned = pc.icp(max_iterations=80, tolerance=1e-8)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-7)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-7)
