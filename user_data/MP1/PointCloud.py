import numpy as np

class PointCloud:
    def __init__(self, scene_points, model_points):
        self.sceneP = scene_points # Scene points in world frame
        self.modelP = model_points # Model points in model frame
        self.alignedP = model_points.copy() # Aligned model points in world frame after ICP
        self.R = None # Rotation matrix
        self.p = None # Translation vector

    def register(self, scene_points=None, model_points=None):
        """
        Computes the rigid transform (R, p) that best aligns model_points to scene_points
        in a least-squares sense (point-to-point; Kabsch/Umeyama without scaling).
        scene_points, model_points: Nx5 arrays with correspondences (self.sceneP[i] <-> self.modelP[i]).

        Returns: R (3x3), p (3,), and rms error.
        """
        if scene_points is not None:
            sceneP = scene_points
        else:
            sceneP = self.sceneP
        if model_points is not None:
            modelP = model_points
        else:
            modelP = self.modelP
        assert sceneP.shape == modelP.shape

        ######### START STUDENT CODE #########
        scene_centroid = np.mean(sceneP, axis=0)
        model_centroid = np.mean(modelP, axis=0)
        centered_scence = sceneP - scene_centroid
        centered_model = modelP - model_centroid
        H = centered_model.T @ centered_scence
        upper, sig, val = np.linalg.svd(H)
        R = val.T @ upper.T
        if np.linalg.det(R) < 0 :
            val[-1, :] *= -1
            R = val.T @ upper.T
        
        p = scene_centroid - R @ model_centroid
        alignedP = (R @ modelP.T).T + p
        ########## END STUDENT CODE ##########

        # RMSE
        rmse = np.sqrt(np.mean(np.sum((alignedP - sceneP)**2, axis=1)))

        return R, p, rmse

    def nearest_neighbor(self, model_points):
        """
        For each point in model_points, find the nearest point in self.sceneP.
        
        Return: idx (indices into self.sceneP) (len(model_points) x 3), d2 (squared distances) (len(model_points),)
        """

        ######### START STUDENT CODE #########
        difference = model_points[:, None, :] - self.sceneP[None, :, :]
        diffsq_matrix = np.sum(difference**2, axis=2)
        idx = np.argmin(diffsq_matrix, axis=1)
        d2 = diffsq_matrix[np.arange(model_points.shape[0]), idx]
        ########## END STUDENT CODE ##########

        return idx, d2

    def icp(self, R_init=None, p_init=None, max_iterations=50, tolerance=1e-6, trim_fraction=0.0):
        """
        Iterative Closest Point (point-to-point).
        - R_init, p_init: optional initial pose guess
        - max_iterations: stop after this number of iterations
        - tolerance: stop when change in RMSE < tolerance
        - trim_fraction: 0.0 -> use all pairs; e.g., 0.2 -> keep closest 80% (robust trimming)

        Returns:
            R (3x3), p (3,), history (list of rmse per iter), transformed_modelP (Nx3)
        """

        # Initialize
        self.R = np.eye(3) if R_init is None else R_init.copy()
        self.p = np.zeros(3) if p_init is None else p_init.copy()

        ######### START STUDENT CODE #########

        history = []
        modelP_trans = (self.R @ self.modelP.T).T + self.p
        for i in range (max_iterations):
            idx, d2 = self.nearest_neighbor(modelP_trans)
            scene_correl = self.sceneP[idx]

            if trim_fraction > 0.0:
                keep = int((1.0 - trim_fraction) * len(d2))
                keep_idx = np.argsort(d2)[:keep]
                scene_correl = scene_correl[keep_idx]
                model_correl = modelP_trans[keep_idx]
            else:
                model_correl = modelP_trans

            R_delta, p_delta, rmse = self.register(scene_correl, model_correl)

            # 4. Update global transform
            self.R = R_delta @ self.R
            self.p = R_delta @ self.p + p_delta

            # 5. Apply to all model points
            modelP_trans = (R_delta @ modelP_trans.T).T + p_delta
            history.append(rmse)
            if (i > 0 and abs(history[-2] - history[-1]) < tolerance):
                break
        
        self.alignedP = modelP_trans
        ########## END STUDENT CODE ##########

        return self.R, self.p, history, self.alignedP
