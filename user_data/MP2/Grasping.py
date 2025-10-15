import numpy as np

class Grasping:
    def __init__(self, scene_points):
        """
        Initialize the scene point cloud.

        Args:
            scene_points (ndarray): Array of shape (N, 3), where N is the number of points,
                                    and each row is a 3D point [x, y, z].
        """
        self.scene_points = np.asarray(scene_points, dtype=np.float32)
        self.N = self.scene_points.shape[0]

    def filter_outliers(self, k=20, std_ratio=0.5):  
        """
        Strict statistical outlier removal:
        - Compute mean distance of each point to its k nearest neighbors.
        - Keep only points within [mu - std_ratio*sigma, mu + std_ratio*sigma].
        (Stricter than just an upper cutoff.)
        - Updates self.scene_points with filtered cloud.

        Args:
            k (int): number of neighbors.
            std_ratio (float): tighter threshold in std deviations.

        Returns:
            filtered_points (ndarray): cleaned point cloud.
            mask (ndarray): boolean mask of inliers.
        """


        # nearest_twenty = self.nearest_neighbor(num_local_point=k)
        # # meandist = []
        # # for i, row in enumerate(nearest_twenty):
        # #     dists = np.array([np.linalg.norm(point - self.scene_points[i]) for point in row])
        # #     meandist.append(np.mean(dists))

        # mean_dists =np.array([np.mean(np.array([np.linalg.norm(point - self.scene_points[i]) for point in row])) for i,row in enumerate(nearest_twenty)])
        # mu = np.mean(mean_dists)
        # sigma = np.std(mean_dists)
        # variation = (std_ratio*sigma)
        # mask = mean_dists > (mu - variation) and mean_dists < (mu + variation)
        # filtered_points = mean_dists[mask]
        # self.scene_points = filtered_points

        # return filtered_points, mask    

        near_twenty_idx = self.nearest_neighbor(num_local_point=k)
        nearest_twenty = self.scene_points[near_twenty_idx, :]
        mean_dists = np.mean(np.linalg.norm(nearest_twenty - self.scene_points[:, None, :], axis=-1), axis=1)
        
        mu = np.mean(mean_dists)
        sigma = np.std(mean_dists)
        variation = (std_ratio*sigma)

        mask = (mean_dists > (mu - variation)) & (mean_dists < (mu + variation))
        filtered_points = self.scene_points[mask]
        self.scene_points = filtered_points
        self.N = self.scene_points.shape[0]

        return filtered_points    
    
    def nearest_neighbor(self, num_local_point=10):
        """
        For each point in self.scene_points, find its nearest neighbors.

        Args:
            num_local_point (int): Number of nearest neighbors to retrieve (including the point itself).

        Returns:
            neighbors_idx (ndarray): Array of shape (N, num_local_point),
                                     where neighbors_idx[i, :] are the indices
                                     of the nearest neighbors of point i.
        """
        # Compute pairwise squared distances (N x N matrix)
        diff = self.scene_points[:, None, :] - self.scene_points[None, :, :]  # shape (N, N, 3)
        dist_sq = np.sum(diff**2, axis=-1)  # shape (N, N)

        # Get indices of the nearest neighbors (sorted by distance)
        neighbors_idx = np.argsort(dist_sq, axis=1)[:, :num_local_point]

        return neighbors_idx
    
    def estimate_normals(self, neighbors_idx):
        """
        Estimate normals and curvature for each point in the point cloud using PCA.

        Args:
            neighbors_idx (ndarray): Array of shape (N, K), neighbor indices for each point.

        Returns:
            normals (ndarray): Array of shape (N, 3), estimated normals (unit length).
        """

        # neighbors = [self.scene_points[neighbors_idx[i]] for i in range(self.N)]
        # mu = np.mean(neighbors, axis=1)
        # # sig = np.std(neighbors, axis=0)
        # neighbors_normed = (neighbors - mu)

        # covariance = [np.cov(neighbors_normed[i]) for i in range(self.N)]
        # eigen_values, eigen_vectors = np.split([np.linalg.eig(covariance[i]) for i in range(self.N)], 2, axis=10

        # norm_vectors = eigen_vectors[np.argmin(eigen_values, axis=0)]
        # norm_vectors = norm_vectors / np.linalg.norm(norm_vectors, axis=0)
        idx = np.where(neighbors_idx == 512)
        normals = np.zeros((self.N, 3))

        # normals = []

        for i in range(self.N):
            neighbors = self.scene_points[neighbors_idx[i]]
            mu = np.mean(neighbors, axis=0)
            eigen_values, eigen_vectors = np.linalg.eig(np.cov((neighbors - mu).T))
            # eigen_vectors = eigen_vectors[np.argmin(eigen_values)]
            # norm = eigen_vectors[:, 0]
            norm_vector = eigen_vectors[:, np.argmin(eigen_values)]
            norm_vector = norm_vector / np.linalg.norm(norm_vector)
            if (norm_vector[2] < 0):
                norm_vector = -norm_vector
            
            normals[i] = norm_vector
            # normals.append(norm_vector) 
        
        return np.array(normals)


    
    def evaluate_grasp_score(self, normals, pairs, k=10):
        """
        Score antipodal pairs by how colinear local normals are with the grasp axis.
        Ignores direction (flipped normals are fine).

        Args:
            normals : (N,3) unit normals
            pairs   : list of (i,j) indices for antipodal contacts
            k       : KNN size around each contact (brute force search)

        Returns:
            scores  : (len(pairs),) array in [0,1], higher = more colinear
        """

        # print(pairs)
        score = np.zeros((len(pairs)))
        for idx, (i,j) in enumerate(pairs):
            xis= normals[i] - normals[j]
            xis = xis / np.linalg.norm(xis)
            sc = (abs(np.dot(normals[i], xis)) + abs(np.dot(normals[j], xis))) / 2.0
            score[idx] = sc
        return score

    def find_antipodal_pairs(self, normals, gripper_width=0.07, angle_thresh=np.pi/12):
        """
        Find antipodal point pairs by checking both +n and -n directions
        to handle flipped normals.

        Args:
            normals (N,3): estimated normals (unit vectors)
            gripper_width (float): max distance between fingers
            angle_thresh (float): tolerance angle (radians) for antipodality

        Returns:
            list of (i, j) index pairs
        """
        # Realized I was trying this with a vectorized solution and uhhhh -> its hard I need like vectorized condittions and get the triangular matrix
        # and other werid indexing so I'm going to start with the loops and not worry about runtime paralellization just yet


        # diff_points = self.scene_points[:, None, :] - self.scene_points[None, :, :]

        # distance_pts = np.linalg.norm(diff_points, axis= -1)
        # axis = diff_points / distance_pts
        # mask = distance_pts < gripper_width & distance_pts > 0
        # axis = axis[mask]
        # # distance_pts = distance_pts[mask]
        # norms_masked = normals[mask]

        # print("find_antipodal_pairs called")
        pairs = []
        for i in range(self.N):
            for j in range(i, self.N):
                diff = self.scene_points[i] - self.scene_points[j]
                dist = np.linalg.norm(diff)
                if (dist >= gripper_width or dist < 1e-8):
                    continue
                axis = diff / dist
                if (abs(np.dot(normals[i], axis)) > np.cos(angle_thresh) and abs(np.dot(normals[j], axis)) > np.cos(angle_thresh)):
                    pairs.append((i,j))

        # print(len(pairs))
        return pairs


    
    def compute_grasp_poses(self, pairs):
        """
        Compute grasp poses for given antipodal pairs, assuming vertical grasp constraint.

        Args:
            pairs          : list of (i, j) indices for antipodal contacts

        Returns:
            grasp_poses : list of dicts with:
                - "pair": (i, j)
                - "R"   : (3,3) rotation matrix
                - "t"   : (3,) translation vector
        """
        # init_pos = np.array([0.2, 0.2, 0.2])
        grasp_poses = []

        for i,j in pairs:

            translation = (self.scene_points[i] + self.scene_points[j]) /2.0
            translation[2] += 0.12
            # translation = translation - init_pos

            z_axis = np.array([0, 0, -1])

            x_axis = (self.scene_points[i] - self.scene_points[j])
            x_axis[2] = 0
            x_axis = x_axis / np.linalg.norm(x_axis)

            y_axis = np.cross(x_axis, z_axis)
            y_axis[1] *= -1
            y_axis = y_axis / np.linalg.norm(y_axis)


            rotation = np.vstack([x_axis, y_axis, z_axis])
            print("rotate: ", rotation)
            print("t: ", translation)

            pose = {"pair" : (i,j), "R" : rotation, "t" : translation}
            grasp_poses.append(pose)
        
        return grasp_poses
