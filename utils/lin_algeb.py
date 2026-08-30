import math
from typing import Tuple, Sequence, List
import numpy as np




class LinAlgeb :

    # --------- Basic axis rotations (rotation around own axis) ----------
    
    @staticmethod
    def Rx(a: float) -> List[List[float]]:
        ca, sa = math.cos(a), math.sin(a)
        return [[1, 0, 0],
                [0, ca, -sa],
                [0, sa,  ca]]
    
    @staticmethod
    def Ry(a: float) -> List[List[float]]:
        ca, sa = math.cos(a), math.sin(a)
        return [[ ca, 0, sa],
                [  0, 1,  0],
                [-sa, 0, ca]]
    
    @staticmethod
    def Rz(a: float) -> List[List[float]]:
        ca, sa = math.cos(a), math.sin(a)
        return [[ca, -sa, 0],
                [sa,  ca, 0],
                [ 0,   0, 1]]



    
    # --------- Basic Matrix functions ----------

    @staticmethod
    def matmul3(A, B):
        # 3x3 matrix * 3x3 matrix
        return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    
    @staticmethod
    def matvec3(R, v):
        # 3x3 matrix * 3x1 vector
        return [sum(R[i][k]*v[k] for k in range(3)) for i in range(3)]

    @staticmethod
    def vec_add(a, b):
        # 3D vect + 3D vect
        return [a[i] + b[i] for i in range(3)]
    
    @staticmethod
    def vec_sub(a, b):
        # 3D vect - 3D vect
        return [a[i] - b[i] for i in range(3)]
    
    @staticmethod
    def RT(R):
        # Transpose of 3x3 matrix
        return [[R[j][i] for j in range(3)] for i in range(3)]

    @staticmethod
    def skew(w):
            return np.array([
                [0,     -w[2],  w[1]],
                [w[2],   0,    -w[0]],
                [-w[1],  w[0],  0]
            ])




    # ---------- Getting R and t from 6 coords ---------------

    def make_rot_trans(tx: float, ty: float, tz: float,
                    rx: float, ry: float, rz: float) -> Tuple[List[List[float]], List[float]]:

        Rx_m = LinAlgeb.Rx(rx)
        Ry_m = LinAlgeb.Ry(ry)
        Rz_m = LinAlgeb.Rz(rz)

        # R = Rz * Ry * Rx multiplying them give us how the rotation matrix expressing how the camera is rotated around its state before rotating
        R = LinAlgeb.matmul3(Rz_m, LinAlgeb.matmul3(Ry_m, Rx_m))
        t = [tx, ty, tz]
        return R, t


    # ------------- Getting homog matrix from R and t --------------

    @staticmethod
    def get_homog_frm_rt(R: Sequence[Sequence[float]], t: Sequence[float]) -> np.ndarray:
        """Build 4x4 homogeneous transformation matrix from R and t."""
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    



    @staticmethod
    def get_Rt_from_homog_matrix(T: np.ndarray):
        R = T[:3, :3]
        t = T[:3, 3]
        return R, t


    @staticmethod
    def get_homog_frm_vect(pose_vector) :
        R,t = LinAlgeb.make_rot_trans(*pose_vector)
        return LinAlgeb.get_homog_frm_rt(R,t)



    
    @staticmethod
    def pose_distance(T1, T2):
        """
        Compute Euclidean distance between two vectors
        """
        return np.linalg.norm(T2 - T1)
    
    
    @staticmethod
    def normalize_t(t_diff: float, t_norm: float) -> float:
        """Normalize a translation dist by another one"""
        if t_norm == 0 or np.isnan(t_norm):
            return float("nan")
        return t_diff / t_norm

    @staticmethod
    def rotation_diff(R_gt: np.ndarray, R_other: np.ndarray) -> float:
        """Angle (in degrees) between two rotation matrices."""
        R_rel = R_gt.T @ R_other
        cos_angle = (np.trace(R_rel) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        return float(np.degrees(angle_rad))



    @staticmethod
    def inverse_mat(pose_mat) :
        return np.linalg.inv(pose_mat)
    


    # ------------- Direct transform frame of a point using homog matrix -------------------

    @staticmethod
    def transform_points_to_world(p_in_cam,
                        cam_homog : np.ndarray) -> List[float]:
        
        # Convert to numpy
        p_in_cam = np.asarray(p_in_cam, dtype=np.float64)

        # flatten to (H*W,3) if H,W,3)
        p_in_cam = p_in_cam.reshape(-1, 3)

        # (Camera -> World)
        # p_world = cur_cam_rot * p_cam + cur_cam_trans
        cam_rot = cam_homog[:3, :3]
        cam_trans = cam_homog[:3, 3]

        points_c = []
        for p in p_in_cam:
            Rp = LinAlgeb.matvec3(cam_rot, p)
            points_c.append(LinAlgeb.vec_add(Rp, cam_trans))
        return np.array(points_c) 


    # ------------ Inverse transform frame of a point using homog matrix --------------------  
     
    @staticmethod 
    def transform_points_to_cam(p_in_world,
                        cam_homog : np.ndarray) -> List[float]:
        
        # Convert to numpy
        p_in_world = np.asarray(p_in_world, dtype=np.float64)

        # flatten to (H*W,3) if H,W,3)
        p_in_world = p_in_world.reshape(-1, 3)

        # (in World -> in Camera)
        # p_cam = cam_rot^T * (p_world - cam_trans)
        cam_rot = cam_homog[:3, :3]
        cam_trans = cam_homog[:3, 3]
        points_c = []
        for p in p_in_world:
            diff = LinAlgeb.vec_sub(p, cam_trans)
            points_c.append(LinAlgeb.matvec3(LinAlgeb.RT(cam_rot), diff))
        return np.array(points_c)    




    @staticmethod
    def get_mat_condition_number(A):

        A = np.asarray(A)

        # ---- Fix IBVS stacked format (n,2,6) -> (2n,6)
        if A.ndim == 3:
            if A.shape[1] == 2:
                A = A.reshape(-1, A.shape[-1])
            else:
                raise ValueError("Unsupported 3D matrix shape")

        # ---- Ensure 2D matrix
        if A.ndim != 2:
            raise ValueError("Input must be a 2D matrix")

        # ---- Compute condition number
        cond_number = np.linalg.cond(A)

        # ---- Interpret result
        if np.isinf(cond_number):
            meaning = "Singular (rank deficient)"

        elif cond_number < 10:
            meaning = "Excellent conditioning"

        elif cond_number < 100:
            meaning = "Good conditioning"

        elif cond_number < 1e3:
            meaning = "Acceptable"

        elif cond_number < 1e5:
            meaning = "Unstable"

        elif cond_number < 1e7:
            meaning = "Very unstable"

        elif cond_number < 1e8:
            meaning = "Severely ill-conditioned"

        else:
            meaning = "Almost linearly dependent"

        #print(f"{meaning}, {cond_number}")
        return cond_number


    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def inverse_sigmoid(y):
        # y must be in (0, 1)
        return np.log(y / (1 - y))


        

    @staticmethod
    def align_clouds_with_trans(cloud1: np.ndarray, cloud2: np.ndarray) -> np.ndarray:
        """
        Compute the translation t such that (cloud2 + t) is closest to cloud1
        in a least-squares sense.
    
        Parameters
        ----------
        cloud1 : (N, 3) array of target points (xyz)
        cloud2 : (N, 3) array of source points (xyz), same correspondence order as cloud1
    
        Returns
        -------
        t : (3,) translation vector to ADD to cloud2 to align it with cloud1
        """
        cloud1 = np.asarray(cloud1, dtype=np.float64)
        cloud2 = np.asarray(cloud2, dtype=np.float64)
    
        if cloud1.shape != cloud2.shape:
            raise ValueError(f"Shape mismatch: {cloud1.shape} vs {cloud2.shape}")
        if cloud1.ndim != 2 or cloud1.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) arrays, got {cloud1.shape}")
    
        centroid1 = cloud1.mean(axis=0)
        centroid2 = cloud2.mean(axis=0)
    
        t = centroid1 - centroid2
        return t


    @staticmethod
    def transform_pose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        """
        Apply homogeneous transformation T2 to pose T1.

        Parameters
        ----------
        T1 : (4,4) np.ndarray
            Original pose.
        T2 : (4,4) np.ndarray
            Transformation to apply.

        Returns
        -------
        (4,4) np.ndarray
            Transformed pose: T2 @ T1.
        """
        return T2 @ T1
        

