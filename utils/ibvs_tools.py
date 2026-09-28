import numpy as np
import os
import sys
from utils.lin_algeb import LinAlgeb
import math
import cv2

from typing import List
import torch
from utils.image_handling import ImageHandling
from utils.gaussians_handling import GaussiansHandling
from utils.poses_handling import PosesHandling
from utils.mesh_handling import MeshHandling
from utils.my_utils import MyUtils
from scipy.ndimage import distance_transform_edt


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(project_root, "accelerated_features"))
from modules.xfeat import XFeat





class IbvsTools : 


    def __init__(self, CAM_W, CAM_H, fx = None, fy = None, cx = None, cy = None): 
        self.CAM_W = CAM_W
        self.CAM_H = CAM_H
        
        if fx is None : 
            self.FX = self.FY = 0.8 * max(CAM_W, CAM_H)
        else :
            self.FX = fx
            self.FY = fy

        if cx is None :
            self.CX = CAM_W / 2.0
            self.CY = CAM_H / 2.0
        else :    
            self.CX = cx
            self.CY = cy
        self.xfeat = XFeat()

        self.dvs_dt = 0.5
        self.dvs_mu = 0.01
        self.dvs_lamda = 1


    @staticmethod
    def update_cam_pose(cam_homog, Vc, dt): 
    
        # Getting w and v vector (expresed in Cf) and the homog_mat in Wf : T
        T = np.array(cam_homog, dtype=float) 
        vx, vy, vz, wx, wy, wz = [float(x) for x in Vc] 
        w = np.array([wx, wy, wz], dtype=float) 
        v = np.array([vx, vy, vz], dtype=float) 
    
        R_cur = T[:3, :3]  # Current rotation matrix in Wf
        t_cur = T[:3, 3]   # Current translation vector in Wf
        
        # Step 1: Update translation
        # v is trans veloc in cam frame, v*dt give the trans vector made by v*dt (still in cam frame ofc)
        v_cam = v * dt
        # turn this v_cam to wf so we can update our homog matrix
        v_world = R_cur @ v_cam
        # Update translation in world frame
        t_new = t_cur + v_world
        
        # Step 2: Update rotation separately
        w_norm = np.linalg.norm(w) 
        I3 = np.eye(3) 
        
        if w_norm < 1e-12: 
            # Rotation is very small, use first-order approximation (tangent space)
            # R_delta ≈ I + skew(w*dt)
            w_skew = LinAlgeb.skew(w * dt)
            R_delta = I3 + w_skew
        else: 
            # Use exponential map for larger rotations
            theta = w_norm * dt 
            u = w / w_norm 
    
            # Skew(u) 
            ux = LinAlgeb.skew(u) 
            
            # Rodrigues rotation formula
            R_delta = I3 + math.sin(theta)*ux + (1.0 - math.cos(theta))*(ux @ ux) 
        
        # Apply rotation update (body frame, so right-multiply)
        R_new = R_cur @ R_delta
        
        # Build new transformation matrix
        new_T = np.eye(4)
        new_T[:3, :3] = R_new
        new_T[:3, 3] = t_new
        
        return new_T




    @staticmethod
    def get_feats_depth(features, depth_map) :
        #the depth map got the cam_img_H, img_W, and the z value
        #the features are list of u,v (pxls coord)
        features_depth = []
        for f  in features :
            
            if(depth_map[f[1],f[0]]>1000) :
                depth_map[f[1],f[0]] = 1000
                print("some features with z > 1000")

            if(depth_map[f[1],f[0]] < 0) :
                print("some features with z < 0")
            
            if(depth_map[f[1],f[0]] == 0) :
                depth_map[f[1],f[0]] = 0.5
            
            features_depth.append(depth_map[f[1],f[0]])

        return np.array(features_depth)



    # -------------- Get errors --------------------------------
    @staticmethod
    def getting_errors(Ss : List[List[float]], Ss_star : List[List[float]]) -> List[float]:      
        return [a - b for row1, row2 in zip(Ss, Ss_star) for a, b in zip(row1, row2)]



    # ------------- Getting L and pseudo L --------------------
    @staticmethod
    def get_interaction_matrix(nbr_features, Ss, Ss_z, f=1):

        Ls = np.zeros((nbr_features, 2, 6), dtype=float)

        for i, s in enumerate(Ss) :
            x=s[0]
            y=s[1]
            Z=Ss_z[i]

            # Build interaction matrix (2x6)
            L = np.zeros((2, 6))
            L[0, 0] = -f / Z
            L[0, 1] = 0.0
            L[0, 2] = x / Z
            L[0, 3] = x * y / f
            L[0, 4] = -(f + (x * x) / f)
            L[0, 5] = y
            L[1, 0] = 0.0
            L[1, 1] = -f / Z
            L[1, 2] = y / Z
            L[1, 3] = f + (y * y) / f
            L[1, 4] = -(x * y) / f
            L[1, 5] = -x

            Ls[i] = L

        return Ls


    @staticmethod
    def get_inter_mat_pseudo_inverse(intr_mat):

        # make sure the intr matrix got N,2,6 shape 
        intr_mat = np.asarray(intr_mat, dtype=float)
        if intr_mat.ndim != 3 or intr_mat.shape[1:] != (2, 6):
            raise ValueError(f"intr_mat must have shape (N,2,6), got {intr_mat.shape}")

        # Flatten to 2D (2N x 6), so we can calc pseudo inv
        N = intr_mat.shape[0]
        L_stack = intr_mat.reshape(2*N, 6)

        # Compute Moore-Penrose pseudo-inverse
        pseudo_intr_mat = np.linalg.pinv(L_stack)  # shape (6 x 2N)

        return pseudo_intr_mat




    def get_Ss_from_uv(self, uv_list):
        Ss = []
        for u, v in uv_list:
            x = (u - self.CX) / self.FX
            y = (v - self.CY) / self.FY
            Ss.append([x, y])
        return np.array(Ss)



    def get_xfeat_kpts(self, img) :
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)  # (1,3,H,W)
        out = self.xfeat.detectAndCompute(tensor, top_k=500)[0]
        kpts = out['keypoints']
        desc = out['descriptors']
        return  kpts, desc












    def compute_image_interaction_matrix(self, cur_depth_map, grad_Ix, grad_Iy, f=1):    

        # Create 2d of just u coords nd 2d of just v coords, both (H, W)
        u_coords, v_coords = np.meshgrid(np.arange(self.CAM_W), np.arange(self.CAM_H))  
        
        # get a 2d of x coords nd y coords in meters , both (H, W)
        x = (u_coords - self.CX) / self.FX
        y = (v_coords - self.CY) / self.FY
        
        # get a copy of depth with 0s as 100
        Z = cur_depth_map.copy()
        Z[Z == 0] = 100
        
        # Flatten everything for computation, from H,W to H*W
        x_flat = x.ravel() 
        y_flat = y.ravel() 
        Z_flat = Z.ravel() 
        grad_Ix_flat = grad_Ix.ravel() 
        grad_Iy_flat = grad_Iy.ravel() 


        # we will do − (Ix_Lx + Iy_Ly)
        # np.stack takes arrays of same shape nd stack nth element of each array in a single array then go to (n+1)th elemnt
        # this will give [[lx1], [lx2], [lx2].. ]

        Ix_Lx = grad_Ix_flat[:, None] * np.stack([
            -f / Z_flat,                   
            np.zeros_like(Z_flat),          
            x_flat / Z_flat,                
            x_flat * y_flat / f,            
            -(f + (x_flat * x_flat) / f),   
            y_flat                          
        ], axis=1)  # (H*W, 6)
        
        Iy_Ly = grad_Iy_flat[:, None] * np.stack([
            np.zeros_like(Z_flat),          
            -f / Z_flat,                    
            y_flat / Z_flat,                
            f + (y_flat * y_flat) / f,      
            -(x_flat * y_flat) / f,         
            -x_flat                         
        ], axis=1)  # (H*W, 6)


        Ls = -(Ix_Lx + Iy_Ly)  # (H*W, 6)    
        return Ls









    #(last_gaussians2, intrins_gs2, des_pose_gs2, des0_img, case_test_path, des0_mask, max_itrs=125)


    def start_dvs_loop(self, last_gaussians2, intrins_gs2, des_pose_gs2, des0_img, case_test_path, mask = None, max_itrs=500) :

        try:

            costs = []
            cur_pose = des_pose_gs2
            des_img = ImageHandling.img_255_to_01(des0_img)
            gray_des = ImageHandling.turn_img_to_gray(des_img)
            S_star = gray_des.flatten()

            for i in range(max_itrs):

                # 1 - Capture current img nd get S
                cur_img, cur_depth = GaussiansHandling.render_gs_pic(*last_gaussians2, cur_pose, intrins_gs2, self.CAM_W, self.CAM_H)

                if mask is not None :
                    cur_img = ImageHandling.apply_mask_on_img(cur_img, mask)

                cur_img = ImageHandling.img_255_to_01(cur_img)
                gray_cur = ImageHandling.turn_img_to_gray(cur_img)            
                S = gray_cur.flatten()

                # 2 - Compute the cost and the diff img for visua
                diff = S - S_star
                cost = diff.T @ diff
                costs.append(cost)
                print(f"Cost {i} :", cost)

                # 3- Checking oscillations
                if len(costs) >= 10:
                    last_10 = costs[-10:]
                    # 1. Oscillation check (if in last 10 iters we didn't improve with 5 pxls)
                    cost_range = max(last_10) - min(last_10)
                    stagnating = cost_range < 5
                    # 2. Divergence
                    first_half_mean = sum(last_10[:5]) / 5
                    second_half_mean = sum(last_10[5:]) / 5
                    diverging = second_half_mean > first_half_mean
                    if stagnating :
                        print("oscillations")
                        return cur_pose
                    if diverging:
                        print("divering")
                        return cur_pose
                    

                if (i % 5) == 0 :
                    current_diff_img = ImageHandling.compute_grayscale_difference(gray_cur, gray_des)
                    ImageHandling.save_img(cur_img, f"img_{i}", f"{case_test_path}/dvs_infos/DVS_frames")
                    ImageHandling.save_img(current_diff_img, f"diff_{i}", f"{case_test_path}/dvs_infos/DVS_diffs")

                # 3 - Compute Gradient and Ls 
                grad_Ix, grad_Iy = ImageHandling.get_grads_visp(gray_cur, self.FX, self.FY)
                Ls = self.compute_image_interaction_matrix(cur_depth, grad_Ix, grad_Iy)
                        
                # 4 - Compute V with GN or LM 
                V = -self.dvs_lamda * np.linalg.solve(Ls.T @ Ls + self.dvs_mu * np.diag(np.diag(Ls.T @ Ls)), Ls.T @ diff)      
                #V = -lamda * np.linalg.pinv(Ls) @ diff
    
                # 5 - Update camera pose & update matplotlib vis data
                cur_pose = IbvsTools.update_cam_pose(cur_pose, V, self.dvs_dt)

            return cur_pose
    

        except KeyboardInterrupt:
            print("\nCtrl+C detected, exiting loop cleanly.")
            os._exit(0)
            return cur_pose








    def border_safe_indices(self, kpts_img1, kpts_img2, mask1, mask2, margin=10):
        """
        Return the indices of matches whose keypoints are safely inside
        the valid regions of BOTH images.

        A match is kept only if both corresponding keypoints:
        - are inside their respective image/mask bounds
        - are at least `margin` pixels away from an invalid mask region

        Returns
        -------
        safe : np.ndarray, shape (N,), dtype=bool
            Boolean mask indicating the matches to keep.
        """

        if mask1 is None :
            mask1 = MyUtils.get_dummy_mask(self.CAM_W, self.CAM_H)

        if mask2 is None :
            mask2 = MyUtils.get_dummy_mask(self.CAM_W, self.CAM_H)

        kpts1 = np.asarray(kpts_img1).astype(np.int32)
        kpts2 = np.asarray(kpts_img2).astype(np.int32)
        mask1 = np.asarray(mask1, dtype=bool)
        mask2 = np.asarray(mask2, dtype=bool)

        # Distance from each valid pixel to the nearest invalid region
        dist1 = distance_transform_edt(mask1)
        dist2 = distance_transform_edt(mask2)

        x1, y1 = kpts1[:, 0], kpts1[:, 1]
        x2, y2 = kpts2[:, 0], kpts2[:, 1]

        h1, w1 = mask1.shape
        h2, w2 = mask2.shape

        # Check that keypoints are inside their corresponding masks
        valid1 = (x1 >= 0) & (x1 < w1) & (y1 >= 0) & (y1 < h1)
        valid2 = (x2 >= 0) & (x2 < w2) & (y2 >= 0) & (y2 < h2)

        # we init considere all kpts invalid
        safe1 = np.zeros(len(kpts1), dtype=bool)
        safe2 = np.zeros(len(kpts2), dtype=bool)

        # then turn the safe ones to true (ones far 10pxls frm borders(masks))
        safe1[valid1] = dist1[y1[valid1], x1[valid1]] >= margin
        safe2[valid2] = dist2[y2[valid2], x2[valid2]] >= margin

        # A match is valid only when both sides are safe
        return safe1 & safe2




    @staticmethod
    def select_distributed_matches(
        matches_cur,
        matches_des,
        img_width,
        img_height,
        top_x):

        """
        Select spatially distributed matches based on desired-image points.

        matches_cur, matches_des: NumPy arrays with shape (N, 2),
        containing coordinates in (u, v) format.
        """
        matches_cur = np.asarray(matches_cur)
        matches_des = np.asarray(matches_des)

        top_x = min(top_x, len(matches_des))

        if top_x == 0:
            return matches_cur[:0], matches_des[:0]

        # Normalize coordinates by the image dimensions.
        scale = np.array([img_width, img_height], dtype=np.float64)
        points = matches_des.astype(np.float64) / scale

        # Start with the point closest to the image centre.
        center = np.array([0.5, 0.5], dtype=np.float64)
        first_idx = np.argmin(np.linalg.norm(points - center, axis=1))

        selected_indices = [first_idx]

        min_distances = np.linalg.norm(
            points - points[first_idx],
            axis=1)

        min_distances[first_idx] = -1

        # Select the point farthest from all previously selected points.
        for _ in range(1, top_x):
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)
            new_distances = np.linalg.norm(points - points[next_idx], axis=1)
            min_distances = np.minimum(min_distances, new_distances)
            min_distances[selected_indices] = -1

        selected_indices = np.asarray(
            selected_indices,
            dtype=np.int64)

        return (matches_cur[selected_indices], matches_des[selected_indices])










    




    @staticmethod
    def filter_matches_ransac(
        matches_cur,
        matches_des,
        ransac_threshold=2.0,
        confidence=0.999
    ):
        """
        Remove geometrically inconsistent matches using fundamental-matrix RANSAC.

        Parameters
        ----------
        matches_cur : np.ndarray, shape (N, 2)
            Matched (u, v) coordinates in the current image.

        matches_des : np.ndarray, shape (N, 2)
            Corresponding coordinates in the desired image.

        ransac_threshold : float
            Maximum epipolar distance in pixels for an inlier.

        confidence : float
            Confidence used by RANSAC.

        Returns
        -------
        inlier_cur : np.ndarray, shape (M, 2)
        inlier_des : np.ndarray, shape (M, 2)
        """

        matches_cur = np.asarray(matches_cur, dtype=np.float32)
        matches_des = np.asarray(matches_des, dtype=np.float32)

        if len(matches_cur) < 8:
            print(f"RANSAC skipped: only {len(matches_cur)} matches.")
            return matches_cur, matches_des

        F, inlier_mask = cv2.findFundamentalMat(
            matches_cur,
            matches_des,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=ransac_threshold,
            confidence=confidence
        )

        if F is None or inlier_mask is None:
            print("RANSAC failed to estimate the fundamental matrix.")
            return matches_cur[:0], matches_des[:0]

        inlier_mask = inlier_mask.ravel().astype(bool)

        inlier_cur = matches_cur[inlier_mask]
        inlier_des = matches_des[inlier_mask]

        print(
            f"RANSAC: {len(inlier_cur)}/{len(matches_cur)} "
            f"matches retained "
            f"({100 * len(inlier_cur) / len(matches_cur):.1f}%)."
        )

        return inlier_cur, inlier_des






    def filtered_matching(self, img1, img2, max_matches, mask1 = None, mask2 = None):
        """
        input :
        -imgs nd mask: 2d npy
        -the mask of the img with holes (in all our matching for now only 1 img got holes)
        -holes_on2, means the second img got holes instead of the first (so we take the mask2 & matches2 into consid for border's safe indices)
        
        -->  border filter with mask, outlier filter ransac, distributed selction
        
        return :
        mtches_img1, mtches_img2 (np (list of corresp 2d coords : ints))
        """

        img1_kpts, img1_desc = self.get_xfeat_kpts(img1)
        img2_kpts, img2_desc = self.get_xfeat_kpts(img2)

        idxs0, idxs1 = self.xfeat.match(img2_desc, img1_desc)
        matches_img2 = img2_kpts[idxs0].detach().cpu().numpy()
        matches_img1 = img1_kpts[idxs1].detach().cpu().numpy()

        if mask1 is None :
            mask1 = MyUtils.get_dummy_mask(self.CAM_W, self.CAM_H)

        if mask2 is None :
            mask2 = MyUtils.get_dummy_mask(self.CAM_W, self.CAM_H)

        safe_indices = self.border_safe_indices(matches_img1, matches_img2, mask1, mask2)
        
        matches_img2 = matches_img2[safe_indices]
        matches_img1 = matches_img1[safe_indices]

        matches_img2, matches_img1 = self.filter_matches_ransac(matches_img2, matches_img1, ransac_threshold=2.0, confidence=0.999)
        
        matches_img2, matches_img1 = IbvsTools.select_distributed_matches(matches_img2, matches_img1, self.CAM_W, self.CAM_H, top_x=max_matches)

        matches_img2 = matches_img2.astype(np.int32)
        matches_img1 = matches_img1.astype(np.int32)

        return matches_img1, matches_img2