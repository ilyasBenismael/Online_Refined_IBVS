import numpy as np
import os
import sys
from utils.lin_algeb import LinAlgeb
import math
from typing import List
import torch
from utils.image_handling import ImageHandling
from utils.gaussians_handling import GaussiansHandling
from utils.poses_handling import PosesHandling
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












    def start_dvs_loop(self, gaussians, intrins_gs1, init_pose, des_img, frames_saving_dir, mask = None, max_itrs=5550, detect_oscill = True) :

        try:
            costs = []
            for i in range(max_itrs):

                if (i==0) :
                    cur_pose = init_pose
                    des_img = ImageHandling.img_255_to_01(des_img)
                    gray_des = ImageHandling.turn_img_to_gray(des_img)
                    S_star = gray_des.flatten()

                # 1 - Capture current img nd get S
                cur_img, cur_depth = GaussiansHandling.render_gs_pic(*gaussians, cur_pose, intrins_gs1, self.CAM_W, self.CAM_H)

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
                    ImageHandling.save_img(cur_img, f"img_{i}", f"{frames_saving_dir}/dvs_infos/DVS_frames")
                    ImageHandling.save_img(current_diff_img, f"diff_{i}", f"{frames_saving_dir}/dvs_infos/DVS_diffs")

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




    @staticmethod
    def border_safe_indices(kpts_xy, mask, margin=10):
        """
        kpts_xy: (N, 2) array or tensor of (x, y) integer pixel coords
        mask: 2D bool array, True = valid/inside, False = empty/border
        margin: min distance (in px) required from the mask border
        """
        if isinstance(kpts_xy, torch.Tensor):
            kpts_xy = kpts_xy.detach().cpu().numpy()
        kpts_xy = np.asarray(kpts_xy).astype(int)

        dist = distance_transform_edt(mask)  # distance to nearest False pixel

        xs = kpts_xy[:, 0]
        ys = kpts_xy[:, 1]
        h, w = mask.shape
        in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)

        safe = np.zeros(len(kpts_xy), dtype=bool)
        safe[in_bounds] = dist[ys[in_bounds], xs[in_bounds]] >= margin

        return safe