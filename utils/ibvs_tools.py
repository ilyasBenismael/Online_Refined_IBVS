import numpy as np
import os
import sys
import numpy as np
from utils.lin_algeb import LinAlgeb
import math
from typing import List
import torch


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(project_root, "accelerated_features"))
from modules.xfeat import XFeat





class IbvsTools : 


    def __init__(self, CAM_W, CAM_H): 
        self.CAM_W = CAM_W
        self.CAM_H = CAM_H
        self.CX = CAM_W / 2.0
        self.CY = CAM_H / 2.0
        self.FX = self.FY = 0.8 * max(CAM_W, CAM_H)
        self.xfeat = XFeat()


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
    def get_interaction_matrix(nbr_features, Ss, Ss_z, f):

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



