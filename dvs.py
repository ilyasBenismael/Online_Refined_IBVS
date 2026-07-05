
import os
import open3d as o3d
import numpy as np
from utils.lin_algeb import LinAlgeb
import matplotlib.pyplot as plt
import math
from utils.image_handling import ImageHandling
from utils.gaussians_handling import GaussiansHandling
import torch
from utils.poses_handling import PosesHandling
from utils.my_utils import MyUtils
import pycolmap
from moge.model.v2 import MoGeModel
import time, path





"""
from pathlib import Path
import shutil

input_dir = Path("/home/user/Bureau/visual_navigation/IBVS_CODE/datasets/360_v2/room/images_2")
output_dir = Path("/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test/room2/sfm/images")
output_dir.mkdir(exist_ok=True)

images = sorted(
    [p for p in input_dir.iterdir()
     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
)

# Keep 2 out of every 5 images
for i, img_path in enumerate(images):
    if i % 5 in (0, 1):
        shutil.copy2(img_path, output_dir / img_path.name)

print(f"Copied {len(list(output_dir.iterdir()))} images")


"""

sfm_path = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test/tree/sfm"
inria_output = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test/garden/inria"
     
PosesHandling.apply_sfm_reconstruction(sfm_path, False)
#GaussiansHandling.run_gs_training(sfm_path=sfm_path, output_path= inria_output, gs_reso = 4)



"""

import os
import open3d as o3d
import numpy as np
from utils.lin_algeb import LinAlgeb
import matplotlib.pyplot as plt
import math
from utils.image_handling import ImageHandling
from utils.gaussians_handling import GaussiansHandling
import torch
from utils.poses_handling import PosesHandling
from utils.my_utils import MyUtils
import pycolmap
from moge.model.v2 import MoGeModel
import time






# Paths___________________________________________________________

# Main_tests path for shortcuts
scene_name = "thehouse"
case_nbr = 2
main_test_path = "my_results/online_ibvs_test"
case_test_path = f"{main_test_path}/{scene_name}/{scene_name}_case{case_nbr}"


# GS1 paths
gs1_ply_path = f"{main_test_path}/{scene_name}/real_scene/{scene_name}.ply"

# All States to be saved, paths
des_imgs_path = f"{case_test_path}/desired_imgs"
real_frames_path = f"{case_test_path}/ibvs_frames/real_frames"  



# Robot camera______________________________________________________
CAM_W, CAM_H = 1332, 876
FX = FY = 0.8 * max(CAM_W, CAM_H)
f=1
CX, CY = CAM_W / 2.0, CAM_H / 2.0



intrins_gs1 = torch.tensor(
    [[FX, 0.0, CX],
        [0.0, FY, CY],
        [0.0, 0.0, 1.0],],
    dtype=torch.float32,
    device="cuda",
).unsqueeze(0)


# DVS Vars_________________________________________________________
dt = 0.5
mu = 0.01
lamda = 1










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











def compute_image_interaction_matrix(cur_depth_map, grad_Ix, grad_Iy):    

    # Create 2d of just u coords nd 2d of just v coords, both (H, W)
    u_coords, v_coords = np.meshgrid(np.arange(CAM_W), np.arange(CAM_H))  
    
    # get a 2d of x coords nd y coords in meters , both (H, W)
    x = (u_coords - CX) / FX
    y = (v_coords - CY) / FY
    
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















def start_dvs_loop(gaussians, init_gs1_pose, des_img) :

    try:
        for i in range(999):

            if (i==0) :
                cur_pose = init_gs1_pose
                des_img = ImageHandling.img_255_to_01(des_img)
                gray_des = ImageHandling.turn_img_to_gray(des_img)
                S_star = gray_des.flatten()

            # 1 - Capture current img nd get S
            cur_img, cur_depth = GaussiansHandling.render_gs_pic(*gaussians, cur_pose, intrins_gs1, CAM_W, CAM_H)
            cur_img = ImageHandling.img_255_to_01(cur_img)
            gray_cur = ImageHandling.turn_img_to_gray(cur_img)            
            S = gray_cur.flatten()

            # 2 - Compute the cost and the diff img for visua
            diff = S - S_star
            cost = diff.T @ diff
            print(f"Cost {i} :", cost)

            if (i % 20) == 0 :
                current_diff_img = ImageHandling.compute_grayscale_difference(gray_cur, gray_des)
                ImageHandling.save_img(cur_img, f"img_{i}", f"{real_frames_path}/DVS")
                ImageHandling.save_img(current_diff_img, f"diff_{i}", f"{real_frames_path}/DVS")

            # 3 - Compute Gradient and Ls 
            grad_Ix, grad_Iy = ImageHandling.get_grads_visp(gray_cur, FX, FY)
            Ls = compute_image_interaction_matrix(cur_depth, grad_Ix, grad_Iy)
                    
            # 4 - Compute V with GN or LM 
            V = -lamda * np.linalg.solve(Ls.T @ Ls + mu * np.diag(np.diag(Ls.T @ Ls)), Ls.T @ diff)      
            #V = -lamda * np.linalg.pinv(Ls) @ diff
   
            # 5 - Update camera pose & update matplotlib vis data
            cur_pose = update_cam_pose(cur_pose, V, dt)
 

    except KeyboardInterrupt:
        print("\nCtrl+C detected, exiting loop cleanly.")
        os._exit(0)













def main() :

    init_pose = np.array([
    [0.08970551, -0.42680273,  0.89988463,  1.46558079],
    [0.99596655,  0.03672936, -0.08186327, -0.7019205 ],
    [ 0.00188728,  0.90359857,  0.42837607,  0.16063543],
    [ 0.     ,     0.    ,      0.    ,      1.        ]])
    

    gaussians = GaussiansHandling.load_gaussians_from_ply(gs1_ply_path)                                                                                     
    des_img = ImageHandling.load_np_img(f"{des_imgs_path}/des101.png")
    start_dvs_loop(gaussians, init_pose, des_img)





if __name__ == "__main__":

    try:
        main()
    finally:
        MyUtils.cleanup()





        



"""

