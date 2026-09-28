import sys
sys.path.append("/home/user/Bureau/visual_navigation/IBVS_CODE/scripts/")
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List
import multiprocessing as mp
import traceback 

import cv2
import emoji
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pycolmap
import torch
from moge.model.v2 import MoGeModel
from plyfile import PlyData
from skimage.metrics import structural_similarity as ssim
from wcwidth import wcswidth

from utils.gaussians_handling import GaussiansHandling
from utils.ibvs_tools import IbvsTools
from utils.image_handling import ImageHandling
from utils.lin_algeb import LinAlgeb
from utils.main_visualizer import MainVisualizer
from utils.scale_optimizer import ScaleOptimizer
from utils.mesh_handling import MeshHandling
from utils.my_utils import MyUtils
from utils.poses_handling import PosesHandling
# Add accelerated_features to our Python paths , so that when featx script gets executed it xill know where to find the modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(project_root, "accelerated_features"))
from modules.xfeat import XFeat 
  


























def main() :

    #__________
    #loop on the moge imgs 
    #_______

    #sfm_path = "/home/user/Bureau/ilyas/3d_art/essaouira_room/sfm"
    #GaussiansHandling.run_gs_training(sfm_path=sfm_path, output_path= f"/home/user/Bureau/visual_navigation/", gs_reso = 2, gs_nbr_itrs = 15000)
    
    

    # Load sfm infos
    sfm_path = "/home/user/Bureau/ilyas/3d_art/essaouira_room/sfm"
    recons = PosesHandling.get_recons(sfm_path)
    K = PosesHandling.get_cam_matrix(recons)
    sfm_points = PosesHandling.get_sparse_points(recons) # list of xyz's
    fx_gs1 = K[0, 0]
    fy_gs1 = K[1, 1]
    cx_gs1 = K[0, 2]
    cy_gs1 = K[1, 2]


    # Cam and rendering infos
    an_img = ImageHandling.load_np_img(f"{sfm_path}/images/1.jpg")
    CAM_H, CAM_W = an_img.shape[:2]
    mesh_handling = MeshHandling(CAM_W, CAM_H, fx=fx_gs1, fy=fy_gs1)

    # Turn f to moge format (fov in deg)
    fov_x_rad = 2 * np.arctan(CAM_W / (2 * fx_gs1))
    fov_x_deg = np.rad2deg(fov_x_rad)
        
    # Load Gs
    gaussians = GaussiansHandling.load_gaussians_from_ply("/home/user/Bureau/ilyas/3d_art/essaouira_room/moge_gaussians.ply")    
    K_gs = GaussiansHandling.turn_cam_matrix_to_gsplat_format(K)

    for i in range(1, 63):
        img_pose = PosesHandling.get_img_sfm_pose(recons, f"{i}.jpg")
        gs_redner, _ = GaussiansHandling.render_gs_pic(*gaussians, img_pose, K_gs, CAM_W, CAM_H)
        ImageHandling.save_img(gs_redner, f"{i}a.jpg", f"{sfm_path}/images")

    
    











    #----------------------

    accumulated_moge_points = np.array([[0,0,0]])
    accumulated_moge_colors = np.array([[0,0,0]])


    # Loop on moge imgs, get the cloud align it, turn the new points to gaussians
    moge_imgs_names = ["5.jpg", "7.jpg", "18.jpg", "20.jpg", "22.jpg", "42.jpg", "50.jpg", "61.jpg"]

    for moge_img_name in moge_imgs_names : 

        # Load the img
        moge_img = ImageHandling.load_np_img(f"{sfm_path}/images/{moge_img_name}")

        # Get the img moge cloud 
        masked_points, masked_colors, all_moge_points, all_moge_colors, moge_mask, moge_intrins = GaussiansHandling.get_moge_points(moge_img, use_fp16_bool=False, fov_x=fov_x_deg) 
            
        # Get the img pose
        img_pose = PosesHandling.get_img_sfm_pose(recons, moge_img_name)

        # Render accum_moges from its pose
        accumulated_moges_o3d = MeshHandling.turn_points_to_o3d(accumulated_moge_points, accumulated_moge_colors)
        moge_img_render, _ = mesh_handling.render_mesh_pic([accumulated_moges_o3d], img_pose)
        
        # Get img txtr (small patch) and render txtr(big patches)
        _, moge_img_txtr = ImageHandling.compute_texturemap_and_mask(moge_img, threshold=0.1)
        _, moge_img_render_txtr = ImageHandling.compute_texturemap_and_mask(moge_img_render, threshold=0.01)
        txtr_mask = moge_img_txtr & (~moge_img_render_txtr)

        # Get moge mask to txtr mask and apply it on flattened points
        final_mask = moge_mask & txtr_mask
        final_mask_flat = final_mask.reshape(-1)  #turning H,W,1 to H*W,1
        all_moge_colors_flat = all_moge_colors.reshape(-1, 3) # turning H,W,3 to H*W,3
        all_moge_points_flat = all_moge_points.reshape(-1, 3).astype(np.float64)
        final_moge_colors = all_moge_colors_flat[final_mask_flat]
        final_moge_points = all_moge_points_flat[final_mask_flat]

        ImageHandling.plot_4_imgs(moge_img, moge_img_txtr, moge_img_render, moge_img_render_txtr)
        ImageHandling.plot_img(txtr_mask, "final_mask")
        
        # Get the 3d-2d points of the img from sfm data
        img_sfm_points_2d, img_sfm_points_3d = PosesHandling.get_2d_3d_points_of_img(recons, moge_img_name)
        # Keep only moge points corresp to sfm points
        x = img_sfm_points_2d[:,0].astype(int)
        y = img_sfm_points_2d[:,1].astype(int)
        sfm_moge_points = all_moge_points[y, x]
        # Apply moge mask on both mogepoints and sfmpoints (to avoid inf values points)
        sfm_mask = moge_mask[y, x]
        sfm_moge_points = sfm_moge_points[sfm_mask]
        img_sfm_points_3d = img_sfm_points_3d[sfm_mask]
        # Get the transfo infos between moge nd sfm_sparse
        _, s, R, t = MeshHandling.align_points(sfm_moge_points, img_sfm_points_3d)  

        # Apply T on the filtered points
        final_moge_points = (s * (R @ final_moge_points.T)).T + t
        accumulated_moge_points = np.concatenate((accumulated_moge_points, final_moge_points), axis=0)
        accumulated_moge_colors = np.concatenate((accumulated_moge_colors, final_moge_colors), axis=0)    


    accumulated_moges_o3d = MeshHandling.turn_points_to_o3d(accumulated_moge_points, accumulated_moge_colors)
    MeshHandling.visualize_scene([accumulated_moges_o3d])

    all_gaussians = GaussiansHandling.turn_points_to_gaussians(accumulated_moge_points, accumulated_moge_colors)
    GaussiansHandling.turn_gaussians_to_ply(all_gaussians, "/home/user/Bureau/ilyas/3d_art/essaouira_room/moge_gaussians.ply")




  
  
  
  
    
if __name__ == "__main__":
    main()
