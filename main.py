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



#_________________________________________________________________________



# Main params
scene_name = "playroom"
case_nbr = 3
CAM_W, CAM_H = 1264, 832
init_img_gs1_name = init_img_name_sfm1 = "DSC05732.jpg"
des_img_name_sfm1 = des_img_name_sfm_gs2 = gt_des_name = "des0.png"


# IBVS Vars
lambda_gain = 0.1
dt_ibvs = 0.03
ibvs_nbr_features = 10
max_ibvs_nbr_itrs_init = 5000
ibvs_pxl_error_conv = 0.02
max_ibvs_nbr_itrs = 100
# Kf vars
kf_motion_ratio = 0.03    # 3% Of screen
kf_motion_nbr_features = 100
ibvs_pxl_error_kf = 0.05 # Last kf to take in consid
# GS vars
gs_reso = 2
gs_nbr_itrs = 50
moge_reso = 4
# Moge vars
moge_model_reso_lvl = 1
moge_depth_edge_threshold = 0.05
# Some configs
np.set_printoptions(precision=2, suppress=False)
xfeat = XFeat()
_xfeat = None



# Main paths
main_test_path = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test"
case_test_path = f"{main_test_path}/{scene_name}/{scene_name}_case{case_nbr}"
# GS1 paths
gs1_sfm_path = f"{main_test_path}/{scene_name}/real_scene/sfm_{scene_name}"
gs1_ply_path = f"{main_test_path}/{scene_name}/real_scene/{scene_name}.ply"
gs1_sfm_aligned_path = f"{case_test_path}/gs1_sfm_aligned"
# Main paths
des_imgs_path = f"{case_test_path}/desired_imgs"
keyframes_path = f"{case_test_path}/keyframes"
sfm_path = f"{case_test_path}/sfm"
gs2s_dir_path = f"{case_test_path}/gs2s"
pre_matches_frames_path = f"{case_test_path}/pre_des_ibvs_matches"
real_frames_path = f"{case_test_path}/ibvs_frames/real_frames"
matches_frames_path = f"{case_test_path}/ibvs_frames/matches_frames"
configs_path = f"{case_test_path}/configs.txt"
results_txt_file = f"{case_test_path}/results.txt"
# Saved elmnts path
moge_path = f"{case_test_path}/init_moges.ply"
init_des_trans_npy_path = f"{case_test_path}/init_des_trans.npy"
des_masks_path = f"{des_imgs_path}/des_masks.npy"
# Saved results path
des0_ibvs_infos_npy_path = f"{case_test_path}/des0_ibvs_infos.npy"  
desGT_ibvs_infos_npy_path = f"{case_test_path}/desGT_ibvs_infos.npy"  
ibvs_infos_npy_path = f"{case_test_path}/ibvs_infos.npy"
des_depths_npy_path = f"{case_test_path}/des_depths.npy"
ibvs_frame_depths_npy_path = f"{case_test_path}/ibvs_frame_depths.npy"
# GS_inria_path for training
gs_inria_path = "gaussian_splatting2"
inria_saved_ply_path = f"{case_test_path}/gs2s/inria_output/point_cloud/iteration_{gs_nbr_itrs}/point_cloud.ply"

# Starting states
did_start = False
trans_exist = False
des0_ready = False
des0_aligned = False 
des_GT_ready = False
ibvs_desGT_converge = True
ibvs_des0_converge = True
get_des_frm_gs = False
ready_for_dualoop = True







def write_init_config_txt(configs_path) :
    
    # Path where you want to save the file
    txt_path = Path(configs_path)

    with open(txt_path, "w") as f:
        f.write("____ initial configs ____\n\n")

        f.write(f"init_img_gs1_name = {init_img_gs1_name}\n\n")

        f.write("# IBVS Vars\n")
        f.write(f"lambda_gain = {lambda_gain}\n")
        f.write(f"dt_ibvs = {dt_ibvs}\n")
        f.write(f"ibvs_nbr_features = {ibvs_nbr_features}\n")
        f.write(f"max_ibvs_nbr_itrs = {max_ibvs_nbr_itrs}\n")
        f.write(f"ibvs_pxl_error_conv = {ibvs_pxl_error_conv}\n\n")

        f.write("# Some configs\n")
        f.write(f"moge_reso = {moge_reso}\n")
        f.write(f"moge_model_reso_lvl = {moge_model_reso_lvl}\n")
        f.write(f"moge_depth_edge_threshold = {moge_depth_edge_threshold}\n")





def get_xfeat_model():
    global _xfeat
    if _xfeat is None:
        _xfeat = XFeat()
    return _xfeat




def get_xfeat_kpts(img, xfeat_model) :
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)  # (1,3,H,W)
    out = xfeat_model.detectAndCompute(tensor, top_k=2048)[0]
    kpts = out['keypoints']
    desc = out['descriptors']
    return  kpts, desc









 
MSG_WIDTH  = 67
NUM_WIDTH  = 10
TIME_WIDTH = 14
 
def _display_width(text: str) -> int:
    """Return true terminal display width, correctly handling emojis and wide chars."""
    cleaned = emoji.replace_emoji(text, replace="__")  # each emoji -> 2-wide placeholder
    w = wcswidth(cleaned)
    return w if w >= 0 else len(cleaned)
 
 
def _pad(text: str, width: int) -> str:
    """Pad text to a fixed display width."""
    padding = width - _display_width(text)
    return text + " " * max(padding, 0)
 
 
def _fmt_time(t: float) -> str:
    """Convert a time.time() float to HH:MM:SS.mmm format."""
    dt = datetime.fromtimestamp(t)
    return dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"

 
def log_to_table(message: str, source: str, dt: float, file_path: str = results_txt_file):
    MSG_WIDTH  = 55
    NUM_WIDTH  = 10
    TIME_WIDTH = 14
 
    SEP = ("+" + "-" * MSG_WIDTH
         + "+" + "-" * MSG_WIDTH
         + "+" + "-" * NUM_WIDTH
         + "+" + "-" * TIME_WIDTH + "+\n")
    BLANK = " " * MSG_WIDTH
 
    dt_str   = f"{dt:.3f}"
    time_str = _fmt_time(time.time())
 
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(SEP)
            f.write(
                f"|{_pad('  IBVS', MSG_WIDTH)}"
                f"|{_pad('  GS', MSG_WIDTH)}"
                f"|{_pad('  DT', NUM_WIDTH)}"
                f"|{_pad('  TIME', TIME_WIDTH)}|\n"
            )
            f.write(SEP)
 
    src = source.strip().lower()
    if src == "ibvs":
        row = (f"|{_pad(' ' + message, MSG_WIDTH)}"
               f"|{BLANK}"
               f"|{_pad(' ' + dt_str, NUM_WIDTH)}"
               f"|{_pad(' ' + time_str, TIME_WIDTH)}|\n")
    elif src == "gs":
        row = (f"|{BLANK}"
               f"|{_pad(' ' + message, MSG_WIDTH)}"
               f"|{_pad(' ' + dt_str, NUM_WIDTH)}"
               f"|{_pad(' ' + time_str, TIME_WIDTH)}|\n")
    else:
        raise ValueError("source must be 'ibvs' or 'gs'")
 
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(row)
        f.write(SEP)













def getposefrmibvs(ibvs_npy_path, itr_nbr):

    ibvs_dict = np.load(ibvs_npy_path, allow_pickle=True).item()

    if itr_nbr not in ibvs_dict:
        raise KeyError(f"Iteration {itr_nbr} not found in '{ibvs_npy_path}'.")

    ibvs_infos = ibvs_dict[itr_nbr]
    cur_pose_gs1 = ibvs_infos[-1]

    return cur_pose_gs1
















def dyna_ibvs(gaussians, intrins_gs1, init_pose_gs1, des_img, des_ibvs_infos_npy_path) :

    try: 
        ibvs_tools = IbvsTools(CAM_W, CAM_H)    
        cur_pose_gs1 = init_pose_gs1
        des_kpts, des_desc = ibvs_tools.get_xfeat_kpts(des_img)
        norm_of_error = None
        i = 0

        des_mask = load_frst_dict_np_elmnt(des_masks_path)

        for i in range(max_ibvs_nbr_itrs_init) :    

            # Render new cur_gs_pic and get its depthmap
            cur_gs1_img, cur_gs1_depth_map = GaussiansHandling.render_gs_pic(*gaussians, T=cur_pose_gs1, K=intrins_gs1, W=CAM_W, H=CAM_H)

            # Match current_gs1 with des_estim using xfeat 
            cur_kpts, cur_desc = ibvs_tools.get_xfeat_kpts(cur_gs1_img)
            idxs0, idxs1 = xfeat.match(cur_desc, des_desc)
            matches_cur = cur_kpts[idxs0]
            matches_des = des_kpts[idxs1]

            # Filter the matches (remove the one on des_mask edges)
            safe_matches_indices = ibvs_tools.border_safe_indices(matches_des, des_mask)  # this gets a tensor turn it to numpy for handling mask edges and all
            safe_matches_indices = torch.from_numpy(safe_matches_indices).to(matches_cur.device) # as border_safe returns numpy indices, we got to turn them to tensors 
            matches_cur = matches_cur[safe_matches_indices]
            matches_des = matches_des[safe_matches_indices]

            matches_cur = matches_cur[:ibvs_nbr_features].to(torch.int).cpu().numpy()
            matches_des = matches_des[:ibvs_nbr_features].to(torch.int).cpu().numpy()
            
            if (len(matches_cur) < 4) :
                raise Exception(f"only {len(matches_cur)} < {ibvs_nbr_features}")

            # 2 - Get cur and des features
            Ss_star = ibvs_tools.get_Ss_from_uv(matches_des)
            Ss_cur = ibvs_tools.get_Ss_from_uv(matches_cur)
            Ss_Z_cur = ibvs_tools.get_feats_depth(matches_cur, cur_gs1_depth_map)

            # 3 - Get the error
            errors = ibvs_tools.getting_errors(Ss_cur, Ss_star)
            errors = np.asarray(errors, dtype=float).reshape(-1)
            norm_of_error = np.linalg.norm(errors)
            print(f"IBVS iter {i} / 2D error : {norm_of_error}")

            if norm_of_error < ibvs_pxl_error_conv :
                print("converged !!")
                break

            # 4 - Get the intr matrix & its pseudo_inv 
            L = ibvs_tools.get_interaction_matrix(len(matches_cur), Ss_cur, Ss_Z_cur, 1)
            L_psinv = ibvs_tools.get_inter_mat_pseudo_inverse(L)

            # 5 - Get V from control law
            V = - lambda_gain * (L_psinv @ errors)  
           
            # 6 - Update cur_cam_pose and update visualization
            cur_pose_gs1 = ibvs_tools.update_cam_pose(cur_pose_gs1, V, dt_ibvs)
 
            # Save the infos of each ibvs iteration : 2d_err, condit_nbr, V, matches, pose_in_glbl_gs
            ibvs_infos = [norm_of_error, LinAlgeb.get_mat_condition_number(L), V, [matches_cur, matches_des],cur_pose_gs1]
            MyUtils.save_arrays_to_npy(des_ibvs_infos_npy_path, i, ibvs_infos)


            if (i % 5) == 0 :
                cur_mtch_gs_img = ImageHandling.draw_matches(matches_cur, matches_des, cur_gs1_img, des_img)
                ImageHandling.save_img(cur_mtch_gs_img, i, pre_matches_frames_path)   
                MyUtils.cleanup()



    except KeyboardInterrupt:
            print("\nCtrl+C detected, exiting loop cleanly.")
            MyUtils.cleanup()

    
    
    




def type_confis_on_txt(config_txt_path):
    with open(config_txt_path, "a") as f:
        f.write("\n\n# Camera & init img\n")
        f.write(f"CAM_W = {CAM_W}\n")
        f.write(f"CAM_H = {CAM_H}\n")
        f.write(f'init_img_gs1_name = "{init_img_gs1_name}"\n')

        f.write("\n# IBVS Vars\n")
        f.write(f"lambda_gain = {lambda_gain}\n")
        f.write(f"dt_ibvs = {dt_ibvs}\n")
        f.write(f"ibvs_nbr_features = {ibvs_nbr_features}\n")
        f.write(f"max_ibvs_nbr_itrs_init = {max_ibvs_nbr_itrs_init}\n")
        f.write(f"ibvs_pxl_error_conv = {ibvs_pxl_error_conv}\n")
        f.write(f"max_ibvs_nbr_itrs = {max_ibvs_nbr_itrs}\n")

        f.write("\n# Kf vars\n")
        f.write(f"kf_motion_ratio = {kf_motion_ratio}\n")
        f.write(f"kf_motion_nbr_features = {kf_motion_nbr_features}\n")
        f.write(f"ibvs_pxl_error_kf = {ibvs_pxl_error_kf}\n")

        f.write("\n# GS vars\n")
        f.write(f"gs_reso = {gs_reso}\n")
        f.write(f"gs_nbr_itrs = {gs_nbr_itrs}\n")
        f.write(f"moge_reso = {moge_reso}\n")

        f.write("\n# Moge vars\n")
        f.write(f"moge_model_reso_lvl = {moge_model_reso_lvl}\n")
        f.write(f"moge_depth_edge_threshold = {moge_depth_edge_threshold}\n")








def get_matched_features(kpts_img1, desc_img1, kpts_img2, desc_img2, nbr_matches) :
    idxs0, idxs1 = xfeat.match(desc_img1, desc_img2)
    matches_img1 = kpts_img1[idxs0]
    matches_img2 = kpts_img2[idxs1]
    matches_img1 = matches_img1[:nbr_matches].to(torch.int).cpu().numpy()
    matches_img2 = matches_img2[:nbr_matches].to(torch.int).cpu().numpy()
    return matches_img1, matches_img2







def get_corresp_3d_point_matches(matches_img1, matches_img2, moge_mask1, moge_mask2, all_moge_points1, all_moge_points2) :

    # get the indices(x,y) of the xfeat matched 2d points on img1 and img2
    x1 = matches_img1[:, 0]
    y1 = matches_img1[:, 1]
    x2 = matches_img2[:, 0]
    y2 = matches_img2[:, 1]

    # return the moge mask for the xfeat points (True for the valid xfeat points) => 1D [T, F, T..] lngth of the xfeat points
    valid1 = moge_mask1[y1, x1]
    valid2 = moge_mask2[y2, x2]
    valid = valid1 & valid2

    H, W = moge_mask1.shape
    all_moge_points1 = all_moge_points1.reshape(H, W, 3)
    all_moge_points2 = all_moge_points2.reshape(H, W, 3)

    # Corresponding 3D points and colors of the valid 2d xfeat matches
    matched_points3d1 = all_moge_points1[y1, x1][valid]
    matched_points3d2 = all_moge_points2[y2, x2][valid]
    return matched_points3d1, matched_points3d2 



def filter_points(points_3d, colors, mask) : 

    points_3d_flat = points_3d.reshape(-1, 3) # turning H,W,3 to H*W,3
    colors_flat = colors.reshape(-1, 3).astype(np.float64)
    mask_flat = mask.reshape(-1)  #turning H,W,1 to H*W,1

    filtered_moge_colors = colors_flat[mask_flat]
    filtered_moge_points = points_3d_flat[mask_flat]

    return filtered_moge_points, filtered_moge_colors




def get_intrins_gs(gs_sfm_path):
    recons = PosesHandling.get_recons(gs_sfm_path)
    K = PosesHandling.get_cam_matrix(recons)
    intrins_gs = GaussiansHandling.turn_cam_matrix_to_gsplat_format(K)
    return intrins_gs




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






def validate_keyframe(last_kf, curr_frame, xfeat, ibvs_itr_nbr,
                      kf_motion_nbr_features=100,
                      motion_thresh_ratio=0.0):

    # 1. Match features
    pts_last, pts_curr = xfeat.match_xfeat(last_kf, curr_frame)

    # 2. Keep top matches
    pts_last = pts_last[:kf_motion_nbr_features]
    pts_curr = pts_curr[:kf_motion_nbr_features]

    if len(pts_last) < 10:
        return False

    # 3. Compute pixel displacement(listof pxls disp)
    disp = np.linalg.norm(pts_curr - pts_last, axis=1)

    # 4. Robust score (median)
    median_disp = np.median(disp)
    #why median (example) :
    #pixels displacemnts = [2, 3, 5, 6, 100]
    #mean = (2+3+5+6+100)/5 = 23.2  (weak against outliers)
    #median = 5 (robuts to outliers)

    # 5. Normalize by image width // get how % the median pxls is in img_width
    img_width = curr_frame.shape[1]
    motion_ratio = median_disp / img_width
    print(f"🔵-{ibvs_itr_nbr} kf motion : {motion_ratio} pxls")

    # 6. Decision
    return motion_ratio > motion_thresh_ratio






def load_frst_dict_np_elmnt(dict_np_path) : 
    data = np.load(dict_np_path, allow_pickle=True)
    dict = data.item()
    elmnt = dict[0]
    return elmnt









def dyna_ibvs_loop(shared, kf_queue, des_img_queue, kf_event, gs1_ply_path, init_pose_gs1, gt_des_pose_gs1, init_img, des_img, fx_gs1, fy_gs1) :

    try:

        msg = f"🔵Ibvs_loop launched"; print(msg); log_to_table(msg, 'ibvs', 0); _t = time.time()
        # Loading gs1
        gaussians1 = GaussiansHandling.load_gaussians_from_ply(gs1_ply_path)
        intrins_gs1 = get_intrins_gs(gs1_sfm_path)
        # get des0 kpnts
        xfeat_model = get_xfeat_model()
        des_kpts, des_desc = get_xfeat_kpts(des_img, xfeat_model)
        # init vars
        pxl_error = pose_error = None
        i = 0
        des_vrsn = 0
        final_kf_sent = False
        cur_pose_gs1 = init_pose_gs1
        last_keyframe = init_img
        ibvs_tools = IbvsTools(CAM_W, CAM_H, fx_gs1, fy_gs1)

        des_mask = load_frst_dict_np_elmnt(des_masks_path)
   
        msg = f"🔵Loaded xfeat and gs1"; dt = time.time()-_t; _t = time.time()
        print(msg); log_to_table(msg, 'ibvs', dt); 

        while not shared["stop"] :
            i+=1;_t = time.time()

            # Check if GS_loop got us new des_img, then update it   
            while not des_img_queue.empty():
                des_img = des_img_queue.get()
                des_vrsn += 1
                des_kpts, des_desc = get_xfeat_kpts(des_img, xfeat_model)
                msg = f"🔵🔵{i}-Got new des_img({des_vrsn})"; print(msg); log_to_table(msg, 'ibvs', 0)

            # Render new cur_gs_pic and get its depth_map
            cur_gs1_img, cur_gs1_depth_map = GaussiansHandling.render_gs_pic(*gaussians1, T=cur_pose_gs1, K=intrins_gs1, W=CAM_W, H=CAM_H)

            # Notify GS_Loop if it's a keyframe
            if validate_keyframe(last_keyframe, cur_gs1_img, xfeat_model, ibvs_itr_nbr=i, kf_motion_nbr_features= kf_motion_nbr_features, motion_thresh_ratio = kf_motion_ratio) :
                last_keyframe = cur_gs1_img
                kf_queue.put([last_keyframe, cur_pose_gs1, False])
                kf_event.set()
                msg = f"🔵🔵{i}-Keyframe was found and sent to gsloop"
                print(msg); log_to_table(msg, 'ibvs', 0)


            # 1 - Match current_gs1_frame with des_img
            cur_kpts, cur_desc = get_xfeat_kpts(cur_gs1_img, xfeat_model)
            idxs0, idxs1 = xfeat_model.match(cur_desc, des_desc)
            matches_cur = cur_kpts[idxs0]
            matches_des = des_kpts[idxs1]

            # Filter the matches (remove the one on des_mask edges)
            safe_matches_indices = IbvsTools.border_safe_indices(matches_des, des_mask)
            matches_cur = matches_cur[safe_matches_indices]
            matches_des = matches_des[safe_matches_indices]

            matches_cur = matches_cur[:ibvs_nbr_features].to(torch.int).cpu().numpy()
            matches_des = matches_des[:ibvs_nbr_features].to(torch.int).cpu().numpy()
            

            # Stop if we got occluded or got no matches
            if (len(matches_cur) < 4) :
                shared["stop"] = True
                msg = f"!!!!🔵{i}- Not enough matches during ibvs" 
                print(msg); log_to_table(msg, 'ibvs', 0)
                cur_mtch_gs_img = ImageHandling.draw_matches(matches_cur, matches_des, cur_gs1_img, des_img)
                ImageHandling.save_img(cur_mtch_gs_img, i, matches_frames_path) 
                raise Exception(f"only {len(matches_cur)} < {ibvs_nbr_features}")

            # 2 - Get ss and depth from the matches (to make intr matrix)
            Ss_star = ibvs_tools.get_Ss_from_uv(matches_des)
            Ss_cur = ibvs_tools.get_Ss_from_uv(matches_cur)
            Ss_Z_cur = ibvs_tools.get_feats_depth(matches_cur, cur_gs1_depth_map)

            # 3 - Get the error vector
            errors = ibvs_tools.getting_errors(Ss_cur, Ss_star)
            errors = np.asarray(errors, dtype=float).reshape(-1)

            # Print the norm of the error
            pxl_error = np.linalg.norm(errors)
            pose_error = LinAlgeb.pose_distance(cur_pose_gs1, gt_des_pose_gs1)
            msg = f"🔵{i}-IBVS: 2d_err:{pxl_error:.4f} | des_img({des_vrsn})"; dt = time.time()-_t; print(f"{msg} || {dt:.3f} s")

            # 4 - Get the intr matrix & its pseudo_inv & calc V with control law
            L = ibvs_tools.get_interaction_matrix(len(matches_cur), Ss_cur, Ss_Z_cur, 1)
            L_psinv = get_inter_mat_pseudo_inverse(L)
            V = - lambda_gain * (L_psinv @ errors)
 
            # 5 - Update cur_cam_pose and update visualization
            cur_pose_gs1 = ibvs_tools.update_cam_pose(cur_pose_gs1, V, dt_ibvs)

            # Check if we converged
            if pxl_error < ibvs_pxl_error_conv :
                shared["stop"] = True

            # Check if we need to send a final kf
            if (pxl_error < ibvs_pxl_error_kf) and (not final_kf_sent) :
                last_keyframe = cur_gs1_img
                kf_queue.put([last_keyframe, cur_pose_gs1, True])
                kf_event.set()
                final_kf_sent = True
                msg = f"🔵🔵{i}-Last kf found"
                print(msg); log_to_table(msg, 'ibvs', 0)

            # Save the infos of each ibvs iteration : 2d_err, 3d_err, condit_nbr, V, matches, homog_pose_in_gs1
            cond_nbr = LinAlgeb.get_mat_condition_number(L)
            ibvs_infos = [pxl_error, pose_error, cond_nbr, V, [matches_cur, matches_des],cur_pose_gs1]
            MyUtils.save_arrays_to_npy(ibvs_infos_npy_path, i, ibvs_infos)

            if (i % 5) == 0 :
                print(msg); log_to_table(msg, 'ibvs', dt)
                cur_mtch_gs_img = ImageHandling.draw_matches(matches_cur, matches_des, cur_gs1_img, des_img)
                ImageHandling.save_img(cur_gs1_img, i, real_frames_path)
                ImageHandling.save_img(cur_mtch_gs_img, i, matches_frames_path)   
                MyUtils.save_arrays_to_npy(ibvs_frame_depths_npy_path, i, cur_gs1_depth_map) 
                MyUtils.cleanup()

    except Exception as e:
            print(f"Error in ibvs_loop {e} !!!!")
            print("[ibvs] FULL TRACEBACK:")
            traceback.print_exc()
            MyUtils.cleanup()
            shared["stop"] = True






def local_gs_loop(shared, kf_queue, des_img_queue, des_img_event, moge_points1_cmfrm, moge_colors1, moge_mask1, init_pose_gs1, moge_des_trans, fx_gs1, fy_gs1, init_img) :

    import queue
    try :

        curr_keyframe = None
        init_pose_gs2 = None
        des_pose_gs2 = None
        scene_scale = None
        intrins_gs2 = None
        moge_points_o3d1 = None
        gs_nbr_itrs = 50  
        inria_saved_ply_path = f"{case_test_path}/gs2s/inria_output/point_cloud/iteration_{gs_nbr_itrs}/point_cloud.ply"
        init = True
        i = 0
        kf_vrsn = 0
        gs_nbr_itrs = 50
        kpts_img1 =  desc_img1 = None
        is_last_kf = False
        

        # get fovx 
        fov_x_rad = 2 * np.arctan(CAM_W / (2 * fx_gs1))
        fov_x_deg = np.rad2deg(fov_x_rad)

        #intrins_gs2 = get_intrins_gs(sfm_path)
        mesh_handling = MeshHandling(CAM_W, CAM_H, fx_gs1, fy_gs1)
        ibvs_tools = IbvsTools(CAM_W, CAM_H, fx_gs1, fy_gs1)

        # loading moge model once in the beg
        _t = time.time()
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda")
        msg = f"🟢{i}-Just Loaded Moge model"; dt = time.time()-_t; print(f"{msg} || {dt:.3f} s")
        log_to_table(msg, 'gs', dt); _t = time.time()


        while not shared["stop"]:            
            curr_keyframe = None

            # if kf_queue is empty we skip this bloc that loads keyframes
            while not kf_queue.empty():
                curr_keyframe_list = kf_queue.get()
                curr_keyframe = curr_keyframe_list[0]
                curr_keyframe_pose_gs1 = curr_keyframe_list[1]
                is_last_kf = curr_keyframe_list[2]
                kf_vrsn += 1
                _t = time.time()
                msg = f"🟢🟢{i}-Received KF{kf_vrsn}"; dt = time.time()-_t
                print(msg); log_to_table(msg, 'gs', dt); _t = time.time()


            # If a keyframe is received do this 
            if curr_keyframe is not None:

                # (this will happen only the first time we get a kf)
                if init :

                    print("🟢 Optimizing scale______________"); _t = time.time()

                    # Turning the mogecloud to wf and getting o3d
                    moge_points1 = LinAlgeb.transform_points_to_world(moge_points1_cmfrm, init_pose_gs1)
                    moge_points_o3d1 = MeshHandling.turn_points_to_o3d(moge_points1, moge_colors1)

                    # Launch optimizer                    
                    optimizer = ScaleOptimizer(case_test_path)
                    best_scale, corresp_error = optimizer.optimize(curr_keyframe_pose_gs1, init_pose_gs1, mesh_handling, moge_points_o3d1, curr_keyframe)
                    scene_scale = best_scale
                    MyUtils.add_line_to_text(configs_path, f"Scale = {best_scale}")
                    msg = f"🟢{i}-Optimized scale using KF_{kf_vrsn}"; dt = time.time()-_t; print(msg); log_to_table(msg, 'gs', dt); _t = time.time()

                    # Scale T0
                    init_pose_gs2 = init_pose_gs1.copy()
                    init_pose_gs2[:3, 3] *= scene_scale

                    # Transform P0
                    moge_points1 = LinAlgeb.transform_points_to_world(moge_points1_cmfrm, init_pose_gs2)
                    moge_points_o3d1 = MeshHandling.turn_points_to_o3d(moge_points1, moge_colors1)

                    # Init Colmap
                    PosesHandling.make_colmap_data(f"{sfm_path}/sparse/0", [0], ["init_img.png"], [init_pose_gs2], CAM_W, CAM_H, fx_gs1, fy_gs1)
                    ImageHandling.save_img(init_img, "init_img", f"{sfm_path}/images")
                    msg = f"🟢{i}-Init Colmap"; dt = time.time()-_t; print(msg); log_to_table(msg, 'gs', dt); _t = time.time()
                                          
                    # Get des_pose_gs2 and load intrin_gs2
                    des_pose_gs2 = LinAlgeb.transform_pose(moge_des_trans, init_pose_gs2)
                    intrins_gs2 = get_intrins_gs(sfm_path)

                    # Saving gs2_0_pre  :
                    # Texture I0 and get its Kpnts
                    _, init_img_txtr_mask = ImageHandling.compute_texturemap_and_mask(init_img, threshold=0.07)
                    kpts_img1, desc_img1 =ibvs_tools.get_xfeat_kpts(init_img)

                    # Filter P0
                    final_mask = moge_mask1 & init_img_txtr_mask
                    filtered_moge_points1, filtered_moge_colors1 = filter_points(moge_points1, moge_colors1, final_mask)
                    # Downsample & Turn P0 to gaussians and save ply 
                    moge_reso = 4
                    filtered_moge_points1 = filtered_moge_points1[::moge_reso]
                    filtered_moge_colors1 = filtered_moge_colors1[::moge_reso]    
                    gaussians0 = GaussiansHandling.turn_points_to_gaussians(filtered_moge_points1, filtered_moge_colors1, scale=0.01)
                    GaussiansHandling.turn_gaussians_to_ply(gaussians0, f"{gs2s_dir_path}/gs2_0_pre.ply")
                    msg = f"🟢{i}-Filter P0 and turn it to gs2_0"; dt = time.time()-_t; print(msg); log_to_table(msg, 'gs', dt); _t = time.time()


                # Get Kf txtr and kpnts
                _, kf_txtred = ImageHandling.compute_texturemap_and_mask(curr_keyframe, threshold=0.07)
                kpts_kf, desc_kf = ibvs_tools.get_xfeat_kpts(curr_keyframe)

                # Scale T_kfi and add to colmap
                curr_keyframe_pose_gs2 = curr_keyframe_pose_gs1.copy()
                curr_keyframe_pose_gs2[:3, 3] *= scene_scale
                PosesHandling.add_imgs_to_colmap([kf_vrsn], [f"keyframe{kf_vrsn}.png"], [curr_keyframe_pose_gs2], f"{sfm_path}/sparse/")
                ImageHandling.save_img(curr_keyframe, f"keyframe{kf_vrsn}", f"{sfm_path}/images")
                msg = f"🟢{i}-Scale Kf_{kf_vrsn} and add it to colmap"; dt = time.time()-_t; print(msg); log_to_table(msg, 'gs', dt); _t = time.time()

                # Get Pi and transform it
                masked_points, masked_colors, all_moge_points, all_moge_colors, moge_mask, moge_intrins = GaussiansHandling.get_moge_points(
                    curr_keyframe, fov_x=fov_x_deg, model_reso_lvl=moge_model_reso_lvl, depth_edge_threshold=moge_depth_edge_threshold, moge_model=moge_model, use_fp16_bool=False) 
                all_moge_points = LinAlgeb.transform_points_to_world(all_moge_points, curr_keyframe_pose_gs2)
                msg = f"🟢{i}-Get Moge cloud of Kf_{kf_vrsn}"; dt = time.time()-_t; print(msg); log_to_table(msg, 'gs', dt); _t = time.time()


                # Align Pi with P0 : ftr matching -> getting corresp 3ds -> umeyama 
                matches_img1, matches_kf = get_matched_features(kpts_img1, desc_img1, kpts_kf, desc_kf, nbr_matches=50)
                matched_3d_points_img1, matched_3d_points_kf = get_corresp_3d_point_matches(
                    matches_img1, matches_kf, moge_mask1, moge_mask, moge_points1, all_moge_points)        
                _, s, R, t = MeshHandling.align_points(matched_3d_points_kf, matched_3d_points_img1) # first cloud in arg is the one to change
                msg = f"🟢{i}-Match and align Kf_{kf_vrsn} with I0 P0"; dt = time.time()-_t; print(msg); log_to_table(msg, 'gs', dt); _t = time.time()


                # Filter Pi (new txtrd pixels) ________________________________________________________

                # step0: load last_local_gs for filtering 
                if init : # m forced to load gs2_0_pre before training it to gs2_0
                    last_gaussians = GaussiansHandling.load_gaussians_from_ply(f"{gs2s_dir_path}/gs2_0_pre.ply") 
                else :
                    last_gaussians = GaussiansHandling.load_gaussians_from_ply(f"{gs2s_dir_path}/gs2_{i}.ply")

                # step1: Render Ii from last_gs or from P0 (if I1), and 
                if init :
                    kf_render, _ = mesh_handling.render_mesh_pic([moge_points_o3d1], curr_keyframe_pose_gs2)
                else : 
                    kf_render, _ = GaussiansHandling.render_gs_pic(*last_gaussians, curr_keyframe_pose_gs2, intrins_gs2, CAM_W, CAM_H)

                # get mask of new txtr pixels
                _, kf_render_txtred = ImageHandling.compute_texturemap_and_mask(kf_render, threshold=0.01)
                txtr_mask = kf_txtred & (~kf_render_txtred)
                final_mask = moge_mask & txtr_mask # moge mask to make sure the pixels have a corresponding moge point

                # save results
                #ImageHandling.save_img(curr_keyframe, f"kf_{i}_0", f"{case_test_path}/keyframes")
                #ImageHandling.save_img(kf_txtred, f"kf_{i}_1txtred", f"{case_test_path}/keyframes")
                #ImageHandling.save_img(kf_render, f"kf_{i}_2render", f"{case_test_path}/keyframes")
                #ImageHandling.save_img(kf_render_txtred, f"kf_{i}_3rndr_txtred", f"{case_test_path}/keyframes")
                #ImageHandling.save_img(final_mask, f"kf_{i}_rndr_4finalmask", f"{case_test_path}/keyframes")

                # Filter Pi (keep new textured pixels points)  
                filtered_moge_points, filtered_moge_colors = filter_points(all_moge_points, all_moge_colors, final_mask)
                
                # Align filtered Pi with P0
                filtered_moge_points = (s * (R @ filtered_moge_points.T)).T + t

                filtered_moge_points = filtered_moge_points[::moge_reso]
                filtered_moge_colors = filtered_moge_colors[::moge_reso]  

                # Turn filtered points to gaussians and merge it with last gaussians
                new_gaussians = GaussiansHandling.turn_points_to_gaussians(filtered_moge_points, filtered_moge_colors, scale=0.007)
                total_gaussians = GaussiansHandling.merge_2_gaussians(last_gaussians, new_gaussians)

                # Save these gaussians to _pre.ply then train it for 50 itrs, then copy output in gs2
                GaussiansHandling.turn_gaussians_to_ply(total_gaussians, f"{gs2s_dir_path}/gs2_{i}_pre.ply")
                msg = f"🟢{i}-Filter cloud of Kf_{kf_vrsn} and add to gs2"; dt = time.time()-_t; print(msg); log_to_table(msg, 'gs', dt); _t = time.time()
                init = False

                
            # the training won't be applied until we get the first Keyframe and add its gaussians to gs2_0
            if not init :
                i+=1
                _t = time.time()

                if(i==1) :
                    gs_nbr_itrs = 150  
                    inria_saved_ply_path = f"{case_test_path}/gs2s/inria_output/point_cloud/iteration_{gs_nbr_itrs}/point_cloud.ply"
                else : 
                    gs_nbr_itrs = 50  
                    inria_saved_ply_path = f"{case_test_path}/gs2s/inria_output/point_cloud/iteration_{gs_nbr_itrs}/point_cloud.ply"

                # train last gs2 and copy the inria saved pointcloud to gs2s 
                GaussiansHandling.run_gs_training(sfm_path=sfm_path,
                    output_path=f"{gs2s_dir_path}/inria_output", gs_reso=gs_reso, gs_nbr_itrs=gs_nbr_itrs)
                shutil.copy2(inria_saved_ply_path, f"{gs2s_dir_path}/gs2_{i}.ply")
                msg = f"🟢{i}-Trained local_gs for {gs_nbr_itrs} iterations"; dt = time.time()-_t; _t = time.time()
                print(msg); log_to_table(msg, 'gs', dt)

                # Load last gs2 & render new des img & notify ibvs loop
                last_gaussians2 = GaussiansHandling.load_gaussians_from_ply(f"{gs2s_dir_path}/gs2_{i}.ply")

                # I we reached the last kf, i will update the des pose (dvs before rendering it), this only happens once with lastkf and then we set it again to false 
                if is_last_kf :
                    des0_img = ImageHandling.load_np_img(f"{des_imgs_path}/des0.png")
                    des0_mask = load_frst_dict_np_elmnt(des_masks_path)
                    des_pose_gs2 = ibvs_tools.start_dvs_loop(last_gaussians2, intrins_gs2, des_pose_gs2, des0_img, case_test_path, des0_mask, max_itrs=125)
                    is_last_kf = False

                des_img, des_depth = GaussiansHandling.render_gs_pic(*last_gaussians2, des_pose_gs2, intrins_gs2, CAM_W, CAM_H)
                ImageHandling.save_img(des_img, f"des{i}", des_imgs_path)
                des_img_queue.put(des_img); des_img_event.set()
                msg = f"🟢🟢{i}-Loading local_gs & Rendering new des_img({i})"; dt = time.time()-_t
                print(msg); log_to_table(msg, 'gs', dt); _t = time.time()

                # Saving infos about des_img
                MyUtils.save_arrays_to_npy(des_depths_npy_path, i, des_depth)



    except Exception as e:
        print(f"Error in gs_loop : {e} !!!!!!!!!!!!!!")
        print("[gs] FULL TRACEBACK:")
        traceback.print_exc()
        MyUtils.cleanup()
        shared["stop"] = True










#_______________________________________________________________________________________________________________________________________










def main() :

  
    # Make the folders
    if not did_start :
        os.mkdir(f"{case_test_path}/desired_imgs")
        os.mkdir(f"{case_test_path}/gs2s")
        os.mkdir(f"{case_test_path}/ibvs_frames")
        os.mkdir(f"{case_test_path}/keyframes")
        os.mkdir(f"{case_test_path}/sfm")
        os.mkdir(f"{case_test_path}/sfm/images")
        os.makedirs(os.path.join(sfm_path, "sparse", "0"), exist_ok=True)
        type_confis_on_txt(configs_path)
     
    # Load moge & gs1 infos
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda") 
    gaussians1 = GaussiansHandling.load_gaussians_from_ply(gs1_ply_path)
    recons1 = PosesHandling.get_recons(gs1_sfm_path)
    K_gs1 = PosesHandling.get_cam_matrix(recons1)
    intrins_gs1 = GaussiansHandling.turn_cam_matrix_to_gsplat_format(K_gs1)
    fx_gs1 = K_gs1[0, 0]
    fy_gs1 = K_gs1[1, 1]
    cx_gs1 = K_gs1[0, 2]
    cy_gs1 = K_gs1[1, 2]

    # Make mesh_handler with same cam as gs1, and prepare fov_x moge
    mesh_handling = MeshHandling(CAM_W, CAM_H, fx=fx_gs1, fy=fy_gs1) # Our mesh handler always considers cx=cy=H,W/2
    fov_x_rad = 2 * np.arctan(CAM_W / (2 * fx_gs1))
    fov_x_deg = np.rad2deg(fov_x_rad)
    #!!!!
    # MoGe considers a pinhole like logic where fx = fy and cx = cx = W,H/2 & Most cams do have fx nearly= fy and cx=cy=W,H/2 
    # If we using a cam with very diffrnt settings (or non-pinhole cam) then we got to find a depth estimator with more flexible intrinsics

    # Capture I0 from T0
    init_img_pose_gs1 = PosesHandling.get_img_sfm_pose(recons1, init_img_gs1_name)
    init_img, _ = GaussiansHandling.render_gs_pic(*gaussians1, init_img_pose_gs1, intrins_gs1, CAM_W, CAM_H)
    ImageHandling.save_img(init_img, "init_img", keyframes_path)

           
    # Get des0 and save P0 and local-gs
    if not des0_ready : 

        # Load init_img from original gs1_sfm (for better quality)
        if not get_des_frm_gs : 
            init_img = ImageHandling.load_np_img(f"{gs1_sfm_path}/images/{init_img_gs1_name}")

        # Get P0 using (intrins of robot cam (gs1))
        masked_points, masked_colors, all_moge_points, all_moge_colors, moge_mask, moge_intrins = GaussiansHandling.get_moge_points(init_img, fov_x=fov_x_deg, model_reso_lvl=moge_model_reso_lvl, depth_edge_threshold=moge_depth_edge_threshold, moge_model=moge_model, use_fp16_bool=False) 
        MyUtils.save_arrays_to_npy(f"{case_test_path}/all_moge_points1.npy", 0, all_moge_points)
        MyUtils.save_arrays_to_npy(f"{case_test_path}/all_moge_colors1.npy", 0, all_moge_colors)
        MyUtils.save_arrays_to_npy(f"{case_test_path}/moge_mask1.npy", 0, moge_mask)

        # Saving  all our intrinsics
        fx_moge = moge_intrins[0, 0].item() * CAM_W
        fy_moge = moge_intrins[1, 1].item() * CAM_H
        cx_moge = moge_intrins[0, 2].item() * CAM_W
        cy_moge = moge_intrins[1, 2].item() * CAM_H
        intrins_line = f"fx_gs1 {fx_gs1} , fy_gs1 {fy_gs1}, cx_gs1 {cx_gs1}, cy_gs1 {cy_gs1} \nfx_moge {fx_moge} , fy_moge {fy_moge}, cx_moge {cx_moge}, cy_moge {cy_moge} \no3d intrins used are same as gs1 ones"
        print(intrins_line)
        MyUtils.add_line_to_text(configs_path, "Intrinsics______")
        MyUtils.add_line_to_text(configs_path, intrins_line)

        # Saving P0
        moge_points_o3d = MeshHandling.turn_points_to_o3d(masked_points, masked_colors)
        MeshHandling.save_o3dpcd(moge_points_o3d, moge_path) 

        # Render I*0 from interactive visualizer using robot_intrinsics (gs1), after getting moge_des_pose
        if trans_exist :
            moge_des_pose = load_frst_dict_np_elmnt(init_des_trans_npy_path)
        else :
            moge_des_pose = mesh_handling.get_cam_pose_from_mesh_view([moge_points_o3d])
        des_estim_img, depth = mesh_handling.render_mesh_pic([moge_points_o3d], moge_des_pose)
        ImageHandling.save_img(des_estim_img, "des0", des_imgs_path)
        ImageHandling.plot_2_imgs(init_img, des_estim_img)    
        
        # Save des0_mask and moge_des_pose
        depth = np.asarray(depth)
        mask = np.isfinite(depth) & (depth > 0) 
        MyUtils.save_arrays_to_npy(des_masks_path, 0, mask)
        MyUtils.save_arrays_to_npy(init_des_trans_npy_path, 0, moge_des_pose)


    if not des0_aligned :
        # copy gs1_sfm and align des1 with it
        MyUtils.copy_any(gs1_sfm_path, gs1_sfm_aligned_path, overwrite=True)
        PosesHandling.align_new_image(f"{des_imgs_path}/des0.png", gs1_sfm_aligned_path, sequential=False)
    

    if not des_GT_ready :
        # Render nd save des_GT  
        recons1_align = PosesHandling.get_recons(gs1_sfm_aligned_path)
        gt_des_pose = PosesHandling.get_img_sfm_pose(recons1_align, "des0.png")
        gt_des_img, _ = GaussiansHandling.render_gs_pic(*gaussians1, gt_des_pose, intrins_gs1, CAM_W, CAM_H)
        des_estim_img = ImageHandling.load_np_img(f"{des_imgs_path}/des0.png")
        ImageHandling.save_img(gt_des_img, "GT_des", des_imgs_path)
        print("GT des saved !")
        ImageHandling.plot_2_imgs(des_estim_img, gt_des_img)

        # Getting des0 mask & save masked GT
        des_masks_dict = np.load(des_masks_path, allow_pickle=True).item()
        des0_mask = des_masks_dict[0]
        gt_des_img_masked = gt_des_img.copy()
        gt_des_img_masked[~des0_mask] = 0
        ImageHandling.save_img(gt_des_img_masked, "gt_des_img_masked", des_imgs_path)

    if not ibvs_desGT_converge :
        gt_des_img = ImageHandling.load_np_img(f"{des_imgs_path}/GT_des.png") 
        dyna_ibvs(gaussians1, intrins_gs1, init_img_pose_gs1, gt_des_img, desGT_ibvs_infos_npy_path)


    if not ibvs_des0_converge :
        des0_img = ImageHandling.load_np_img(f"{des_imgs_path}/des0.png") 
        dyna_ibvs(gaussians1, intrins_gs1, init_img_pose_gs1, des0_img, des0_ibvs_infos_npy_path)


    #______________________________________________________________


    if ready_for_dualoop :

        # Start multi-process elements // We can use fork or spawn (fork is faster and works on ubuntu)
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        shared = manager.dict()
        shared["stop"] = False
        kf_queue = ctx.Queue()
        des_img_queue = ctx.Queue()
        kf_event = ctx.Event()
        des_img_event = ctx.Event()

        # Get I*0 to start ibvs, init_frame for ibvs next frame selection
        des_img = ImageHandling.load_np_img(f"{des_imgs_path}/des0.png")
        init_img = ImageHandling.load_np_img(f"{keyframes_path}/init_img.png")
        
        # Get init pose in gs1 for ibvs start, and des poses in gs1 for GT
        gs1_recon = pycolmap.Reconstruction(f"{gs1_sfm_aligned_path}/sparse/0")
        init_pose_gs1 = PosesHandling.get_img_sfm_pose(gs1_recon, init_img_name_sfm1)
        gt_des_pose_gs1 = PosesHandling.get_img_sfm_pose(gs1_recon,gt_des_name)

        # Get des_moge_transf
        moge_des_trans = load_frst_dict_np_elmnt(init_des_trans_npy_path)

        # Load clouds1 infos
        all_moge_points1 = load_frst_dict_np_elmnt(f"{case_test_path}/all_moge_points1.npy")
        all_moge_colors1 = load_frst_dict_np_elmnt(f"{case_test_path}/all_moge_colors1.npy")
        moge_mask1 = load_frst_dict_np_elmnt(f"{case_test_path}/moge_mask1.npy")

        # Starting the ibvs and gs loops after getting all the infos we need
        p1 = ctx.Process(target=dyna_ibvs_loop, args=(shared, kf_queue, des_img_queue, kf_event, gs1_ply_path, init_pose_gs1, gt_des_pose_gs1, init_img, des_img, fx_gs1, fy_gs1))
        p2 = ctx.Process(target=local_gs_loop, args=(shared, kf_queue, des_img_queue, des_img_event, all_moge_points1, all_moge_colors1, moge_mask1, init_pose_gs1, moge_des_trans, fx_gs1, fy_gs1, init_img))
        p1.start()
        p2.start()
        p1.join()
        p2.join()

    return








    
if __name__ == "__main__":
    main()












