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
from utils.incremental_tsdf import IncrementalTSDFMesher

# Add accelerated_features to our Python paths , so that when featx script gets executed it xill know where to find the modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(project_root, "accelerated_features"))
from modules.xfeat import XFeat 



#_________________________________________________________________________



# Main params
scene_name = "livingroom"
case_nbr = 1
CAM_W, CAM_H = 1557, 1038 
init_img_gs1_name = init_img_name_sfm1 = "DSCF4816.JPG"    
des_img_name_sfm1 = des_img_name_sfm_gs2 = gt_des_name = "des0.png"
diff_lvl = 1

# IBVS Vars
lambda_gain = 0.1
dt_ibvs = 0.03
ibvs_nbr_features = 10
max_ibvs_nbr_itrs_init = 5000
ibvs_pxl_error_conv = 0.02
max_ibvs_nbr_itrs = 100
# Kf vars
kf_motion_ratio = 0.03    # 3% of screen
kf_motion_nbr_features = 100
ibvs_pxl_error_kf = 0.05 # Last kf to take in consid
scale = 0.4
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
parent_case_test_path = f"{main_test_path}/{scene_name}/{scene_name}_case{case_nbr}"
# GS1 paths
gs1_sfm_path = f"{main_test_path}/{scene_name}/real_scene/sfm_{scene_name}"
gs1_ply_path = f"{main_test_path}/{scene_name}/real_scene/{scene_name}.ply"
gs1_sfm_aligned_path = f"{parent_case_test_path}/gs1_sfm_aligned"
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
des_masks_path = f"{case_test_path}/desired_imgs/des_masks.npy"
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
did_start = True
trans_exist = True
des0_ready = True
des0_aligned = True 
des_GT_ready = True
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




 
def log_to_table(message: str, source: str, dt: float, file_path = None):

    if file_path is None:
        file_path = results_txt_file


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
















def dyna_ibvs(gaussians, intrins_gs1, init_pose_gs1, des_img, des_ibvs_infos_npy_path, gt_matches_frames_path) :

    try: 
        ibvs_tools = IbvsTools(CAM_W, CAM_H)    
        cur_pose_gs1 = init_pose_gs1
        norm_of_error = None
        i = 0

        #des_mask = load_frst_dict_np_elmnt(des_masks_path)
        des_mask = MyUtils.get_dummy_mask(CAM_W, CAM_H)

        for i in range(max_ibvs_nbr_itrs_init) :    

            # Render new cur_gs_pic and get its depthmap
            cur_gs1_img, cur_gs1_depth_map = GaussiansHandling.render_gs_pic(*gaussians, T=cur_pose_gs1, K=intrins_gs1, W=CAM_W, H=CAM_H)

            matches_cur, matches_des = ibvs_tools.filtered_matching(cur_gs1_img, des_img, ibvs_nbr_features, mask1=None, mask2=None)

            if len(matches_cur) < 4:
                raise Exception(
                    f"Only {len(matches_cur)} matches are available.")

            # Important: pixel indices must be integers before depth-map indexing.
            matches_cur = matches_cur.astype(np.int32)
            matches_des = matches_des.astype(np.int32)

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


            if (i % 20) == 0 :
                cur_mtch_gs_img = ImageHandling.draw_matches(matches_cur, matches_des, cur_gs1_img, des_img)
                ImageHandling.save_img(cur_mtch_gs_img, i, gt_matches_frames_path)   
                MyUtils.cleanup()



    except KeyboardInterrupt:
            print("\nCtrl+C detected, exiting loop cleanly.")
            MyUtils.cleanup()

    
    
    




def type_configs_on_txt(config_txt_path):
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








def get_matched_features(kpts_img1, desc_img1, kpts_img2, desc_img2, fx_gs1, fy_gs1, nbr_matches) :

    idxs0, idxs1 = xfeat.match(desc_img1, desc_img2)
    matches_img1 = kpts_img1[idxs0].detach().cpu().numpy()
    matches_img2 = kpts_img2[idxs1].detach().cpu().numpy()

    ibvs_tools = IbvsTools(CAM_W, CAM_H, fx_gs1, fy_gs1)
    matches_img1, matches_img2 = ibvs_tools.filter_matches_ransac(matches_img1, matches_img2, ransac_threshold=2.0, confidence=0.999)

    matches_img1, matches_img2 = IbvsTools.select_distributed_matches(
        matches_img1,
        matches_img2,
        CAM_W,
        CAM_H,
        top_x=nbr_matches)
    
    return matches_img1, matches_img2






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









def local_gs_loop(shared, kf_queue, des_img_queue, des_img_event, moge_points1_cmfrm, moge_colors1, moge_mask1, init_pose_gs1, moge_des_trans, fx_gs1, fy_gs1, init_img, new_case_test_path) :

    import queue
    try :

        # update again the paths inside this process, cuz this parallel process creates its own environement 
        # and didn't take into consid the changes that were done in the global_pahts (it's for an old envirnmnt)
        update_case_paths(new_case_test_path)

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

        K_intrs = PosesHandling.get_K(fx_gs1, fy_gs1, (CAM_W/2.0) ,(CAM_H/2.0))

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

                    print("🟢🟢🟢🟢🟢🟢🟢 Optimizing scale______________"); _t = time.time()

                    # Turning the mogecloud to wf and getting o3d
                    moge_points1 = LinAlgeb.transform_points_to_world(moge_points1_cmfrm, init_pose_gs1)
                    moge_points_o3d1 = MeshHandling.turn_points_to_o3d(moge_points1, moge_colors1)

                    # Launch optimizer                    
                    optimizer = ScaleOptimizer(case_test_path)
                    #best_scale, corresp_error = optimizer.optimize(curr_keyframe_pose_gs1, init_pose_gs1, mesh_handling, moge_points_o3d1, curr_keyframe)
                    init_kf_moge_trans = LinAlgeb.get_relative_transfo_c2w(init_pose_gs1, curr_keyframe_pose_gs1)
                    best_scale = optimizer.optimize2(init_img, curr_keyframe, moge_points1_cmfrm, moge_mask1, ibvs_tools, K_intrs, init_kf_moge_trans)
                    scene_scale = best_scale
                    MyUtils.add_line_to_text(configs_path, f"Scale = {scene_scale}")
                    msg = f"🟢{i}-Optimized scale using KF_{kf_vrsn}"; dt = time.time()-_t; print(msg); log_to_table(msg, 'gs', dt); _t = time.time()
                    #scene_scale = 0.3865
                    
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

                # Scale T_kfi and add to colmap
                curr_keyframe_pose_gs2 = curr_keyframe_pose_gs1.copy()
                curr_keyframe_pose_gs2[:3, 3] *= scene_scale
                PosesHandling.add_imgs_to_colmap([kf_vrsn], [f"keyframe{kf_vrsn}.png"], [curr_keyframe_pose_gs2], f"{sfm_path}/sparse/")
                ImageHandling.save_img(curr_keyframe, f"keyframe{kf_vrsn}", f"{sfm_path}/images")
                msg = f"🟢{i}-Scale Kf_{kf_vrsn} and add it to colmap"; dt = time.time()-_t; print(msg); log_to_table(msg, 'gs', dt); _t = time.time()

                if shared["stop"]:
                    break

                # Get Pi and transform it
                masked_points, masked_colors, all_moge_points, all_moge_colors, moge_mask, moge_intrins = GaussiansHandling.get_moge_points(
                    curr_keyframe, fov_x=fov_x_deg, model_reso_lvl=moge_model_reso_lvl, depth_edge_threshold=moge_depth_edge_threshold, moge_model=moge_model, use_fp16_bool=False) 
                all_moge_points = LinAlgeb.transform_points_to_world(all_moge_points, curr_keyframe_pose_gs2)
                msg = f"🟢{i}-Get Moge cloud of Kf_{kf_vrsn}"; dt = time.time()-_t; print(msg); log_to_table(msg, 'gs', dt); _t = time.time()


                # Align Pi with P0 : ftr matching -> getting corresp 3ds -> umeyama 
                matches_img1, matches_kf = ibvs_tools.filtered_matching(init_img, curr_keyframe, max_matches = kf_motion_nbr_features, mask1 = None, mask2 = None) #both maks to None cuz both kf nd init are frm gs1 with no depthedges 
                matched_3d_points_img1, matched_3d_points_kf = MeshHandling.get_corresp_3d_point_matches(matches_img1, matches_kf, moge_mask1, moge_mask, moge_points1, all_moge_points)        
                _, s, R, t = MeshHandling.align_points(matched_3d_points_kf, matched_3d_points_img1) # first cloud in arg is the one to change
                msg = f"🟢{i}-Match and align Kf_{kf_vrsn} with I0 P0"; dt = time.time()-_t; print(msg); log_to_table(msg, 'gs', dt); _t = time.time()

                if shared["stop"]:
                    break

                # Filter Pi (new txtrd pixels) ________________________________________________________

                # step0: load last_local_gs for filtering 
                if init : # m forced to load gs2_0_pre before training it to gs2_0
                    last_gaussians = GaussiansHandling.load_gaussians_from_ply(f"{gs2s_dir_path}/gs2_0_pre.ply") 
                else :
                    last_gaussians = GaussiansHandling.load_gaussians_from_ply(f"{gs2s_dir_path}/gs2_{i}.ply")

                if shared["stop"]:
                    break

                # step1: Render Ii from last_gs or from P0 (if I1), and 
                if init :
                    kf_render, _ = mesh_handling.render_mesh_pic([moge_points_o3d1], curr_keyframe_pose_gs2)
                else : 
                    kf_render, _ = GaussiansHandling.render_gs_pic(*last_gaussians, curr_keyframe_pose_gs2, intrins_gs2, CAM_W, CAM_H)

                if shared["stop"]:
                    break

                # get mask of new txtr pixels
                _, kf_render_txtred = ImageHandling.compute_texturemap_and_mask(kf_render, threshold=0.01)
                txtr_mask = kf_txtred & (~kf_render_txtred)
                final_mask = moge_mask & txtr_mask # moge mask to make sure the pixels have a corresponding moge point

                if shared["stop"]:
                    break

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

                if shared["stop"]:
                    break

                # Turn filtered points to gaussians and merge it with last gaussians
                new_gaussians = GaussiansHandling.turn_points_to_gaussians(filtered_moge_points, filtered_moge_colors, scale=0.007)
                total_gaussians = GaussiansHandling.merge_2_gaussians(last_gaussians, new_gaussians)

                if shared["stop"]:
                    break

                # Save these gaussians to _pre.ply then train it for 50 itrs, then copy output in gs2
                GaussiansHandling.turn_gaussians_to_ply(total_gaussians, f"{gs2s_dir_path}/gs2_{i}_pre.ply")
                msg = f"🟢{i}-Filter cloud of Kf_{kf_vrsn} and add to gs2"; dt = time.time()-_t; print(msg); log_to_table(msg, 'gs', dt); _t = time.time()
                init = False

                if shared["stop"]:
                    break

                
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

                if shared["stop"]:
                    break

                # train last gs2 and copy the inria saved pointcloud to gs2s 
                GaussiansHandling.run_gs_training(sfm_path=sfm_path,
                    output_path=f"{gs2s_dir_path}/inria_output", gs_reso=gs_reso,
                    gs_nbr_itrs=gs_nbr_itrs, curr_case_gs2_path=gs2s_dir_path)
                shutil.copy2(inria_saved_ply_path, f"{gs2s_dir_path}/gs2_{i}.ply")
                msg = f"🟢{i}-Trained local_gs for {gs_nbr_itrs} iterations"; dt = time.time()-_t; _t = time.time()
                print(msg); log_to_table(msg, 'gs', dt)

                if shared["stop"]:
                    break

                # Load last gs2 & render new des img & notify ibvs loop
                last_gaussians2 = GaussiansHandling.load_gaussians_from_ply(f"{gs2s_dir_path}/gs2_{i}.ply")

                # I we reached the last kf, i will update the des pose (dvs before rendering it), this only happens once with lastkf and then we set it again to false 
                if is_last_kf :
                    des0_img = ImageHandling.load_np_img(f"{des_imgs_path}/des0.png")
                    des0_mask = load_frst_dict_np_elmnt(des_masks_path)
                    des_pose_gs2 = ibvs_tools.start_dvs_loop(last_gaussians2, intrins_gs2, des_pose_gs2, des0_img, case_test_path, des0_mask, max_itrs=125)
                    is_last_kf = False

                if shared["stop"]:
                    break

                des_img, des_depth = GaussiansHandling.render_gs_pic(*last_gaussians2, des_pose_gs2, intrins_gs2, CAM_W, CAM_H)

                if shared["stop"]:
                    break

                des_mask = ImageHandling.get_mask_frm_depth(des_depth)
                ImageHandling.save_img(des_img, f"des{i}", des_imgs_path)

                if shared["stop"]:
                    break

                des_img_queue.put([des_img, des_mask]); des_img_event.set()
                msg = f"🟢🟢{i}-Loading local_gs & Rendering new des_img({i})"; dt = time.time()-_t
                print(msg); log_to_table(msg, 'gs', dt); _t = time.time()

                if shared["stop"]:
                    break

                # Saving infos about des_img
                MyUtils.save_arrays_to_npy(des_depths_npy_path, i, des_depth)


    except Exception as error:
        error_traceback = traceback.format_exc()
        print(f"Error in IBVS loop: {error}")
        print(error_traceback)
        shared["stop"] = True
        save_case_error(new_case_test_path, "dyna_ibvs_loop", error_traceback)
        MyUtils.cleanup()
        raise







def dyna_ibvs_loop(shared, kf_queue, des_img_queue, kf_event, gs1_ply_path, init_pose_gs1, gt_des_pose_gs1, init_img, des_img, fx_gs1, fy_gs1, new_case_test_path) :

    try:

        # update again the paths inside this process, cuz this parallel process creates its own environement 
        # and didn't take into consid the changes that were done in the global_pahts (it's for an old envirnmnt)
        update_case_paths(new_case_test_path)

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
                des_img_infos = des_img_queue.get()
                des_img = des_img_infos[0]
                des_mask = des_img_infos[1]
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
            cur_kpts, cur_desc = ibvs_tools.get_xfeat_kpts(cur_gs1_img)
            idxs0, idxs1 = xfeat.match(cur_desc, des_desc)
            matches_cur = cur_kpts[idxs0].detach().cpu().numpy()
            matches_des = des_kpts[idxs1].detach().cpu().numpy()

            safe_indices = ibvs_tools.border_safe_indices(matches_cur, matches_des, mask1=None, mask2=des_mask)

            matches_cur = matches_cur[safe_indices]
            matches_des = matches_des[safe_indices]

            matches_cur, matches_des = ibvs_tools.filter_matches_ransac(matches_cur, matches_des, ransac_threshold=2.0, confidence=0.999)

            matches_cur, matches_des = IbvsTools.select_distributed_matches(
                matches_cur,
                matches_des,
                CAM_W,
                CAM_H,
                top_x=ibvs_nbr_features)

            matches_cur = matches_cur.astype(np.int32)
            matches_des = matches_des.astype(np.int32)

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


            # Check if we need to send a final kf
            if (pxl_error < ibvs_pxl_error_kf) and (not final_kf_sent) :
                last_keyframe = cur_gs1_img
                kf_queue.put([last_keyframe, cur_pose_gs1, True])
                kf_event.set()
                final_kf_sent = True
                msg = f"🔵🔵{i}-Last kf found"
                print(msg); log_to_table(msg, 'ibvs', 0)


            # 4 - Get the intr matrix & its pseudo_inv & calc V with control law
            L = ibvs_tools.get_interaction_matrix(len(matches_cur), Ss_cur, Ss_Z_cur, 1)
            L_psinv = get_inter_mat_pseudo_inverse(L)
            V = - lambda_gain * (L_psinv @ errors)
 
            # 5 - Update cur_cam_pose and update visualization
            cur_pose_gs1 = ibvs_tools.update_cam_pose(cur_pose_gs1, V, dt_ibvs)

            # Check if we converged
            if pxl_error < ibvs_pxl_error_conv :
                shared["stop"] = True


            # Save the infos of each ibvs iteration : 2d_err, 3d_err, condit_nbr, V, matches, homog_pose_in_gs1
            cond_nbr = LinAlgeb.get_mat_condition_number(L)
            ibvs_infos = [pxl_error, pose_error, cond_nbr, V, [matches_cur, matches_des],cur_pose_gs1]
            MyUtils.save_arrays_to_npy(ibvs_infos_npy_path, i, ibvs_infos)

            if (i % 50) == 0 :
                print(msg); log_to_table(msg, 'ibvs', dt)
                cur_mtch_gs_img = ImageHandling.draw_matches(matches_cur, matches_des, cur_gs1_img, des_img)
                ImageHandling.save_img(cur_gs1_img, i, real_frames_path)
                ImageHandling.save_img(cur_mtch_gs_img, i, matches_frames_path)   
                MyUtils.save_arrays_to_npy(ibvs_frame_depths_npy_path, i, cur_gs1_depth_map) 
                MyUtils.cleanup()

    except Exception as error:
        error_traceback = traceback.format_exc()
        print(f"Error in GS loop: {error}")
        print(error_traceback)
        shared["stop"] = True
        save_case_error(new_case_test_path, "local_gs_loop", error_traceback)
        MyUtils.cleanup()
        raise













# the dense cloud only (representation)
def local_cloud_loop(shared, kf_queue, des_img_queue, des_img_event, moge_points1_cmfrm, moge_colors1, moge_mask1, init_pose_gs1, moge_des_trans, fx_gs1, fy_gs1, init_img, new_case_test_path) :

    import queue
    try :

        curr_keyframe = None
        init_pose_gs2 = None
        des_pose_gs2 = None
        scene_scale = None
        accumulated_moge_points = accumulated_moge_colors = accum_moge_points_o3d = None
        init = True
        i = 0
        kf_vrsn = 0
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


        # update again the paths inside this process 
        # cuz this parallel process creates its own environement and didn't take into consid the changes that were done in the global_pahts (it's for an old envirnmnt)
        update_case_paths(new_case_test_path)

        
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
                i+=1

                # (this will happen only the first time we get a kf)
                if init :

                    print("🟢 Optimizing scale______________"); _t = time.time()

                    # Turning the mogecloud to wf and get its o3d
                    moge_points1 = LinAlgeb.transform_points_to_world(moge_points1_cmfrm, init_pose_gs1)
                    moge_points_o3d1 = MeshHandling.turn_points_to_o3d(moge_points1, moge_colors1)

                    
                    # Launch optimizer   
                                   
                    optimizer = ScaleOptimizer(case_test_path)
                    best_scale, corresp_error = optimizer.optimize(curr_keyframe_pose_gs1, init_pose_gs1, mesh_handling, moge_points_o3d1, curr_keyframe)
                    scene_scale = best_scale
                    MyUtils.add_line_to_text(configs_path, f"Scale = {scene_scale}")
                    msg = f"🟢{i}-Optimized scale using KF_{kf_vrsn}"; dt = time.time()-_t; print(msg); log_to_table(msg, 'gs', dt); _t = time.time()
                    
                    #scene_scale = scale

                    # Scale the init pose of gs1 to get the right pose in gs2
                    init_pose_gs2 = init_pose_gs1.copy()
                    init_pose_gs2[:3, 3] *= scene_scale

                    # Transform P0 to the new pose
                    moge_points1 = LinAlgeb.transform_points_to_world(moge_points1_cmfrm, init_pose_gs2)
                    moge_points_o3d1 = MeshHandling.turn_points_to_o3d(moge_points1, moge_colors1)

                    # init the accum_3dpoints which are in gs2 frame ofc
                    accumulated_moge_points = moge_points1.reshape(-1, 3)
                    accumulated_moge_colors = moge_colors1.reshape(-1, 3)
                    # apply mask on em 
                    moge_mask1_flat = moge_mask1.reshape(-1)
                    accumulated_moge_points = accumulated_moge_points[moge_mask1_flat]
                    accumulated_moge_colors = accumulated_moge_colors[moge_mask1_flat]
                                                    
                    # Turn the desired pose Td to the frame of gs2 (scaled gs1)
                    des_pose_gs2 = LinAlgeb.transform_pose(moge_des_trans, init_pose_gs2)

                    # Get I0's Kpnts
                    kpts_img1, desc_img1 =ibvs_tools.get_xfeat_kpts(init_img)


                # Get Kf's kpnts
                kpts_kf, desc_kf = ibvs_tools.get_xfeat_kpts(curr_keyframe)

                # Scale the KF pose and put it in gs2 frame
                curr_keyframe_pose_gs2 = curr_keyframe_pose_gs1.copy()
                curr_keyframe_pose_gs2[:3, 3] *= scene_scale

                # Get the Kf's Pi and put it in frame
                masked_points, masked_colors, all_moge_points, all_moge_colors, moge_mask, moge_intrins = GaussiansHandling.get_moge_points(
                    curr_keyframe, fov_x=fov_x_deg, model_reso_lvl=moge_model_reso_lvl, depth_edge_threshold=moge_depth_edge_threshold, moge_model=moge_model, use_fp16_bool=False) 
                all_moge_points = LinAlgeb.transform_points_to_world(all_moge_points, curr_keyframe_pose_gs2)
                all_moge_points_o3d = MeshHandling.turn_points_to_o3d(all_moge_points, all_moge_colors)
                msg = f"🟢{i}-Get Moge cloud of Kf_{kf_vrsn}"; dt = time.time()-_t; print(msg); log_to_table(msg, 'gs', dt); _t = time.time()


                # Align Pi with P0 : ftr matching -> getting corresp 3ds -> umeyama 
                """                
                matches_img1, matches_kf = get_matched_features(kpts_img1, desc_img1, kpts_kf, desc_kf, nbr_matches=50)
                matched_3d_points_img1, matched_3d_points_kf = get_corresp_3d_point_matches(
                    matches_img1, matches_kf, moge_mask1, moge_mask, moge_points1, all_moge_points)        
                _, s, R, t = MeshHandling.align_points(matched_3d_points_kf, matched_3d_points_img1) # first cloud in arg is the one to change
                msg = f"🟢{i}-Match and align Kf_{kf_vrsn} with I0 P0"; dt = time.time()-_t; print(msg); log_to_table(msg, 'gs', dt); _t = time.time()

                matched_img = ImageHandling.draw_matches(matches_img1, matches_kf, init_img, curr_keyframe)
                ImageHandling.save_img(matched_img, f"matches_{i}", keyframes_path)
                """

                # Filter Pi to keep the new points only and add to cloud_gs2 ________________________________________________________               
                accum_moge_points_o3d = MeshHandling.turn_points_to_o3d(accumulated_moge_points, accumulated_moge_colors)
                kf_render, kf_render_depth = mesh_handling.render_mesh_pic([accum_moge_points_o3d], curr_keyframe_pose_gs2)

                # get mask of new txtr pixels
                existing_region_mask = ImageHandling.get_mask_frm_depth(kf_render_depth)
                new_region_mask = ~existing_region_mask

                """
                R, t, = MeshHandling.align_clouds_icp(
                    source_cloud=overlap_cloud_o3d,
                    target_cloud=moge_points_o3d1,
                    max_correspondence_distance=0.05,
                    max_iterations=50,
                    voxel_size=0.02)
                """

                # Filter Pi (keep new  pixels points)  
                filtered_moge_points, filtered_moge_colors = filter_points(all_moge_points, all_moge_colors, new_region_mask)


                # Align filtered Pi with P0
                #filtered_moge_points = (s * (R @ filtered_moge_points.T)).T + t
                
                accumulated_moge_points = np.concatenate((accumulated_moge_points, filtered_moge_points), axis=0)
                accumulated_moge_colors = np.concatenate((accumulated_moge_colors, filtered_moge_colors), axis=0)  
                accum_moge_points_o3d = MeshHandling.turn_points_to_o3d(accumulated_moge_points, accumulated_moge_colors) 
                init = False              

                # If we reached the last kf, i will update the des pose (dvs before rendering it), this only happens once with lastkf and then we set it again to false 
                if is_last_kf :
                    des0_img = ImageHandling.load_np_img(f"{des_imgs_path}/des0.png")
                    des0_mask = load_frst_dict_np_elmnt(des_masks_path)
                    des_pose_gs2 = ibvs_tools.start_dvs_loop(accum_moge_points_o3d, des_pose_gs2, des0_img, case_test_path, mesh_handling, des0_mask, max_itrs=125)
                    is_last_kf = False

                des_img, des_depth = mesh_handling.render_mesh_pic([accum_moge_points_o3d], des_pose_gs2)
                des_mask = ImageHandling.get_mask_frm_depth(des_depth)
                ImageHandling.save_img(des_img, f"des{i}", des_imgs_path)
                des_img_queue.put([des_img, des_mask]); des_img_event.set()
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










# the incrmntl mesh (representation)
def local_mesh_loop(shared, kf_queue, des_img_queue, des_img_event, moge_points1_cmfrm, moge_colors1, moge_mask1, init_pose_gs1, moge_des_trans, fx_gs1, fy_gs1, init_img, intrins_K, new_case_test_path) :

    import queue
    try :

        curr_keyframe = None
        init_pose_gs2 = None
        des_pose_gs2 = None
        scene_scale = None
        current_mesh = None
        init = True
        i = 0
        kf_vrsn = 0
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


        # Initialize the TSD_mesher
        import torch
        print(
            f"PyTorch allocated: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(
            f"PyTorch reserved: "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB")


        mesher = IncrementalTSDFMesher(
            intrinsic=intrins_K,
            voxel_size=0.01,
            depth_min=0.05,
            depth_max=10.0,
            block_resolution=16,
            block_count=5000,
            device="CPU:0",
            pose_is_camera_to_world=True,
        )


        # update again the paths inside this process 
        # cuz this parallel process creates its own environement and didn't take into consid the changes that were done in the global_pahts (it's for an old envirnmnt)
        update_case_paths(new_case_test_path)

        while not shared["stop"]:            
            curr_keyframe = None

            # if kf_queue is empty we skip this block that loads keyframes
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
                i+=1

                # (this will happen only the first time we get a kf)
                if init :

                    print("🟢 Optimizing scale______________"); _t = time.time()

                    # Turning the mogecloud to wf and get its o3d
                    moge_points1 = LinAlgeb.transform_points_to_world(moge_points1_cmfrm, init_pose_gs1)
                    moge_points_o3d1 = MeshHandling.turn_points_to_o3d(moge_points1, moge_colors1)

                    # Launch optimizer             
                    
                    optimizer = ScaleOptimizer(case_test_path)
                    best_scale, corresp_error = optimizer.optimize(curr_keyframe_pose_gs1, init_pose_gs1, mesh_handling, moge_points_o3d1, curr_keyframe)
                    scene_scale = best_scale
                    MyUtils.add_line_to_text(configs_path, f"Scale = {scene_scale}")
                    msg = f"🟢{i}-Optimized scale using KF_{kf_vrsn}"; dt = time.time()-_t; print(msg); log_to_table(msg, 'gs', dt); _t = time.time()
                    
                    #scene_scale = scale

                    # Scale the init pose of gs1 to get the right pose in gs2
                    init_pose_gs2 = init_pose_gs1.copy()
                    init_pose_gs2[:3, 3] *= scene_scale

                    # Transform P0 to the new pose
                    moge_points1 = LinAlgeb.transform_points_to_world(moge_points1_cmfrm, init_pose_gs2)
                    moge_points_o3d1 = MeshHandling.turn_points_to_o3d(moge_points1, moge_colors1)

                    # init the accum_3dpoints which are in gs2 frame ofc
                    accumulated_moge_points = moge_points1.reshape(-1, 3)
                    accumulated_moge_colors = moge_colors1.reshape(-1, 3)
                    # apply mask on em 
                    moge_mask1_flat = moge_mask1.reshape(-1)
                    accumulated_moge_points = accumulated_moge_points[moge_mask1_flat]
                    accumulated_moge_colors = accumulated_moge_colors[moge_mask1_flat]
                                                    
                    # Turn the desired pose Td to the frame of gs2 (scaled gs1)
                    des_pose_gs2 = LinAlgeb.transform_pose(moge_des_trans, init_pose_gs2)

                    # Update vgb using init img
                    mesher.integrate_frame(image_rgb= init_img, points_cam= moge_points1_cmfrm, moge_mask= moge_mask1, pose= init_pose_gs2)

                    init = False



                # Scale the KF pose and put it in gs2 frame
                curr_keyframe_pose_gs2 = curr_keyframe_pose_gs1.copy()
                curr_keyframe_pose_gs2[:3, 3] *= scene_scale

                # Get the Kf's Pi and put it in frame
                masked_points, masked_colors, all_moge_points_cam, all_moge_colors, moge_mask, moge_intrins = GaussiansHandling.get_moge_points(
                    curr_keyframe, fov_x=fov_x_deg, model_reso_lvl=moge_model_reso_lvl, depth_edge_threshold=moge_depth_edge_threshold, moge_model=moge_model, use_fp16_bool=False) 
                all_moge_points = LinAlgeb.transform_points_to_world(all_moge_points_cam, curr_keyframe_pose_gs2)
                msg = f"🟢{i}-Get Moge cloud of Kf_{kf_vrsn}"; dt = time.time()-_t; print(msg); log_to_table(msg, 'gs', dt); _t = time.time()


                #_________________________ Update the vgb using the new KF + extract and update current mesh
                
                
                current_mesh = mesher.extract_mesh(extract_on_cpu=False)
                kf_render, kf_render_depth = mesh_handling.render_mesh_pic_withnoshad([current_mesh], curr_keyframe_pose_gs2)
                #ImageHandling.plot_2_imgs(kf_render, kf_render_depth)

                # get mask of new txtr pixels
                existing_region_mask = ImageHandling.get_mask_frm_depth(kf_render_depth)
                new_region_mask = ~existing_region_mask
                
                mesher.integrate_frame(image_rgb= curr_keyframe, points_cam= all_moge_points_cam, moge_mask= new_region_mask, pose= curr_keyframe_pose_gs2)
                current_mesh = mesher.extract_mesh(extract_on_cpu=False)



                o3d.io.write_triangle_mesh(
                    f"{keyframes_path}/meshi.ply",
                    current_mesh,
                    write_ascii=False,
                    write_vertex_normals=True,
                    write_vertex_colors=True)


                # If we reached the last kf, i will update the des pose (dvs before rendering it), this only happens once with lastkf and then we set it again to false 
                if is_last_kf :
                    des0_img = ImageHandling.load_np_img(f"{des_imgs_path}/des0.png")
                    des0_mask = load_frst_dict_np_elmnt(des_masks_path)
                    des_pose_gs2 = ibvs_tools.start_dvs_loop(current_mesh, des_pose_gs2, des0_img, case_test_path, mesh_handling, des0_mask, max_itrs=125)
                    is_last_kf = False


                if i > 2 : 
                    des_img, des_depth = mesh_handling.render_mesh_pic_withnoshad([current_mesh], des_pose_gs2)
                    des_mask = ImageHandling.get_mask_frm_depth(des_depth)
                    ImageHandling.save_img(des_img, f"des{i}", des_imgs_path)
                    des_img_queue.put([des_img, des_mask]); des_img_event.set()
                    msg = f"🟢🟢{i}-Loading local_gs & Rendering new des_img({i})"; dt = time.time()-_t
                    print(msg); log_to_table(msg, 'gs', dt); _t = time.time()

                    # Saving infos about des_img
                    MyUtils.save_arrays_to_npy(des_depths_npy_path, i, des_depth)

                init = False



    except Exception as e:
        print(f"Error in gs_loop : {e} !!!!!!!!!!!!!!")
        print("[gs] FULL TRACEBACK:")
        traceback.print_exc()
        MyUtils.cleanup()
        shared["stop"] = True





#_______________________________________________________________________________________________________________________________________







def generate_random_des_pose(init_img, all_moge_points, moge_mask, mesh_handling: MeshHandling, init_moge_points_o3d, diff_lvl) :

    """
    if diff_lvl == 1 :
        trans_range = [0.2, 0.4]
        z_rot_range = [-5, 5]
        focal_range = [-0.1, 0.1] 

    elif diff_lvl == 2 :
        trans_range  = [0.4, 0.6]
        z_rot_range = [-10, 10] 
        focal_range = [-0.3, 0.3] 

    else :
        trans_range = [0.6, 0.8]
        z_rot_range = [-15, 15] 
        focal_range = [-0.5, 0.5] 
    """

    if diff_lvl == 1 :
        trans_range  = [0.4, 0.6]
        z_rot_range = [-10, 10] 
        focal_range = [-0.3, 0.3] 

    elif diff_lvl == 2 :
        trans_range = [0.6, 0.8]
        z_rot_range = [-15, 15] 
        focal_range = [-0.5, 0.5] 

    else :
        trans_range = [0.8, 1]
        z_rot_range = [-15, 15] 
        focal_range = [-0.7, 0.7] 


    # Apply a translation of 'entered range'
    trans_init_pose = LinAlgeb.apply_random_translation(trans_range)
    trans_img, _ = mesh_handling.render_mesh_pic([init_moge_points_o3d], trans_init_pose)

    # Make the pose look back again to the focal point (aftr getting it first)
    focal_point = MeshHandling.get_focal_point(all_moge_points, moge_mask)
    trans_init_pose_oldfoc = LinAlgeb.look_at_focal_point(trans_init_pose, focal_point)
    trans_img_oldfoc, _ = mesh_handling.render_mesh_pic([init_moge_points_o3d], trans_init_pose_oldfoc)
    
    # Add noise to focal point and make the cam look at it
    new_focal_point = LinAlgeb.perturb_focal_point(focal_point, focal_range)
    trans_init_pose_newfoc = LinAlgeb.look_at_focal_point(trans_init_pose_oldfoc, new_focal_point)
    trans_img_newfoc, _ = mesh_handling.render_mesh_pic([init_moge_points_o3d], trans_init_pose_newfoc)
    #ImageHandling.plot_4_imgs(init_img, trans_img, trans_img_oldfoc, trans_img_newfoc, "initial img", "translated img", "trans_img_oldfoc", "trans_img_newfoc")

    # Add a random rot around Z
    trans_init_pose_newfoc_rotz = LinAlgeb.apply_random_z_rotation(trans_init_pose_newfoc, z_rot_range)
    trans_img_newfoc_rotz, _ = mesh_handling.render_mesh_pic([init_moge_points_o3d], trans_init_pose_newfoc_rotz)
    #ImageHandling.plot_2_imgs(init_img, trans_img_newfoc_rotz, "init_img", "trans_img_newfoc_rotz")

    return trans_init_pose_newfoc_rotz









def get_lvl_des_paths(case_path, levels=None, tst_nbrs=None, des_type="des0"):
    """
    Return selected desired-image paths.

    Examples
    --------
    levels=1, tst_nbrs=3, des_type="gt_des"
    levels=[1, 2], tst_nbrs=[1, 4, 7], des_type="des0"
    levels=None, tst_nbrs=None -> get everything from levels 1, 2 and 3
    """
    if des_type not in ("des0", "gt_des"):
        raise ValueError("des_type must be 'des0' or 'gt_des'.")

    if levels is None:
        levels = [1, 2, 3]
    elif isinstance(levels, int):
        levels = [levels]

    if isinstance(tst_nbrs, int):
        tst_nbrs = [tst_nbrs]

    des_paths = []

    for diff_lvl in levels:
        level_path = Path(f"{case_path}/all_lvl_des") / f"level_{diff_lvl}"

        if tst_nbrs is None:
            level_paths = sorted(
                level_path.glob(f"*_{des_type}.png"),
                key=lambda path: int(path.stem.split("_")[0])
            )
        else:
            level_paths = [
                level_path / f"{tst_nbr}_{des_type}.png"
                for tst_nbr in tst_nbrs
            ]

        for path in level_paths:
            if path.exists():
                des_paths.append(str(path))
            else:
                print(f"Image not found: {path}")

    return des_paths






#_______________________________________________________________

def get_level_and_case(img_path):
    img_path = Path(img_path)

    diff_lvl = int(img_path.parent.name.split("_")[-1])
    case_nbr = int(img_path.stem.split("_")[0])

    return diff_lvl, case_nbr





#____________________________________________________________________




def prepare_test_case(des_img_path, case_test_path):
    """
    this function expects the input des_path to be inside : 
    case_test_path/all_lvl_des/level_'x'/'case_nbr'_des0.png with :
    'case_nbr'_desmask.npy and 'case_nbr'_des0.png and 'case_nbr'_des_moge_pose.npy

    and creates :
        case_test_path/level_x/case_'case_nbr'/ then inside it copy the corresponding elements
    """
        

    des_img_path = Path(des_img_path)
    case_test_path = Path(case_test_path)

    # Extract level and test number.
    diff_lvl = int(des_img_path.parent.name.split("_")[-1])
    tst_nbr = int(des_img_path.stem.split("_")[0])

    # Source folder:
    # all_lvl_des/level_1/
    source_lvl_path = des_img_path.parent

    # New test folder:
    # case_test_path/level_1/case_1/
    new_case_path = (
        case_test_path /
        f"level_{diff_lvl}" /
        f"case_{tst_nbr}"
    )

    desired_imgs_path = new_case_path / "desired_imgs"
    gs2s_path = new_case_path / "gs2s"
    ibvs_frames_path = new_case_path / "ibvs_frames"
    keyframes_path = new_case_path / "keyframes"
    sfm_path = new_case_path / "sfm"

    # Create all directories.
    for path in [
        desired_imgs_path,
        gs2s_path,
        ibvs_frames_path,
        keyframes_path,
        sfm_path / "images",
        sfm_path / "sparse" / "0"]:
        path.mkdir(parents=True, exist_ok=True)

    # Source files.
    des0_source = source_lvl_path / f"{tst_nbr}_des0.png"
    mask_source = source_lvl_path / f"{tst_nbr}_desmask.npy"
    pose_source = source_lvl_path / f"{tst_nbr}_des_moge_pose.npy"
    gt_des_source = source_lvl_path / f"{tst_nbr}_gt_des.png"
    matches_source = source_lvl_path / f"{tst_nbr}_matches_folder"

    # Copy and rename files.
    shutil.copy2(
        des0_source,
        desired_imgs_path / "des0.png")

    shutil.copy2(
        mask_source,
        desired_imgs_path / "des_masks.npy")

    shutil.copy2(
        pose_source,
        new_case_path / "init_des_trans.npy")

    # Copy the ground-truth desired image too.
    if gt_des_source.exists():
        shutil.copy2(
            gt_des_source,
            desired_imgs_path / "gt_des.png")

    # Copy the complete matches folder.
    if matches_source.exists():
        shutil.copytree(
            matches_source,
            new_case_path / "matches_folder",
            dirs_exist_ok=True)

    print(
        f"Prepared level {diff_lvl}, test {tst_nbr}:\n"
        f"{new_case_path}")

    type_configs_on_txt(configs_path)

    return new_case_path






#_________________________


def update_case_paths(new_case_test_path):

    global case_test_path
    global gs1_sfm_aligned_path
    global des_imgs_path, keyframes_path, sfm_path, gs2s_dir_path
    global pre_matches_frames_path, real_frames_path, matches_frames_path
    global configs_path, results_txt_file
    global moge_path, init_des_trans_npy_path, des_masks_path
    global des0_ibvs_infos_npy_path, desGT_ibvs_infos_npy_path
    global ibvs_infos_npy_path, des_depths_npy_path
    global ibvs_frame_depths_npy_path, inria_saved_ply_path

    case_test_path = str(new_case_test_path)

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

    # Saved elements
    moge_path = f"{case_test_path}/init_moges.ply"
    init_des_trans_npy_path = f"{case_test_path}/init_des_trans.npy"
    des_masks_path = f"{case_test_path}/desired_imgs/des_masks.npy"

    # Saved results
    des0_ibvs_infos_npy_path = f"{case_test_path}/des0_ibvs_infos.npy"
    desGT_ibvs_infos_npy_path = f"{case_test_path}/desGT_ibvs_infos.npy"
    ibvs_infos_npy_path = f"{case_test_path}/ibvs_infos.npy"
    des_depths_npy_path = f"{case_test_path}/des_depths.npy"
    ibvs_frame_depths_npy_path = f"{case_test_path}/ibvs_frame_depths.npy"

    # Gaussian Splatting output
    inria_saved_ply_path = (
        f"{case_test_path}/gs2s/inria_output/"
        f"point_cloud/iteration_{gs_nbr_itrs}/point_cloud.ply"
    )








def save_case_error(case_path, source, error_text):
    error_path = Path(case_path) / "errors.txt"
    error_path.parent.mkdir(parents=True, exist_ok=True)

    with open(error_path, "a") as file:
        file.write("\n" + "=" * 80 + "\n")
        file.write(f"Time: {datetime.now()}\n")
        file.write(f"Source: {source}\n")
        file.write(error_text)
        file.write("\n")








def main() :

    # Load Main compos and elements _______________________________________________________________________________________

    # Loading moge, gs1, gs1_intrins, making the mesh_handling, 
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

    # init img & init pose gs1
    init_img = ImageHandling.load_np_img(f"{gs1_sfm_path}/images/{init_img_gs1_name}")
    init_img_pose_gs1 = PosesHandling.get_img_sfm_pose(recons1, init_img_gs1_name)




    # Generate des imgs from different levels _________________________________________________________________________________

    """
    # Getting P0 (using gs1_intrins)
    
    masked_points, masked_colors, all_moge_points, all_moge_colors, moge_mask, moge_intrins = GaussiansHandling.get_moge_points(
        init_img, fov_x=fov_x_deg, model_reso_lvl=moge_model_reso_lvl, depth_edge_threshold=moge_depth_edge_threshold, moge_model=moge_model, use_fp16_bool=False) 
    init_moge_points_o3d =  MeshHandling.turn_points_to_o3d(masked_points, masked_colors)

    ImageHandling.save_img(init_img, "init_img", case_test_path)
    MeshHandling.save_o3dpcd(init_moge_points_o3d, f"{case_test_path}/init_moge.ply")

    
    for diff_lvl in range(1, 4):          # Levels 1, 2, 3
        lvl_path = Path(f"{case_test_path}/all_lvl_des") / f"level_{diff_lvl}"
        lvl_path.mkdir(parents=True, exist_ok=True)

        for tst_nbr in range(1, 11):      # Tests 1 to 10
            print(f"Generating level {diff_lvl}, test {tst_nbr}")

            moge_des_pose = generate_random_des_pose(
                init_img, all_moge_points,
                moge_mask, mesh_handling,
                init_moge_points_o3d, diff_lvl)

            des_estim_img, depth = mesh_handling.render_mesh_pic([init_moge_points_o3d], moge_des_pose)

            # Save initimg, moge0, des0_mask, moge_des_pose
            ImageHandling.save_img(des_estim_img, f"{tst_nbr}_des0", str(lvl_path))
            des_mask = ImageHandling.get_mask_frm_depth(depth) 
            MyUtils.save_arrays_to_npy(f"{case_test_path}/all_lvl_des/level_{diff_lvl}/{tst_nbr}_desmask.npy", 0, des_mask)
            MyUtils.save_arrays_to_npy(f"{case_test_path}/all_lvl_des/level_{diff_lvl}/{tst_nbr}_des_moge_pose.npy", 0, moge_des_pose)
    """  
       
    
    

    # Align all saved des0 imgs frm all lvls ________________________________________________________

    """
    # Get all estimated desired images.
    des_imgs_paths = get_lvl_des_paths(case_test_path)
    print("des_imgs_paths  : ", des_imgs_paths)
    
    # Give every image a unique name for COLMAP.
    alignment_inputs_path = Path(case_test_path) / "alignment_inputs"
    alignment_inputs_path.mkdir(parents=True, exist_ok=True)

    aligned_names = {}
    unique_des_paths = []

    for des_path in map(Path, des_imgs_paths):
        unique_name = f"{des_path.parent.name}_{des_path.name}"
        unique_path = alignment_inputs_path / unique_name
        shutil.copy2(des_path, unique_path)

        aligned_names[str(des_path)] = unique_name
        unique_des_paths.append(str(unique_path))

    # Align all images.
    MyUtils.copy_any(gs1_sfm_path, gs1_sfm_aligned_path, overwrite=True)
    PosesHandling.align_new_images(unique_des_paths, gs1_sfm_aligned_path, sequential=True)
    """
    




    # After alignement, we render the GT corresps to all des0 and save ______________________________________________________________

    """    
    # Get all estimated desired images.
    des_imgs_paths = get_lvl_des_paths(case_test_path)

    # Recreate the same original-path -> COLMAP-name mapping.
    des_paths = [Path(path) for path in des_imgs_paths]
    aligned_names = {
        str(des_path): f"{des_path.parent.name}_{des_path.name}"
        for des_path in des_paths}

    recons1_align = PosesHandling.get_recons(gs1_sfm_aligned_path)

    # Render and save all ground-truth desired images.
    for des_path in des_paths:
        aligned_name = aligned_names[str(des_path)]
        tst_nbr = des_path.stem.replace("_des0", "")

        try:
            gt_des_pose = PosesHandling.get_img_sfm_pose(recons1_align, aligned_name)
        except Exception as error:
            print(f"Could not get pose for {aligned_name}: {error}")
            continue

        gt_des_img, _ = GaussiansHandling.render_gs_pic(*gaussians1, gt_des_pose, intrins_gs1, CAM_W, CAM_H)
        ImageHandling.save_img(gt_des_img, f"{tst_nbr}_gt_des", str(des_path.parent))

        print(f"Saved {des_path.parent.name}/{tst_nbr}_gt_des.png")
    """
    


    # Apply IBVS on GT des imgs _________________________________________________________________________________________________________
    
    """
    # pick the des i want, make matches folder next to each one     
    gt_des_paths = get_lvl_des_paths(case_test_path, [3], [1, 3, 6], "gt_des")

    for gt_des_path in map(Path, gt_des_paths) :
        gt_des_img = ImageHandling.load_np_img(gt_des_path)

        # make matches_dir, gt_ibvs_path
        diff_lvl = int(gt_des_path.parent.name.split("_")[-1])
        tst_nbr = int(gt_des_path.stem.split("_")[0])
        matches_save_path = gt_des_path.parent / f"{tst_nbr}_matches_folder"
        matches_save_path.mkdir(parents=True, exist_ok=True)
        gt_ibvs_path = gt_des_path.parent / f"{tst_nbr}_gt_ibvs.npy"

        # run dyna ibvs on gt_des 
        print(f"Running level {diff_lvl}, test {tst_nbr}")
        try : 
            dyna_ibvs(gaussians1, intrins_gs1, init_img_pose_gs1, gt_des_img, gt_ibvs_path, matches_save_path)

        except Exception as error:
            print(
                f"\nError in gt_ibvs for level {diff_lvl}, "
                f"test {tst_nbr}: {type(error).__name__}: {error}")
            traceback.print_exc()
            continue
    
    return
    """


    
    # Prepare the case test folders for the clean desired imgs _________________________________________________________________________________
    
    des_imgs_paths = get_lvl_des_paths(case_test_path, [2, 3], [1, 3, 6])
    for des_img_path in des_imgs_paths : 

        try :

            # Change the paths to the new_case_test_path
            diff_lvl, case_nbr = get_level_and_case(des_img_path)
            new_case_test_path = f"{parent_case_test_path}/level_{diff_lvl}/case_{case_nbr}"
            update_case_paths(new_case_test_path)

            # make initial main folders and files
            prepare_test_case(des_img_path, parent_case_test_path)

            # Load init_img from original gs1_sfm (for better quality)
            if not get_des_frm_gs : 
                init_img = ImageHandling.load_np_img(f"{gs1_sfm_path}/images/{init_img_gs1_name}")
        
            # Get P0 using (intrins of robot cam (gs1))
            masked_points, masked_colors, all_moge_points, all_moge_colors, moge_mask, moge_intrins = GaussiansHandling.get_moge_points(init_img, fov_x=fov_x_deg, model_reso_lvl=moge_model_reso_lvl, depth_edge_threshold=moge_depth_edge_threshold, moge_model=moge_model, use_fp16_bool=False) 
            MyUtils.save_arrays_to_npy(f"{parent_case_test_path}/all_moge_points1.npy", 0, all_moge_points)
            MyUtils.save_arrays_to_npy(f"{parent_case_test_path}/all_moge_colors1.npy", 0, all_moge_colors)
            MyUtils.save_arrays_to_npy(f"{parent_case_test_path}/moge_mask1.npy", 0, moge_mask)
        
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
        
            # Load the pose
            moge_des_pose = load_frst_dict_np_elmnt(init_des_trans_npy_path)
            des_estim_img, depth = mesh_handling.render_mesh_pic([moge_points_o3d], moge_des_pose)
                
            #______________________________________________________________

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
            init_img = ImageHandling.load_np_img(f"{parent_case_test_path}/init_img.png")

            # Get init pose in gs1 for ibvs start, and des poses in gs1 for GT
            gs1_recon = pycolmap.Reconstruction(f"{gs1_sfm_aligned_path}/sparse/0")
            init_pose_gs1 = PosesHandling.get_img_sfm_pose(gs1_recon, init_img_name_sfm1)
            gt_des_pose_gs1 = PosesHandling.get_img_sfm_pose(gs1_recon, f"level_{diff_lvl}_{case_nbr}_des0.png")
            intrins_K = PosesHandling.get_cam_matrix(gs1_recon)

            # Get des_moge_transf
            moge_des_trans = load_frst_dict_np_elmnt(init_des_trans_npy_path)

            # Load clouds1 infos
            all_moge_points1 = load_frst_dict_np_elmnt(f"{parent_case_test_path}/all_moge_points1.npy")
            all_moge_colors1 = load_frst_dict_np_elmnt(f"{parent_case_test_path}/all_moge_colors1.npy")
            moge_mask1 = load_frst_dict_np_elmnt(f"{parent_case_test_path}/moge_mask1.npy")

            # Starting the ibvs and gs loops after getting all the infos we need
            p1 = ctx.Process(target=dyna_ibvs_loop, args=(shared, kf_queue, des_img_queue, kf_event, gs1_ply_path, init_pose_gs1, gt_des_pose_gs1, init_img, des_img, fx_gs1, fy_gs1, new_case_test_path))
            p2 = ctx.Process(target=local_gs_loop, args=(shared, kf_queue, des_img_queue, des_img_event, all_moge_points1, all_moge_colors1, moge_mask1, init_pose_gs1, moge_des_trans, fx_gs1, fy_gs1, init_img, new_case_test_path))
            p1.start()
            p2.start()
            p1.join()
            p2.join()


            # Check whether they finished successfully.
            if p1.exitcode != 0 or p2.exitcode != 0:
                message = (
                    f"Child-process failure\n"
                    f"IBVS exit code: {p1.exitcode}\n"
                    f"GS exit code: {p2.exitcode}\n")
                save_case_error( new_case_test_path, "main process", message)
                print(f"Failed level {diff_lvl}, case {case_nbr}: " f"IBVS={p1.exitcode}, GS={p2.exitcode}")
                continue
            print(f"Completed level {diff_lvl}, " f"case {case_nbr} successfully")

        except KeyboardInterrupt:
            print("\nExecution interrupted by the user.")

            if "shared" in locals():
                shared["stop"] = True
            if "p1" in locals() and p1.is_alive():
                p1.terminate()
                p1.join()
            if "p2" in locals() and p2.is_alive():
                p2.terminate()
                p2.join()
            raise


        except Exception:
            error_traceback = traceback.format_exc()
            print(f"Error in main loop for level {diff_lvl}, case {case_nbr}")
            print(error_traceback)
            save_case_error(new_case_test_path, "main iteration", error_traceback)

            # Move to the next desired image.
            continue

        finally:
            if "manager" in locals():
                manager.shutdown()

            

    return




    






    
    #____________________________________________________________________________________________________________________________________________________________
    
    



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
        intrins_K = PosesHandling.get_cam_matrix(gs1_recon)

        # Get des_moge_transf
        moge_des_trans = load_frst_dict_np_elmnt(init_des_trans_npy_path)

        # Load clouds1 infos
        all_moge_points1 = load_frst_dict_np_elmnt(f"{case_test_path}/all_moge_points1.npy")
        all_moge_colors1 = load_frst_dict_np_elmnt(f"{case_test_path}/all_moge_colors1.npy")
        moge_mask1 = load_frst_dict_np_elmnt(f"{case_test_path}/moge_mask1.npy")

        # Starting the ibvs and gs loops after getting all the infos we need
        p1 = ctx.Process(target=dyna_ibvs_loop, args=(shared, kf_queue, des_img_queue, kf_event, gs1_ply_path, init_pose_gs1, gt_des_pose_gs1, init_img, des_img, fx_gs1, fy_gs1))
        p2 = ctx.Process(target=local_mesh_loop, args=(shared, kf_queue, des_img_queue, des_img_event, all_moge_points1, all_moge_colors1, moge_mask1, init_pose_gs1, moge_des_trans, fx_gs1, fy_gs1, init_img, intrins_K))
        p1.start()
        p2.start()
        p1.join()
        p2.join()

    return








    
if __name__ == "__main__":
    main()











