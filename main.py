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
import torch
import cv2
import emoji
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import queue  
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

"""
scene_name = "playroom"
case_nbr = 1
CAM_W, CAM_H = 1264, 832
init_img_gs1_name = init_img_name_sfm1 = "DSC05576.jpg"
"""

#"""
scene_name = "livingroom"
case_nbr = 1
CAM_W, CAM_H = 1557, 1038
init_img_gs1_name = init_img_name_sfm1 = "DSCF4956.JPG"
#"""

"""
scene_name = "thehouse"
case_nbr = 1
CAM_W, CAM_H = 1332, 876
init_img_gs1_name = init_img_name_sfm1 = "IMG_6377.jpg"
"""


des_img_name_sfm1 = des_img_name_sfm_gs2 = gt_des_name = "des0.png"
diff_lvl = 1

# IBVS Vars
lambda_gain = 0.2
dt_ibvs = 0.03
ibvs_nbr_features = 50
max_ibvs_nbr_itrs_init = 5000
ibvs_pxl_error_conv = 0.01
max_ibvs_nbr_itrs = 100

# Kf vars
kf_motion_ratio = 0.03    # 3% of screen
kf_motion_nbr_features = 100
ibvs_pxl_error_kf = pxl_err_to_refi = 0.05 # ibvs err to apply des0 refinement
pxl_err_to_morekf = 0.2 # ibvs err to make kf selection more frequent
scale = 0.4

# GS vars
gs_reso = 2
gs_nbr_itrs = 50
moge_reso = 1

# Moge vars
moge_model_reso_lvl = 1
moge_depth_edge_threshold = 0.02

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
matches_frames_path = f"{case_test_path}/ibvs_frames"
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




def get_xfeat_kpts(img, xfeat_model, max_kpts=1000) :
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)  # (1,3,H,W)
    out = xfeat_model.detectAndCompute(tensor, top_k=max_kpts)[0]
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




 
def log_to_table(message: str, source: str, dt = 0, file_path = None):

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

        #des_mask = MyUtils.load_frst_dict_np_elmnt(des_masks_path)
        des_mask = MyUtils.get_dummy_mask(CAM_W, CAM_H)

        # get des kpts once (des_img doesn't change in this loop)
        xfeat_model = get_xfeat_model()
        des_kpts, des_desc = get_xfeat_kpts(des_img, xfeat_model)
        dists_cur = dists_des = None

        for i in range(max_ibvs_nbr_itrs_init) :    

            # Render new cur_gs_pic and get its depthmap
            cur_gs1_img, cur_gs1_depth_map = GaussiansHandling.render_gs_pic(*gaussians, T=cur_pose_gs1, K=intrins_gs1, W=CAM_W, H=CAM_H)

            # 1 - Match cur_gs1_img with des_img, only each x iterations (reused in between)
        
            cur_kpts, cur_desc = ibvs_tools.get_xfeat_kpts(cur_gs1_img)

            idxs0, idxs1 = xfeat.match(cur_desc, des_desc)

            matches_cur = cur_kpts[idxs0].detach().cpu().numpy()
            matches_des = des_kpts[idxs1].detach().cpu().numpy()

            safe_indices, dists_cur, dists_des = ibvs_tools.border_safe_indices(
                matches_cur, matches_des,
                mask1=None, mask2=None,
                dist1=dists_cur, dist2=dists_des)
            
            matches_cur = matches_cur[safe_indices]
            matches_des = matches_des[safe_indices]

            matches_cur, matches_des = ibvs_tools.filter_matches_ransac(
                matches_cur, matches_des, ransac_threshold=2.0, confidence=0.999)

            matches_cur, matches_des = IbvsTools.select_distributed_matches(
                matches_cur, matches_des, CAM_W, CAM_H, top_x=ibvs_nbr_features)

            # Important: pixel indices must be integers before depth-map indexing.
            matches_cur = matches_cur.astype(np.int32)
            matches_des = matches_des.astype(np.int32)

            if len(matches_cur) < 4:
                raise Exception(
                    f"Only {len(matches_cur)} matches are available.")

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

    idxs0, idxs1 = xfeat.match(desc_img1, desc_img2, min_cossim=0.85)
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







def validate_keyframe(last_kf_kpts, last_kf_desc, cur_kpts, cur_desc, xfeat, ibvs_itr_nbr, kf_motion_nbr_features=100, motion_thresh_ratio=0.03):
    """
    Decide whether the current frame is sufficiently far from the last keyframe.

    Current features are reused from the IBVS matching, avoiding another
    XFeat feature extraction.
    """

    # Match cached last-keyframe descriptors with current descriptors
    idxs_last, idxs_cur = xfeat.match(last_kf_desc, cur_desc, min_cossim=0.835)

    pts_last = last_kf_kpts[idxs_last].detach().cpu().numpy()
    pts_cur = cur_kpts[idxs_cur].detach().cpu().numpy()

    if len(pts_last) < 10:
        return False

    # The returned matches are not necessarily ordered by quality
    if len(pts_last) > kf_motion_nbr_features:
        selected = np.linspace(0, len(pts_last) - 1, kf_motion_nbr_features, dtype=np.int32)
        pts_last = pts_last[selected]
        pts_cur = pts_cur[selected]

    # Robust image-space displacement relative to the last keyframe
    displacements = np.linalg.norm(pts_cur - pts_last, axis=1)
    median_displacement = np.median(displacements)
    motion_ratio = median_displacement / CAM_W

    print(f"🔵-{ibvs_itr_nbr} KF motion: {motion_ratio:.3%}")

    return motion_ratio > motion_thresh_ratio







def local_gs_loop(shared, stop_event, kf_queue, des_img_queue, des_img_event, moge_points1_cmfrm, moge_colors1, moge_mask1, moge_normals1, moge_des_trans, moge_intrins, init_img, new_case_test_path):
    import queue

    try:
        # Update the paths inside this process (a parallel process does not inherit later changes to the globals)
        update_case_paths(new_case_test_path)

        # Init Vars
        gs_nbr_itrs = 150
        inria_saved_ply_path = f"{case_test_path}/gs2s/inria_output/point_cloud/iteration_{gs_nbr_itrs}/point_cloud.ply"
        i = kf_vrsn = last_kf_vrsn = 0
        init = True
        is_last_kf = False

        # cam_init is the origin, so the init pose is identity and the MoGe cam frame IS the world (gs2) frame.
        init_pose_gs2 = LinAlgeb.get_eye_matrix(4)
        des_pose_gs2 = moge_des_trans
        moge_points1 = moge_points1_cmfrm

        # Intrinsics and helper objects
        fx_moge = moge_intrins[0, 0]; fy_moge = moge_intrins[1, 1]; cx_moge = moge_intrins[0, 2]; cy_moge = moge_intrins[1, 2]
        mesh_handling = MeshHandling(CAM_W, CAM_H, fx_moge, fy_moge, cx_moge, cy_moge)
        ibvs_tools = IbvsTools(CAM_W, CAM_H, fx_moge, fy_moge, cx_moge, cy_moge)
        fov_x_rad = 2 * np.arctan(CAM_W / (2 * fx_moge))
        fov_x_deg = np.rad2deg(fov_x_rad)
        intrins_gs2 = None

        # Load the MoGe model once at the beginning.
        _t = time.time()
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda")
        msg = f"🟢{i}-Just Loaded Moge model"; dt = time.time() - _t; print(f"{msg} || {dt:.3f} s"); log_to_table(msg, "gs", dt)

        # Last-kf infos used for PnP pose estimation (init image is the first "last kf")
        last_kf_pose = init_pose_gs2
        last_kf_cloud_cam = moge_points1_cmfrm
        last_keyframe = init_img
        last_kf_moge_mask = moge_mask1

        while not shared["stop"] and not stop_event.is_set():
            curr_keyframe = None

            # Get the latest keyframe (drain the queue, keep the newest)
            while True:
                try:
                    curr_keyframe_list = kf_queue.get(timeout=0.2)
                except queue.Empty:
                    break
                curr_keyframe = curr_keyframe_list[0]
                is_last_kf = is_last_kf or curr_keyframe_list[2]
                kf_vrsn += 1
                i += 1
                msg = f"🟢🟢{i}-Received KF{kf_vrsn}"; print(msg); log_to_table(msg, "gs", )

            if curr_keyframe is None:
                continue

            # ------------------------------------------------------------------
            # First keyframe only: init COLMAP, des pose and gaussians from P0
            # ------------------------------------------------------------------
            if init:
                
                # Initialize COLMAP with the init image at the origin
                _t = time.time()
                PosesHandling.make_colmap_data(f"{sfm_path}/sparse/0", [0], ["init_img.png"], [init_pose_gs2], CAM_W, CAM_H, fx_moge, fy_moge)
                ImageHandling.save_img(init_img, "init_img", f"{sfm_path}/images")
                msg = f"🟢{i}-Init Colmap"; dt = time.time() - _t; print(msg); log_to_table(msg, "gs", dt); _t = time.time()

                # Save des pose and load gs2 intrinsics
                MyUtils.save_arrays_to_npy(f"{case_test_path}/des_pose_gs2.npy", 0, des_pose_gs2)
                intrins_gs2 = PosesHandling.get_intrins_gs(sfm_path)
 
                # Keep textured pixels, plus patch centers that belong to new areas
                _, txtr_mask = ImageHandling.compute_texturemap_and_mask(init_img, threshold=0.08)

               # Turn img to gaussians based on txtr
                gaussians0 = GaussiansHandling.turn_image_to_gaussians(init_img, moge_points1, moge_mask1, txtr_mask, moge_colors1)
                init_rndr, _ = GaussiansHandling.render_gs_pic(*gaussians0, init_pose_gs2, intrins_gs2, CAM_W, CAM_H)

                GaussiansHandling.turn_gaussians_to_ply(gaussians0, f"{gs2s_dir_path}/gs2_0_pre.ply")
                msg = f"🟢{i}-Init gaussians from P0"; dt = time.time() - _t; print(msg); log_to_table(msg, "gs", dt)

                ImageHandling.save_img(init_img, f"0_0_init_img", keyframes_path)
                ImageHandling.save_img(txtr_mask, f"0_1_txtr_init", keyframes_path)
                ImageHandling.save_img(init_rndr, f"0_2_init_rndr", keyframes_path)

            # ------------------------------------------------------------------
            # Keyframe pose estimation with PnP
            # ------------------------------------------------------------------

            ImageHandling.save_img(curr_keyframe, f"{kf_vrsn}_0_keyframe", keyframes_path)

            _t = time.time()
            # 2D matches between last kf and current kf, then their corresponding 3D points (in last kf cam frame)
            matched_2ds_last_kf, matched_2ds_curr_kf = ibvs_tools.filtered_matching(last_keyframe, curr_keyframe, max_matches=80, mask1=last_kf_moge_mask)
            matched_3dpoints_lastkf_cam = MeshHandling.get_corresp_3d_points(matched_2ds_last_kf, last_kf_moge_mask, last_kf_cloud_cam)
            
            # Initial PnP: the 3D points are in the previous keyframe's camera frame.
            # OpenCV returns the transform from that frame to the current camera frame.
            R, t, inlier_indices = PosesHandling.estimate_pose_pnp_ransac(matched_3dpoints_lastkf_cam, matched_2ds_curr_kf, moge_intrins)

            if inlier_indices is None or len(inlier_indices) < 6:
                raise ValueError(f"PnP failed for KF_{kf_vrsn}: not enough inliers")

            # Invert PnP to get the current camera-to-world pose relative to the previous camera.
            R_relative = R.T; t_relative = (-R.T @ np.asarray(t).reshape(3, 1)).ravel()
            pnp_pose_relative = LinAlgeb.get_homog_frm_rt(R_relative, t_relative)
            # Get the pose in world frame
            pnp_pose_world = np.asarray(last_kf_pose, dtype=np.float64) @ pnp_pose_relative
            

            # Load the latest local 3DGS. Its Gaussian positions are in the shared world frame.
            if init:
                last_gaussians = GaussiansHandling.load_gaussians_from_ply(f"{gs2s_dir_path}/gs2_0_pre.ply")
                refined_pose_world = pnp_pose_world   

            else:
                #last_gaussians = GaussiansHandling.load_gaussians_from_ply(f"{gs2s_dir_path}/gs2_{last_kf_vrsn}.ply")
                
                kf_render_pnp, _ = GaussiansHandling.render_gs_pic(*last_gaussians, curr_keyframe_pose_gs2, intrins_gs2, CAM_W, CAM_H)
                ImageHandling.save_img(kf_render_pnp, f"{kf_vrsn}_1_kf_render_pnp", keyframes_path)

                # Refine the world-frame PnP pose by matching the Gaussian render to the real keyframe.
                """
                refined_pose_world, loss_history = PosesHandling.optimize_keyframe_pose(
                initial_pose_np=pnp_pose_world, keyframe_np=curr_keyframe, means=means, quats=quats, scales=scales, opacities=opacities, sh=sh, K=moge_intrins, W=CAM_W, H=CAM_H)
                """
                means, quats, scales, opacities, sh = last_gaussians
                print("!!!!!! gs nd pose refinement started !!!!!!!!!!")
                refined_pose_world, last_gaussians, loss_history = PosesHandling.optimize_keyframe_pose_and_gaussians(pnp_pose_world, curr_keyframe, means=means, quats=quats, scales=scales, opacities=opacities, sh=sh, K=moge_intrins, W=CAM_W, H=CAM_H)
                print(loss_history)
                               
           # Make sure it's npy 4*4
            curr_keyframe_pose_gs2 = np.asarray(refined_pose_world, dtype=np.float64).reshape(4, 4)
            msg = f"🟢{i}-Refined pose of Kf_{kf_vrsn}"; dt = time.time() - _t; print(msg); log_to_table(msg, "gs", dt); _t = time.time()

            # Add the keyframe to COLMAP.
            PosesHandling.add_imgs_to_colmap([kf_vrsn], [f"keyframe{kf_vrsn}.png"], [curr_keyframe_pose_gs2], f"{sfm_path}/sparse/")
            ImageHandling.save_img(curr_keyframe, f"keyframe{kf_vrsn}", f"{sfm_path}/images")
            msg = f"🟢{i}-Add Kf_{kf_vrsn} to colmap"; dt = time.time() - _t; print(msg); log_to_table(msg, "gs", dt); _t = time.time()


            # ------------------------------------------------------------------
            # MoGe cloud of the keyframe + z-offset alignment with the initial cloud
            # ------------------------------------------------------------------
            
            _, _, all_moge_points_cam, all_moge_colors, moge_mask, _, normals = GaussiansHandling.get_moge_points(
                curr_keyframe, fov_x=fov_x_deg, model_reso_lvl=moge_model_reso_lvl, depth_edge_threshold=moge_depth_edge_threshold, moge_model=moge_model, use_fp16_bool=False)
            all_moge_points = LinAlgeb.transform_points_to_world(all_moge_points_cam, curr_keyframe_pose_gs2)

            # Filtered matches between the initial image and the current keyframe, and their 3D correspondences
            matches_img1, matches_kf = ibvs_tools.filtered_matching(init_img, curr_keyframe, max_matches=80, mask1=moge_mask1)
            matches_img = ImageHandling.draw_matches(matches_img1, matches_kf, init_img, curr_keyframe)
            ImageHandling.save_img(matches_img, f"{kf_vrsn}_8_kf_alignmnt_mtches", keyframes_path)
            matched_3d_points_img1, matched_3d_points_kf = MeshHandling.get_corresp_3d_point_matches(
                matches_img1, matches_kf, moge_mask1, moge_mask, moge_points1, all_moge_points)

            # Estimate the Z offset and correct the full current cloud
            all_moge_points, _ = MeshHandling.estimate_and_apply_z_offset(
                cloud1=matched_3d_points_img1, cloud2=matched_3d_points_kf, cloud2_full=all_moge_points, pose2_ofthecamcloud=curr_keyframe_pose_gs2)


            # Get txtr mask + new picel mask (opacity) ______________________
            # Detecting pixels of new areas, by low opacity 3dgs rendering
            kf_render_pose_refined, kf_rendered_depth, alpha_map = GaussiansHandling.render_gs_pic(*last_gaussians, curr_keyframe_pose_gs2, intrins_gs2, CAM_W, CAM_H, rndr_alpha=True)
            low_alpha_mask = GaussiansHandling.get_low_alpha_mask(alpha_map, alpha_threshold=0.5)
            ImageHandling.save_img(kf_render_pose_refined, f"{kf_vrsn}_2_kf_render_pose_refined", keyframes_path)

            _, txtr_mask = ImageHandling.compute_texturemap_and_mask(curr_keyframe, threshold=0.08)
            final_mask = low_alpha_mask & moge_mask
            ImageHandling.save_img(final_mask, f"{kf_vrsn}_3_final_mask", keyframes_path)

            # Turning the new points 
            new_gaussians = GaussiansHandling.turn_image_to_gaussians(curr_keyframe, all_moge_points, final_mask, txtr_mask, all_moge_colors)
            total_gaussians = GaussiansHandling.merge_2_gaussians(last_gaussians, new_gaussians)
            kf_render_init, _ = GaussiansHandling.render_gs_pic(*total_gaussians, curr_keyframe_pose_gs2, intrins_gs2, CAM_W, CAM_H)
            ImageHandling.save_img(kf_render_init, f"{kf_vrsn}_4_kf_render_after_gs_init", keyframes_path)

            msg = f"🟢{i}-Turn surfaces to gaussians nd merge"; dt = time.time() - _t; print(msg); log_to_table(msg, "gs", dt); _t = time.time()

            last_gaussians2 = total_gaussians

            # Save pre-gaussians and train them
            #"""
            GaussiansHandling.turn_gaussians_to_ply(total_gaussians, f"{gs2s_dir_path}/gs2_{kf_vrsn}_pre.ply")
            GaussiansHandling.run_gs_training(sfm_path=sfm_path, output_path=f"{gs2s_dir_path}/inria_output", gs_reso=gs_reso, gs_nbr_itrs=gs_nbr_itrs, curr_case_gs2_path=gs2s_dir_path)
            shutil.copy2(inria_saved_ply_path, f"{gs2s_dir_path}/gs2_{kf_vrsn}.ply")
            msg = f"🟢{i}-Trained local_gs for {gs_nbr_itrs} iterations"; dt = time.time() - _t; print(msg); log_to_table(msg, "gs", dt); _t = time.time()
            #"""
            
            # Load the latest GS2
            #last_gaussians2 = GaussiansHandling.load_gaussians_from_ply(f"{gs2s_dir_path}/gs2_{kf_vrsn}.ply")
            
            kf_render_aftr, _ = GaussiansHandling.render_gs_pic(*last_gaussians2, curr_keyframe_pose_gs2, intrins_gs2, CAM_W, CAM_H)
            ImageHandling.save_img(kf_render_aftr, f"{kf_vrsn}_4_kf_render_after_gs_train", keyframes_path)

            # On the last KF, update the desired pose once using DVS
            if is_last_kf:
                _t = time.time()
                des0_img = ImageHandling.load_np_img(f"{des_imgs_path}/des0.png")
                des0_mask = MyUtils.load_frst_dict_np_elmnt(des_masks_path)
                des_pose_gs2 = ibvs_tools.start_dvs_loop_gs(last_gaussians2, intrins_gs2, des_pose_gs2, des0_img, case_test_path, des0_mask, max_itrs=50)
                is_last_kf = False
                msg = f"🟢{i}-DVS to update des pose"; dt = time.time() - _t; print(msg); log_to_table(msg, "gs", dt); _t = time.time()
                MyUtils.save_arrays_to_npy(f"{case_test_path}/des_pose_gs2_dvs.npy", 0, des_pose_gs2)

            # Render the new des img and send it to the servo loop
            des_img, des_depth = GaussiansHandling.render_gs_pic(*last_gaussians2, des_pose_gs2, intrins_gs2, CAM_W, CAM_H)
            des_mask = ImageHandling.get_mask_frm_depth(des_depth)
            ImageHandling.save_img(des_img, f"des{i}", des_imgs_path)
            des_img_queue.put([des_img, des_mask])
            des_img_event.set()
            msg = f"🟢{i}-Render new des_img and send it to servo loop"; dt = time.time() - _t; print(msg); log_to_table(msg, "gs", dt)

            # Update last-kf infos for the next PnP (z-corrected cloud, back in the cam frame of this kf)
            last_kf_cloud_cam = LinAlgeb.transform_points_to_world(all_moge_points, np.linalg.inv(curr_keyframe_pose_gs2))
            last_kf_pose = curr_keyframe_pose_gs2
            last_keyframe = curr_keyframe
            last_kf_moge_mask = moge_mask

            init = False
            last_kf_vrsn = kf_vrsn


            last_gaussians = last_gaussians2

            # Save desired-image depth information
            MyUtils.save_arrays_to_npy(des_depths_npy_path, i, des_depth)
            MyUtils.cleanup()

            if stop_event.is_set():
                break

    except Exception as error:
        error_traceback = traceback.format_exc()
        print(f"Error in GS loop: {error}")
        print(error_traceback)
        shared["stop"] = True
        stop_event.set()
        save_case_error(new_case_test_path, "local_gs_loop", error_traceback)
        MyUtils.cleanup()
        raise








def dyna_ibvs_loop(shared, stop_event, kf_queue, des_img_queue, kf_event, gs1_ply_path, init_pose_gs1, gt_des_pose_gs1, init_img, des_img, fx_gs1, fy_gs1, new_case_test_path) :

    try:

        # update again the paths inside this process, cuz this parallel process creates its own environement 
        # and didn't take into consid the changes that were done in the global_pahts (it's for an old envirnmnt)
        update_case_paths(new_case_test_path)

        msg = f"🔵Ibvs_loop launched"; print(msg); log_to_table(msg, 'ibvs', 0); _t = time.time()
        
        # Loading gs1
        gaussians1 = GaussiansHandling.load_gaussians_from_ply(gs1_ply_path)
        intrins_gs1 = PosesHandling.get_intrins_gs(gs1_sfm_path)
        last_kf_sent = False
        
        # get des0 kpnts
        xfeat_model = get_xfeat_model()
        des_kpts, des_desc = get_xfeat_kpts(des_img, xfeat_model)
        last_kf_kpts, last_kf_desc = get_xfeat_kpts(init_img, xfeat_model)
        dists_cur = dists_des = None
        
        # init vars
        pxl_error = pose_error = None
        ibvs_infos_storage = {}
        #ibvs_depths_storage = {}
        i = 0
        des_vrsn = 0
        kf_motion_ratio = 0.03
        #final_kf_sent = False
        cur_pose_gs1 = init_pose_gs1
        last_keyframe = init_img
        ibvs_tools = IbvsTools(CAM_W, CAM_H, fx_gs1, fy_gs1)

        des_mask = MyUtils.load_frst_dict_np_elmnt(des_masks_path)
   
        msg = f"🔵Loaded xfeat and gs1"; dt = time.time()-_t; _t = time.time()
        print(msg); log_to_table(msg, 'ibvs', dt)

        while not stop_event.is_set():
            i+=1; _t = time.time()

            # Check if GS_loop got us new des_img, then update it   
            while not des_img_queue.empty():
                des_img_infos = des_img_queue.get()
                des_img = des_img_infos[0]
                des_mask = des_img_infos[1]
                des_vrsn += 1
                des_kpts, des_desc = get_xfeat_kpts(des_img, xfeat_model)
                dists_des = None
                msg = f"🔵🔵{i}-Got new des_img({des_vrsn})"; print(msg); log_to_table(msg, 'ibvs', 0)

            # Render new cur_gs_pic and get its depth_map
            cur_gs1_img, cur_gs1_depth_map = GaussiansHandling.render_gs_pic(*gaussians1, T=cur_pose_gs1, K=intrins_gs1, W=CAM_W, H=CAM_H)

            # 1 - Match current_gs1_frame with des_img
            cur_kpts, cur_desc = ibvs_tools.get_xfeat_kpts(cur_gs1_img)

            idxs0, idxs1 = xfeat.match(cur_desc, des_desc)
            #print(f"{i}: raw matches = {len(idxs0)}")

            matches_cur = cur_kpts[idxs0].detach().cpu().numpy()
            matches_des = des_kpts[idxs1].detach().cpu().numpy()

            safe_indices, dists_cur, dists_des = ibvs_tools.border_safe_indices(
                matches_cur, matches_des,
                mask1=None, mask2=des_mask,
                dist1=dists_cur, dist2=dists_des)
            matches_cur = matches_cur[safe_indices]
            matches_des = matches_des[safe_indices]
            #print(f"{i}: after border filter = {len(matches_cur)}")

            matches_cur, matches_des = ibvs_tools.filter_matches_ransac(
                matches_cur, matches_des, ransac_threshold=2.0, confidence=0.999)
            #print(f"{i}: after RANSAC = {len(matches_cur)}")

            matches_cur, matches_des = IbvsTools.select_distributed_matches(
                matches_cur, matches_des, CAM_W, CAM_H,
                top_x=ibvs_nbr_features)
            #print(f"{i}: after distribution = {len(matches_cur)}")

            matches_cur = matches_cur.astype(np.int32)
            matches_des = matches_des.astype(np.int32)


            # Notify GS_Loop if it's a keyframe
            if ((i % 20) == 0) and validate_keyframe(last_kf_kpts, last_kf_desc, cur_kpts, cur_desc, xfeat_model, ibvs_itr_nbr=i, kf_motion_nbr_features= kf_motion_nbr_features, motion_thresh_ratio = kf_motion_ratio) :
                last_keyframe = cur_gs1_img
                last_kf_kpts = cur_kpts
                last_kf_desc = cur_desc
                         
                if (pxl_error < pxl_err_to_refi) and (not last_kf_sent):
                    kf_queue.put([last_keyframe, cur_pose_gs1, True])
                    last_kf_sent= True
                else :
                    kf_queue.put([last_keyframe, cur_pose_gs1, False])
                      
                kf_event.set()
                msg = f"🔵🔵{i}-Keyframe was found and sent to gsloop"
                print(msg); log_to_table(msg, 'ibvs', 0)


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
            pxl_error = np.linalg.norm(errors)
            pose_error = LinAlgeb.pose_distance(cur_pose_gs1, gt_des_pose_gs1)  

            if pxl_error < pxl_err_to_morekf :
                kf_motion_ratio = 0.002

            # 4 - Get the intr matrix & its pseudo_inv & calc V with control law
            L = ibvs_tools.get_interaction_matrix(len(matches_cur), Ss_cur, Ss_Z_cur, 1)
            L_psinv = get_inter_mat_pseudo_inverse(L)
            V = - lambda_gain * (L_psinv @ errors)
 
            # 5 - Update cur_cam_pose and update visualization
            cur_pose_gs1 = ibvs_tools.update_cam_pose(cur_pose_gs1, V, dt_ibvs)

            # Save the information of the current iteration
            cond_nbr = LinAlgeb.get_mat_condition_number(L)
            ibvs_infos = [pxl_error, pose_error, cond_nbr, V, [matches_cur, matches_des], cur_pose_gs1]
            MyUtils.accumulate_array(ibvs_infos_storage, i, ibvs_infos)


            if (i % 20) == 0:          
                msg = f"🔵{i}-IBVS: 2d_err:{pxl_error:.4f} | des_img({des_vrsn})"; dt = time.time()-_t; print(msg)
                log_to_table(msg, "ibvs", dt)

                cur_mtch_gs_img = ImageHandling.draw_matches(matches_cur, matches_des, cur_gs1_img, des_img)
                ImageHandling.save_img(cur_mtch_gs_img, i, matches_frames_path)

                # Save depth only each 50itrs, then release its storage, cuz it takes the whole storage
                """
                MyUtils.accumulate_array(ibvs_depths_storage, i, cur_gs1_depth_map)
                MyUtils.save_accumulated_arrays(ibvs_frame_depths_npy_path, ibvs_depths_storage)
                ibvs_depths_storage.clear()
                """

            # Check convergence after accumulating the current iteration
            if pxl_error < ibvs_pxl_error_conv:
                stop_event.set()
                kf_event.set()
                break


    except Exception as error:
        error_traceback = traceback.format_exc()
        print(f"Error in GS loop: {error}")
        print(error_traceback)
        shared["stop"] = True
        stop_event.set()
        save_case_error(new_case_test_path, "local_gs_loop", error_traceback)
        raise

    finally:
        try:
            MyUtils.save_accumulated_arrays(ibvs_infos_npy_path, ibvs_infos_storage)
            #MyUtils.save_accumulated_arrays(ibvs_frame_depths_npy_path, ibvs_depths_storage)
            ibvs_infos_storage.clear()
            #ibvs_depths_storage.clear()
        except Exception as save_error:
            print(f"Could not save accumulated IBVS results: {save_error}")







def local_mesh_loop(shared, stop_event, kf_queue, des_img_queue, des_img_event, moge_points1_cmfrm, moge_colors1, moge_mask1, moge_des_trans, moge_intrins, init_img, intrins_K, new_case_test_path):

    import os
    import queue

    try:
        curr_keyframe = None
        current_mesh = None
        init = True
        i = 0
        kf_vrsn = 0
        is_last_kf = False

        # cam_init is the origin: the init pose is identity and the MoGe cam frame is the world frame
        init_pose_gs2 = LinAlgeb.get_eye_matrix(4)
        des_pose_gs2 = moge_des_trans
        moge_points1 = moge_points1_cmfrm

        # get the intrinsics and make the object classes
        fx_moge = moge_intrins[0, 0]; fy_moge = moge_intrins[1, 1]; cx_moge = moge_intrins[0, 2]; cy_moge = moge_intrins[1, 2]
        mesh_handling = MeshHandling(CAM_W, CAM_H, fx_moge, fy_moge, cx_moge, cy_moge)
        ibvs_tools = IbvsTools(CAM_W, CAM_H, fx_moge, fy_moge, cx_moge, cy_moge)

        # get fovx
        fov_x_rad = 2 * np.arctan(CAM_W / (2 * fx_moge))
        fov_x_deg = np.rad2deg(fov_x_rad)

        # init the last kf infos with the init img infos (used for the pose estimation of the next kf)
        last_kf_pose = init_pose_gs2
        last_kf_cloud_cam = moge_points1_cmfrm
        last_keyframe = init_img
        last_kf_moge_mask = moge_mask1

        # loading moge model once in the beg
        _t = time.time()
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda")
        msg = f"🟢{i}-Just Loaded Moge model"; dt = time.time() - _t
        print(f"{msg} || {dt:.3f} s"); log_to_table(msg, "gs", dt); _t = time.time()

        # Initialize the TSD_mesher
        mesher = IncrementalTSDFMesher(
            intrinsic=intrins_K,
            voxel_size=0.01,
            depth_min=0.05,
            depth_max=10.0, block_resolution=16, block_count=5000, device="CPU:0", pose_is_camera_to_world=True)

        # update again the paths inside this process
        # cuz this parallel process creates its own environement and didn't take into consid the changes that were done in the global_pahts (it's for an old envirnmnt)
        update_case_paths(new_case_test_path)

        while not shared["stop"] and not stop_event.is_set():
            curr_keyframe = None
            batch_last_flag = False

            # if kf_queue is empty we skip this block that loads keyframes
            while True:
                try:
                    curr_keyframe_list = kf_queue.get(timeout=0.2)
                except queue.Empty:
                    break

                curr_keyframe = curr_keyframe_list[0]
                batch_last_flag = batch_last_flag or bool(curr_keyframe_list[2])
                kf_vrsn += 1
                i += 1
                msg = f"🟢🟢{i}-Received KF{kf_vrsn}"; print(msg); log_to_table(msg, "gs")

            if curr_keyframe is None:
                continue

            is_last_kf = is_last_kf or batch_last_flag

            if stop_event.is_set():
                break

            # If this is the first keyframe, initialize the TSDF.
            if init:
                # init the accum_3dpoints which are in gs2 frame ofc
                accumulated_moge_points = moge_points1.reshape(-1, 3)
                accumulated_moge_colors = moge_colors1.reshape(-1, 3)

                # apply mask on em
                moge_mask1_flat = moge_mask1.reshape(-1)
                accumulated_moge_points = accumulated_moge_points[moge_mask1_flat]
                accumulated_moge_colors = accumulated_moge_colors[moge_mask1_flat]

                # Update vgb using init img
                _t = time.time()
                mesher.integrate_frame(image_rgb=init_img, points_cam=moge_points1_cmfrm, moge_mask=moge_mask1, pose=init_pose_gs2)
                msg = f"🟢{i}-Initial TSDF integration"; dt = time.time() - _t
                print(msg); log_to_table(msg, "gs", dt)
                init = False

            # Estimate the keyframe pose with PnP
            _t = time.time()
            ImageHandling.save_img(curr_keyframe, f"kf_{kf_vrsn}_1_kf", f"{new_case_test_path}/keyframes")

            # Do 2d matches between the last kf and the current kf and get the corresp 3ds
            matched_2ds_last_kf, matched_2ds_curr_kf = ibvs_tools.filtered_matching(last_keyframe, curr_keyframe, max_matches=80, mask1=last_kf_moge_mask, min_cossim=0.8)
            matched_img = ImageHandling.draw_matches(matched_2ds_last_kf, matched_2ds_curr_kf, last_keyframe, curr_keyframe)
            ImageHandling.save_img(matched_img, f"kf_{kf_vrsn}_3_matches", f"{new_case_test_path}/keyframes")
            matched_3dpoints_lastkf_cam = MeshHandling.get_corresp_3d_points(matched_2ds_last_kf, last_kf_moge_mask, last_kf_cloud_cam)

            # Get the initial PnP (R,t) and refine it using its inliers
            R, t, inlier_indices = PosesHandling.estimate_pose_pnp_ransac(matched_3dpoints_lastkf_cam, matched_2ds_curr_kf, moge_intrins)
            if inlier_indices is None or len(inlier_indices) < 6:
                raise ValueError(f"PnP failed for KF{kf_vrsn}: not enough inliers")
            R_refined, t_refined = PosesHandling.refine_pose_pnp(matched_3dpoints_lastkf_cam, matched_2ds_curr_kf, moge_intrins, R, t, inlier_indices)

            # Get the corresp world pose of the kf
            R_new = R_refined.T; t_new = (-R_refined.T @ np.asarray(t_refined).reshape(3, 1)).ravel()
            pose_refined = LinAlgeb.get_homog_frm_rt(R_new, t_new)
            curr_keyframe_pose_gs2 = LinAlgeb.transform_pose(last_kf_pose, pose_refined)
            msg = f"🟢{i}-PnP pose of KF{kf_vrsn}"; dt = time.time() - _t
            print(msg); log_to_table(msg, "gs", dt); _t = time.time()

            # Get the Kf's Pi and put it in frame
            masked_points, masked_colors, all_moge_points_cam, all_moge_colors, moge_mask, _, normals = GaussiansHandling.get_moge_points(
                curr_keyframe, fov_x=fov_x_deg, model_reso_lvl=moge_model_reso_lvl, depth_edge_threshold=moge_depth_edge_threshold, moge_model=moge_model, use_fp16_bool=False
            )
            all_moge_points = LinAlgeb.transform_points_to_world(all_moge_points_cam, curr_keyframe_pose_gs2)

            # Get corresp 3d matches P(0) and P(kfi)
            matches_img1, matches_kf = ibvs_tools.filtered_matching(init_img, curr_keyframe, max_matches=80, mask1=moge_mask1, min_cossim=0.8)
            matched_3d_points_img1, matched_3d_points_kf = MeshHandling.get_corresp_3d_point_matches(
                matches_img1, matches_kf, moge_mask1, moge_mask, moge_points1, all_moge_points
            )
            matches_img = ImageHandling.draw_matches(matches_img1, matches_kf, init_img, curr_keyframe)
            ImageHandling.save_img(matches_img, f"kf_alignmnt_mtches_{kf_vrsn}", f"{case_test_path}/kfs_matches")

            if stop_event.is_set():
                break

            # Estimate the Z offset and correct the full current cloud.
            all_moge_points, _ = MeshHandling.estimate_and_apply_z_offset(
                cloud1=matched_3d_points_img1, cloud2=matched_3d_points_kf, cloud2_full=all_moge_points, pose2_ofthecamcloud=curr_keyframe_pose_gs2)
            all_moge_points_cam = LinAlgeb.transform_points_to_world(all_moge_points, np.linalg.inv(curr_keyframe_pose_gs2))
            msg = f"🟢{i}-Moge cloud of KF{kf_vrsn} and z alignement"; dt = time.time() - _t
            print(msg); log_to_table(msg, "gs", dt); _t = time.time()

            # update the last kf infos for the next kf pose estimation
            last_kf_cloud_cam = all_moge_points_cam
            last_kf_pose = curr_keyframe_pose_gs2
            last_keyframe = curr_keyframe
            last_kf_moge_mask = moge_mask

            if stop_event.is_set():
                break

            #_________________________ Update the vgb using the new KF + extract and update current mesh
            current_mesh = mesher.extract_mesh(extract_on_cpu=False)
            kf_render, kf_render_depth = mesh_handling.render_mesh_pic_withnoshad([current_mesh], curr_keyframe_pose_gs2)

            # get mask of new txtr pixels
            existing_region_mask = ImageHandling.get_mask_frm_depth(kf_render_depth)
            new_region_mask = np.asarray(moge_mask, dtype=bool) & ~np.asarray(existing_region_mask, dtype=bool)

            mesher.integrate_frame(image_rgb=curr_keyframe, points_cam=all_moge_points_cam, moge_mask=new_region_mask, pose=curr_keyframe_pose_gs2)
            current_mesh = mesher.extract_mesh(extract_on_cpu=False)

            os.makedirs(keyframes_path, exist_ok=True)
            ply_path = f"{keyframes_path}/meshi.ply"
            o3d.io.write_triangle_mesh(ply_path, current_mesh, write_ascii=False, write_vertex_normals=True, write_vertex_colors=True)
            msg = f"🟢{i}-Integrated KF{kf_vrsn} in the TSDF and extracted the mesh"; dt = time.time() - _t
            print(msg); log_to_table(msg, "gs", dt); _t = time.time()

            # If we reached the last kf, i will update the des pose (dvs before rendering it), this only happens once with lastkf and then we set it again to false
            if is_last_kf:
                des0_img = ImageHandling.load_np_img(f"{des_imgs_path}/des0.png")
                des0_mask = MyUtils.load_frst_dict_np_elmnt(des_masks_path)
                des_pose_gs2 = ibvs_tools.start_dvs_loop_o3d(current_mesh, mesh_handling, des_pose_gs2, des0_img, case_test_path, des0_mask)
                is_last_kf = False
                msg = f"🟢{i}-DVS to update des pose"; dt = time.time() - _t
                print(msg); log_to_table(msg, "gs", dt); _t = time.time()

            # Render the new des img and send it to the servo loop (even if stop_event was set during DVS)
            if len(current_mesh.vertices) > 0:
                des_img, des_depth = mesh_handling.render_mesh_pic_withnoshad([current_mesh], des_pose_gs2)
                des_mask = ImageHandling.get_mask_frm_depth(des_depth)
                ImageHandling.save_img(des_img, f"des{i}", des_imgs_path)
                des_img_queue.put([des_img, des_mask])
                des_img_event.set()
                MyUtils.save_arrays_to_npy(des_depths_npy_path, i, des_depth)
                msg = f"🟢{i}-Rendered new desired image"; dt = time.time() - _t
                print(msg); log_to_table(msg, "gs", dt)
            else:
                print(f"🟢{i}-Skipping desired render: extracted mesh is empty")

            MyUtils.cleanup()

            if stop_event.is_set():
                break

    except Exception as e:
        print(f"Error in local_mesh_loop: {e} !!!!!!!!!!!!!!")
        print("[mesh] FULL TRACEBACK:")
        traceback.print_exc()
        MyUtils.cleanup()
        shared["stop"] = True






def local_cloud_loop(shared, stop_event, kf_queue, des_img_queue, des_img_event, moge_points1_cmfrm, masked_moge_points1_cmfrm, moge_colors1, masked_moge_colors1, moge_mask1, moge_des_trans, moge_intrins, init_img, new_case_test_path):
    import queue

    try:
        
        # Configs ___________________________________________________________
        
        # Update paths
        update_case_paths(new_case_test_path)

        # Initialize variables
        des_pose_gs2 = moge_des_trans # des pose gs2 is the relative init des trans since cam_init is origin!
        i = kf_vrsn = 0
        is_last_kf = False

        # Get intrinscs and make object classes
        fx_moge = moge_intrins[0, 0]; fy_moge = moge_intrins[1, 1]; cx_moge = moge_intrins[0, 2]; cy_moge = moge_intrins[1, 2]
        mesh_handling = MeshHandling(CAM_W, CAM_H, fx_moge, fy_moge, cx_moge, cy_moge)
        ibvs_tools = IbvsTools(CAM_W, CAM_H, fx_moge, fy_moge, cx_moge, cy_moge)  
        # Get fovx
        fov_x_rad = 2 * np.arctan(CAM_W / (2 * fx_moge))
        fov_x_deg = np.rad2deg(fov_x_rad)

        # Load MoGe
        _t = time.time()
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda")
        msg = f"🟢{i}-Just Loaded Moge model"; dt = time.time() - _t
        print(f"{msg} || {dt:.3f} s"); log_to_table(msg, "gs", dt); _t = time.time()

        # Init img infos as last kf infos 
        init_pose_gs2 = LinAlgeb.get_eye_matrix(4)
        last_kf_pose=init_pose_gs2; last_kf_cloud_cam=moge_points1_cmfrm; last_keyframe=init_img; last_kf_moge_mask=moge_mask1
        accumulated_moge_points = masked_moge_points1_cmfrm # cam frame is world frame since cam_init is origin !
        accumulated_moge_colors = masked_moge_colors1
        accum_moge_points_o3d = MeshHandling.turn_points_to_o3d(accumulated_moge_points, accumulated_moge_colors)
        accum_moge_points_o3d_init = accum_moge_points_o3d
              



        # Launching the cloud loop___________________________________________________________________

        while not shared["stop"] and not stop_event.is_set():
            curr_keyframe = None

            # Get Last Keyframe _____________________________________________________________________

            while True:
                try:
                    curr_keyframe_list = kf_queue.get(timeout=0.2)
                except queue.Empty:
                    break
                curr_keyframe = curr_keyframe_list[0]
                is_last_kf = is_last_kf or curr_keyframe_list[2]
                kf_vrsn += 1; i += 1
                msg = f"🟢🟢{i}-Received KF{kf_vrsn}"; print(msg); log_to_table(msg, "gs")

            if curr_keyframe is None:
                continue


            
            # Estimating Keyframe pose via PnP _____________________________________

            ImageHandling.save_img(curr_keyframe, f"kf_{kf_vrsn}_1_kf", f"{new_case_test_path}/keyframes")

            # Getting 2D-3D pairs between this kf img and last kf's cloud 
            matched_2ds_last_kf, matched_2ds_curr_kf = ibvs_tools.filtered_matching(last_keyframe, curr_keyframe, max_matches= 80, mask1= last_kf_moge_mask, min_cossim=0.8) # we used last kf mask here so we good
            matched_img = ImageHandling.draw_matches(matched_2ds_last_kf, matched_2ds_curr_kf, last_keyframe, curr_keyframe)
            ImageHandling.save_img(matched_img, f"kf_{kf_vrsn}_3_matches", f"{new_case_test_path}/keyframes")
            matched_3dpoints_lastkf_cam = MeshHandling.get_corresp_3d_points(matched_2ds_last_kf, last_kf_moge_mask, last_kf_cloud_cam)
            
            # Get the initial PnP (R,t) 
            R, t, _ = PosesHandling.estimate_pose_pnp_ransac(matched_3dpoints_lastkf_cam, matched_2ds_curr_kf, moge_intrins)
            
            # Get the corresp world pose & render and save
            R_new = R.T; t_new = (-R.T @ np.asarray(t).reshape(3, 1)).ravel()
            pose_refined = LinAlgeb.get_homog_frm_rt(R_new, t_new); 
            curr_kf_pose_gs2 = LinAlgeb.transform_pose(last_kf_pose, pose_refined)
            kf_render_refined, _ = mesh_handling.render_mesh_pic([accum_moge_points_o3d], curr_kf_pose_gs2)
            ImageHandling.save_img(kf_render_refined, f"kf_{kf_vrsn}_2_rndr_refined", f"{new_case_test_path}/keyframes")

             
            # Get keyframe's Cloud and ________________________________

            masked_moge_points_cam_kf, masked_moge_colors_kf, all_moge_points_cam_kf, all_moge_colors_kf, moge_mask_kf, _, _ = GaussiansHandling.get_moge_points(
                curr_keyframe,fov_x=fov_x_deg, model_reso_lvl=moge_model_reso_lvl, depth_edge_threshold=moge_depth_edge_threshold, moge_model=moge_model, use_fp16_bool=False)
            masked_moge_points_kf_world = LinAlgeb.transform_points_to_world(masked_moge_points_cam_kf, curr_kf_pose_gs2)
            all_moge_points_kf_world = LinAlgeb.transform_points_to_world(all_moge_points_cam_kf, curr_kf_pose_gs2)             
            masked_moge_points_kf_world_o3d = MeshHandling.turn_points_to_o3d(masked_moge_points_kf_world, masked_moge_colors_kf)


            # Align it with old accum ones via ICP ____________________
            
            R, t = MeshHandling.align_clouds_icp(masked_moge_points_kf_world_o3d, accum_moge_points_o3d_init)
            all_moge_points_kf_world = LinAlgeb.apply_rt_on_3dpoints(all_moge_points_kf_world, R, t)
            all_moge_points_kf_world_o3d = MeshHandling.turn_points_to_o3d(all_moge_points_kf_world, all_moge_colors_kf)

            #MeshHandling.visualize_scene([accum_moge_points_o3d, all_moge_points_kf_world_o3d])
            
            # Get new pixels mask of the curr keyframe afte rendering accum cloud _________________________
            
            _, kf_render_depth = mesh_handling.render_mesh_pic([accum_moge_points_o3d], curr_kf_pose_gs2)
            existing_region_mask = ImageHandling.get_mask_frm_depth(kf_render_depth)
            new_region_mask = moge_mask_kf & ~existing_region_mask

            # Get the new corresp cloud and add it to the accum cloud
            new_points_world = all_moge_points_kf_world[new_region_mask]
            new_colors = all_moge_colors_kf[new_region_mask]
            accumulated_moge_points = np.concatenate((accumulated_moge_points, new_points_world), axis=0)
            accumulated_moge_colors = np.concatenate((accumulated_moge_colors, new_colors), axis=0)
            accum_moge_points_o3d = MeshHandling.turn_points_to_o3d(accumulated_moge_points, accumulated_moge_colors)

            # Update last k infos for next kf pose estimation
            last_kf_cloud_cam = LinAlgeb.transform_points_to_world(all_moge_points_kf_world, np.linalg.inv(curr_kf_pose_gs2))
            last_kf_pose = curr_kf_pose_gs2
            last_keyframe = curr_keyframe
            last_kf_moge_mask = moge_mask_kf

            # If this was the last kf we apply a DVS and update the des pose in gs2
            if is_last_kf:
                des0_img = ImageHandling.load_np_img(f"{des_imgs_path}/des0.png")
                des0_mask = MyUtils.load_frst_dict_np_elmnt(des_masks_path)
                des_pose_gs2 = ibvs_tools.start_dvs_loop_o3d(accum_moge_points_o3d, mesh_handling, des_pose_gs2, des0_img, case_test_path, des0_mask)
                is_last_kf = False

            # render des img of the new accum cloud and send to servo loop
            des_img, des_depth = mesh_handling.render_mesh_pic([accum_moge_points_o3d], des_pose_gs2)
            des_mask = ImageHandling.get_mask_frm_depth(des_depth)
            ImageHandling.save_img(des_img, f"des{i}", des_imgs_path)
            des_img_queue.put([des_img, des_mask]); des_img_event.set()
            msg = f"🟢🟢{i}-Rendered new desired image"; dt = time.time() - _t
            print(msg); log_to_table(msg, "gs", dt); _t = time.time()

            MyUtils.save_arrays_to_npy(des_depths_npy_path, i, des_depth)
            MyUtils.cleanup()

            if stop_event.is_set():
                break

    except Exception as e:
        print(f"Error in local_cloud_loop: {e}")
        print("[cloud] FULL TRACEBACK:")
        traceback.print_exc()
        MyUtils.cleanup()
        shared["stop"] = True








def local_gs_loop(shared, stop_event, kf_queue, des_img_queue, des_img_event, moge_points1_cmfrm, moge_colors1, moge_mask1, moge_normals1, moge_des_trans, moge_intrins, init_img, new_case_test_path):
    import queue

    try:
        # Update the paths inside this process (a parallel process does not inherit later changes to the globals)
        update_case_paths(new_case_test_path)
        gs_init_path = f"{new_case_test_path}/results/gs_init"
        #gs_refi_path = f"{new_case_test_path}/results/gs_refi"

        # Init Vars
        gs_nbr_itrs = 50
        i = kf_vrsn  = 0
        init = True
        is_last_kf = False
        poses_dict = {}; kf_names = []
        init_img_name = "init_img.png"
        des0_img = ImageHandling.load_np_img(f"{des_imgs_path}/des0.png")
        des0_mask = MyUtils.load_frst_dict_np_elmnt(des_masks_path)
                
        # cam_init is the origin, so the init pose is identity and the MoGe cam frame IS the world (gs2) frame.
        init_pose_gs2 = LinAlgeb.get_eye_matrix(4)
        des_pose_gs2 = moge_des_trans
        moge_points1 = moge_points1_cmfrm
        poses_dict[init_img_name] = init_pose_gs2
        ImageHandling.save_img(init_img, f"init_img", keyframes_path)
        #moge_points1_o3d = MeshHandling.turn_points_to_o3d(moge_points1, moge_colors1)

        # Intrinsics and helper objects
        fx_moge = moge_intrins[0, 0]; fy_moge = moge_intrins[1, 1]; cx_moge = moge_intrins[0, 2]; cy_moge = moge_intrins[1, 2]
        intrins_gs2 = GaussiansHandling.turn_cam_matrix_to_gsplat_format(moge_intrins)
        mesh_handling = MeshHandling(CAM_W, CAM_H, fx_moge, fy_moge, cx_moge, cy_moge)
        ibvs_tools = IbvsTools(CAM_W, CAM_H, fx_moge, fy_moge, cx_moge, cy_moge)
        fov_x_rad = 2 * np.arctan(CAM_W / (2 * fx_moge))
        fov_x_deg = np.rad2deg(fov_x_rad)

        # Load the MoGe model once at the beginning.
        _t = time.time()
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda")
        msg = f"🟢{i}-Just Loaded Moge model"; dt = time.time() - _t; print(f"{msg} || {dt:.3f} s"); log_to_table(msg, "gs", dt)

        # Last-kf infos used for PnP pose estimation (init image is the first "last kf")
        last_kf_pose = init_pose_gs2
        last_kf_cloud_cam = moge_points1_cmfrm
        last_keyframe = init_img
        last_kf_moge_mask = moge_mask1

        while not shared["stop"] and not stop_event.is_set():
            curr_keyframe = None

            # Get the latest keyframe (drain the queue, keep the newest)
            while True:
                try:
                    curr_keyframe_list = kf_queue.get(timeout=0.2)
                except queue.Empty:
                    break
                curr_keyframe = curr_keyframe_list[0]
                is_last_kf = is_last_kf or curr_keyframe_list[2]
                kf_vrsn += 1
                i += 1
                msg = f"🟢🟢{i}-Received KF{kf_vrsn}"; print(msg); log_to_table(msg, "gs"); _t = time.time()

            if curr_keyframe is None:
                continue


            # We init GS using P0 for the init keyframe____________________

            if init:
                # Turn img to gaussians based on txtr                    
                _, txtr_mask = ImageHandling.compute_texturemap_and_mask(init_img, threshold=0.08)
                local_gaussians = GaussiansHandling.turn_image_to_gaussians(init_img, moge_points1, moge_mask1, txtr_mask, moge_colors1)
                GaussiansHandling.turn_gaussians_to_ply(local_gaussians, f"{gs2s_dir_path}/gs2_0_pre.ply")
                msg = f"🟢{i}-Init gaussians from P0"; dt = time.time() - _t; print(msg); log_to_table(msg, "gs", dt)   

            
            # Keyframe pose estimation with PnP ____________________________________

            # Init kf pose with Pnp (current keyframe X last keyframe's cloud)
            matched_2ds_last_kf, matched_2ds_curr_kf = ibvs_tools.filtered_matching(last_keyframe, curr_keyframe, max_matches=80, mask1=last_kf_moge_mask) # 2D matches
            matched_3dpoints_lastkf_cam = MeshHandling.get_corresp_3d_points(matched_2ds_last_kf, last_kf_moge_mask, last_kf_cloud_cam) # corresp 3d points of last kf cloud
            R, t, _ = PosesHandling.estimate_pose_pnp_ransac(matched_3dpoints_lastkf_cam, matched_2ds_curr_kf, moge_intrins) # PnP R,t using the pairs
           
            # Get the kf pose in gs2 frame based on the R,t we got
            R_relative = R.T; t_relative = (-R.T @ np.asarray(t).reshape(3, 1)).ravel()
            pnp_pose_relative = LinAlgeb.get_homog_frm_rt(R_relative, t_relative)
            refined_pose_world = np.asarray(last_kf_pose, dtype=np.float64) @ pnp_pose_relative
            msg = f"🟢{i}-init pose of Kf_{kf_vrsn} via pnp"; dt = time.time() - _t; print(msg); log_to_table(msg, "gs", dt); _t = time.time()

            kf_render_pnp , _ = GaussiansHandling.render_gs_pic(*local_gaussians, refined_pose_world, intrins_gs2, CAM_W, CAM_H) # °°°°°°°
            

            # Keyframe pose refinement with diff_rendering ____________________________________
            if not init :
                refined_pose_world, _ = PosesHandling.optimize_keyframe_pose(
                initial_pose_np=refined_pose_world, frame=curr_keyframe, gaussians=local_gaussians, K=intrins_gs2, W=CAM_W, H=CAM_H, iterations=10)
                msg = f"🟢{i}-Refined pose of Kf_{kf_vrsn} via diff_rend"; dt = time.time() - _t; print(msg); log_to_table(msg, "gs", dt); _t = time.time()
                

            curr_keyframe_pose_gs2 = np.asarray(refined_pose_world, dtype=np.float64).reshape(4, 4)
            kf_name = f"keyframe{kf_vrsn}"; poses_dict[kf_name] = curr_keyframe_pose_gs2; kf_names.append(kf_name)
            kf_render_diffrntl , _ = GaussiansHandling.render_gs_pic(*local_gaussians, curr_keyframe_pose_gs2, intrins_gs2, CAM_W, CAM_H) # °°°°°°°
            ImageHandling.save_img(curr_keyframe, f"keyframe{kf_vrsn}", keyframes_path)
             

            # Get kf cloud and align it with z correction against init cloud _____________________________________
            
            # get kf moge cloud
            _, _, all_moge_points_cam, all_moge_colors, moge_mask, _, normals = GaussiansHandling.get_moge_points(
                curr_keyframe, fov_x=fov_x_deg, model_reso_lvl=moge_model_reso_lvl, depth_edge_threshold=moge_depth_edge_threshold, moge_model=moge_model, use_fp16_bool=False)
            all_moge_points = LinAlgeb.transform_points_to_world(all_moge_points_cam, curr_keyframe_pose_gs2)


            # 2d matches between the initial image and the current keyframe, and their 3D correspondences
            matches_img1, matches_kf = ibvs_tools.filtered_matching(init_img, curr_keyframe, max_matches=80, mask1=moge_mask1)
            matched_3d_points_img1, matched_3d_points_kf = MeshHandling.get_corresp_3d_point_matches(
                matches_img1, matches_kf, moge_mask1, moge_mask, moge_points1, all_moge_points)

            all_moge_points_o3d = MeshHandling.turn_points_to_o3d(all_moge_points, all_moge_colors)
            matches_cloud_img = ImageHandling.draw_matches(matches_img1, matches_kf, init_img, curr_keyframe) # °°°°°°°
            clouds_img_init, _ = mesh_handling.render_mesh_pic([all_moge_points_o3d], curr_keyframe_pose_gs2) # °°°°°°°°°
            
            # Estimate the Z offset and correct the full current cloud
            all_moge_points, _ = MeshHandling.estimate_and_apply_z_offset(
                cloud1=matched_3d_points_img1, cloud2=matched_3d_points_kf, cloud2_full=all_moge_points, pose2_ofthecamcloud=curr_keyframe_pose_gs2)
            msg = f"🟢{i}-Got kf {kf_vrsn} cloud and align it"; dt = time.time() - _t; print(msg); log_to_table(msg, "gs", dt); _t = time.time()

            all_moge_points_o3d = MeshHandling.turn_points_to_o3d(all_moge_points, all_moge_colors)
            clouds_img_after, _ = mesh_handling.render_mesh_pic([all_moge_points_o3d], curr_keyframe_pose_gs2) # °°°°°°°°°


            # Get txtr mask + new pixel mask (opacity) ___________________________

            # Get opacity mask 
            _, _, alpha_map = GaussiansHandling.render_gs_pic(*local_gaussians, curr_keyframe_pose_gs2, intrins_gs2, CAM_W, CAM_H, rndr_alpha=True)
            low_alpha_mask = GaussiansHandling.get_low_alpha_mask(alpha_map, alpha_threshold=0.5)

            # Get texture mask
            _, txtr_mask = ImageHandling.compute_texturemap_and_mask(curr_keyframe, threshold=0.08)
            final_mask = low_alpha_mask & moge_mask # °°°°°°°°°°°°°

            # Turning the new points to gaussians and merge
            new_gaussians = GaussiansHandling.turn_image_to_gaussians(curr_keyframe, all_moge_points, final_mask, txtr_mask, all_moge_colors)
            local_gaussians = GaussiansHandling.merge_2_gaussians(local_gaussians, new_gaussians)
            msg = f"🟢{i}-Turn new pointsclouds to gaussians nd merge"; dt = time.time() - _t; print(msg); log_to_table(msg, "gs", dt); _t = time.time()

            kf_render_merge, _ = GaussiansHandling.render_gs_pic(*local_gaussians, curr_keyframe_pose_gs2, intrins_gs2, CAM_W, CAM_H) # °°°°°°°
            des_render_merge, _ = GaussiansHandling.render_gs_pic(*local_gaussians, des_pose_gs2, intrins_gs2, CAM_W, CAM_H) # °°°°°°°°°°°°


            # Train gaussians with poses ______________________________

            poses_dict, local_gaussians, _ = GaussiansHandling.optimize_poses_gaussians(
            kf_names, poses_dict, keyframes_path, intrins_gs2, local_gaussians, number_iters=gs_nbr_itrs, last_2_kfs_weight=4)
            curr_keyframe_pose_gs2 = poses_dict[kf_name]
            msg = f"🟢{i}-Trained local_gs for {gs_nbr_itrs} iterations"; dt = time.time() - _t; print(msg); log_to_table(msg, "gs", dt); _t = time.time()

            kf_render_final, _ = GaussiansHandling.render_gs_pic(*local_gaussians, curr_keyframe_pose_gs2, intrins_gs2, CAM_W, CAM_H) # °°°°°°°

            """
            # Saving all keyframes render for the curr keyframe refinement
            for kf_name, kf_pose in poses_dict.items():
                kf_name = kf_name.removesuffix(".png")
                keyframe_render, _ = GaussiansHandling.render_gs_pic(*local_gaussians, kf_pose, intrins_gs2, CAM_W, CAM_H)
                keyframe = ImageHandling.load_np_img(f"{keyframes_path}/{kf_name}.png")
                ImageHandling.save_img(keyframe, f"{kf_name}_c_original_kf", f"{gs_refi_path}/{kf_vrsn}")
                ImageHandling.save_img(keyframe_render, f"{kf_name}_d_render", f"{gs_refi_path}/{kf_vrsn}")
            """

            # On the last KF, update the desired pose once using dif_rendring on local-gs
            if is_last_kf:
                _t = time.time()
                des_pose_gs2, _ = PosesHandling.optimize_keyframe_pose(des_pose_gs2, des0_img, local_gaussians, intrins_gs2, CAM_W, CAM_H, iterations=200, frame_mask=des0_mask)
                is_last_kf = False
                msg = f"🟢{i}-Refined des0 pose with diffrntl_rndring"; dt = time.time() - _t; print(msg); log_to_table(msg, "gs", dt); _t = time.time()
                MyUtils.save_arrays_to_npy(f"{case_test_path}/des_pose_gs2_dvs.npy", 0, des_pose_gs2)

            # Render the new des img and send it to the servo loop
            des_img, des_depth = GaussiansHandling.render_gs_pic(*local_gaussians, des_pose_gs2, intrins_gs2, CAM_W, CAM_H)
            des_mask = ImageHandling.get_mask_frm_depth(des_depth)
            ImageHandling.save_img(des_img, f"des{i}", des_imgs_path)
            des_img_queue.put([des_img, des_mask]); des_img_event.set()
            msg = f"🟢{i}-Render new des_img and send it to servo loop"; dt = time.time() - _t; print(msg); log_to_table(msg, "gs", dt)

            # Update last-kf infos for the next PnP (z-corrected cloud, back in the cam frame of this kf)
            last_kf_cloud_cam = LinAlgeb.transform_points_to_world(all_moge_points, np.linalg.inv(curr_keyframe_pose_gs2))
            last_kf_pose = curr_keyframe_pose_gs2
            last_keyframe = curr_keyframe
            last_kf_moge_mask = moge_mask
            init = False


            # Saving all results infos ____________________
                             
            ImageHandling.save_img(matches_cloud_img, f"1_matches_cloud_img", f"{gs_init_path}/{kf_vrsn}")
            ImageHandling.save_img(final_mask, f"2_final_mask", f"{gs_init_path}/{kf_vrsn}")
            ImageHandling.save_img(clouds_img_init, f"3_clouds_img_init", f"{gs_init_path}/{kf_vrsn}")
            ImageHandling.save_img(clouds_img_after, f"4_clouds_img_after", f"{gs_init_path}/{kf_vrsn}")
            ImageHandling.save_img(curr_keyframe, f"5_original_keyframe", f"{gs_init_path}/{kf_vrsn}")
            ImageHandling.save_img(kf_render_diffrntl, f"6_kf_render_pnp", f"{gs_init_path}/{kf_vrsn}")
            ImageHandling.save_img(kf_render_pnp, f"7_kf_render_diffrntl", f"{gs_init_path}/{kf_vrsn}")
            ImageHandling.save_img(curr_keyframe, f"8_original_keyframe", f"{gs_init_path}/{kf_vrsn}")
            ImageHandling.save_img(kf_render_merge, f"9_kf_render_merge", f"{gs_init_path}/{kf_vrsn}")
            ImageHandling.save_img(kf_render_final, f"10_kf_render_final", f"{gs_init_path}/{kf_vrsn}")
            ImageHandling.save_img(curr_keyframe, f"11_original_keyframe", f"{gs_init_path}/{kf_vrsn}")
            ImageHandling.save_img(des_render_merge, f"12_des_render_merge", f"{gs_init_path}/{kf_vrsn}")
            ImageHandling.save_img(des_img, f"13_des_render_final", f"{gs_init_path}/{kf_vrsn}")
            ImageHandling.save_img(des0_img, f"14_des0", f"{gs_init_path}/{kf_vrsn}")


            # Save desired-image depth information
            #MyUtils.save_arrays_to_npy(des_depths_npy_path, i, des_depth)
            MyUtils.cleanup()

            if stop_event.is_set():
                break

    except Exception as error:
        error_traceback = traceback.format_exc()
        print(f"Error in GS loop: {error}")
        print(error_traceback)
        shared["stop"] = True
        stop_event.set()
        save_case_error(new_case_test_path, "local_gs_loop", error_traceback)
        MyUtils.cleanup()
        raise











def generate_random_des_pose(init_img, all_moge_points, moge_mask, mesh_handling: MeshHandling, init_moge_points_o3d, diff_lvl) :


    if diff_lvl == 1 :
        trans_range  = [0.1, 0.4]
        z_rot_range = [-10, 10] 
        focal_range = [-0.35, 0.35] 

    elif diff_lvl == 2 :
        trans_range = [0.4, 0.7]
        z_rot_range = [-15, 15] 
        focal_range = [-0.65, 0.65] 

    elif diff_lvl == 3 :
        trans_range = [0.7, 1]
        z_rot_range = [-20, 20] 
        focal_range = [-1, 1]

    else :     
        trans_range = [1, 1.3]
        z_rot_range = [-22, 22] 
        focal_range = [-1.2, 1.2]

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
    #keyframes_path = new_case_path / "keyframes"
    sfm_path = new_case_path / "sfm"

    # Create all directories.
    for path in [
        desired_imgs_path,
        gs2s_path,
        ibvs_frames_path,
        sfm_path / "images",
        sfm_path / "sparse" / "0"]:
        path.mkdir(parents=True, exist_ok=True)

    # Source files.
    des0_source = source_lvl_path / f"{tst_nbr}_des0.png"
    mask_source = source_lvl_path / f"{tst_nbr}_desmask.npy"
    pose_source = source_lvl_path / f"{tst_nbr}_des_moge_pose.npy"
    gt_des_source = source_lvl_path / f"{tst_nbr}_gt_des.png"

    # Copy and rename files.
    shutil.copy2(
        des0_source,
        #f"{case_test_path}/level_{diff_lvl}/case_{tst_nbr}_old/desired_imgs/des0.png",
        desired_imgs_path / "des0.png")

    shutil.copy2(
        mask_source,
        #f"{case_test_path}/level_{diff_lvl}/case_{tst_nbr}_old/desired_imgs/des_masks.npy",
        desired_imgs_path / "des_masks.npy")

    shutil.copy2(
        pose_source,
        #f"{case_test_path}/level_{diff_lvl}/case_{tst_nbr}_old/init_des_trans.npy",
        new_case_path / "init_des_trans.npy")

    shutil.copy2(
        gt_des_source,
        #f"{case_test_path}/level_{diff_lvl}/case_{tst_nbr}_old/desired_imgs/gt_des.png",
        desired_imgs_path / "gt_des.png")
    

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
    global pre_matches_frames_path, matches_frames_path
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
    matches_frames_path = f"{case_test_path}/ibvs_frames"
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







def testing_gs_pose_refinement() :

    # Configs
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda") 
    scene_path = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test/new_tests/livingroom"
    test_path = f"{scene_path}/case_32"
    keyframe_names = ["keyframe2.png", "keyframe4.png", "keyframe6.png","keyframe8.png", "keyframe9.png", "keyframe10.png", 
                      "keyframe11.png", "keyframe12.png", "keyframe13.png", "keyframe14.png", "keyframe15.png", "keyframe16.png"]
    
    poses_dict = {}
    init_img_name = "init_img.png"
    keyframe_names_received = []
    des_img_pose = MyUtils.load_frst_dict_np_elmnt(f"{test_path}/init_des_trans.npy")

    # Load init img infos
    init_img = ImageHandling.load_np_img(f"{test_path}/{init_img_name}")
    init_pose = LinAlgeb.get_eye_matrix(4)
    poses_dict[init_img_name] = init_pose


    # Getting P0
    masked_moge_points_cam_init, masked_moge_colors_init, all_moge_points_cam_init, all_moge_colors_init, moge_mask_init, moge_intrins, normals_init = GaussiansHandling.get_moge_points(
            init_img, model_reso_lvl=moge_model_reso_lvl, depth_edge_threshold=moge_depth_edge_threshold, moge_model=moge_model, use_fp16_bool=False) 

    # Getting Intrinsics
    fx_moge = moge_intrins[0, 0].item() * CAM_W
    fy_moge = moge_intrins[1, 1].item() * CAM_H
    cx_moge = moge_intrins[0, 2].item() * CAM_W
    cy_moge = moge_intrins[1, 2].item() * CAM_H
    moge_intrins = np.array([
    [fx_moge, 0.0, cx_moge],
    [0.0, fy_moge, cy_moge],
    [0.0, 0.0, 1.0]], dtype=np.float64)
    fov_x_rad = 2 * np.arctan(CAM_W / (2 * fx_moge))
    fov_x_deg = np.rad2deg(fov_x_rad)
    moge_intrins_gs = GaussiansHandling.turn_cam_matrix_to_gsplat_format(moge_intrins)
    ibvs_tools = IbvsTools(CAM_W, CAM_H, fx=fx_moge, fy=fy_moge, cx=cx_moge, cy=cy_moge)
    
    # Init GS
    _, txtr_mask = ImageHandling.compute_texturemap_and_mask(init_img, threshold=0.08)
    gaussians0 = GaussiansHandling.turn_image_to_gaussians(init_img, all_moge_points_cam_init, moge_mask_init, txtr_mask, all_moge_colors_init)


    refi_step=0
    nbr_itrs = 150
    for keyframe_name in keyframe_names :
        main_kf_name=keyframe_name.removesuffix(".png")
        refi_step+=1

        # Load the keyframe
        keyframe_names_received.append(keyframe_name)
        keyframe = ImageHandling.load_np_img(f"{test_path}/{keyframe_name}")

        # Init kf pose via pnp
        matched_2ds_init, matched_2ds_curr_kf = ibvs_tools.filtered_matching(init_img, keyframe, max_matches=80, mask1=moge_mask_init)
        init_3ds_matching_kf = MeshHandling.get_corresp_3d_points(matched_2ds_init, moge_mask_init, all_moge_points_cam_init)
        R, t, _ = PosesHandling.estimate_pose_pnp_ransac(init_3ds_matching_kf, matched_2ds_curr_kf, moge_intrins)
        R_relative = R.T; t_relative = (-R.T @ np.asarray(t).reshape(3, 1)).ravel()
        pnp_pose_relative = LinAlgeb.get_homog_frm_rt(R_relative, t_relative)
        refined_kf_pose = np.asarray(init_pose, dtype=np.float64) @ pnp_pose_relative

        # Refine pose with diff_rend
        #refined_kf_pose, loss_history = PosesHandling.optimize_keyframe_pose(
        #                initial_pose_np=refined_kf_pose, frame=keyframe, gaussians=gaussians0, K=moge_intrins, W=CAM_W, H=CAM_H, iterations=50)
        
        # Add the pose to the dictio
        poses_dict[keyframe_name] = refined_kf_pose

        # Get MoGe Points of keyframe
        _, _, all_moge_points_cam, all_moge_colors, moge_mask, _, _ = GaussiansHandling.get_moge_points(
            keyframe, fov_x=fov_x_deg, model_reso_lvl=moge_model_reso_lvl, depth_edge_threshold=moge_depth_edge_threshold, moge_model=moge_model, use_fp16_bool=False)
        all_moge_points = LinAlgeb.transform_points_to_world(all_moge_points_cam, refined_kf_pose)

        # Align its cloud with init moge
        matched_3d_points_img1, matched_3d_points_kf = MeshHandling.get_corresp_3d_point_matches(
            matched_2ds_init, matched_2ds_curr_kf, moge_mask_init, moge_mask, all_moge_points_cam_init, all_moge_points)
        all_moge_points, _ = MeshHandling.estimate_and_apply_z_offset(
            cloud1=matched_3d_points_img1, cloud2=matched_3d_points_kf, cloud2_full=all_moge_points, pose2_ofthecamcloud=refined_kf_pose)

        # Get low opacity & texture mask
        _, _, alpha_map = GaussiansHandling.render_gs_pic(*gaussians0, refined_kf_pose, moge_intrins_gs, CAM_W, CAM_H, rndr_alpha=True)
        low_alpha_mask = GaussiansHandling.get_low_alpha_mask(alpha_map, alpha_threshold=0.5)
        _, txtr_mask = ImageHandling.compute_texturemap_and_mask(keyframe, threshold=0.08)
        final_mask = low_alpha_mask & moge_mask

        # Turning the pixels to gaussians using opacity and texture mask
        new_gaussians = GaussiansHandling.turn_image_to_gaussians(keyframe, all_moge_points, final_mask, txtr_mask, all_moge_colors)
        gaussians0 = GaussiansHandling.merge_2_gaussians(gaussians0, new_gaussians)
       
        # Train all gaussians and all kf poses
        poses_dict, gaussians0, _ = GaussiansHandling.optimize_poses_gaussians(
            keyframe_names_received, poses_dict, test_path, moge_intrins_gs, gaussians0, number_iters=nbr_itrs, last_2_kfs_weight=1, pose_learning_rate=0.005)
       
        # Saving des img for that kf refinement
        des_render, _ = GaussiansHandling.render_gs_pic(*gaussians0, des_img_pose, moge_intrins_gs, CAM_W, CAM_H)
        ImageHandling.save_img(des_render, f"des_render", f"{test_path}/{main_kf_name}")

        # Saving all keyframes render for the curr keyframe refinement
        for kf_name, kf_pose in poses_dict.items():
            kf_name = kf_name.removesuffix(".png")
            keyframe_render2, _ = GaussiansHandling.render_gs_pic(*gaussians0, kf_pose, moge_intrins_gs, CAM_W, CAM_H)
            ImageHandling.save_img(keyframe_render2, f"{kf_name}_render", f"{test_path}/{main_kf_name}")


    # Saving the final desired_img
    des_img = ImageHandling.load_np_img(f"{test_path}/des0.png")
    des0_mask = MyUtils.load_frst_dict_np_elmnt(f"{test_path}/des_masks.npy")
    des_img_pose, _ = PosesHandling.optimize_keyframe_pose(des_img_pose, des_img, gaussians0, moge_intrins_gs, CAM_W, CAM_H, iterations=300, frame_mask=des0_mask)
    des_render, _ = GaussiansHandling.render_gs_pic(*gaussians0, des_img_pose, moge_intrins_gs, CAM_W, CAM_H)
    ImageHandling.save_img(des_render, f"des_render_after", test_path)



#______________________________________________________________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________________________________________________________________________________________________


def main() :

    """
    testing_gs_pose_refinement()
    return
    """

    # Load Main compos and elements _______________________________________________________________________________________  

    # Loading moge, gs1, gs1_intrins, making the mesh_handling, 
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda") 
    gaussians1 = GaussiansHandling.load_gaussians_from_ply(gs1_ply_path)
    gs1_recon = PosesHandling.get_recons(gs1_sfm_path)
    K_gs1 = PosesHandling.get_cam_matrix(gs1_recon)
    intrins_gs1 = GaussiansHandling.turn_cam_matrix_to_gsplat_format(K_gs1)


         
     # Improve the GT des img _____________________________________________________________________
  
    """
    # Paths
    scene_path = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test/new_tests/playroom"
    test_path = f"{scene_path}/case_19"
    global_gs_sfm_align_path = f"{scene_path}/gs1_sfm_aligned"
  
    gt_des_name = "level_4_19_des0.png"
  
    # Get desimg
    des_img = ImageHandling.load_np_img(f"{test_path}/des0.png")
    des0_mask = MyUtils.load_frst_dict_np_elmnt(f"{test_path}/des_masks.npy")
  
    # Gt init pose
    global_gs_align_recons = PosesHandling.get_recons(global_gs_sfm_align_path)
    des_gt_pose = PosesHandling.get_img_sfm_pose(global_gs_align_recons, gt_des_name)
    
    # optimize des0's gt pose on gs1
    des_gt_pose, _ = PosesHandling.optimize_keyframe_pose(des_gt_pose, des_img, gaussians1, intrins_gs1, CAM_W, CAM_H, iterations=2000, ssim_weight=0.2, frame_mask=des0_mask, learning_rate=1e-4, downscale=6)
  
    des_render, _ = GaussiansHandling.render_gs_pic(*gaussians1, des_gt_pose, intrins_gs1, CAM_W, CAM_H)
    ImageHandling.save_img(des_render, f"gt_des_dw_2ks", test_path)

    return
    """
    

    #"""
    fx_gs1 = fx_moge = K_gs1[0, 0]
    fy_gs1 = fy_moge = K_gs1[1, 1]
    cx_gs1 = cx_moge = K_gs1[0, 2]
    cy_gs1 = cy_moge = K_gs1[1, 2]   
    #"""

    """
    fx_gs1 = fx_moge = 727.6869659423828
    fy_gs1 = fy_moge = 727.6869812011719
    cx_gs1 = cx_moge = 632.0
    cy_gs1 = cy_moge = 416.0 

    moge_intrins  = np.array([
    [fx_gs1, 0.0, cx_gs1],
    [0.0, fy_gs1, cy_gs1],
    [0.0, 0.0, 1.0]], dtype=np.float64)
    """
    
    # Get fovx
    #"""
    fov_x_rad = 2 * np.arctan(CAM_W / (2 * fx_gs1))
    fov_x_deg = np.rad2deg(fov_x_rad)
    #"""
    
    # Init img & init pose gs1 & P0
    #"""
    init_img = ImageHandling.load_np_img(f"{gs1_sfm_path}/images/{init_img_gs1_name}")  
    init_img_pose_gs1 = PosesHandling.get_img_sfm_pose(gs1_recon, init_img_gs1_name)
    #"""

    # Getting P0
    #"""
    masked_moge_points_cam_init, masked_moge_colors_init, all_moge_points_cam_init, all_moge_colors_init, moge_mask_init, _ , normals_init = GaussiansHandling.get_moge_points(
        init_img, model_reso_lvl=moge_model_reso_lvl, depth_edge_threshold=moge_depth_edge_threshold, moge_model=moge_model, use_fp16_bool=False, fov_x=fov_x_deg) 
    init_moge_points_o3d =  MeshHandling.turn_points_to_o3d(masked_moge_points_cam_init, masked_moge_colors_init)
    #"""

    """
    masked_moge_points_cam_init, masked_moge_colors_init, all_moge_points_cam_init, all_moge_colors_init, moge_mask_init, moge_intrins, normals_init = GaussiansHandling.get_moge_points(
            init_img, model_reso_lvl=moge_model_reso_lvl, depth_edge_threshold=moge_depth_edge_threshold, moge_model=moge_model, use_fp16_bool=False) 
    init_moge_points_o3d =  MeshHandling.turn_points_to_o3d(masked_moge_points_cam_init, masked_moge_colors_init)        
    """
    
    
    # Get the intrins of moge
    """
    fx_gs1 = fx_moge = moge_intrins[0, 0].item() * CAM_W
    fy_gs1 = fy_moge = moge_intrins[1, 1].item() * CAM_H
    cx_gs1 = cx_moge = moge_intrins[0, 2].item() * CAM_W
    cy_gs1 = cy_moge = moge_intrins[1, 2].item() * CAM_H
    """
    
    #"""
    moge_intrins = np.array([
    [fx_moge, 0.0, cx_moge],
    [0.0, fy_moge, cy_moge],
    [0.0, 0.0, 1.0]], dtype=np.float64)
    #"""
    
    """
    mesh_handling = MeshHandling(CAM_W, CAM_H, fx=fx_moge, fy=fy_moge, cx=cx_moge, cy=cy_moge) 
    ibvs_tools = IbvsTools(CAM_W, CAM_H, fx=fx_moge, fy=fy_moge, cx=cx_moge, cy=cy_moge)
    """



    """
    new_case_test_path = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test/thehouse/thehouse_case1/level_3/case_3_old/"
    des_img = ImageHandling.load_np_img(f"{new_case_test_path}/desired_imgs/des0.png")
    des0_ibvs_infos_npy_path = f"{new_case_test_path}/des0_ibvs_infos.npy"
    dyna_ibvs(gaussians1, intrins_gs1, init_img_pose_gs1, des_img, des0_ibvs_infos_npy_path, f"{new_case_test_path}/des0_matches")
    return
    """


    #___________ for visua 
    """
    surfaces= MyUtils.load_frst_dict_np_elmnt("/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/other_tests/gs_init_frm_nmrls_test/livingroom/surfaces.npy")
    mesh_handling.render_planar_structure_imgs(all_moge_points_cam_init, surfaces, visualize=True)
    return
    """


    # Saving init infos_______________________
    
    #"""
    # Save init img, and P0.o3d
    ImageHandling.save_img(init_img, "init_img", case_test_path)  
    
    # Save P0 infos
    MeshHandling.save_o3dpcd(init_moge_points_o3d, moge_path) 
    MyUtils.save_arrays_to_npy(f"{parent_case_test_path}/all_moge_points1.npy", 0, all_moge_points_cam_init)
    MyUtils.save_arrays_to_npy(f"{parent_case_test_path}/all_moge_colors1.npy", 0, all_moge_colors_init)
    MyUtils.save_arrays_to_npy(f"{parent_case_test_path}/moge_mask1.npy", 0, moge_mask_init)

    # Adding intrs to text    
    intrins_line = f"fx_gs1 {fx_gs1} , fy_gs1 {fy_gs1}, cx_gs1 {cx_gs1}, cy_gs1 {cy_gs1} \nfx_moge {fx_moge} , fy_moge {fy_moge}, cx_moge {cx_moge}, cy_moge {cy_moge} \no3d intrins used are same as gs1 ones"
    MyUtils.add_line_to_text(configs_path, "Intrinsics______")
    MyUtils.add_line_to_text(configs_path, intrins_line)
    #"""



    # Generate des imgs from different levels _________________________________________________________________________________

    """
    #for diff_lvl in range(1, 4):          # Levels 1, 2, 3
    for diff_lvl in range(4, 5):     
        lvl_path = Path(f"{case_test_path}/all_lvl_des") / f"level_{diff_lvl}"
        lvl_path.mkdir(parents=True, exist_ok=True)

        #for tst_nbr in range(1, 11):      # Tests 1 to 10
        for tst_nbr in range(1, 101):      # Tests 1 to 10
            print(f"Generating level {diff_lvl}, test {tst_nbr}")

            moge_des_pose = generate_random_des_pose(
                init_img, all_moge_points_cam_init,
                moge_mask_init, mesh_handling,
                init_moge_points_o3d, diff_lvl)

            des_estim_img, depth = mesh_handling.render_mesh_pic([init_moge_points_o3d], moge_des_pose)

            # Save initimg, moge0, des0_mask, moge_des_pose
            ImageHandling.save_img(des_estim_img, f"{tst_nbr}_des0", str(lvl_path))
            des_mask = ImageHandling.get_mask_frm_depth(depth) 
            MyUtils.save_arrays_to_npy(f"{case_test_path}/all_lvl_des/level_{diff_lvl}/{tst_nbr}_desmask.npy", 0, des_mask)
            MyUtils.save_arrays_to_npy(f"{case_test_path}/all_lvl_des/level_{diff_lvl}/{tst_nbr}_des_moge_pose.npy", 0, moge_des_pose)
      
    #return
    """


    

    # Align all saved des0 imgs frm all lvls ________________________________________________________

    """
    # Get all estimated desired images.
    des_imgs_paths = get_lvl_des_paths(case_test_path, levels=[4], tst_nbrs=range(1, 101))
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
    if not (os.path.isdir(gs1_sfm_aligned_path)) :
        MyUtils.copy_any(gs1_sfm_path, gs1_sfm_aligned_path, overwrite=True)
    PosesHandling.align_new_images(unique_des_paths, gs1_sfm_aligned_path, sequential=True)
    
    #return
    """
    




    # After alignement, we render the GT corresps to all des0 and save ______________________________________________________________

    """    
    # Get all estimated desired images.
    des_imgs_paths = get_lvl_des_paths(case_test_path, levels=4, tst_nbrs=range(1, 101))

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
    
    return
    """
    

    # Apply IBVS on GT des imgs _________________________________________________________________________________________________________  
    """
    # pick the des i want, make matches folder next to each one     
    gt_des_paths = get_lvl_des_paths(case_test_path, levels=[4], tst_nbrs=[1, 4, 7, 12, 15, 16, 27, 34, 42, 46], des_type= "gt_des")

    for gt_des_path in map(Path, gt_des_paths):
        diff_lvl = int(gt_des_path.parent.name.split("_")[-1])
        tst_nbr = int(gt_des_path.stem.split("_")[0])
        matches_save_path = gt_des_path.parent / f"{tst_nbr}_matches_folder"

        # Skip images whose matches folder already exists
        if matches_save_path.exists():
            continue

        matches_save_path.mkdir(parents=True, exist_ok=True)
        gt_ibvs_path = gt_des_path.parent / f"{tst_nbr}_gt_ibvs.npy"
        gt_des_img = ImageHandling.load_np_img(gt_des_path)

        # Run dyna IBVS on gt_des
        print(f"Running level {diff_lvl}, test {tst_nbr}")
        try:
            dyna_ibvs(
                gaussians1, intrins_gs1, init_img_pose_gs1,
                gt_des_img, gt_ibvs_path, matches_save_path)
        except Exception as error:
            print(
                f"\nError in gt_ibvs for level {diff_lvl}, "
                f"test {tst_nbr}: {type(error).__name__}: {error}")
            traceback.print_exc()
            continue
    
    return
    """


    # Updating the Gt poses via diffr rendering ______________________________________________________________
    """
    des_imgs_paths = get_lvl_des_paths(case_test_path, levels=[4], tst_nbrs=[44, 98])
    for des_img_path in des_imgs_paths : 

        diff_lvl, case_nbr = get_level_and_case(des_img_path)
        new_case_test_path = f"{parent_case_test_path}/level_{diff_lvl}/case_{case_nbr}"
        update_case_paths(new_case_test_path)

        all_des_path = f"{parent_case_test_path}/all_lvl_des/level_{diff_lvl}"
        
        # loading des0 and its mask
        des0_img = ImageHandling.load_np_img(f"{all_des_path}/{case_nbr}_des0.png")
        des0_mask = MyUtils.load_frst_dict_np_elmnt(f"{all_des_path}/{case_nbr}_desmask.npy")

        # Load Gt_des pose
        gs1_recon_align = PosesHandling.get_recons(gs1_sfm_aligned_path)
        gt_des_pose_gs1 = PosesHandling.get_img_sfm_pose(gs1_recon_align, f"level_{diff_lvl}_{case_nbr}_des0.png")

        gt_des_img, _ = GaussiansHandling.render_gs_pic(*gaussians1, gt_des_pose_gs1, intrins_gs1, CAM_W, CAM_H)

        ImageHandling.plot_2_imgs(des0_img, gt_des_img)

        refined_gt_des_pose_gs1, loss_histo = PosesHandling.optimize_keyframe_pose(gt_des_pose_gs1, des0_img, *gaussians1, intrins_gs1, CAM_W, CAM_H, iterations=500)
        refined_gt_des_img, _ = GaussiansHandling.render_gs_pic(*gaussians1, refined_gt_des_pose_gs1, intrins_gs1, CAM_W, CAM_H)
        print(loss_histo)
        ImageHandling.save_img(refined_gt_des_img, f"{case_nbr}_aRefined_gt.png", all_des_path)        

    return
    """


 
    # Prepare the case test folders for the clean desired imgs _________________________________________________________________________________
    
    des_imgs_paths = get_lvl_des_paths(case_test_path, levels=[4], tst_nbrs=[38])
     
    for des_img_path in des_imgs_paths : 

        try :

            # Change the paths to the new_case_test_path
            diff_lvl, case_nbr = get_level_and_case(des_img_path)
            new_case_test_path = f"{parent_case_test_path}/level_{diff_lvl}/case_{case_nbr}"
            
            #"""
            if os.path.isdir(new_case_test_path):
                 continue
            #"""
            
            update_case_paths(new_case_test_path)

            # make initial main folders and files
            prepare_test_case(des_img_path, parent_case_test_path)
            
            # Load the des pose
            moge_des_trans = MyUtils.load_frst_dict_np_elmnt(init_des_trans_npy_path)
            #des_estim_img, depth = mesh_handling.render_mesh_pic([moge_points_o3d], moge_des_trans)
                
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
            stop_event = ctx.Event()

            # Get I*0 to start ibvs, init_frame for ibvs next frame selection
            des_img = ImageHandling.load_np_img(f"{des_imgs_path}/des0.png")

            # Get init pose in gs1 for ibvs start, and des poses in gs1 for GT
            gs1_recon_align = PosesHandling.get_recons(gs1_sfm_aligned_path)
            gt_des_pose_gs1 = PosesHandling.get_img_sfm_pose(gs1_recon_align, f"level_{diff_lvl}_{case_nbr}_des0.png")

            # Load clouds1 infos
            """
            all_moge_points_init = MyUtils.load_frst_dict_np_elmnt(f"{parent_case_test_path}/all_moge_points1.npy")
            all_moge_colors1 = MyUtils.load_frst_dict_np_elmnt(f"{parent_case_test_path}/all_moge_colors1.npy")
            moge_mask1 = MyUtils.load_frst_dict_np_elmnt(f"{parent_case_test_path}/moge_mask1.npy")
            """
            
            des0_ibvs_infos_npy_path = f"{new_case_test_path}/des0_ibvs_infos.npy"
            #dyna_ibvs(gaussians1, intrins_gs1, init_img_pose_gs1, des_img, des0_ibvs_infos_npy_path, f"{new_case_test_path}/des0_matches")

            # Starting the ibvs and gs loops after getting all the infos we need
            p1 = ctx.Process(target=dyna_ibvs_loop, args=(shared, stop_event, kf_queue, des_img_queue, kf_event, gs1_ply_path, init_img_pose_gs1, gt_des_pose_gs1, init_img, des_img, fx_moge, fy_moge, new_case_test_path))
            #p2 = ctx.Process(target=local_cloud_loop, args=(shared, stop_event, kf_queue, des_img_queue, des_img_event, all_moge_points_cam_init, masked_moge_points_cam_init, all_moge_colors_init, masked_moge_colors_init, moge_mask_init, moge_des_trans, moge_intrins, init_img, new_case_test_path))
            #p2 = ctx.Process(target=local_mesh_loop, args=(shared, stop_event, kf_queue, des_img_queue, des_img_event, all_moge_points_cam_init, all_moge_colors_init, moge_mask_init, moge_des_trans, moge_intrins, init_img, K_gs1, new_case_test_path))
            p2 = ctx.Process(target=local_gs_loop, args=(shared, stop_event, kf_queue, des_img_queue, des_img_event, all_moge_points_cam_init, all_moge_colors_init, moge_mask_init, normals_init, moge_des_trans, moge_intrins, init_img, new_case_test_path))

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




    











    
if __name__ == "__main__":
    main()

























    """
    # The init gs tests ___________________________________________________________________________________________
    #______________________________________________________________________________________________________________


    # Configs
    save_path = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/other_tests/gs_init_frm_nmrls_test/playroom"
    new_case_test_path = f"{parent_case_test_path}/level_1/case_6"
    update_case_paths(new_case_test_path) 
    recons_gs2 = PosesHandling.get_recons(sfm_path)
    
    # init pose in gs2 nd align moges
    init_pose_gs2 = PosesHandling.get_img_sfm_pose(recons_gs2, "init_img.png")
    all_moge_points_init = LinAlgeb.transform_points_to_world(all_moge_points_cam_init, init_pose_gs2)
    all_moge_points_init_o3d = MeshHandling.turn_points_to_o3d(all_moge_points_init, all_moge_colors_init)

    # getting kf and kf's gs2_pose
    curr_keyframe = ImageHandling.load_np_img(f"{sfm_path}/images/keyframe1.png")
    kf_pose_gs2 = PosesHandling.get_img_sfm_pose(recons_gs2, "keyframe1.png")

    # getting kf's moge points nd align it with gs2
    masked_moge_points_cam_kf, masked_moge_colors_kf, all_moge_points_cam_kf, all_moge_colors_kf, moge_mask_kf, moge_intrins, normals_kf = GaussiansHandling.get_moge_points(
        curr_keyframe, fov_x=fov_x_deg, model_reso_lvl=moge_model_reso_lvl, depth_edge_threshold=moge_depth_edge_threshold,moge_model=moge_model) 
    all_moge_points_kf = LinAlgeb.transform_points_to_world(all_moge_points_cam_kf, kf_pose_gs2)

    # load current gaussian and render kf and get the alpha mask
    gaussians_init = GaussiansHandling.load_gaussians_from_ply(f"/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/other_tests/gs_init_frm_nmrls_test/playroom/new_gs/init/gaussians.ply")
    kf_gs_render, kf_gs_depth, alpha_map = GaussiansHandling.render_gs_pic(*gaussians_init, kf_pose_gs2, intrins_gs1, CAM_W, CAM_H, rndr_alpha=True)
    alpha_mask = GaussiansHandling.get_low_alpha_mask(alpha_map, alpha_threshold=0.5)
    final_kf_mask = alpha_mask & moge_mask_kf
    """
    
    # re align kf moge with gs2 : ftr matching -> getting corresp 3ds -> umeyama 
    """
    matches_init, matches_kf = ibvs_tools.filtered_matching(init_img, curr_keyframe, max_matches=30) # both maks to def None cuz both kf nd init are frm gs1 with no depthedges 
    matched_3d_points_init, matched_3d_points_kf = MeshHandling.get_corresp_3d_point_matches(matches_init, matches_kf, moge_mask_init, moge_mask_kf, all_moge_points_init, all_moge_points_kf)        
    _, s, R, t = MeshHandling.align_points(matched_3d_points_kf, matched_3d_points_init)
    all_moge_points_kf_fltnd = LinAlgeb.reshape_to_flat(all_moge_points_kf)
    all_moge_points_kf_fltnd = (s * (R @ all_moge_points_kf_fltnd.T)).T + t
    all_moge_points_kf = LinAlgeb.reshape_to_image(all_moge_points_kf_fltnd, CAM_H, CAM_W)
    """


    # get surfaces of the new points of the keyframe
    """
    normal_groups, normal_label_map = MeshHandling.group_points_by_normals(points=all_moge_points_kf, normals=normals_kf, valid_mask=final_kf_mask)
    organized_sub_surfaces = MeshHandling.organize_planar_subsurfaces(points=all_moge_points_kf, label_map=normal_label_map)
    #surfaces_img, subsurfaces_img = mesh_handling.render_planar_structure_imgs(all_moge_points_kf_aligned, organized_sub_surfaces, kf_pose_gs2)
    MyUtils.save_arrays_to_npy(f"{save_path}/organized_sub_surfaces_kf.npy", 0, organized_sub_surfaces)   
    #return
    """

    """
    surfaces_kf = MyUtils.load_frst_dict_np_elmnt(f"{save_path}/organized_sub_surfaces_kf.npy") 
    surfaces_kf_img, subsurfaces_kf_img = mesh_handling.render_planar_structure_imgs(all_moge_points_kf, surfaces_kf, kf_pose_gs2)
    ImageHandling.save_img(subsurfaces_kf_img, "subsurfaces_kf_img", save_path)
    
    masked_kf = ImageHandling.apply_mask_on_img(curr_keyframe, final_kf_mask)
    ImageHandling.save_img(masked_kf, "masked_kf", save_path)

    texture_map_kf, texture_mask_kf = ImageHandling.compute_texturemap_and_mask(curr_keyframe)
    ImageHandling.save_img(texture_mask_kf, "txtr_mask_kf", save_path)


    gaussians_kf = GaussiansHandling.turn_surfaces_to_gaussians(points=all_moge_points_kf, colors=all_moge_colors_kf, organized_subsurfaces=surfaces_kf, 
                                                             texture_mask=texture_mask_kf)

    gaussians = GaussiansHandling.merge_2_gaussians(gaussians_init, gaussians_kf)

    kf_render_new_gs, _ = GaussiansHandling.render_gs_pic(*gaussians, kf_pose_gs2, intrins_gs1, CAM_W, CAM_H)
    ImageHandling.save_img(kf_render_new_gs, "kf_render_new_gs", save_path)

    GaussiansHandling.turn_gaussians_to_ply(gaussians, f"{save_path}/gaussians.ply")

    return
    """



    # Turning surfaces to gaussians ____________________________________________________________
    """
    surfaces_init = MyUtils.load_frst_dict_np_elmnt(f"{save_path}/organized_sub_surfaces.npy")   
    texture_map, texture_mask = ImageHandling.compute_texturemap_and_mask(init_img)
    ImageHandling.save_img(texture_mask, "txtr_mask", save_path)
    gaussians = GaussiansHandling.turn_surfaces_to_gaussians(points=all_moge_points_init_scaled, colors=all_moge_colors_init, organized_subsurfaces=surfaces_init, 
                                                             texture_mask=texture_mask, scale_multiplier=1.5)
    GaussiansHandling.turn_gaussians_to_ply(gaussians, f"{save_path}/gaussians.ply")
    #return
    """



    # Render a representation from all keyframes____________________________________________________________________________________

    """
    save_path_rndrs = f"{save_path}/new_gs/init3/"
    
    # Get des pose gs2
    init_des_trans = MyUtils.load_frst_dict_np_elmnt(init_des_trans_npy_path) 
    des_pose_gs2 = LinAlgeb.transform_pose(init_des_trans, init_pose_gs2)

    # Load the saved gaussian
    gaussians = GaussiansHandling.load_gaussians_from_ply(f"{save_path}/gaussians.ply")

    # Render the desired img and save it
    #des_rndr, _ = mesh_handling.render_mesh_pic([all_moge_points_scaled_o3d], des_pose_gs2)
    des_rndr, _ = GaussiansHandling.render_gs_pic(*gaussians, des_pose_gs2, intrins_gs1, CAM_W, CAM_H)
    ImageHandling.save_img(des_rndr, "des_rndr", save_path_rndrs)

    # Render all keyframes and save them
    keyframe_names=["init_img.png", "keyframe1.png", "keyframe2.png", "keyframe3.png", "keyframe4.png", "keyframe5.png"] 
    for kf_name in keyframe_names :
        kf_pose_gs2 = PosesHandling.get_img_sfm_pose(recons_gs2, kf_name)
        #kf_rndr, _ = mesh_handling.render_mesh_pic([all_moge_points_scaled_o3d], kf_pose_gs2)
        kf_rndr, _ = GaussiansHandling.render_gs_pic(*gaussians, kf_pose_gs2, intrins_gs1, CAM_W, CAM_H)
        ImageHandling.save_img(kf_rndr, f"{kf_name}_rndr", save_path_rndrs)

    return
    """



    # Training a gs ____________________________________________________________________________
    """
    GaussiansHandling.run_gs_training(sfm_path=sfm_path, output_path=f"{save_path}/inria_output", gs_reso=gs_reso, gs_nbr_itrs=150, curr_case_gs2_path=f"{save_path}/gaussians.ply")
    return
    """
    #________________________________________________________________________________________________





