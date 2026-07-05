import multiprocessing as mp
import os
import sys
import open3d as o3d
import numpy as np
from utils.lin_algeb import LinAlgeb
import matplotlib.pyplot as plt
import math
from utils.main_visualizer import MainVisualizer
from utils.image_handling import ImageHandling
from utils.gaussians_handling import GaussiansHandling
import torch
from typing import List
from utils.poses_handling import PosesHandling
from utils.my_utils import MyUtils
from utils.mesh_handling import MeshHandling
import pycolmap
import shutil
from moge.model.v2 import MoGeModel
import time
from datetime import datetime
from utils.ibvs_tools import IbvsTools
from wcwidth import wcswidth
import emoji


# Add accelerated_features to our Python paths , so that when featx script gets executed it will know where to find the modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(project_root, "accelerated_features"))
from modules.xfeat import XFeat






# Paths___________________________________________________________

# Main_tests path for shortcuts
scene_name = "kitchen"
case_nbr = 1
main_test_path = "my_results/online_ibvs_test"
case_test_path = f"{main_test_path}/{scene_name}/{scene_name}_case{case_nbr}"


# Results txt files
results_txt_file = f"{case_test_path}/results.txt"
ibvs_infos_npy_path = f"{case_test_path}/ibvs_infos.npy"
des_depths_npy_path = f"{case_test_path}/des_depths.npy"
ibvs_frame_depths_npy_path = f"{case_test_path}/ibvs_frame_depths.npy"


# GS_inria_path for training
gs_inria_path = "gaussian_splatting2"
gs_nbr_itrs = 50
inria_saved_ply_path = f"{case_test_path}/gs2s/inria_output/point_cloud/iteration_{gs_nbr_itrs}/point_cloud.ply"


# GS1 paths
gs1_sfm_path = f"{case_test_path}/gs1_sfm_aligned"
gs1_ply_path = f"{main_test_path}/{scene_name}/real_scene/{scene_name}.ply"

# All States to be saved, paths
gs2s_dir_path = f"{case_test_path}/gs2s"
sfms_path = f"{case_test_path}/sfms"
des_imgs_path = f"{case_test_path}/desired_imgs"
keyframes_path = f"{case_test_path}/keyframes"
real_frames_path = f"{case_test_path}/ibvs_frames/real_frames"
matches_frames_path = f"{case_test_path}/ibvs_frames/matches_frames"

# init & des imgs info
init_img_name_sfm1 = "DSCF5893.JPG"
des_img_name_sfm1 = des_img_name_sfm_gs2 = gt_des_name = "des0.png"


# Configs__________________________________________________________
np.set_printoptions(precision=8, suppress=False)
_xfeat = None


# Robot camera______________________________________________________
CAM_W, CAM_H =  3115, 2076


# IBVS Vars_________________________________________________________
lambda_gain = 0.1
dt_ibvs = 0.03
ibvs_nbr_features = 10
gs_reso = 1
moge_reso = 4
max_ibvs_nbr_itrs = 100
kf_motion_ratio = 0.03    # 3% of screen
kf_motion_nbr_features = 100
ibvs_pxl_error_conv = 0.02
ibvs_pxl_error_kf = 0.05



def get_xfeat_model():
    global _xfeat
    if _xfeat is None:
        _xfeat = XFeat()
    return _xfeat





def get_intrins_gs1(gs1_sfm_path):
    recons1 = PosesHandling.get_recons(gs1_sfm_path)
    K = PosesHandling.get_cam_matrix(recons1)
    intrins_gs1 = GaussiansHandling.turn_cam_matrix_to_gsplat_format(K)
    return intrins_gs1






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






def remove_edge_features(coords, mask, nbr_features, radius=2):
    H, W = mask.shape
    valid_indices = []

    for i, (u, v) in enumerate(coords):
        if 0 <= u < W and 0 <= v < H:

            u_min = max(u - radius, 0)
            u_max = min(u + radius + 1, W)
            v_min = max(v - radius, 0)
            v_max = min(v + radius + 1, H)

            neighborhood = mask[v_min:v_max, u_min:u_max]

            if not np.any(neighborhood):  # all False
                valid_indices.append(i)

                if len(valid_indices) >= nbr_features:
                    break

    if len(valid_indices) < nbr_features:
        raise ValueError(
            f"Only {len(valid_indices)} valid features found, "
            f"but {nbr_features} were requested."
        )

    return valid_indices







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




def load_all_trajectories(folder_path, n):
    all_trajectories = []

    for i in range(1, n + 1):
        filename = f"ibvs_poses_{i}.npy"
        filepath = os.path.join(folder_path, filename)

        traj = np.load(filepath, allow_pickle=True)
        all_trajectories.append(traj)

    return all_trajectories








def get_xfeat_kpts(img) :
    xfeat_model = get_xfeat_model()
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)  # (1,3,H,W)
    out = xfeat_model.detectAndCompute(tensor, top_k=2048)[0]
    kpts = out['keypoints']
    desc = out['descriptors']
    return  kpts, desc













# _____ applying moge
"""img = ImageHandling.load_np_img(f"{keyframes_path}/keyframe2.png")
moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda")
mskd_points, mskd_colors , all_moge_points, all_moge_colors, moge_mask  = GaussiansHandling.get_moge_points(img, depth_edge_threshold=0.005, moge_model = moge_model, use_fp16_bool=False, model_reso_lvl=2)
all_moge_points_o3d = MeshHandling.turn_points_to_o3d(mskd_points, mskd_colors)
MeshHandling.visualize_scene([all_moge_points_o3d])
return"""



#_______ rendering a gs
"""gaussians2 = GaussiansHandling.load_gaussians_from_ply(f"{gs2s_dir_path}/gs2_50_101.ply")
recons2 = pycolmap.Reconstruction(f"{sfms_path}/sfm100/sparse/0")
K = PosesHandling.get_cam_matrix(recons2)
intrins_gs2 = GaussiansHandling.turn_cam_matrix_to_gsplat_format(K)
keyframe_names= ["keyframe1.png", "keyframe2.png", "keyframe10.png", "keyframe20.png", "keyframe30.png", "keyframe40.png", "keyframe50.png", "keyframe60.png", "keyframe70.png", "keyframe80.png", "keyframe90.png", "keyframe100.png"]
for keyframe_name in keyframe_names :
    keyframe_pose = PosesHandling.get_img_sfm_pose(recons2, keyframe_name)
    rendered_keyframe, _ = GaussiansHandling.render_gs_pic(*gaussians2, keyframe_pose, intrins_gs2, CAM_W, CAM_H)
    ImageHandling.save_img(rendered_keyframe, keyframe_name, des_imgs_path)
return"""




#___visualizing the aligned points (moge nd sfm) :

"""#_____________________________________________________________

 sfm_moge_colors = all_moge_colors[y, x]
 sfm_moge_colors = sfm_moge_colors[sfm_mask]
 sfm_moge_points_flat = sfm_moge_points.reshape(-1, 3)
 sfm_moge_colors_flat = sfm_moge_colors.reshape(-1, 3)
 print(f"{sfm_moge_points_flat.shape} // {sfm_moge_colors_flat.shape}
 sfm_moge_points_flat = (s * (R @ sfm_moge_points_flat.T)).T +
 sfm_moge_points_o3d = MeshHandling.turn_points_to_o3d(sfm_moge_points_flat, sfm_moge_colors_flat)
 img_sfm_points_3d_o3d = MeshHandling.turn_points_to_o3d(img_sfm_points_3d)
 MeshHandling.visualize_scene([sfm_moge_points_o3d, img_sfm_points_3d_o3d
 retun
#_____________________________________________________________"""





 
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











def dyna_ibvs_loop(shared, kf_queue, des_img_queue, kf_event, gs1_ply_path, init_pose_gs1, gt_des_pose_gs1, init_keyframe, des_img) :

    try:
        msg = f"🔵Ibvs_loop launched"; print(msg)
        log_to_table(msg, 'ibvs', 0)
        _t = time.time()
        gaussians1 = GaussiansHandling.load_gaussians_from_ply(gs1_ply_path)
        intrins_gs1 = get_intrins_gs1(gs1_sfm_path)
        xfeat_model = get_xfeat_model()
        cur_pose_gs1 = init_pose_gs1
        last_keyframe = init_keyframe
        des_kpts, des_desc = get_xfeat_kpts(des_img)
        pxl_error = pose_error = None
        i = 0
        des_vrsn = 0
        final_kf_sent = False
        msg = f"🔵Loaded xfeat and gs1"; dt = time.time()-_t; print(f"{msg} || {dt:.3f} s"); _t = time.time()
        log_to_table(msg, 'ibvs', dt)
        ibvs_tools = IbvsTools(CAM_W, CAM_H)

        while not shared["stop"] :
            i+=1;_t = time.time()
            
            # Check if GS_loop got us new des_img, then update it
            if not des_img_queue.empty():
                while not des_img_queue.empty():
                    des_img = des_img_queue.get()
                    des_vrsn += 1
                    des_kpts, des_desc = get_xfeat_kpts(des_img)
                    msg = f"🔵🔵{i}-Got new des_img({des_vrsn})"; print(msg)
                    log_to_table(msg, 'ibvs', 0)


            # Render new cur_gs_pic and get its depth_map
            cur_gs1_img, cur_gs1_depth_map = GaussiansHandling.render_gs_pic(*gaussians1, T=cur_pose_gs1, K=intrins_gs1, W=CAM_W, H=CAM_H)

            # Notify GS_Loop if it's a keyframe
            if validate_keyframe(last_keyframe, cur_gs1_img, xfeat_model, ibvs_itr_nbr=i, kf_motion_nbr_features= kf_motion_nbr_features, motion_thresh_ratio = kf_motion_ratio) :
                last_keyframe = cur_gs1_img
                kf_queue.put(last_keyframe)
                kf_event.set()
                msg = f"🔵🔵{i}-Keyframe was found and sent to gsloop"; print(msg)
                log_to_table(msg, 'ibvs', 0)

            # 1 - Match current_gs1_frame with des_img
            cur_kpts, cur_desc = get_xfeat_kpts(cur_gs1_img)
            idxs0, idxs1 = xfeat_model.match(cur_desc, des_desc)
            matches_cur = cur_kpts[idxs0]
            matches_des = des_kpts[idxs1]
            matches_cur = matches_cur[:ibvs_nbr_features].to(torch.int).cpu().numpy()
            matches_des = matches_des[:ibvs_nbr_features].to(torch.int).cpu().numpy()


            # Stop if we got occluded or got no matches
            if (len(matches_cur) < 4) :
                shared["stop"] = True
                msg = f"!!!!🔵{i}- Not enough matches during ibvs"; print(msg)
                log_to_table(msg, 'ibvs', 0)
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
            cur_pose_gs1 = update_cam_pose(cur_pose_gs1, V, dt_ibvs)

            # Check if we converged
            if pxl_error < ibvs_pxl_error_conv :
                shared["stop"] = True

            # Check if we need to send a final kf
            if (pxl_error < ibvs_pxl_error_kf) and (not final_kf_sent) :
                last_keyframe = cur_gs1_img
                kf_queue.put(last_keyframe)
                kf_event.set()
                final_kf_sent = True
                msg = f"🔵🔵{i}-Last kf found"; print(msg)
                log_to_table(msg, 'ibvs', 0)

            # Save the infos of each ibvs iteration : 2d_err, 3d_err, condit_nbr, V, matches, pose_in_glbl_gs
            ibvs_infos = [pxl_error, pose_error, LinAlgeb.get_mat_condition_number(L), V, [matches_cur, matches_des],cur_pose_gs1]
            MyUtils.save_arrays_to_npy(ibvs_infos_npy_path, i, ibvs_infos)


            if (i % 5) == 0 :
                log_to_table(msg, 'ibvs', dt)
                cur_mtch_gs_img = ImageHandling.draw_matches(matches_cur, matches_des, cur_gs1_img, des_img)
                ImageHandling.save_img(cur_gs1_img, i, real_frames_path)
                ImageHandling.save_img(cur_mtch_gs_img, i, matches_frames_path)   
                MyUtils.save_arrays_to_npy(ibvs_frame_depths_npy_path, i, cur_gs1_depth_map) 
                MyUtils.cleanup()


    except Exception as e:
            print(f"Error in ibvs_loop {e} !!!!")
            MyUtils.cleanup()
            shared["stop"] = True










def local_gs_loop(shared, kf_queue, des_img_queue, des_img_event) :

    try : 

        import queue
        msg = f"🟢GS_loop launched"; print(msg)
        log_to_table(msg, 'gs', 0); _t = time.time()
        i=0
        kf_vrsn = 0
        last_kf_versn = 0
        moge_model = None
        curr_recons2 = pycolmap.Reconstruction(f"{sfms_path}/sfm0/sparse/0")
        K = PosesHandling.get_cam_matrix(curr_recons2)
        intrins_gs2 = GaussiansHandling.turn_cam_matrix_to_gsplat_format(K)    
        des_pose_gs2 = PosesHandling.get_img_sfm_pose(curr_recons2, des_img_name_sfm_gs2)
        msg = f"🟢{i}-Loaded local_gs infos"; dt = time.time()-_t; print(f"{msg} || {dt:.3f} s")
        log_to_table(msg, 'gs', dt); _t = time.time()        

        while not shared["stop"]:
            i+=1
            # Looping on all kfs (getting last one (probably there is only one keyframe, cuz we can't find more than kf that fast before updating gs)
            curr_keyframe = None
            while not kf_queue.empty():
                curr_keyframe = kf_queue.get()
                kf_vrsn += 1
                msg = f"🟢🟢{i}-Received KF{kf_vrsn}"; dt = time.time()-_t; print(f"{msg} || {dt:.3f} s")
                log_to_table(msg, 'gs', dt); _t = time.time()


            if curr_keyframe is not None:
    
                # Loading moge model once in the beg
                if moge_model is None:
                    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda")
                    msg = f"🟢{i}-Just Loaded Moge model"; dt = time.time()-_t; print(f"{msg} || {dt:.3f} s")
                    log_to_table(msg, 'gs', dt); _t = time.time()

                # Saving this keyframe
                ImageHandling.save_img(curr_keyframe, f"keyframe{kf_vrsn}", keyframes_path)
                msg = f"🟢{i}-Saved KF({kf_vrsn})"; dt = time.time()-_t; print(f"{msg} || {dt:.3f} s"); _t = time.time()

                # Get last saved sfm and align kf with it and save
                MyUtils.copy_any(f"{sfms_path}/sfm{last_kf_versn}", f"{sfms_path}/sfm{kf_vrsn}")
                PosesHandling.align_new_image(new_image_path=f"{keyframes_path}/keyframe{kf_vrsn}.png", sfm_path=f"{sfms_path}/sfm{kf_vrsn}")            
                last_kf_versn = kf_vrsn
                msg = f"🟢{i}-Just aligned KF({kf_vrsn})"; dt = time.time()-_t; print(f"{msg} || {dt:.3f} s")
                log_to_table(msg, 'gs', dt); _t = time.time()
                
                # Get the kf pose in local_gs 
                curr_recons2 = pycolmap.Reconstruction(f"{sfms_path}/sfm{kf_vrsn}/sparse/0")
                try : 
                    curr_keyframe_pose_gs2 = PosesHandling.get_img_sfm_pose(curr_recons2, f"keyframe{kf_vrsn}.png")
                except :
                    msg = f"🟢{i} !!!!! kf {kf_vrsn} was not aligned"; print(msg)
                    log_to_table(msg, 'gs', 0)
                    continue #most cmn errror is img not being aligned, if so we just skip this iteration, and wait for the
                
                # Get rendered_kf
                last_gaussians2 = GaussiansHandling.load_gaussians_from_ply(f"{gs2s_dir_path}/gs2_{i-1}.ply")
                curr_keyframe_gs2_render, _ = GaussiansHandling.render_gs_pic(*last_gaussians2, curr_keyframe_pose_gs2, intrins_gs2, CAM_W, CAM_H)
                msg = f"🟢{i}-Loaded local_gs & rendered KF'({kf_vrsn})"; dt = time.time()-_t; print(f"{msg} || {dt:.3f} s")
                log_to_table(msg, 'gs', dt); _t = time.time()

                # Get txtr map from original kf and render kf, and compute new textured pixels mask
                _, txtr_mask_kf = ImageHandling.compute_texturemap_and_mask(curr_keyframe, threshold=0.05)
                _, txtr_mask_rndr_kf = ImageHandling.compute_texturemap_and_mask(curr_keyframe_gs2_render, threshold=0.01)
                txtr_mask = txtr_mask_kf & (~txtr_mask_rndr_kf)
                msg = f"🟢{i}-Computed texture masks of KFs({kf_vrsn})"; dt = time.time()-_t; print(f"{msg} || {dt:.3f} s")
                log_to_table(msg, 'gs', dt); _t = time.time()

                # Get moge points from kf to align with sfm nd add to local_gs)
                _, _, all_moge_points, all_moge_colors, moge_mask  = GaussiansHandling.get_moge_points(curr_keyframe, depth_edge_threshold=0.05, moge_model = moge_model, use_fp16_bool=False, model_reso_lvl=2)
                final_mask = moge_mask & txtr_mask    # update final mask with moge mask, cuz the pixels with inf 3d value should be avoided
                msg = f"🟢{i}-Got moge points of KF({kf_vrsn})"; dt = time.time()-_t; print(f"{msg} || {dt:.3f} s")
                log_to_table(msg, 'gs', dt); _t = time.time()

                # Get curr_frame sfm's 2ds 3ds to align moge points
                img_sfm_points_2d, img_sfm_points_3d = PosesHandling.get_2d_3d_points_of_img(curr_recons2, f"keyframe{kf_vrsn}.png")

                # Keep only moge points corresp to sfm (& removing masked points (inf values..))
                x = img_sfm_points_2d[:,0].astype(int)
                y = img_sfm_points_2d[:,1].astype(int)
                sfm_moge_points = all_moge_points[y, x]
                sfm_mask = moge_mask[y, x]  # We didn't filter only txtrd ones cuz we want max points for gd alignement
                sfm_moge_points = sfm_moge_points[sfm_mask]
                img_sfm_points_3d = img_sfm_points_3d[sfm_mask]

                # Get the transformation infos between sfm-cloud and moge-cloud
                _, s, R, t = MeshHandling.align_points(sfm_moge_points, img_sfm_points_3d)


                #___________________________________________
                """
                sfm_moge_points_o3d = MeshHandling.turn_points_to_o3d(sfm_moge_points)
                img_sfm_points_3d_o3d = MeshHandling.turn_points_to_o3d(img_sfm_points_3d)
                MeshHandling.visualize_scene([sfm_moge_points_o3d, img_sfm_points_3d_o3d])
                """
                #___________________________________________


                # Flatten all moge points and filter them (keep trusted+textured ones to turn to gaussians)
                all_moge_colors_flat = all_moge_colors.reshape(-1, 3) # turning H,W,3 to H*W,3
                all_moge_points_flat = all_moge_points.reshape(-1, 3).astype(np.float64)
                final_mask_flat = final_mask.reshape(-1)  #turning H,W,1 to H*W,1
                final_moge_colors = all_moge_colors_flat[final_mask_flat]
                final_moge_points = all_moge_points_flat[final_mask_flat]

                # Apply the calculated transformation on the final-clean moge points
                final_moge_points = (s * (R @ final_moge_points.T)).T + t
                msg = f"🟢{i}-Filter & Align moge points with sfm, KF({kf_vrsn})"; dt = time.time()-_t; print(f"{msg} || {dt:.3f} s")
                log_to_table(msg, 'gs', dt); _t = time.time()

                # Downsample, Turn points to gaussians, merge with old ones
                final_moge_points = final_moge_points[::moge_reso]
                final_moge_colors = final_moge_colors[::moge_reso]
                new_gaussians = GaussiansHandling.turn_points_to_gaussians(final_moge_points, final_moge_colors)
                new_gaussians = GaussiansHandling.merge_2_gaussians(last_gaussians2, new_gaussians)
                msg = f"🟢{i}-Turn points to gaussians, KF({kf_vrsn})"; dt = time.time()-_t; print(f"{msg} || {dt:.3f} s"); _t = time.time()
                log_to_table(msg, 'gs', dt)

                # Saving mid-des-img too and the gs2 ply file
                des_img_mid, _ = GaussiansHandling.render_gs_pic(*new_gaussians, des_pose_gs2, intrins_gs2, CAM_W, CAM_H)
                ImageHandling.save_img(des_img_mid, f"des{i-1}_plus", des_imgs_path)
                GaussiansHandling.turn_gaussians_to_ply(new_gaussians, f"{gs2s_dir_path}/gs2_{i}_pre.ply")
                msg = f"🟢{i}-Render & Saving mid_des_img, KF({kf_vrsn})"; dt = time.time()-_t; print(f"{msg} || {dt:.3f} s"); _t = time.time()
                log_to_table(msg, 'gs', dt)

                

            # Train last gs2_pre then save new gs2
            _t = time.time()
            GaussiansHandling.run_gs_training(sfm_path=f"/home/user/Bureau/visual_navigation/IBVS_CODE/{sfms_path}/sfm{kf_vrsn}", output_path= f"/home/user/Bureau/visual_navigation/IBVS_CODE/{gs2s_dir_path}/inria_output", gs_reso = gs_reso, gs_nbr_itrs = gs_nbr_itrs)
            shutil.copy2(inria_saved_ply_path, f"{gs2s_dir_path}/gs2_{i}.ply")
            msg = f"🟢{i}-Trained local_gs for {gs_nbr_itrs} iterations"; dt = time.time()-_t; print(f"{msg} || {dt:.3f} s"); _t = time.time()
            log_to_table(msg, 'gs', dt)

            # Loading the last saved gs2 & render new des img & notify ibvs loop
            last_gaussians2 = GaussiansHandling.load_gaussians_from_ply(f"{gs2s_dir_path}/gs2_{i}.ply")
            des_img, des_depth = GaussiansHandling.render_gs_pic(*last_gaussians2, des_pose_gs2, intrins_gs2, CAM_W, CAM_H)
            ImageHandling.save_img(des_img, f"des{i}", des_imgs_path)
            des_img_queue.put(des_img)
            des_img_event.set()

            #saving some infos about des_img
            msg = f"🟢🟢{i}-Loading local_gs & Rendering new des_img({i})"; dt = time.time()-_t; print(f"{msg} || {dt:.3f} s")
            log_to_table(msg, 'gs', dt); _t = time.time()
            MyUtils.save_arrays_to_npy(des_depths_npy_path, i, des_depth)

            if (i % 5) == 0 :
                MyUtils.cleanup()

    except Exception as e:
        MyUtils.cleanup()
        print(f"Error in gs_loop : {e} !!!!!!!!!!!!!!!!!")
        shared["stop"] = True         














def main() :


    # Start multi-process elements  // we can use fork or spawn (fork is faster and works on ubuntu)
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    shared = manager.dict()
    shared["stop"] = False
    kf_queue = ctx.Queue()
    des_img_queue = ctx.Queue()
    kf_event = ctx.Event()
    des_img_event = ctx.Event()

    # Get init des_img and gt_des_img
    des_img = ImageHandling.load_np_img(f"{des_imgs_path}/des0.png")
    curr_keyframe = ImageHandling.load_np_img(f"{keyframes_path}/init_img.png")
    init_keyframes = []

    # Get init/cur and des poses in gs1
    gs1_recon = pycolmap.Reconstruction(f"{gs1_sfm_path}/sparse/0")
    init_img_pose_gs1 = PosesHandling.get_img_sfm_pose(gs1_recon, init_img_name_sfm1)
    gt_des_pose_gs1 = PosesHandling.get_img_sfm_pose(gs1_recon,gt_des_name)

    # Starting the ibvs and gs loops after getting all the infos we need
    p1 = ctx.Process(target=dyna_ibvs_loop, args=(shared, kf_queue, des_img_queue, kf_event, gs1_ply_path, init_img_pose_gs1, gt_des_pose_gs1, curr_keyframe, des_img))
    p2 = ctx.Process(target=local_gs_loop, args=(shared, kf_queue, des_img_queue, des_img_event))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    return




    for i in range(2, 102):

        _iter_start = time.time()
        print(f"✅ Iteration N°{i} ____________________________________________________________________")
        blank(7)
        log(f"Iteration N°{i}", big=True)

        got_kf = False

        print("✅ Starting IBVS ______________________________________________")
        _t = time.time()

        # Apply IBVS for 50 iterations starting from curr_kf(i-1)
        curr_keyframe_pose_gs1, got_kf, ibvs_itrs, img_error, pose_error = start_dyna_ibvs_loop(gaussians1, curr_keyframe_pose_gs1, gt_des_pose_gs1, curr_keyframe, des_img, i)
        _ibvs_dur = time.time() - _t
        log(f"IBVS of {ibvs_itrs} iterations", _ibvs_dur, big=True)
        log(f"2D error : {img_error}")
        log(f"3D error : {pose_error}")
        log(f"Last Pose : {curr_keyframe_pose_gs1}")
        #keyframes_poses.append(ibvs_poses)

        if got_kf :

            # Get new pose and render curr_kf(i)
            curr_keyframe, _ = GaussiansHandling.render_gs_pic(*gaussians1, curr_keyframe_pose_gs1, intrins_gs1, CAM_W, CAM_H)
            ImageHandling.save_img(curr_keyframe, f"keyframe{i}", keyframes_path)

            # if init KFs just wait until we make the sfm1
            if i < len(init_keyframes) :
                ImageHandling.save_img(curr_keyframe, f"keyframe{i}", f"{sfms_path}/sfm1/images")

            # got to make sure thatin this begining i always return keyframes

            #__________________________________________________________

            # Get new sfm(i) with aligned current_keyframe
            print("✅ Aligning new keyframe _________________________________________ ")

            MyUtils.copy_any(f"{sfms_path}/sfm{last_kf_nbr}", f"{sfms_path}/sfm{i}")

            _t = time.time()
            PosesHandling.align_new_image(new_image_path=f"{keyframes_path}/keyframe{i}.png", sfm_path=f"{sfms_path}/sfm{i}")
            curr_recons2 = pycolmap.Reconstruction(f"{sfms_path}/sfm{i}/sparse/0")
            curr_keyframe_pose_gs2 = PosesHandling.get_img_sfm_pose(curr_recons2, f"keyframe{i}.png")
            log("Align new keyframe with GS2", time.time() - _t, big=True)


            log("Getting new gaussians from keyframe", big=True)
            print("✅ Getting Moge Points from keyframe.. ")
            _t = time.time()
            mskd_points, mskd_colors, all_moge_points, all_moge_colors, moge_mask  = GaussiansHandling.get_moge_points(curr_keyframe, depth_edge_threshold=0.05, moge_model = moge_model, use_fp16_bool=False, model_reso_lvl=2)


            log("Getting Moge Points from keyframe..", time.time() - _t)
            # Get moge points and texture map
            print("✅ Getting texture Map from keyframe.. ")
            _t = time.time()
            _, txtr_mask = ImageHandling.compute_texturemap_and_mask(curr_keyframe, threshold=0.05)

            log("Getting texture Map from keyframe..", time.time() - _t)


            if (i > 2) :

                _t = time.time()
                # render current keyframe from last gs2
                curr_keyframe_gs2_render, _ = GaussiansHandling.render_gs_pic(*last_gaussians2, curr_keyframe_pose_gs2, intrins_gs2, CAM_W, CAM_H)
                #ImageHandling.save_img(curr_keyframe_gs2_render, f"keyframe{i}_render" , keyframes_path)

                # compute final txtr map
                print("✅ Getting Texture map from keyframe's render.. ")
                _, txtr_mask_rndr = ImageHandling.compute_texturemap_and_mask(curr_keyframe_gs2_render, threshold=0.01)
                txtr_mask = txtr_mask & (~txtr_mask_rndr)
                log("Getting Texture map from keyframe's render..", time.time() - _t)

                #plot_img = ImageHandling.plot_4_imgs(curr_keyframe, curr_keyframe_gs2_render, txtr_mask, txtr_mask_rndr, "real_frame", "rendered frame", "final mask", "texture to avoid", show=False, save=True)
                #ImageHandling.save_img(plot_img, f"keyframe{i}_txtr_maps", keyframes_path)

            final_mask = moge_mask & txtr_mask

            # Get curr_frame sfm's 2Ds 3Ds
            print("✅ Getting keyframe sfm's 2d-3d points.. ")
            _t = time.time()
            img_sfm_points_2d, img_sfm_points_3d = PosesHandling.get_2d_3d_points_of_img(curr_recons2, f"keyframe{i}.png")
            log("Getting keyframe sfm's 2d-3d points..", time.time() - _t)

            # Keep only moge points corresp to sfm (& removing masked points (inf values..))
            print("✅ Filter moge points corresponding to sfm 3d points.. ")
            _t = time.time()
            x = img_sfm_points_2d[:,0].astype(int)
            y = img_sfm_points_2d[:,1].astype(int)
            sfm_moge_points = all_moge_points[y, x]
            sfm_mask = moge_mask[y, x]
            sfm_moge_points = sfm_moge_points[sfm_mask]
            img_sfm_points_3d = img_sfm_points_3d[sfm_mask]
            log("Filter moge points corresponding to sfm 3d points..", time.time() - _t)

            # Get the transformation infos between the 2 clouds
            print("✅ Aligning Moge points using Umeyama")
            _t = time.time()
            _, s, R, t = MeshHandling.align_points(sfm_moge_points, img_sfm_points_3d)
            log("Aligning Moge points using Umeyama", time.time() - _t)

            # Flatten all moge points and filter them (keep trusted+textured ones to turn to gaussians)
            print("✅ Filtering Moge points to turn to gaussians")
            _t = time.time()
            all_moge_colors_flat = all_moge_colors.reshape(-1, 3) #turning H,W,3 to H*W,3
            all_moge_points_flat = all_moge_points.reshape(-1, 3).astype(np.float64)
            final_mask_flat = final_mask.reshape(-1)  #turning H,W,1 to H*W,1

            final_moge_colors = all_moge_colors_flat[final_mask_flat]
            final_moge_points = all_moge_points_flat[final_mask_flat]

            # Apply the calculated transformation on the clean moge points
            final_moge_points = (s * (R @ final_moge_points.T)).T + t

            # Downsample, Turn points to gaussians, merge with old ones, and save
            final_moge_points = final_moge_points[::moge_reso]
            final_moge_colors = final_moge_colors[::moge_reso]

            log("Filtering Moge points to turn to gaussians", time.time() - _t)


            print("✅ Turning MoGepoints to Gaussians")
            _t = time.time()
            new_gaussians = GaussiansHandling.turn_points_to_gaussians(final_moge_points, final_moge_colors)
            log("Turning MoGepoints to Gaussians", time.time() - _t)

            if i > 2 :
                print("✅ Merging the new & old Gaussians_____________________________")
                _t = time.time()
                new_gaussians = GaussiansHandling.merge_2_gaussians(last_gaussians2, new_gaussians)
                log("Merging the new & old Gaussians", time.time() - _t, big=True)

            # Saving mid des img too
            des_img_mid, _ = GaussiansHandling.render_gs_pic(*new_gaussians, des_pose_gs2, intrins_gs2, CAM_W, CAM_H)
            ImageHandling.save_img(des_img_mid, f"des{i-1}_plus", des_imgs_path)

            print("✅ Saving new gaussians as gs2_ply")
            _t = time.time()
            GaussiansHandling.turn_gaussians_to_ply(new_gaussians, f"{gs2s_dir_path}/gs2_1_{i}.ply")

            last_kf_nbr = i


        # Train gs2(i)_1 then save gs2(i)_200
        print(f"✅ Training GS2 for {gs_nbr_itrs} itrs ________________________________")
        _t = time.time()
        GaussiansHandling.run_gs_training(sfm_path=f"/home/user/Bureau/visual_navigation/IBVS_CODE/{sfms_path}/sfm{last_kf_nbr}", output_path= f"/home/user/Bureau/visual_navigation/IBVS_CODE/{gs2s_dir_path}/inria_output", gs_reso= gs_reso, gs_nbr_itrs= gs_nbr_itrs)
        log(f"Training GS2 for {gs_nbr_itrs} itrs", time.time() - _t, big=True)


        print("✅ Loading the trained gs2 and render new des_img_____________________________")
        _t = time.time()

        # the gs2_200 gets saved in inria output we copy it to gs2s with name i
        shutil.copy2(inria_saved_ply_path, f"{gs2s_dir_path}/gs2_{gs_nbr_itrs}_{i}.ply")

        # Loading new gaussians
        last_gaussians2 = GaussiansHandling.load_gaussians_from_ply(f"{gs2s_dir_path}/gs2_{gs_nbr_itrs}_{i}.ply")

        # Render new des img from gs2
        des_img, _ = GaussiansHandling.render_gs_pic(*last_gaussians2, des_pose_gs2, intrins_gs2, CAM_W, CAM_H)
        ImageHandling.save_img(des_img, f"des{i}", des_imgs_path)
        log("Loading the trained gs2 and render new des_img",  time.time() - _t, big=True)

        _iter_dur = time.time() - _iter_start
        log(f"Iteration N°{i} TOTAL TIME: {_iter_dur:.2f}s", big=True)

        if (i % 3) == 0 :
            MyUtils.cleanup()

        save_log()


    return






    """
    # functions for results printing in text file
    log_lines = []

    def log(msg, duration=None, big=False):
        if big:
            prefix = "\n" + "_" * 70 + "\n"
            msg = f"✅ {msg}"
        else:
            prefix = ""
            msg = f"✔ {msg}"

        if duration is not None:
            line = f"{prefix}{msg}  ⏱ {duration:.2f}s"
        else:
            line = f"{prefix}{msg}"
        log_lines.append(line)

    def blank(n=1):
        log_lines.extend([""] * n)

    def save_log():
        with open(results_txt_file, "w") as f:
            f.write("\n".join(log_lines))




    log("Configs", big=True)
    log(f"Camera : CAM_W {CAM_W}, CAM_H {CAM_H}, f {FX}")
    log(f"IBVS :lambda {lambda_gain}, dt {dt}, iterations {max_ibvs_nbr_itrs}, features {ibvs_nbr_features}")
    log(f"Keyframe : motion_ratio {kf_motion_ratio}, features {kf_motion_nbr_features}")
    log(f"GS : moge_reso {moge_reso}, gs_reso {gs_reso}, gs_iterations {gs_nbr_itrs}")
    blank(2)
    """




if __name__ == "__main__":

    try:
        main()
    finally:
        MyUtils.cleanup()
