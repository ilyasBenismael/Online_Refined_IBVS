import open3d as o3d
import numpy as np
import torch
from utils.lin_algeb import LinAlgeb
from utils.gaussians_handling import GaussiansHandling
from plyfile import PlyData
from utils.mesh_handling import MeshHandling
from utils.poses_handling import PosesHandling
from utils.image_handling import ImageHandling
from utils.my_utils import MyUtils
from utils.ibvs_tools import IbvsTools 
import pycolmap
import os ,sys
from moge.model.v2 import MoGeModel
import numpy as np
from pathlib import Path

from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib.pyplot as plt

# Add accelerated_features to our Python paths , so that when featx script gets executed it xill know where to find the modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(project_root, "accelerated_features"))
from modules.xfeat import XFeat 



# Main_tests path for shortcuts
scene_name = "thehouse"
case_nbr = 3
CAM_W, CAM_H = 1332, 876
main_test_path = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test"
case_test_path = f"{main_test_path}/{scene_name}/{scene_name}_case{case_nbr}"

# GS1 paths
gs1_sfm_path = f"{main_test_path}/{scene_name}/real_scene/sfm_{scene_name}"
gs1_ply_path = f"{main_test_path}/{scene_name}/real_scene/{scene_name}.ply"
gs1_sfm_aligned_path = f"{case_test_path}/gs1_sfm_aligned"
configs_path = f"{case_test_path}/configs.txt"

# All States to be saved, paths
des_imgs_path = f"{case_test_path}/desired_imgs"
keyframes_path = f"{case_test_path}/keyframes"
sfms_path = f"{case_test_path}/sfms"
gs2s_dir_path = f"{case_test_path}/gs2s"
moge_path = f"{case_test_path}/init_moges.ply"
init_des_trans_npy_path = f"{case_test_path}/init_des_trans.npy"
matches_frames_path = f"{case_test_path}/pre_des_ibvs_matches"
des0_ibvs_infos_npy_path = f"{case_test_path}/des0_ibvs_infos.npy"  
desGT_ibvs_infos_npy_path = f"{case_test_path}/desGT_ibvs_infos.npy"  

# Metrics_path 
des_masks_path = f"{des_imgs_path}/des_masks.npy"
init_des_sim_path = f"{des_imgs_path}/init_des_sim.npy"
all_des_sim_path = f"{des_imgs_path}/all_des_sim.npy"

# o3d Renderer
FX = FY = 0.8 * max(CAM_W, CAM_H)
f=1
CX, CY = CAM_W / 2.0, CAM_H / 2.0
intrins_o3d = o3d.camera.PinholeCameraIntrinsic(
    width=CAM_W,
    height=CAM_H,
    fx=FX,
    fy=FY,
    cx=CX,
    cy=CY
)


# Starting states
did_start = True
trans_exist = True
des0_ready = False
des0_aligned = False 
des_GT_ready = True
ibvs_desGT_converge = True
ibvs_des0_converge = True
neighbour_imgs_ready = False
sfm0_ready = False
init_img_gs2_name = "init_img.png"
get_des_frm_gs = True

init_img_gs1_name = "IMG_6393.jpg"
neighbour_imgs_name = ["IMG_6299.jpg", "IMG_6300.jpg", "IMG_6379.jpg", "IMG_6380.jpg", "IMG_6394.jpg", "IMG_6403.jpg", "IMG_6453.jpg", "IMG_6460.jpg", "IMG_6461.jpg", "IMG_6475.jpg", "IMG_6506.jpg", "IMG_6507.jpg"]

# IBVS Vars
lambda_gain = 0.1
dt_ibvs = 0.03
ibvs_nbr_features = 10
max_ibvs_nbr_itrs = 5000
ibvs_pxl_error_conv = 0.02


# Some configs
np.set_printoptions(precision=2, suppress=False)
xfeat = XFeat()
ibvs_nbr_features = 10
moge_reso = 4
moge_model_reso_lvl = 1
moge_depth_edge_threshold = 0.01




def write_init_config_txt(configs_path) :
    
    # Path where you want to save the file
    txt_path = Path(configs_path)

    with open(txt_path, "w") as f:
        f.write("____ initial configs ____\n\n")

        f.write(f"init_img_gs1_name = {init_img_gs1_name}\n")
        f.write(f"neighbour_imgs_name = {neighbour_imgs_name}\n\n")

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


def get_xfeat_kpts(img) :
    xfeat_model = get_xfeat_model()
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)  # (1,3,H,W)
    out = xfeat_model.detectAndCompute(tensor, top_k=2048)[0]
    kpts = out['keypoints']
    desc = out['descriptors']
    return  kpts, desc









def get_cam_pose_from_mesh_view(mesh):

    def move_top(vis):
        ctr = vis.get_view_control()
        ctr.camera_local_translate(0.0, 0.0, -0.01)
        return False
    

    def move_bot(vis):
        ctr = vis.get_view_control()
        ctr.camera_local_translate(0.0, 0.0, 0.01)
        return False


    def move_left(vis):
        ctr = vis.get_view_control()
        ctr.camera_local_translate(0.0, -0.01, 0.0)
        return False


    def move_right(vis):
        ctr = vis.get_view_control()
        ctr.camera_local_translate(0.0, 0.01, 0.0)
        return False


    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Choose ur pose",
                      width=CAM_W,
                      height=CAM_H)

    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True

    # get viewcontrol -> to pinhole -> change intrinsics to same robot cam intrinsics
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    cam_params.intrinsic = intrins_o3d  
    ctr.convert_from_pinhole_camera_parameters(
        cam_params,
        allow_arbitrary=True
    )
    # ---------------------------------------

    vis.register_key_callback(ord("W"), move_top)
    vis.register_key_callback(ord("S"), move_bot)
    vis.register_key_callback(ord("A"), move_right)
    vis.register_key_callback(ord("D"), move_left)
    vis.run()

    # After user moves camera, extract pose
    vc = vis.get_view_control()
    cam_params = vc.convert_to_pinhole_camera_parameters()
    T_wc = cam_params.extrinsic
    T_cw = np.linalg.inv(T_wc)

    vis.destroy_window()
    return T_cw





    
def render_mesh_pic(mesh, extrins):

    # defining extrins and T_cw_final params 
    extrins = np.linalg.inv(extrins)

    # making our pincamparams objct
    cam_params = o3d.camera.PinholeCameraParameters()
    cam_params.intrinsic = intrins_o3d
    cam_params.extrinsic = extrins

    # ---------- Create visualizer ----------
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=CAM_W, height=CAM_H, visible=False)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.0, 0.0, 0.0])  # black
    vis.add_geometry(mesh)


    # ---------- Apply camera ----------
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(
        cam_params,
        allow_arbitrary=True
    )

    # ---------- Render once ----------
    vis.poll_events()
    vis.update_renderer()

    # ---------- Capture image ----------
    img = vis.capture_screen_float_buffer()
    depth = vis.capture_depth_float_buffer()
    vis.destroy_window()

    return np.asarray(img), np.asarray(depth) 






def get_sfm_poses(recon) : 

    homog_poses = {}
    for image in recon.images.values():
        if not image.has_pose:
            continue
        T_cw = image.cam_from_world()   # Rigid3d
        R = T_cw.rotation.matrix()      # (3,3)
        t = T_cw.translation            # (3,)
        T_h = LinAlgeb.get_homog_frm_rt(R,t)
        homog_poses[image.name] = T_h

    return homog_poses





def load_gaussians_from_ply(path, device="cuda"):

    # Read PLY
    ply = PlyData.read(path)["vertex"]
    N = ply.count
    print(f"Loaded {N} vertices from {path}\n")

    # -----------------------
    # Means
    # -----------------------
    means = torch.stack([
        torch.from_numpy(ply["x"]),
        torch.from_numpy(ply["y"]),
        torch.from_numpy(ply["z"]),
    ], dim=1).float().to(device)

    # -----------------------
    # Scales (log → real)
    # -----------------------
    scales_log = torch.stack([
        torch.from_numpy(ply["scale_0"]),
        torch.from_numpy(ply["scale_1"]),
        torch.from_numpy(ply["scale_2"]),
    ], dim=1).float().to(device)
    scales = torch.exp(scales_log)

    # -----------------------
    # Rotation (normalize quaternion)
    # -----------------------
    quats = torch.stack([
        torch.from_numpy(ply["rot_0"]),
        torch.from_numpy(ply["rot_1"]),
        torch.from_numpy(ply["rot_2"]),
        torch.from_numpy(ply["rot_3"]),
    ], dim=1).float().to(device)
    quats = quats / torch.norm(quats, dim=1, keepdim=True)

    # -----------------------
    # Opacity (inverse sigmoid → alpha)
    # -----------------------
    opacity_param = torch.from_numpy(ply["opacity"]).float().to(device)
    opacities = torch.sigmoid(opacity_param)

    # -----------------------
    # Spherical Harmonics (flexible: degree 0 or higher)
    # -----------------------
    f_dc = torch.stack([
        torch.from_numpy(ply["f_dc_0"]),
        torch.from_numpy(ply["f_dc_1"]),
        torch.from_numpy(ply["f_dc_2"]),
    ], dim=1)  # (N, 3)

    ply_data = ply.data
    rest_keys = sorted(
        [k for k in ply_data.dtype.names if k.startswith("f_rest_")],
        key=lambda k: int(k.split("_")[-1])   # sort by index: f_rest_0, f_rest_1, ...
    )

    if len(rest_keys) == 0:
        # SH degree 0: only DC component, reshape directly
        sh = f_dc.unsqueeze(1).float().to(device)  # (N, 1, 3)
    else:
        f_rest = torch.stack([
            torch.from_numpy(ply_data[k]) for k in rest_keys
        ], dim=1)  # (N, num_rest_coeffs)

        sh = torch.cat([f_dc, f_rest], dim=1)  # (N, total_coeffs)
        sh = sh.view(N, -1, 3).float().to(device)  # (N, num_coeffs_per_channel, 3)

    return [means, quats, scales, opacities, sh]









def dyna_ibvs(gaussians, intrins_gs1, init_pose_gs1, des_img, des_ibvs_infos_npy_path) :

    try: 
        ibvs_tools = IbvsTools(CAM_W, CAM_H)    
        cur_pose_gs1 = init_pose_gs1
        des_kpts, des_desc = ibvs_tools.get_xfeat_kpts(des_img)
        norm_of_error = None
        i = 0

        for i in range(max_ibvs_nbr_itrs) :    

            # Render new cur_gs_pic and get its depthmap
            cur_gs1_img, cur_gs1_depth_map = GaussiansHandling.render_gs_pic(*gaussians, T=cur_pose_gs1, K=intrins_gs1, W=CAM_W, H=CAM_H)

            # Match current_gs1 with des_estim using xfeat 
            cur_kpts, cur_desc = ibvs_tools.get_xfeat_kpts(cur_gs1_img)
            idxs0, idxs1 = xfeat.match(cur_desc, des_desc)
            matches_cur = cur_kpts[idxs0]
            matches_des = des_kpts[idxs1]
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
                ImageHandling.save_img(cur_mtch_gs_img, i, matches_frames_path)   
                MyUtils.cleanup()



    except KeyboardInterrupt:
            print("\nCtrl+C detected, exiting loop cleanly.")
            MyUtils.cleanup()






























#_______________________________________________________________________________________________________________________________________



def main() :

    sfm_path = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test/train/real_scene/sfm_train"
    PosesHandling.apply_sfm_reconstruction(sfm_path, True)
    return

    # Make the folders
    if not did_start :
        os.mkdir(f"{case_test_path}/desired_imgs")
        os.mkdir(f"{case_test_path}/gs2s")
        os.mkdir(f"{case_test_path}/ibvs_frames")
        os.mkdir(f"{case_test_path}/keyframes")
        os.mkdir(f"{case_test_path}/sfms")
        write_init_config_txt(configs_path)

    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda")
     
    # Load gaussians1 & Get gs1 camera 
    gaussians1 = load_gaussians_from_ply(gs1_ply_path)
    recons1 = PosesHandling.get_recons(gs1_sfm_path)
    K = PosesHandling.get_cam_matrix(recons1)
    intrins_gs1 = GaussiansHandling.turn_cam_matrix_to_gsplat_format(K)

    # Get init_pose_gs1 & render kf1 & save it
    init_img_pose_gs1 = PosesHandling.get_img_sfm_pose(recons1, init_img_gs1_name)
    init_img, _ = GaussiansHandling.render_gs_pic(*gaussians1, init_img_pose_gs1, intrins_gs1, CAM_W, CAM_H)
    ImageHandling.save_img(init_img, "init_img", keyframes_path)
    ImageHandling.save_img(init_img, "init_img", f"{sfms_path}/sfm0/images")


    # Load moge_img from original gs1_sfm (for better quality)
    if not get_des_frm_gs : 
        init_img = ImageHandling.load_np_img(f"{gs1_sfm_path}/images/{init_img_gs1_name}")
           

    if not des0_ready : 
 
        # Apply Moge on it
        masked_points, masked_colors, all_moge_points, all_moge_colors, moge_mask = GaussiansHandling.get_moge_points(init_img, model_reso_lvl=moge_model_reso_lvl, depth_edge_threshold=moge_depth_edge_threshold, moge_model=moge_model, use_fp16_bool=False) 
        moge_points_o3d = MeshHandling.turn_points_to_o3d(masked_points, masked_colors)
        MeshHandling.save_o3dpcd(moge_points_o3d, moge_path) 
        
        # Manually render des0 from init_img moge points, save it to des_imgs and to sfm0 || or load an already saved trans
        if trans_exist :
            trans_data = np.load(init_des_trans_npy_path, allow_pickle=True)
            trans_dict = trans_data.item()
            moge_des_pose = trans_dict[0]
        else :
            moge_des_pose = get_cam_pose_from_mesh_view(moge_points_o3d)

        des_estim_img, depth = render_mesh_pic(moge_points_o3d, moge_des_pose)
        ImageHandling.save_img(des_estim_img, "des0", des_imgs_path)
        ImageHandling.save_img(des_estim_img, "des0", f"{sfms_path}/sfm0/images")
        ImageHandling.plot_2_imgs(init_img, des_estim_img)    
        
        # getting des0 mask nd save it + saving the T we did
        depth = np.asarray(depth)
        mask = np.isfinite(depth) & (depth > 0) 
        MyUtils.save_arrays_to_npy(des_masks_path, 0, mask)
        MyUtils.save_arrays_to_npy(init_des_trans_npy_path, 0, moge_des_pose)


    if not des0_aligned :
        # copy gs1_sfm and align des1 with it
        MyUtils.copy_any(gs1_sfm_path, gs1_sfm_aligned_path, overwrite=True)
        PosesHandling.align_new_image(f"{des_imgs_path}/des0.png", gs1_sfm_aligned_path)
    

    if not des_GT_ready :
        # Render nd save des_GT  
        recons1_align = PosesHandling.get_recons(gs1_sfm_aligned_path)
        gt_des_pose = PosesHandling.get_img_sfm_pose(recons1_align, "des0.png")
        gt_des_img, _ = GaussiansHandling.render_gs_pic(*gaussians1, gt_des_pose, intrins_gs1, CAM_W, CAM_H)
        des_estim_img = ImageHandling.load_np_img(f"{des_imgs_path}/des0.png")
        ImageHandling.save_img(gt_des_img, "GT_des", des_imgs_path)
        print("GT des saved !")
        ImageHandling.plot_2_imgs(des_estim_img, gt_des_img)

        # getting des0 mask & save masked GT
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



  

    if not neighbour_imgs_ready :
        i=0
        for img_name in neighbour_imgs_name :
            i+=1
            img_pose = PosesHandling.get_img_sfm_pose(recons1, img_name)
            img_gs1, _ = GaussiansHandling.render_gs_pic(*gaussians1, img_pose, intrins_gs1, CAM_W, CAM_H)
            ImageHandling.save_img(img_gs1, f"nghbr_{i}", f"{sfms_path}/sfm0/images")


    if not sfm0_ready :
        # at this point I got to make sfm ready (onl sparse folder will be filled)
        path = f"{sfms_path}/sfm0/sparse"
        if not os.path.exists(path):
            os.mkdir(path)
        PosesHandling.apply_sfm_reconstruction(f"{sfms_path}/sfm0", sequential=True)



    # Get that initial img, get the moge we got from it
    masked_points, masked_colors, all_moge_points, all_moge_colors, moge_mask = GaussiansHandling.get_moge_points(init_img, model_reso_lvl = moge_model_reso_lvl, depth_edge_threshold=moge_depth_edge_threshold, moge_model=moge_model, use_fp16_bool = False)
    
    # Get txtr map from original kf
    _, txtr_mask = ImageHandling.compute_texturemap_and_mask(init_img, threshold=0.05)
    final_mask = moge_mask & txtr_mask    # update final mask with moge mask, cuz the pixels with inf 3d value should be avoide
    
    # Get the sfm0 i just saved => get the 2d 3ds of initial_img 
    curr_recons2 = pycolmap.Reconstruction(f"{sfms_path}/sfm0/sparse/0")    
    img_sfm_points_2d, img_sfm_points_3d = PosesHandling.get_2d_3d_points_of_img(curr_recons2, init_img_gs2_name)

    img_sfm_points_3d_o3d = MeshHandling.turn_points_to_o3d(img_sfm_points_3d)

    # Keep only moge points corresp to sfm (& removing masked points (inf values..))
    x = img_sfm_points_2d[:,0].astype(int)
    y = img_sfm_points_2d[:,1].astype(int)

    # Clamp to image bounds to avoid index errors
    x = np.clip(x, 0, all_moge_points.shape[1] - 1)
    y = np.clip(y, 0, all_moge_points.shape[0] - 1)

    sfm_moge_points = all_moge_points[y, x]
    sfm_mask = moge_mask[y, x]

    sfm_moge_points = sfm_moge_points[sfm_mask]
    img_sfm_points_3d_filt = img_sfm_points_3d[sfm_mask]

    # Get the transformation infos between sfm-cloud and moge-cloud
    _, s, R, t = MeshHandling.align_points(sfm_moge_points, img_sfm_points_3d_filt)

    # Flatten all moge points and filter them
    all_moge_colors_flat = all_moge_colors.reshape(-1, 3)
    all_moge_points_flat = all_moge_points.reshape(-1, 3).astype(np.float64)
    final_mask_flat = final_mask.reshape(-1)
    final_moge_colors = all_moge_colors_flat[final_mask_flat]
    final_moge_points = all_moge_points_flat[final_mask_flat]

    # Apply the calculated transformation on the final-clean moge points
    final_moge_points = (s * (R @ final_moge_points.T)).T + t

    final_moge_points_o3d = MeshHandling.turn_points_to_o3d(final_moge_points, final_moge_colors)
    MeshHandling.visualize_scene([final_moge_points_o3d, img_sfm_points_3d_o3d])
    mesh_handling = MeshHandling(CAM_W, CAM_H)
    des_pose_gs2 = PosesHandling.get_img_sfm_pose(curr_recons2, "des0.png")
    des_img_gs2, _ = mesh_handling.render_mesh_pic([final_moge_points_o3d], des_pose_gs2)
    ImageHandling.plot_img(des_img_gs2)

    # Downsample, turn to gaussians, save
    final_moge_points = final_moge_points[::moge_reso]
    final_moge_colors = final_moge_colors[::moge_reso]

    new_gaussians = GaussiansHandling.turn_points_to_gaussians(final_moge_points, final_moge_colors, scale=0.03)
    GaussiansHandling.turn_gaussians_to_ply(new_gaussians, f"{gs2s_dir_path}/gs2_0.ply")
    print(f"[Output] Saved gs2_0.ply to {gs2s_dir_path}")

    
if __name__ == "__main__":
    main()





























"""
    img = ImageHandling.load_np_img("/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test/thehouse/thehouse_case2/desired_imgs/des0.png")
    white_mask = np.all(img >= 240, axis=-1)
    img[white_mask] = 0
    ImageHandling.save_img(img, "des0a.png", "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test/thehouse/thehouse_case2/desired_imgs")
    return
"""



"""

    import os
    from pathlib import Path
    import cv2
    import numpy as np
    from skimage.metrics import structural_similarity as ssim

    gs_plys = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test/playroom/playroom_case1/gs2s"
    sfm_path = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test/playroom/playroom_case1/sfms/sfm1"
    masks_folder = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test/playroom/playroom_case1/desired_imgs/c"
    gt_des_img = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test/playroom/playroom_case1/desired_imgs/GT_des.png"

    gray_gt = cv2.imread(gt_des_img, cv2.IMREAD_GRAYSCALE)
    gt_des_img = ImageHandling.load_np_img(gt_des_img)

    # get infos about the local-gs
    recons_local = PosesHandling.get_recons(sfm_path)
    K = PosesHandling.get_cam_matrix(recons_local)
    intrins_local = GaussiansHandling.turn_cam_matrix_to_gsplat_format(K)
    print("got infos about the local-gs")    

    # get infos about des-img pose
    des_new_pose = PosesHandling.get_img_sfm_pose(recons_local, "des0.png")
    
    

    for i in range(48) :
        gaussians = GaussiansHandling.load_gaussians_from_ply(f"{gs_plys}/gs2_{i}.ply")
        des_render, depth = GaussiansHandling.render_gs_pic(*gaussians, des_new_pose, intrins_local, CAM_W, CAM_H)
        mask = depth > 1e-6
        mask_img = (mask * 255).astype(np.uint8)
        gray_des = cv2.cvtColor(des_render, cv2.COLOR_RGB2GRAY)

        coverage_perc = 100.0 * mask.mean()
        print(f"Coverage_ {i}: {coverage_perc:.2f}%")

        score, ssim_map = ssim(
        gray_des,
        gray_gt,
        data_range=255,
        full=True)

        masked_score = ssim_map[mask].mean()
        masked_score = ssim_map.mean()
        print(f"SSIM {i}:", masked_score)    
        gray_des = (gray_des * 255).clip(0, 255).astype(np.uint8)
    
        diff = ImageHandling.compute_grayscale_difference(gray_gt, gray_des)
        score = diff.mean()
        print(f"pixel diff {i}: {score}")

    return

"""










"""
    if not init_imgs_ready :
        if manual_init_imgs :
            # Make the square imgs and check them
            mvmnt_length = 0.5
            nghbr1_pose = init_img_pose_gs1 @ LinAlgeb.get_homog_frm_vect([mvmnt_length,0,0,0,0,0])
            nghbr1_img, _ = GaussiansHandling.render_gs_pic(*gaussians1, nghbr1_pose, intrins_gs1, CAM_W, CAM_H)
            nghbr2_pose =  nghbr1_pose @ LinAlgeb.get_homog_frm_vect([0,mvmnt_length,0,0,0,0])
            nghbr2_img, _ = GaussiansHandling.render_gs_pic(*gaussians1, nghbr2_pose, intrins_gs1, CAM_W, CAM_H)
            nghbr3_pose = nghbr2_pose @ LinAlgeb.get_homog_frm_vect([-mvmnt_length,0,0,0,0,0])
            nghbr3_img, _ = GaussiansHandling.render_gs_pic(*gaussians1, nghbr3_pose, intrins_gs1, CAM_W, CAM_H)
            ImageHandling.plot_4_imgs(init_img, nghbr1_img, nghbr2_img, nghbr3_img)
        else :
            keyframes = get_init_kfs_from_dyna_ibvs_loop(gaussians1, init_img_pose_gs1, init_img, des_estim_img)
            a = 0
            for kf in keyframes :
                a += 1
                ImageHandling.save_img(kf, f"init_img_{a}", keyframes_path)
"""
    



