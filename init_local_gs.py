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



# Add accelerated_features to our Python paths , so that when featx script gets executed it xill know where to find the modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(project_root, "accelerated_features"))
from modules.xfeat import XFeat 




# Main_tests path for shortcuts
scene_name = "playroom"
case_nbr = 1
main_test_path = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test"
case_test_path = f"{main_test_path}/{scene_name}/{scene_name}_case{case_nbr}"


# GS1 paths
gs1_sfm_path = f"{main_test_path}/{scene_name}/real_scene/sfm_{scene_name}"
gs1_ply_path = f"{main_test_path}/{scene_name}/real_scene/{scene_name}.ply"
gs1_sfm_aligned_path = f"{case_test_path}/gs1_sfm_aligned"

# All States to be saved, paths
des_imgs_path = f"{case_test_path}/desired_imgs"
keyframes_path = f"{case_test_path}/keyframes"
sfms_path = f"{case_test_path}/sfms"
gs2s_dir_path = f"{case_test_path}/gs2s"


# Robot camera
CAM_W, CAM_H = 1264, 832  
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

intrins_o3d = o3d.camera.PinholeCameraIntrinsic(
    width=CAM_W,
    height=CAM_H,
    fx=FX,
    fy=FY,
    cx=CX,
    cy=CY
)


# Some configs
np.set_printoptions(precision=2, suppress=False)
xfeat = XFeat()
lambda_gain = 0.03
dt = 0.03
ibvs_nbr_features = 10
gs_reso = 1
moge_reso = 4
max_ibvs_nbr_itrs = 100
kf_motion_ratio = 0.1    # 5% of screen
kf_motion_nbr_features = 100



# Starting states :
des1_ready = True
des1_aligned = True 
des1_GT_ready = True
init_imgs_ready = True
manual_init_imgs = False
sfm0_ready = False
init_img_gs1_name = "DSC05590.jpg"








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







def validate_keyframe(last_kf, curr_frame, xfeat,
                      kf_motion_nbr_features=100,
                      motion_thresh_ratio=0.03):
    
    # by default motion ration is 10% disp in img width

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

    # 6. Decision
    return motion_ratio > motion_thresh_ratio







def get_init_kfs_from_dyna_ibvs_loop(gaussians, init_pose_gs1, last_keyframe_gs1, des_estim_img, max_ibvs_nbr_itrs = 5000, nbr_of_kfs = 2) :

    try: 

        keyframes = []
        ibvs_tools = IbvsTools(CAM_W, CAM_H)    
        cur_pose_gs1 = init_pose_gs1
        des_kpts, des_desc = ibvs_tools.get_xfeat_kpts(des_estim_img)
        norm_of_error = pose_error = None
        i = 0

        for i in range(max_ibvs_nbr_itrs) :    

            # Render new cur_gs_pic and get its depthmap
            cur_gs1_img, cur_gs1_depth_map = GaussiansHandling.render_gs_pic(*gaussians, T=cur_pose_gs1, K=intrins_gs1, W=CAM_W, H=CAM_H)

            # Check if it's a keyframe
            if validate_keyframe(last_keyframe_gs1, cur_gs1_img, xfeat, kf_motion_nbr_features= kf_motion_nbr_features, motion_thresh_ratio = kf_motion_ratio) :
                keyframes.append(cur_gs1_img)
                print(f"Keyframe_{i} found")

                last_keyframe_gs1 = cur_gs1_img

                if len(keyframes) == nbr_of_kfs :
                    return keyframes
                

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

            # 4 - Get the intr matrix & its pseudo_inv 
            L = ibvs_tools.get_interaction_matrix(len(matches_cur), Ss_cur, Ss_Z_cur, 1)
            L_psinv = ibvs_tools.get_inter_mat_pseudo_inverse(L)

            # 5 - Get V from control law
            V = - lambda_gain * (L_psinv @ errors)  
           
            # 6 - Update cur_cam_pose and update visualization
            cur_pose_gs1 = ibvs_tools.update_cam_pose(cur_pose_gs1, V, dt)
 
        print("!!!!!! no keyframes found, ibvs is done")
        return keyframes


    except KeyboardInterrupt:
            print("\nCtrl+C detected, exiting loop cleanly.")
            MyUtils.cleanup()
            return keyframes













def main() :


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
    
    
    if not des1_ready : 
        # Load moge_img from gs1_sfm
        init_img = ImageHandling.load_np_img(f"{keyframes_path}/init_img.png")    
        
        # Apply Moge on it
        masked_points, masked_colors, all_moge_points, all_moge_colors, moge_mask = GaussiansHandling.get_moge_points(init_img, model_reso_lvl = 1, use_fp16_bool = False) 
        moge_points_o3d = MeshHandling.turn_points_to_o3d(masked_points, masked_colors) 
        
        # get des1 from kf1 moge points
        moge_des_pose = get_cam_pose_from_mesh_view(moge_points_o3d)
        des_estim_img, _ = render_mesh_pic(moge_points_o3d, moge_des_pose)
        ImageHandling.save_img(des_estim_img, "des0", des_imgs_path)
        ImageHandling.plot_2_imgs(init_img, des_estim_img)     


    if not des1_aligned :
        # copy gs1_sfm and align des1 with it
        MyUtils.copy_any(gs1_sfm_path, gs1_sfm_aligned_path, overwrite=True)
        PosesHandling.align_new_image(f"{des_imgs_path}/des0.png", gs1_sfm_aligned_path)
    

    if not des1_GT_ready :
        # Render nd save des_GT  
        recons1_align = PosesHandling.get_recons(gs1_sfm_aligned_path)
        gt_des_pose = PosesHandling.get_img_sfm_pose(recons1_align, "des0.png")
        gt_des_img, _ = GaussiansHandling.render_gs_pic(*gaussians1, gt_des_pose, intrins_gs1, CAM_W, CAM_H)
        des_estim_img = ImageHandling.load_np_img(f"{des_imgs_path}/des0.png")
        ImageHandling.save_img(gt_des_img, "GT_des", des_imgs_path)

    des_estim_img = ImageHandling.load_np_img(f"{des_imgs_path}/des0.png")

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


    if not sfm0_ready :
        # at this point I got to make sfm ready (onl sparse folder will be filled)
        PosesHandling.apply_sfm_reconstruction(f"{sfms_path}/sfm0", sequential=False)






    # Get that initial img, get the moge we got from it
    masked_points, masked_colors, all_moge_points, all_moge_colors, moge_mask = GaussiansHandling.get_moge_points(init_img, depth_edge_threshold=0.05, moge_model = moge_model, use_fp16_bool=False, model_reso_lvl=2)
    
    # Get txtr map from original kf
    _, txtr_mask = ImageHandling.compute_texturemap_and_mask(init_img, threshold=0.05)
    final_mask = moge_mask & txtr_mask    # update final mask with moge mask, cuz the pixels with inf 3d value should be avoide
    # Get the sfm0 i just saved => get the 2d 3ds of initial_img 
    curr_recons2 = pycolmap.Reconstruction(f"{sfms_path}/sfm0/sparse/0")    
    img_sfm_points_2d, img_sfm_points_3d = PosesHandling.get_2d_3d_points_of_img(curr_recons2, f"init_img.png")

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

    # Downsample, turn to gaussians, save
    final_moge_points = final_moge_points[::moge_reso]
    final_moge_colors = final_moge_colors[::moge_reso]

    new_gaussians = GaussiansHandling.turn_points_to_gaussians(final_moge_points, final_moge_colors)
    GaussiansHandling.turn_gaussians_to_ply(new_gaussians, f"{gs2s_dir_path}/gs2_0.ply")
    print(f"[Output] Saved gs2_0.ply to {gs2s_dir_path}")



    
if __name__ == "__main__":
    main()