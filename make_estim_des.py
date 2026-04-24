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
import pycolmap



# Paths 


# Main_tests path for shortcuts

scene_name = "thehouse"
case_nbr = 3
main_test_path = "my_results/online_ibvs_test"
case_test_path = f"{main_test_path}/{scene_name}/{scene_name}_case{case_nbr}"


# GS1 paths
gs1_sfm_path = f"{main_test_path}/{scene_name}/real_scene/sfm_{scene_name}"
gs1_ply_path = f"{main_test_path}/{scene_name}/real_scene/{scene_name}.ply"
gs1_sfm_aligned_path = f"{case_test_path}/gs1_sfm_aligned"

# All States to be saved, paths
des_imgs_path = f"{case_test_path}/desired_imgs"
keyframes_path = f"{case_test_path}/keyframes"
sfms_path = f"{case_test_path}/sfms"

#/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test/thehouse/thehouse_case2/

# Robot camera
CAM_W, CAM_H = 1332, 876  
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


np.set_printoptions(precision=2, suppress=False)











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






def main() :

    des1_ready = False
    des1_aligned = False 
    des1_GT_ready = False
    init_img_gs1_name = "IMG_6385.jpg"

    # make init_img_name.txt
    with open("init_img_name.txt", "w") as f:
        f.write(init_img_gs1_name + "\n")

    # Load gaussians1 & Get gs1 camera 
    gaussians1 = load_gaussians_from_ply(gs1_ply_path)
    recons1 = PosesHandling.get_recons(gs1_sfm_path)
    K = PosesHandling.get_cam_matrix(recons1)
    intrins_gs1 = GaussiansHandling.turn_cam_matrix_to_gsplat_format(K)

    # Get init_pose_gs1 & render kf1 & save it
    init_img_pose_gs1 = PosesHandling.get_img_sfm_pose(recons1, init_img_gs1_name)
    keyframe1, _ = GaussiansHandling.render_gs_pic(*gaussians1, init_img_pose_gs1, intrins_gs1, CAM_W, CAM_H)
    ImageHandling.save_img(keyframe1, "keyframe1", keyframes_path)
    
    
    if not des1_ready : 
        # Load moge_img from gs1_sfm
        keyframe1 = ImageHandling.load_np_img(f"{keyframes_path}/keyframe1.png")    
        
        # Apply Moge on it
        masked_points, masked_colors, all_moge_points, all_moge_colors, moge_mask = GaussiansHandling.get_moge_points(keyframe1, model_reso_lvl = 1, use_fp16_bool = False) 
        moge_points_o3d = MeshHandling.turn_points_to_o3d(masked_points, masked_colors) 
        
        # get des1 from kf1 moge points
        moge_des_pose = get_cam_pose_from_mesh_view(moge_points_o3d)
        des_estim_img, _ = render_mesh_pic(moge_points_o3d, moge_des_pose)
        ImageHandling.save_img(des_estim_img, "des1", des_imgs_path)
        ImageHandling.plot_2_imgs(keyframe1, des_estim_img)     


    if not des1_aligned :
        # copy gs1_sfm and align des1 with it
        MyUtils.copy_any(gs1_sfm_path, gs1_sfm_aligned_path, overwrite=True)
        PosesHandling.align_new_image(f"{des_imgs_path}/des1.png", gs1_sfm_aligned_path)
    

    if not des1_GT_ready :
        # Render nd save des_GT  
        recons1_align = PosesHandling.get_recons(gs1_sfm_aligned_path)
        gt_des_pose = PosesHandling.get_img_sfm_pose(recons1_align, "des1.png")
        gt_des_img, _ = GaussiansHandling.render_gs_pic(*gaussians1, gt_des_pose, intrins_gs1, CAM_W, CAM_H)
        
        des_estim_img = ImageHandling.load_np_img(f"{des_imgs_path}/des1.png")
        ImageHandling.save_img(gt_des_img, "GT_des", des_imgs_path)
            

    # Align all of them
    #PosesHandling.apply_sfm_reconstruction(f"{sfms_path}/sfm1")
  

    # Make the square imgs and check them
    mvmnt_length = 0.5
    nghbr1_pose = init_img_pose_gs1 @ LinAlgeb.get_homog_frm_vect([mvmnt_length,0,0,0,0,0])
    nghbr1_img, _ = GaussiansHandling.render_gs_pic(*gaussians1, nghbr1_pose, intrins_gs1, CAM_W, CAM_H)
    nghbr2_pose =  nghbr1_pose @ LinAlgeb.get_homog_frm_vect([0,mvmnt_length,0,0,0,0])
    nghbr2_img, _ = GaussiansHandling.render_gs_pic(*gaussians1, nghbr2_pose, intrins_gs1, CAM_W, CAM_H)
    nghbr3_pose = nghbr2_pose @ LinAlgeb.get_homog_frm_vect([-mvmnt_length,0,0,0,0,0])
    nghbr3_img, _ = GaussiansHandling.render_gs_pic(*gaussians1, nghbr3_pose, intrins_gs1, CAM_W, CAM_H)
    ImageHandling.plot_4_imgs(keyframe1, nghbr1_img, nghbr2_img, nghbr3_img)


    # Make sfm1 folder, put init des and nghbrs in it 
    PosesHandling.create_sfm_structure(sfms_path, "sfm1")
    MyUtils.copy_any(f"{keyframes_path}/keyframe1.png", f"{sfms_path}/sfm1/images/keyframe1.png", True)
    MyUtils.copy_any(f"{des_imgs_path}/des1.png", f"{sfms_path}/sfm1/images/des1.png", True)
    ImageHandling.save_img(nghbr1_img, "nghbr1", f"{sfms_path}/sfm1/images")
    ImageHandling.save_img(nghbr2_img, "nghbr2", f"{sfms_path}/sfm1/images")
    ImageHandling.save_img(nghbr3_img, "nghbr3", f"{sfms_path}/sfm1/images")

    # Align all of them
    PosesHandling.apply_sfm_reconstruction(f"{sfms_path}/sfm1")
        

if __name__ == "__main__":
    main()