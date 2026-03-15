import open3d as o3d
import numpy as np
import torch
from utils.lin_algeb import LinAlgeb
from utils.gaussians_handling import GaussiansHandling
from plyfile import PlyData
from utils.image_handling import ImageHandling
import pycolmap



# Paths 
case_path = "my_results/playroom/case_1"
gs1_colmap_path = "my_results/playroom/sfm_playroom_aligned/"
gs1_sparse_path = f"{gs1_colmap_path}/sparse/1"
gs1_images_path = f"{gs1_colmap_path}/images"
gs1_path = "my_results/playroom/playroom.ply"


# Robot camera
CAM_W, CAM_H = 1264, 832
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
        T_h = LinAlgeb.get_homog_matrix(R,t)
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

    is_estim_des_ready = True 
    estim_des_name = "estim_des.png"
    
    # If estim_des already aligned with GS1 we get GT
    if(is_estim_des_ready) : 
        
        # Get moge_img
        estim_des_img = ImageHandling.load_np_img(f"{case_path}/{estim_des_name}")    
        
        # Load gaussians and init_pose from colmap data
        gaussians = load_gaussians_from_ply(gs1_path)
        recon = pycolmap.Reconstruction(gs1_sparse_path)
        poses_dict = get_sfm_poses(recon)
        gt_des_pose = np.linalg.inv(poses_dict[estim_des_name])

        # Get a camera K matrix (first camera)
        camera_id = list(recon.cameras.keys())[0]
        camera = recon.cameras[camera_id]
        K = camera.calibration_matrix()
        # Turn it to gsplat format
        intrins_gs1 = (
        torch.from_numpy(K)          
        .float()                     
        .to("cuda")                  
        .unsqueeze(0)                
    )
        # Render init pic 
        gt_des_img, _ = GaussiansHandling.render_gs_pic(*gaussians, gt_des_pose, intrins_gs1, CAM_W, CAM_H)
        ImageHandling.plot_2_imgs(estim_des_img, gt_des_img)
        ImageHandling.save_img(gt_des_img, "GT_des", case_path)
        

    else :

        # Loading initial image    
        init_real_img_path = f"{case_path}/real_init.png"
        init_real_img = ImageHandling.load_np_img(init_real_img_path)

        # Apply Moge on it
        moge_points_o3d, _, _ = GaussiansHandling.get_moge_points(init_real_img) 

        # Choose & save estim_des from moge points
        moge_des_pose = get_cam_pose_from_mesh_view(moge_points_o3d)
        des_estim_img, _ = render_mesh_pic(moge_points_o3d, moge_des_pose)
        ImageHandling.save_img(des_estim_img, "estim_des", case_path)
        ImageHandling.plot_2_imgs(init_real_img, des_estim_img)    







if __name__ == "__main__":
    main()