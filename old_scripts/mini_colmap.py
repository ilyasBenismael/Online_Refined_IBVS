import os
import sys
import pycolmap
import open3d as o3d
import numpy as np
import shutil
from utils.lin_algeb import LinAlgeb
import cv2
import matplotlib.pyplot as plt
import math
from plyfile import PlyData, PlyElement
from PIL import Image
from datetime import datetime
from scripts.utils.main_visualizer import LiveOptimizationVisualizer
import torch
from typing import Tuple, Sequence, List, Union
from moge.model.v2 import MoGeModel
from gsplat import rasterization
import utils3d 

# Add accelerated_features to our Python paths , so that when featx script gets executed it xill know where to find the modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(project_root, "accelerated_features"))
from modules.xfeat import XFeat 






#paths
mesh_path = "meshes/office2.glb"
o3d_frames_path = "frames/o3d"
colmap_keyframes_path = "sfm_scenes/mycolmap_1img_nomoge/images"
colmap_sparse_path = "sfm_scenes/mycolmap_1img_nomoge/sparse/0"
moge_points_save_path = "moge_points/office2_0.ply"


original_gs_output = "/home/user/Bureau/visual_navigation/IBVS_CODE/gaussian_splatting/output/e28587b4-1/point_cloud"
gs_plys_path = "gs_scenes/gs_results/allmoges_1img/plys"
gs_states_path = "gs_scenes/gs_results/allmoges_1img/renders/train"
gs_unseen_states_path = "gs_scenes/gs_results/allmoges_1img/renders/test"


# robot camera
CAM_W, CAM_H = 1200, 1000
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

intrins_gs = torch.tensor(
    [[FX, 0.0, CX],
        [0.0, FY, CY],
        [0.0, 0.0, 1.0],],
    dtype=torch.float32,
    device="cuda",
).unsqueeze(0)








def render_gs_pic(means, quats, scales, opacities, sh, T, K, W, H):

    T = get_gs_viewmat(T)

    image, alpha, meta = rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,
    colors=sh,
    viewmats=T,
    Ks=K,
    width=W,
    height=H,
    sh_degree=None,
    rasterize_mode="antialiased",
    render_mode="RGB+ED",
)

    img   = image[0, ..., :-1].detach().cpu().numpy() 
    depth = image[0, ..., -1].detach().cpu().numpy() 
        
    return img, depth






def get_gs_viewmat(T, device="cuda"):
    # Taking normal pose (cam relative to wrld) and turn it to (wrld telative to cam)
    T = torch.tensor(T, dtype=torch.float32, device=device)
    viewmat = torch.linalg.inv(T)
    return viewmat.unsqueeze(0)







def bring_plys_frm_gs_output(input_folder, output_folder):

    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        # Check valid iteration folder
        if os.path.isdir(folder_path) and folder_name.startswith("iteration_"):
            ply_path = os.path.join(folder_path, "point_cloud.ply")
            if os.path.isfile(ply_path):
                # Extract i from "iteration_i"
                try:
                    iteration_number = folder_name.split("_")[-1]
                    new_filename = f"{iteration_number}.ply"
                    destination_path = os.path.join(output_folder, new_filename)
                    shutil.copy2(ply_path, destination_path)
                except Exception as e:
                    print(f"Error processing {folder_name}: {e}")
            else:
                print(f"No point_cloud.ply found in {folder_path}")






def render_plys_and_save(
    mesh,
    ply_folder,
    output_folder,
    poses,
    intrins_gs,
    CAM_W,
    CAM_H
):
    ply_files = sorted(
        [f for f in os.listdir(ply_folder) if f.endswith(".ply")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    for f in ply_files:
        i = os.path.splitext(f)[0]
        ply_path = os.path.join(ply_folder, f)

        means, quats, scales, opacities, sh = load_gaussians_from_ply(ply_path)

        for j, pose in enumerate(poses, start=1):
            pose_folder = os.path.join(output_folder, str(j))
            os.makedirs(pose_folder, exist_ok=True)

            img, _ = render_gs_pic(
                means, quats, scales, opacities, sh,
                pose, intrins_gs, CAM_W, CAM_H
            )
            real_mesh_img, _ = render_mesh_pic(mesh, pose)

            save_img(img, i, pose_folder)
            save_img(real_mesh_img, "real_mesh_img", pose_folder)





def load_gaussians_from_ply(path, device="cuda"):


    C0 = 0.28209479177387814  # SH Y00 normalization constant

    ply = PlyData.read(path)["vertex"]
    N = ply.count
    print(f"\nLoaded {N} vertices from {path}")

    # ---------------------------------------------------
    # Means
    # ---------------------------------------------------
    means = torch.stack([
        torch.from_numpy(ply["x"]),
        torch.from_numpy(ply["y"]),
        torch.from_numpy(ply["z"]),
    ], dim=1).float().to(device)

    # ---------------------------------------------------
    # Scales (log → real)
    # ---------------------------------------------------
    scales_log = torch.stack([
        torch.from_numpy(ply["scale_0"]),
        torch.from_numpy(ply["scale_1"]),
        torch.from_numpy(ply["scale_2"]),
    ], dim=1).float().to(device)
    scales = torch.exp(scales_log)

    # ---------------------------------------------------
    # Rotation (normalize quaternion)
    # ---------------------------------------------------
    quats = torch.stack([
        torch.from_numpy(ply["rot_0"]),
        torch.from_numpy(ply["rot_1"]),
        torch.from_numpy(ply["rot_2"]),
        torch.from_numpy(ply["rot_3"]),
    ], dim=1).float().to(device)
    quats = quats / torch.norm(quats, dim=1, keepdim=True)

    # ---------------------------------------------------
    # Opacity (inverse sigmoid → alpha)
    # ---------------------------------------------------
    opacity_param = torch.from_numpy(ply["opacity"]).float().to(device)
    opacities = torch.sigmoid(opacity_param)

    # ---------------------------------------------------
    # SH coefficients reconstruction
    # ---------------------------------------------------
    ply_data = ply.data

    # DC band (stored as SH coefficient)
    f_dc = torch.stack([
        torch.from_numpy(ply["f_dc_0"]),
        torch.from_numpy(ply["f_dc_1"]),
        torch.from_numpy(ply["f_dc_2"]),
    ], dim=1).float()  # (N,3)

    # Sort rest keys numerically
    rest_keys = sorted(
        [k for k in ply_data.dtype.names if k.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1])
    )

    f_rest_raw = torch.stack(
        [torch.from_numpy(ply_data[k]) for k in rest_keys],
        dim=1
    ).float()  # (N,45)

    f_rest = f_rest_raw.view(N, -1, 3)  # (N,15,3)

    # Concatenate to full SH tensor (N,16,3)
    sh = torch.cat([f_dc.unsqueeze(1), f_rest], dim=1)

    # ---------------------------------------------------
    # Convert DC SH back to RGB
    # ---------------------------------------------------
    rgb = sh[:, 0, :] * C0 + 0.5
    rgb = torch.clamp(rgb, 0.0, 1.0).to(device)


    return means, quats, scales, opacities, rgb




def save_img(img, title, folder_path, assume_rgb=True):
    os.makedirs(folder_path, exist_ok=True)
    img = np.asarray(img, dtype=np.float32)
    img_u8 = (img * 255 if img.max() <= 1.0 else img).clip(0, 255).astype(np.uint8)
    if assume_rgb:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(folder_path, f"{title}.png"), img_u8)






def plot_img(des_img, title) :
    plt.imshow(des_img)
    plt.axis("off")
    plt.suptitle(title)
    plt.show()




def plot_2_imgs(img1, img2, title1="", title2="", suptitle=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(img1)
    axs[0].axis("off")
    axs[0].set_title(title1)

    axs[1].imshow(img2)
    axs[1].axis("off")
    axs[1].set_title(title2)

    if suptitle is not None:
        fig.suptitle(suptitle)

    plt.tight_layout()
    plt.show()






def load_np_img(img_path) :
    img = Image.open(img_path).convert("RGB")
    return np.array(img)




def visualize_scene(scene_compos) :
        o3d.visualization.draw_geometries(
        scene_compos,
        window_name="The scene",
        width=800,
        height=800,
        mesh_show_back_face=True)





def get_moge_points(img, threshold=0.01):
    
    # Loading MoGe
    device = "cuda"
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    model.eval()

    # Making sure img is in 0-1
    if img.max() > 1.0:
        img = img / 255.0

    # turn img to torch and apply moge
    image = torch.from_numpy(img).float().to(device).permute(2, 0, 1)
    output = model.infer(image)


    points = output["points"].cpu().numpy()  # (H, W, 3)
    depth = output["depth"].cpu().numpy()
    mask = output["mask"].cpu().numpy()

    #check the edges (big depth diffs) and add it to the mask area to remove
    edge_mask = utils3d.np.depth_map_edge(depth, rtol=threshold)
    mask_cleaned = mask & (~edge_mask)
       
    # get colors frm img and flatten all (colors, points, masks)
    colors = img.astype(np.float64)
    colors_flat = colors.reshape(-1, 3)
    pts_flat = points.reshape(-1, 3).astype(np.float64)
    valid_flat = mask_cleaned.reshape(-1)

    # save clean points and colors
    final_points = pts_flat[valid_flat]
    final_colors = colors_flat[valid_flat]

    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(final_points)
    o3d_points.colors = o3d.utility.Vector3dVector(final_colors)
    o3d.io.write_point_cloud(moge_points_save_path, o3d_points)

    # Return FULL arrays (H, W, 3) for indexing by pixel coordinates
    return o3d_points, final_points, final_colors, mask_cleaned








def rotmat_to_quaternion_hamilton(R):

    q = np.empty(4)
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[0] = 0.25 / s
        q[1] = (R[2,1] - R[1,2]) * s
        q[2] = (R[0,2] - R[2,0]) * s
        q[3] = (R[1,0] - R[0,1]) * s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            q[0] = (R[2,1] - R[1,2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0,1] + R[1,0]) / s
            q[3] = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            q[0] = (R[0,2] - R[2,0]) / s
            q[1] = (R[0,1] + R[1,0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            q[0] = (R[1,0] - R[0,1]) / s
            q[1] = (R[0,2] + R[2,0]) / s
            q[2] = (R[1,2] + R[2,1]) / s
            q[3] = 0.25 * s

    q /= np.linalg.norm(q)
    return q





    """ this function takees a list of image_names and a list of poses (with corresponding indices) and the 3d point cloud correspond only 
    to the first image and pose in the list, we first use the mask to keep just valid 3dpoints and get their corresponding uvs in the image_1, 
    ofc i have one camera shared with all images, but we go through all images : inversing them first getting t and R (turn to hamilton) (tvec nd qvec)
    give for image id the indexe in list+1(starting from 1)..., qvec and tvec, and cam_id of my only cam and name from image_names,
    and the second line only the first image has the corresponding points, but next cams have no 2d points blank second line """






def write_colmap_data(
    xyz_flat,
    rgb_flat,
    mask_2d,
    poses,
    image_names,
    sparse_path,
    width,
    height,
    f,
    cx,
    cy
):
    
    os.makedirs(sparse_path, exist_ok=True)
    ys, xs = np.where(mask_2d)
    N = xyz_flat.shape[0]
    uvs = np.stack([xs.astype(float), ys.astype(float)], axis=1)
    n_imgs = len(poses)


    # ---------------- cameras.txt ----------------
    with open(os.path.join(sparse_path, "cameras.txt"), "w") as f_cam:
        f_cam.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f_cam.write("1 PINHOLE {} {} {} {} {} {}\n".format(
            width, height, f, f, cx, cy
        ))


    # ---------------- images.txt ----------------
    
    # we considere all images got points3d as uvs of the initial img (only img1 is accurate ofc others no but 2D-3D isn't impo in gs training)

    with open(os.path.join(sparse_path, "images.txt"), "w") as f_img:

        f_img.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f_img.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for idx, (pose, name) in enumerate(zip(poses, image_names)):
            image_id = idx + 1

            w2c = np.linalg.inv(pose)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            qvec = rotmat_to_quaternion_hamilton(R)

            # Pose line
            f_img.write(
                f"{image_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
                f"{t[0]} {t[1]} {t[2]} 1 {name}\n"
            )

            # Same 2D points for all images
            entries = [
                f"{uvs[i][0]} {uvs[i][1]} {i+1}"
                for i in range(N)
            ]
            f_img.write(" ".join(entries) + "\n")



    # ---------------- points3D.txt ----------------

    # each point we considere it is tracked in all 
    with open(os.path.join(sparse_path, "points3D.txt"), "w") as f_pts:
        f_pts.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i in range(N):
            pid = i + 1
            x, y, z = xyz_flat[i]
            r, g, b = rgb_flat[i]

            r = int(np.clip(r * 255.0, 0, 255))
            g = int(np.clip(g * 255.0, 0, 255))
            b = int(np.clip(b * 255.0, 0, 255))

            # each 3d point line got exist in all images (img_ids) as indexe i
            track_entries = " ".join(
                f"{img_id} {i}"
                for img_id in range(1, n_imgs + 1)
            )

            f_pts.write(
                f"{pid} {x} {y} {z} "
                f"{r} {g} {b} "
                f"0.0 {track_entries}\n"
            )
    print("colmap data saved")        





    
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





def get_cam_pose_from_mesh_view(mesh, msg):

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=msg, width=800, height=800)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    
    vis.run()
    
    # Get camera parameters
    vc = vis.get_view_control()
    cam_params = vc.convert_to_pinhole_camera_parameters()
    
    # Extract pose turn if frm wrld relative to cam into cam relative to world 
    T_wc = cam_params.extrinsic  
    T_cw = np.linalg.inv(T_wc)

    vis.destroy_window()
    return T_cw




def get_homog(pose_vector) :
    R,t = LinAlgeb.make_rot_trans(*pose_vector)
    return LinAlgeb.get_homog_matrix(R,t)





def load_mesh(path, scale, lambert) :
    
    # loading the mesh, enable post to render the colored texture, (no normals for lambertian)
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    print(f"Number of vertices: {len(mesh.vertices)}")

    if(not lambert) :
        mesh.compute_vertex_normals()  

    # center the mesh, and scaling it
    mesh.translate(-mesh.get_center()) 
    mesh.scale(scale , center=mesh.get_center()) #this center is just about mesh position after scaling (keepin it in the center here)

    return mesh 





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




# Getting random well distributed points from moge pointcloud
def get_random_points_frm_moge(points, num_samples=25):

    # Flatten to (H*W, 3)
    pts_flat = points.reshape(-1, 3)
    
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(pts_flat).all(axis=1)
    pts_flat = pts_flat[valid_mask]
    
    if len(pts_flat) < num_samples:
        return pts_flat
    
    # Initialize: start with a random point
    sampled_indices = [np.random.randint(0, len(pts_flat))]
    sampled_points = [pts_flat[sampled_indices[0]]]
    
    # Distance from each point to the nearest sampled point
    min_distances = np.full(len(pts_flat), np.inf)
    
    # Farthest Point Sampling
    for _ in range(num_samples - 1):
        
        # Update distances to nearest sampled point
        last_point = pts_flat[sampled_indices[-1]]
        distances = np.linalg.norm(pts_flat - last_point, axis=1)
        min_distances = np.minimum(min_distances, distances)
        
        # Select the farthest point
        farthest_idx = np.argmax(min_distances)
        sampled_indices.append(farthest_idx)
        sampled_points.append(pts_flat[farthest_idx])
    
    return np.array(sampled_points)







def main() :

    """ # Load office_2 make it metric with 1/4, make it lambert
    scale = (1/4) 
    mesh = load_mesh(mesh_path, scale, True)
    
    # Load our fixed initial pose and make it as origin
    init_pose = np.load("numpy_data/init_pose.npy")
    mesh.transform(np.linalg.inv(init_pose))
    init_pose = np.eye(4)


    # making sure of the reconstruction by rendering from our own colmap poses
    # 2-loading the sfm-cameras and 3d points
    recon = pycolmap.Reconstruction("/home/user/Bureau/visual_navigation/IBVS_CODE/sfm4/sparse/0")
    poses_dict = get_sfm_poses(recon)
    for image_name, pose in poses_dict.items():
        pose_inv = np.linalg.inv(pose)
        mesh_img, _ = render_mesh_pic(mesh, pose_inv)
        # remove extension if needed
        name = os.path.splitext(image_name)[0]
        save_img(mesh_img, name, "/home/user/Bureau/visual_navigation/IBVS_CODE/sfm5/images/imgs")

    

    #chosing unseen poses
    nbr_of_poses = 5
    for i in range(nbr_of_poses) :
        pose = get_cam_pose_from_mesh_view(mesh, f"choose pose {i+1}/{nbr_of_poses}")
        list_of_poses.append(pose)
    list_of_poses = np.save("numpy_data/list_of_unseen_poses.npy", list_of_poses)


    #list_of_poses = np.load("numpy_data/list_of_unseen_poses.npy")

    # Load the poses and images of already saved images
    list_of_poses = np.load("numpy_data/list_of_poses.npy")

    # bring plys from gs output folder to our project folder
    bring_plys_frm_gs_output(original_gs_output, gs_plys_path)
    # render the plys from poses and save them to see states
    render_plys_and_save(mesh, gs_plys_path, gs_states_path, list_of_poses, intrins_gs, CAM_W, CAM_H)"""

    # Load office_2 make it metric with 1/4, make it lambert
    scale = (1/4) 
    mesh = load_mesh(mesh_path, scale, True)

    
    # Load our fixed initial pose and make it as origin
    init_pose = np.load("numpy_data/init_pose.npy")
    mesh.transform(np.linalg.inv(init_pose))
    init_pose = np.eye(4)


    # Take initial mesh pic save it to sfm/images as 1
    init_mesh_img, _ = render_mesh_pic(mesh, init_pose)
    save_img(init_mesh_img, 1, colmap_keyframes_path)
    list_of_poses = [init_pose]
    images_names = ["1.png"]




    # Apply moge on the init  img || load ready mogepoints
    """moge_points_o3d, moge_points, moge_colors, moge_mask = get_moge_points(init_mesh_img) # moge scene dist from cam is not accurate
    np.save("numpy_data/moge_points.npy", moge_points)
    np.save("numpy_data/moge_colors.npy", moge_colors)
    np.save("numpy_data/moge_mask.npy", moge_mask)"""
    
    moge_points = np.load("numpy_data/moge_points.npy")
    moge_colors = np.load("numpy_data/moge_colors.npy")
    moge_mask = np.load("numpy_data/moge_mask.npy")
    moge_points_o3d = o3d.io.read_point_cloud(moge_points_save_path)


    # choosing random 50 points form moge as sfm3D
    random_points = get_random_points_frm_moge(moge_points, num_samples=50)
    N = random_points.shape[0]
    moge_points = random_points
    # Fake colors (all black, float in [0,1])
    moge_colors = np.zeros((N, 3), dtype=np.float32)
    # Fake mask with shape (N,1), all True
    moge_mask = np.ones((N, 1), dtype=bool)


    """# Loop and get poses nd save em, capture their img nd save them and save their names
    nbr_of_poses = 11
    for i in range(nbr_of_poses) :
        pose = get_cam_pose_from_mesh_view(mesh, f"choose pose {i}/{nbr_of_poses}")
        list_of_poses.append(pose)
        img, _ = render_mesh_pic(mesh, pose)
        save_img(img, (i+2), colmap_keyframes_path)
        images_names.append(f"{i+2}.png")

    np.save("numpy_data/list_of_poses.npy", list_of_poses)
    np.save("numpy_data/images_names.npy", images_names)

    # Load the poses and images of already saved images
    list_of_poses = np.load("numpy_data/list_of_poses.npy")
    images_names = np.load("numpy_data/images_names.npy")"""



    # cams and images have same indices ! and first one is the one with corresponding moge points
    write_colmap_data(
    moge_points,
    moge_colors,
    moge_mask,
    list_of_poses,
    images_names,
    colmap_sparse_path,
    CAM_W,
    CAM_H,
    FX,
    CX,
    CY)






if __name__ == "__main__":
    main()