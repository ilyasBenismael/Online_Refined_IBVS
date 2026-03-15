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
gs_frames_path = "frames/gs2"
moge_points_save_path = "moge_points/office2_0.ply"
gs_save_path = "gs_scenes/init_gs_office2"


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

# whole scene visualizer camera
CAM_W2, CAM_H2 = 400, 300
FX2 = FY2 = 0.8 * max(CAM_W2, CAM_H2)
CX2, CY2 = CAM_W2 / 2.0, CAM_H2 / 2.0
intrins2_o3d = o3d.camera.PinholeCameraIntrinsic(
    width=CAM_W2,
    height=CAM_H2,
    fx=FX2,
    fy=FY2,
    cx=CX2,
    cy=CY2
)

np.set_printoptions(precision=2, suppress=False)










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





def get_uv_from_Ss(Ss):
    uv_list = []
    for x, y in Ss:
        u = CX + FX * x 
        v = CY + FY * y
        u = int(np.clip(round(u), 0, CAM_W - 1))
        v = int(np.clip(round(v), 0, CAM_H - 1))
        uv_list.append([u, v])
    return np.array(uv_list)






def save_img(img, title, folder_path, assume_rgb=True):
    os.makedirs(folder_path, exist_ok=True)
    img = np.asarray(img, dtype=np.float32)
    img_u8 = (img * 255 if img.max() <= 1.0 else img).clip(0, 255).astype(np.uint8)
    if assume_rgb:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(folder_path, f"{title}.png"), img_u8)



def compute_grayscale_difference(img1, img2, normalize=True):
    img1_float = img1.astype(np.float32)
    img2_float = img2.astype(np.float32)
    
    diff = np.abs(img1_float - img2_float)
    
    if normalize:
        if diff.max() > 0:
            diff = (diff / diff.max() * 255).astype(np.uint8)
        else:
            diff = diff.astype(np.uint8)
    else:
        diff = diff.astype(np.uint8)
    
    if diff.ndim == 2:
        diff = diff[:, :, np.newaxis]
    
    return diff








def get_sift_features(img):

    # Convert normalized RGB to uint8 format
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Convert RGB to grayscale
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    
    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=500)
    
    # Detect keypoints and compute descriptors
    kp, des = sift.detectAndCompute(gray, None)

    return kp, des




def get_matches(nbr_features, kp1, des1, kp2, des2, cur_img, des_img, show) :

    cur_img_uint8 = (cur_img * 255).astype(np.uint8)
    des_img_uint8 = (des_img * 255).astype(np.uint8)
    
    # Match features using BFMatcher with L2 norm (for SIFT)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort matches by distance (best matches first) and keep top 4
    matches = sorted(matches, key=lambda x: x.distance)[:nbr_features]

    if len(matches) < nbr_features:
        print(f"got just {len(matches)} matches") 
        #ValueError(f"Expected exactly 4 matches, but got {len(matches)}")
    
    cur_features = []
    des_features = []
    
    # Each match got the query index (index of kp from img1) and its corresp train index (index of kp from img2)
    for m in matches:
        idx1 = m.queryIdx
        idx2 = m.trainIdx
        (x1, y1) = kp1[idx1].pt
        cur_features.append([int(x1), int(y1)])
        (x2, y2) = kp2[idx2].pt
        des_features.append([int(x2), int(y2)])

    if show :
        # Draw top 4 matches
        result = cv2.drawMatches(cur_img_uint8, kp1, des_img_uint8, kp2, 
                                matches, None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.figure(figsize=(15, 6), facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('white')
        plt.imshow(result)
        plt.title(f'SIFT Feature Matches', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()  

    return cur_features, des_features    
    





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



# -------------- Get errors --------------------------------
def getting_errors(Ss : List[List[float]], Ss_star : List[List[float]]) -> List[float]:  
    
    return [a - b for row1, row2 in zip(Ss, Ss_star) for a, b in zip(row1, row2)]



# ------------- Getting L and pseudo L --------------------
def get_interaction_matrix(nbr_features, Ss, Ss_z, f):

    Ls = np.zeros((nbr_features, 2, 6), dtype=float)

    for i, s in enumerate(Ss) :
        x=s[0]
        y=s[1]
        Z=Ss_z[i]

        # Build interaction matrix (2x6)
        L = np.zeros((2, 6))
        L[0, 0] = -f / Z
        L[0, 1] = 0.0
        L[0, 2] = x / Z
        L[0, 3] = x * y / f
        L[0, 4] = -(f + (x * x) / f)
        L[0, 5] = y
        L[1, 0] = 0.0
        L[1, 1] = -f / Z
        L[1, 2] = y / Z
        L[1, 3] = f + (y * y) / f
        L[1, 4] = -(x * y) / f
        L[1, 5] = -x

        Ls[i] = L

    return Ls


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







def get_Ss_from_points(points_in_cam: List[Sequence[float]]) -> List[List[float]]:
    s = []
    for p in points_in_cam :
        X, Y, Z = p
        if Z == 0 or Z < 0 :
            Z = 0.5
        x = f * (X / Z)
        y = f * (Y / Z)
        s.append([x, y])
    return s





def filter_valid_points(candidate_points, cur_pose, des_pose, nbr):

    points_in_cur_cam = LinAlgeb.transform_points_to_cam(candidate_points, cur_pose)
    points_in_des_cam = LinAlgeb.transform_points_to_cam(candidate_points, des_pose)

    # Positive depth in both cameras
    valid_mask = (points_in_cur_cam[:, 2] > 0) & (points_in_des_cam[:, 2] > 0)

    valid_indices = np.where(valid_mask)[0]

    # check if enough points are detected
    if len(valid_indices) < nbr:
        raise ValueError(
            f"Only {len(valid_indices)} valid points detected, "
            f"but {nbr} required."
        )

    selected_indices = valid_indices[:nbr]

    return candidate_points[selected_indices]







def get_Ss_from_uv(uv_list):
    Ss = []
    for u, v in uv_list:
        x = (u - CX) / FX
        y = (v - CY) / FY
        Ss.append([x, y])
    return np.array(Ss)





def draw_matches(cur_img, des_img, pts1, pts2):

    # making sure the matches are equal
    assert len(pts1) == len(pts2), "Point lists must have same length"
    n = len(pts1)

    cur_img = cur_img.copy()
    des_img = des_img.copy()

    # Ensure uint8 (format waited by open_cv [0-255])
    if cur_img.dtype != np.uint8:
        cur_img = (cur_img * 255).astype(np.uint8)
    if des_img.dtype != np.uint8:
        des_img = (des_img * 255).astype(np.uint8)

    # Generate n random colors
    colors = (np.random.rand(n, 3) * 255).astype(np.uint8)

    #for each elemnt of the uvs : get a color frm colors and draw it in the uv elmnt on both imgs
    for k, ((u1, v1), (u2, v2)) in enumerate(zip(pts1, pts2)):
        r, g, b = colors[k]
        color_bgr = (int(b), int(g), int(r))
        # filled colored circles on uv
        cv2.circle(cur_img, (int(u1), int(v1)), 6, color_bgr, -1)
        cv2.circle(des_img, (int(u2), int(v2)), 6, color_bgr, -1)

    return cur_img, des_img







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







def init_gaussians_from_points(
    xyz,
    rgb,
    ply_path,
    scale=0.005,
    alpha=0.9,
    sh_degree=2,
    device="cuda",
):

    # Flatten inputs
    xyz = xyz.reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)
    N = xyz.shape[0]

    # Normalize moge_colors
    if rgb.max() > 1.0:
        rgb = rgb / 255.0


    # Get f_dc from rgb
    f_dc = (rgb - 0.5) / 0.28209479177387814

    # Defining the nmbr of sh coeffs
    num_sh = (sh_degree + 1) ** 2

    #______ init opacity
    # we optimize a free opacity number but we render its sigmoid(always in 0-1), in plywe save the optimized number (inverse_sigmoid)
    opacity_logit = LinAlgeb.inverse_sigmoid(alpha)

    #______ init opacity
    # we optimize free scale numbers but render (exp(scale) - always positive), what gsplat needs (ply save log number, the optimized one)
    scale_log = np.log(scale)


    # ______PLY file structure (GS-compatible)
    
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),          # Gaussian center
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),       # Unused normals
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),  # SH DC
        ("opacity", "f4"),                              # Opacity logit
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),  # log-scales
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),  # quaternion
    ]

    # Higher-order SH coefficients set to 0
    for i in range(num_sh - 1):
        for c in range(3):
            dtype.append((f"f_rest_{3*i + c}", "f4"))

    data = np.empty(N, dtype=dtype)



    #____________  Fill PLY data

    # Positions
    data["x"], data["y"], data["z"] = xyz.T

    # Normals are unused by GS → set to zero
    data["nx"] = data["ny"] = data["nz"] = 0.0

    # SH DC coefficients
    data["f_dc_0"] = f_dc[:, 0]
    data["f_dc_1"] = f_dc[:, 1]
    data["f_dc_2"] = f_dc[:, 2]

    # Opacity
    data["opacity"] = opacity_logit

    # Scales
    data["scale_0"] = scale_log
    data["scale_1"] = scale_log
    data["scale_2"] = scale_log

    # Identity rotation quaternion
    data["rot_0"] = 1.0
    data["rot_1"] = 0.0
    data["rot_2"] = 0.0
    data["rot_3"] = 0.0

    # Higher-order SH coefficients start at zero
    for i in range(num_sh - 1):
        for c in range(3):
            data[f"f_rest_{3*i + c}"] = 0.0

    # Save ply file
    PlyData([PlyElement.describe(data, "vertex")]).write(ply_path)


    # Turn these gaussians properties to tensors
    means = torch.from_numpy(xyz).float().to(device)
    scales = torch.full((N, 3), scale, device=device)
    quats = torch.zeros((N, 4), device=device)
    quats[:, 0] = 1.0  # identity rotation
    opacities = torch.full((N,), alpha, device=device)

    # SH tensor layout: [N, num_sh, 3]
    sh = torch.zeros((N, num_sh, 3), device=device)
    sh[:, 0, :] = torch.from_numpy(f_dc).to(device)

    return means, quats, scales, opacities, sh




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
    sh_degree=2,
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







def turn_points_to_o3d(points) :
    
    n = len(points)
    colors = plt.cm.hsv(np.linspace(0, 1, n))[:, :3]
    spheres=[]
    i=-1

    for point in points:
        
        i+=1
        
        # Create sphere mesh
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        
        # Translate sphere to the point location
        sphere.translate(point)
        
        # Assign color to the sphere
        sphere.paint_uniform_color(colors[i])
        
        # Add to list
        spheres.append(sphere)

    # If you want to combine all spheres into one mesh (optional)
    combined_spheres = o3d.geometry.TriangleMesh()
    for sphere in spheres:
        combined_spheres += sphere

    return combined_spheres




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







def get_3d_from_uv(feats, Zs) :
    # I have a list of fezats [u,v]s and a list of corresp Zs
    # Turning uvs to XYZs and x*Z gives X and y*Z gives Y
    Ss = get_Ss_from_uv(feats)
    Ss = np.array(Ss)      # shape (N,2)
    Zs = np.array(Zs)      # shape (N,)
    XYZs = np.column_stack((Ss * Zs[:, None], Zs))
    return XYZs







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





def rotmat_to_quaternion_hamilton(R):
    """
    Convert 3x3 rotation matrix to Hamilton quaternion (qw, qx, qy, qz).

    Hamilton convention:
        q = w + xi + yj + zk
    Order in COLMAP:
        (qw, qx, qy, qz)
    """

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
    poses,            # list of poses
    image_names,      # list of image names
    sparse_path,
    width,
    height,
    f,
    cx,
    cy
):

    os.makedirs(sparse_path, exist_ok=True)

    # ---------------------------------------------------
    # 1 — UVs from mask (ONLY for first image)
    # ---------------------------------------------------
    ys, xs = np.where(mask_2d)

    assert xyz_flat.shape[0] == xs.shape[0]
    assert rgb_flat.shape[0] == xs.shape[0]

    N = xyz_flat.shape[0]
    uvs = np.stack([xs.astype(float), ys.astype(float)], axis=1)

    # ---------------------------------------------------
    # 2 — cameras.txt (single shared camera)
    # ---------------------------------------------------
    with open(os.path.join(sparse_path, "cameras.txt"), "w") as f_cam:
        f_cam.write("# Camera list\n")
        f_cam.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f_cam.write("1 PINHOLE {} {} {} {} {} {}\n".format(
            width, height, f, f, cx, cy
        ))

    # ---------------------------------------------------
    # 3 — images.txt (multiple images)
    # ---------------------------------------------------
    with open(os.path.join(sparse_path, "images.txt"), "w") as f_img:
        f_img.write("# Image list\n")
        f_img.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f_img.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for idx, (pose, name) in enumerate(zip(poses, image_names)):
            image_id = idx + 1

            w2c = np.linalg.inv(pose)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            qvec = rotmat_to_quaternion_hamilton(R)

            # First line (pose)
            f_img.write(
                f"{image_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
                f"{t[0]} {t[1]} {t[2]} 1 {name}\n"
            )

            # Second line
            if image_id == 1:
                entries = [
                    f"{uvs[i][0]} {uvs[i][1]} {i+1}"
                    for i in range(N)
                ]
                f_img.write(" ".join(entries))
            f_img.write("\n")

    # ---------------------------------------------------
    # 4 — points3D.txt
    # ---------------------------------------------------
    with open(os.path.join(sparse_path, "points3D.txt"), "w") as f_pts:
        f_pts.write("# 3D point list\n")
        f_pts.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")

        for i in range(N):
            pid = i + 1
            x, y, z = xyz_flat[i]
            r, g, b = rgb_flat[i]

            r = int(np.clip(r * 255.0, 0, 255))
            g = int(np.clip(g * 255.0, 0, 255))
            b = int(np.clip(b * 255.0, 0, 255))

            # Only observed in image 1
            f_pts.write(
                f"{pid} {x} {y} {z} "
                f"{r} {g} {b} "
                f"0.0 1 {i}\n"
            )





def bring_plys_frm_gs_output(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

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
    ply_folder,
    output_folder,
    init_pose,
    des_gs_pose,
    intrins_gs,
    CAM_W,
    CAM_H
):
    os.makedirs(os.path.join(output_folder, "init"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "des"), exist_ok=True)

    ply_files = sorted(
        [f for f in os.listdir(ply_folder) if f.endswith(".ply")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    for f in ply_files:
        i = os.path.splitext(f)[0]
        ply_path = os.path.join(ply_folder, f)

        means, quats, scales, opacities, sh = load_gaussians_from_ply(ply_path)

        img, _ = render_gs_pic(
            means, quats, scales, opacities, sh,
            init_pose, intrins_gs, CAM_W, CAM_H
        )
        save_img(img, f"{i}_init", os.path.join(output_folder, "init"))

        img, _ = render_gs_pic(
            means, quats, scales, opacities, sh,
            des_gs_pose, intrins_gs, CAM_W, CAM_H
        )
        save_img(img, f"{i}_des", os.path.join(output_folder, "des"))






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
        f_cam.write("1 PINHOLE {} {} {} {} {} {}\n".format(
            width, height, f, f, cx, cy
        ))

    # ---------------- images.txt ----------------
    with open(os.path.join(sparse_path, "images.txt"), "w") as f_img:

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
    with open(os.path.join(sparse_path, "points3D.txt"), "w") as f_pts:

        for i in range(N):
            pid = i + 1
            x, y, z = xyz_flat[i]
            r, g, b = rgb_flat[i]

            r = int(np.clip(r * 255.0, 0, 255))
            g = int(np.clip(g * 255.0, 0, 255))
            b = int(np.clip(b * 255.0, 0, 255))

            # Track across all images
            track_entries = " ".join(
                f"{img_id} {i}"
                for img_id in range(1, n_imgs + 1)
            )

            f_pts.write(
                f"{pid} {x} {y} {z} "
                f"{r} {g} {b} "
                f"0.0 {track_entries}\n"
            )




def create_mesh_keyframes(mesh, N, col_keyframes_path):
    poses = []
    image_names = []

    os.makedirs(col_keyframes_path, exist_ok=True)

    for i in range(1, N + 1):
        pose = get_cam_pose_from_mesh_view(mesh, f"choose pose{i}")
        img, _ = render_mesh_pic(mesh, pose)
        save_img(img, str(i), col_keyframes_path)

        poses.append(pose)
        image_names.append(f"{i}.png")

    return poses, image_names










def main() :
    
    # Fixed Vars :
    lambda_gain = 0.3
    dt = 0.1
    all_nbr_ftrs = 250
    nbr_features = 10

    # Scale the mesh to meters, for office_2 nearly 1 meter corresp to 4 units
    scale = (1/4) 

    # Load a 3d mesh nearly in meters
    mesh = load_mesh(mesh_path, scale, False)

    
    # Visualize and choose a real initial pose || use the saved one
    """init_pose = get_cam_pose_from_mesh_view(mesh)
    np.save("numpy_data/init_pose.npy", init_pose)"""
    init_pose = np.load("numpy_data/init_pose.npy")


    # Move init_pose to origin
    mesh.transform(np.linalg.inv(init_pose))
    init_pose = np.eye(4)


    # Take initial mesh pic 
    init_mesh_img, _ = render_mesh_pic(mesh, init_pose)
   

    # Apply moge on the init real img || load ready mogepoints
    moge_points_o3d, moge_points, moge_colors, moge_mask = get_moge_points(init_mesh_img) # moge scene dist from cam is not accurate
    np.save("numpy_data/moge_points.npy", moge_points)
    np.save("numpy_data/moge_colors.npy", moge_colors)
    np.save("numpy_data/moge_mask.npy", moge_mask)
    moge_points = np.load("numpy_data/moge_points.npy")
    moge_colors = np.load("numpy_data/moge_colors.npy")
    moge_mask = np.load("numpy_data/moge_mask.npy")
    moge_points_o3d = o3d.io.read_point_cloud(moge_points_save_path)

     # Choosing a des_pose from mesh
    des_gs_pose = get_cam_pose_from_mesh_view(moge_points_o3d, "choose des pose")

   
#__________________________________________________


    # Visualize all
    #visualize_scene([moge_points_o3d, mesh])  

 
    # Init gaussians from moge_points & Rendering gs initial pose img 
    gaussians_list = init_gaussians_from_points(moge_points, moge_colors, gs_save_path)    
    init_gs_img, init_gs_depth = render_gs_pic(*gaussians_list, T=init_pose, K=intrins_gs, W=CAM_W, H=CAM_H)


    # Check init depth map, get contours from it
    edge_mask = utils3d.np.depth_map_edge(init_gs_depth, rtol=0.008) # shape H*W
    #plot_2_imgs(init_gs_depth, edge_mask, "gs_depth", "depth_edges")

    # Render des_gs_img and des_mesh_img
    des_gs_img, _ = render_gs_pic(*gaussians_list, T=des_gs_pose, K=intrins_gs, W=CAM_W, H=CAM_H)
    gray_des_gs_img = 0.299 * des_gs_img[:, :, 0] + 0.587 * des_gs_img[:, :, 1] + 0.114 * des_gs_img[:, :, 2]
    des_mesh_img, _ = render_mesh_pic(mesh, des_gs_pose)
    #des_gs_img = des_mesh_img
    plot_2_imgs(init_mesh_img, init_gs_img, "init_mesh_img",  "init_gs_img")
    plot_2_imgs(des_mesh_img, des_gs_img, "des_mesh_img",  "des_gs_img")
    plot_2_imgs(init_gs_img, des_gs_img, "init_gs_img",  "des_gs_img")
    save_img(des_mesh_img, "desired", o3d_frames_path)
    save_img(des_gs_img, "desired", gs_frames_path)



    # Getting random n points from moge points
    all_candidate_points = get_random_points_frm_moge(moge_points, num_samples=all_nbr_ftrs)

    # Starting IBVS loop
    cur_gs_pose = init_pose
    matp_vis = LiveOptimizationVisualizer(init_gs_img)





 
    try:    
        for i in range(999) :

            # ok now we got the initial 3d points we will keep them for the rest of the work, we have their yv in des, we woll get their ss_sat

            # 1 - Get first n valid elmnts(in front of both cams) frm candidates and optio visualize
            valid_points_wf = filter_valid_points(all_candidate_points, init_pose, des_gs_pose, nbr_features)

            # 2 - get points in cur and des cam and their ss features
            valid_points_cf = LinAlgeb.transform_points_to_cam(valid_points_wf, des_gs_pose)
            Ss_star = get_Ss_from_points(valid_points_cf)
            pxls_star = get_uv_from_Ss(Ss_star)
            valid_points_cf = LinAlgeb.transform_points_to_cam(valid_points_wf, cur_gs_pose)
            Ss = get_Ss_from_points(valid_points_cf)
            pxls = get_uv_from_Ss(Ss)

            # 3 - Rendering current img, nd get depth exprssd in camera frame with cam frame units 
            cur_gs_img, cur_gs_depth_map = render_gs_pic(*gaussians_list, T=cur_gs_pose, K=intrins_gs, W=CAM_W, H=CAM_H)
            gray_cur_gs_img = 0.299 * cur_gs_img[:, :, 0] + 0.587 * cur_gs_img[:, :, 1] + 0.114 * cur_gs_img[:, :, 2]
            cur_gs_img_mtchs, des_gs_img_mtchs = draw_matches(cur_gs_img, des_gs_img, pxls, pxls_star)
            save_img(cur_gs_img_mtchs,i, gs_frames_path)

            # Render frm same pose the mesh pic
            cur_mesh_img, _ = render_mesh_pic(mesh, cur_gs_pose)
            
            # 4 - Get corresp uvs and their depth in meters
            cur_feats_uv = get_uv_from_Ss(Ss)
            Ss_Z = get_feats_depth(cur_feats_uv, cur_gs_depth_map)

            # 5 - Getting list of errors nd reshaping it
            errors = getting_errors(Ss, Ss_star) 
            errors = np.asarray(errors, dtype=float).reshape(-1)

            # 6 - Print the norm of the error and get the diff of imgs
            norm_of_error = np.linalg.norm(errors) 
            current_diff_img = compute_grayscale_difference(gray_cur_gs_img, gray_des_gs_img)
            print(f"error {i} :", norm_of_error)
            
            # 7 - Get the intr matrix nd its pseudo_inv 
            L = get_interaction_matrix(nbr_features, Ss, Ss_Z, 1)
            LinAlgeb.print_mat_condition_number(L)
            L_psinv = get_inter_mat_pseudo_inverse(L)


            # 8 - Control law
            if norm_of_error < 0.05 :
                lambda_gain = 0.1
            if norm_of_error < 0.0001 :
                break        
            V = - lambda_gain * (L_psinv @ errors)  
           
            # 9 - Apply the velocity for dt and update cur_cam_pose and update visualization
            cur_gs_pose = update_cam_pose(cur_gs_pose, V, dt)
            matp_vis.update(i, current_diff_img, cur_gs_img, cur_mesh_img, des_gs_img, V, norm_of_error)

            # 10 - Updating the valid points (probably we will return same old first valid ones )
            valid_points_wf = filter_valid_points(all_candidate_points, cur_gs_pose, des_gs_pose, nbr_features)

            if(i==0) :
                valid_points_o3d = turn_points_to_o3d(valid_points_wf)
                visualize_scene([moge_points_o3d, valid_points_o3d])

    
    except KeyboardInterrupt:
            print("\nCtrl+C detected, exiting loop cleanly.")
            matp_vis.close()
            os._exit(0)

    


if __name__ == "__main__":
    main()