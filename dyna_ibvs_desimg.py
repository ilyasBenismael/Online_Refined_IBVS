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
from utils.mesh_handling import MeshHandling
from plyfile import PlyData, PlyElement
import pycolmap

# Add accelerated_features to our Python paths , so that when featx script gets executed it xill know where to find the modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(project_root, "accelerated_features"))
from modules.xfeat import XFeat 






# Paths
case_path = "my_results/playroom/Tests/case_1"
gs1_path = "my_results/playroom/playroom.ply"

gs1_colmap_path = "my_results/playroom/sfm_playroom_aligned/"
gs1_sparse_path = f"{gs1_colmap_path}/sparse/1"
gs1_images_path = f"{gs1_colmap_path}/images"

real_frames_path = f"{case_path}/saved_frames/real_frames"
matches_frames_path = f"{case_path}/saved_frames/matches_real_target"

gs2_path = f"{case_path}/gs2/gs2.ply"
gs2_colmap_path = f"{case_path}/gs2/sfm_aligned"
gs2_images_path = f"{gs2_colmap_path}/images"
gs2_sparse_path = f"{gs2_colmap_path}/sparse/1"


UPDATE_INTERVAL_MS = 5 #ms


# robot camera
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

"""
intrins_gs1 = torch.tensor(
    [[FX, 0.0, CX],
        [0.0, FY, CY],
        [0.0, 0.0, 1.0],],
    dtype=torch.float32,
    device="cuda",
).unsqueeze(0)
"""

np.set_printoptions(precision=2, suppress=False)

# Fixed Vars :
lambda_gain = 0.1
dt = 0.05
nbr_features = 10
iterations = 999







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




def get_Ss_from_uv(uv_list):
    Ss = []
    for u, v in uv_list:
        x = (u - CX) / FX
        y = (v - CY) / FY
        Ss.append([x, y])
    return np.array(Ss)




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





def start_dyna_ibvs_loop(gaussians, init_pose_gs1, des_pose_gs1, init_img, des_gt_img, des_estim_img) :
    
    try: 
        condition_nbr_list = np.array([])

        for i in range(iterations) :
            
            if(i == 0 ) :
                xfeat = XFeat()
                cur_pose_gs1 = init_pose_gs1
                matp_vis = MainVisualizer(init_img)
                fig, ax, trajectory = MainVisualizer.init_traject_visualizer(init_pose_gs1, des_pose_gs1)
            
            # Update 3d traject visualizer
            MainVisualizer.update_traject_visualizer(ax, init_pose_gs1, des_pose_gs1, cur_pose_gs1, trajectory)
            
            # Render new cur_gs_pic and get its depthmap
            cur_gs1_img, cur_gs_depth_map = GaussiansHandling.render_gs_pic(*gaussians, T=cur_pose_gs1, K=intrins_gs1, W=CAM_W, H=CAM_H)

            # Match current_gs1 with des_estim using xfeat 
            all_matches_cur, all_matches_des = xfeat.match_xfeat(cur_gs1_img, des_estim_img)
            matches_cur = (all_matches_cur[:nbr_features]).astype(int)
            matches_des = (all_matches_des[:nbr_features]).astype(int) 
            
            if (len(matches_cur) < nbr_features) :
                raise Exception(f"only {len(matches_cur)} < {nbr_features}")

            cur_mtch_gs_img = ImageHandling.draw_matches(matches_cur, matches_des, cur_gs1_img, des_estim_img)
            ImageHandling.save_img(cur_gs1_img, i, real_frames_path)
            ImageHandling.save_img(cur_mtch_gs_img, i, matches_frames_path)

            # 2 - Get cur and des features
            Ss_star = get_Ss_from_uv(matches_des)
            Ss_cur = get_Ss_from_uv(matches_cur)
            Ss_Z_cur = get_feats_depth(matches_cur, cur_gs_depth_map)

            # 3 - Get the error
            errors = getting_errors(Ss_cur, Ss_star)
            errors = np.asarray(errors, dtype=float).reshape(-1)

            # Print the norm of the error
            norm_of_error = np.linalg.norm(errors)
            pose_error = LinAlgeb.pose_distance(cur_pose_gs1, des_pose_gs1)
            print("Iteration :", i)
            print(f"2D error: {norm_of_error:.4f}")
            print(f"Pose error is: {pose_error:.4f}")

            # 4 - Get the intr matrix & its pseudo_inv 
            L = get_interaction_matrix(nbr_features, Ss_cur, Ss_Z_cur, 1)
            condition_nbr_list = np.append(condition_nbr_list, LinAlgeb.get_mat_condition_number(L))
            L_psinv = get_inter_mat_pseudo_inverse(L)

            # 5 - Choose lambda and calculate V with control law
            if norm_of_error < 0.0001 :
                break        
            V = - lambda_gain * (L_psinv @ errors)  
           
            # 6 - Update cur_cam_pose and update visualization
            cur_pose_gs1 = update_cam_pose(cur_pose_gs1, V, dt)
            matp_vis.update(i, cur_mtch_gs_img, cur_gs1_img, des_gt_img, des_estim_img, V, norm_of_error)

            if ((i%50)==0) :
                print("mean:", condition_nbr_list.mean())
                print("min:", condition_nbr_list.min())
                print("max:", condition_nbr_list.max())

        plt.ioff()
        plt.show()

    except KeyboardInterrupt:
            print("\nCtrl+C detected, exiting loop cleanly.")
            matp_vis.close()
            os._exit(0)
            plt.ioff()
            plt.show()








def turn_o3dply_to_gsply(pcd, output_path):
    """
    Convert Open3D point cloud to PLY format:

    float x y z
    float nx ny nz
    uchar red green blue

    Keeps the same number of points.
    """

    # compute normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05,
            max_nn=30
        )
    )
    pcd.normalize_normals()

    # extract arrays
    points = np.asarray(pcd.points).astype(np.float32)
    normals = np.asarray(pcd.normals).astype(np.float32)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    n = len(points)

    vertices = np.empty(n, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])

    vertices['x'] = points[:,0]
    vertices['y'] = points[:,1]
    vertices['z'] = points[:,2]

    vertices['nx'] = normals[:,0]
    vertices['ny'] = normals[:,1]
    vertices['nz'] = normals[:,2]

    vertices['red'] = colors[:,0]
    vertices['green'] = colors[:,1]
    vertices['blue'] = colors[:,2]

    ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
    ply.write(output_path)

    print(f"Saved: {output_path} with {n} vertices")



def turn_points(points, colors = None) :
    if (colors is None) :
        n = len(points)
        colors = np.tile([1.0, 0.0, 0.0], (n, 1)).astype(np.float64)  # red
    o3d_points = o3d.geometry.PointCloud()
    o3d_points.points = o3d.utility.Vector3dVector(points)
    o3d_points.colors = o3d.utility.Vector3dVector(colors)
    return o3d_points




def align_points(points1, points2):

    # centroids
    c1 = points1.mean(axis=0)
    c2 = points2.mean(axis=0)

    X1 = points1 - c1
    X2 = points2 - c2

    # rotation
    H = X1.T @ X2
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # reflection fix
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    # scale
    scale = np.sum(S) / np.sum(X1**2)

    # translation
    t = c2 - scale * R @ c1

    # transform points1
    points1_aligned = (scale * (R @ points1.T)).T + t

    return points1_aligned, scale, R, t





def main() :

    ply_o3d_path = "my_results/playroom/Tests/case_1/gs2/sfm_moge/sparse/points3D.ply"
    pcd = o3d.io.read_point_cloud(ply_o3d_path)
    ply_sfm_path = "my_results/playroom/Tests/case_1/gs2/sfm_moge/sparse/points3D2.ply" 
    turn_o3dply_to_gsply(pcd, ply_sfm_path)
    pass
    return

    #____________ aliginng moge with sfm ____________________
    img_name = "135.png"

    # Getting image and its moge points
    imgg = ImageHandling.load_np_img(f"my_results/playroom/Tests/case_1/gs2/sfm/images/{img_name}")
    H, W, _ = imgg.shape
    print("W,H", W, H)
    moge_points_o3d, moge_points, moge_colors, all_moge_points, all_moge_colors = GaussiansHandling.get_moge_points(imgg)
    all_moge_points_o3d = turn_points(all_moge_points.reshape(-1, 3), all_moge_colors.reshape(-1, 3))


    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5, origin=[0,0,0]
)
    compos = [world_frame]

    # Getting all sfm imgs poses
    recon = pycolmap.Reconstruction(gs2_sparse_path)
      
    homog_poses = {}
    for image in recon.images.values():
        if not image.has_pose:
            print(image.name)
            continue
        T_cw = image.cam_from_world()   # Rigid3d
        R = T_cw.rotation.matrix()      # (3,3)
        t = T_cw.translation            # (3,)
        T_h = LinAlgeb.get_homog_frm_rt(R,t)
        homog_poses[image.name] = T_h
    #all_sfm_points = np.array([p.xyz for p in recon.points3D.values()]) 
    
    # Get the img pose and make its frame  
    img_pose = homog_poses[img_name]
    img_pose = np.linalg.inv(img_pose)
    cam_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    cam_axis.transform(img_pose)
    compos.append(cam_axis)

    # turn moge points in front of cam and translate and scalee
    moge_points_o3d = turn_points(moge_points, moge_colors)
    #compos.append(moge_points_o3d)
    moge_points_o3d.scale(15.0, center=(0,0,0))

    # Get the img corresponding 2d features and 3d points
    image = recon.find_image_with_name(img_name)
    img_sfm_points_2d = []
    img_sfm_points_3d = []

    for p2D in image.points2D:
        if p2D.has_point3D():  # only keep triangulated points
            xy = p2D.xy                    # (x,y) pixel
            p3D = recon.point3D(p2D.point3D_id)
            xyz = p3D.xyz                  # 3D coordinate
            img_sfm_points_2d.append(xy)
            img_sfm_points_3d.append(xyz)
    img_sfm_points_2d = np.array(img_sfm_points_2d).astype(int)   # shape (N,2)
    img_sfm_points_3d = np.array(img_sfm_points_3d)   # shape (N,3)

    # turn img sfm points to o3d and add to compos 
    img_sfm_points_3d_o3d = turn_points(img_sfm_points_3d, None)
    compos.append(img_sfm_points_3d_o3d)

    # keep only moge points corresp to sfm
    x = img_sfm_points_2d[:,0]
    y = img_sfm_points_2d[:,1]
    sfm_moge_colors = all_moge_colors[y, x] # it's flattened
    sfm_moge_points = all_moge_points[y, x] # it's flattened
    sfm_moge_points_o3d = turn_points(sfm_moge_points, sfm_moge_colors)
    print("sfm_moge_points_shape : ", sfm_moge_points.shape)
    print("img_sfm_points_3d_shape : ",img_sfm_points_3d.shape)


    _, s, R, t = align_points(sfm_moge_points, img_sfm_points_3d)
    moge_points = (s * (R @ moge_points.T)).T + t
    moge_o3d = turn_points(moge_points, moge_colors)
    o3d.io.write_point_cloud(f"{case_path}/points3D.ply", moge_o3d)
    compos.append(moge_o3d)

    MeshHandling.visualize_scene(compos) 

    return





    is_gs2_ready = True
    #_________________________________ IBVS on GS1 __________________________

    
    # get init_real_img     
    init_img_name = "DSC05590.jpg"
    init_img_real = ImageHandling.load_np_img(f"{gs1_images_path}/{init_img_name}")

    # get des gt
    des_gt_name = "GT_des.png"
    des_gt_img = ImageHandling.load_np_img(f"{case_path}/{des_gt_name}")
    
    # get des estim
    des_estim_name = "estim_des.png"
    des_estim_img = ImageHandling.load_np_img(f"{gs1_images_path}/{des_estim_name}")

    # Load gaussians, init_pose, gt_des_pose from colmap data
    gaussians = GaussiansHandling.load_gaussians_from_ply(gs1_path)
    recon = pycolmap.Reconstruction(gs1_sparse_path)
    poses_dict = PosesHandling.get_sfm_poses(recon)
    init_pose_gs1 = np.linalg.inv(poses_dict[init_img_name])
    des_pose_gs1 = np.linalg.inv(poses_dict[des_estim_name])

    # Render init pic 
    ImageHandling.plot_2_imgs(des_estim_img, des_gt_img)
    


    if (is_gs2_ready ==  False) :
        # Starting IBVS 
        start_dyna_ibvs_loop(gaussians, init_pose_gs1, des_pose_gs1, init_img_real, des_gt_img, des_estim_img)   

    else : 
        #____________________________  IBVS on GS2 _______________________________
        
        # Load gs2 gaussians and estim pose of estimdesimg from colmap data2
        gaussians2 = GaussiansHandling.load_gaussians_from_ply(gs2_path)
        recon2 = pycolmap.Reconstruction(gs2_sparse_path)

        # Get a camera K matrix (first camera)
        camera_id = list(recon2.cameras.keys())[0]
        camera = recon2.cameras[camera_id]
        K = camera.calibration_matrix()
        intrins_gs2 = (
        torch.from_numpy(K)          
        .float()                     
        .to("cuda")                  
        .unsqueeze(0)                
    )
        # get des_estim img's pose in gs2
        poses_dict2 = PosesHandling.get_sfm_poses(recon2)
        des_pose_gs2 = np.linalg.inv(poses_dict2[des_estim_name])
        
        # render that initial estim img from gs2 to see how it looks like from there
        des_estim_img2, depth = GaussiansHandling.render_gs_pic(*gaussians2, des_pose_gs2, intrins_gs2, CAM_W, CAM_H)

        ImageHandling.plot_2_imgs(des_estim_img2, des_estim_img2)
        ImageHandling.save_img(des_estim_img2, "estim_des_2", case_path)        
        
        # Starting IBVS 
        start_dyna_ibvs_loop(gaussians, init_pose_gs1, des_pose_gs1, init_img_real, des_gt_img, des_estim_img2)  


        












if __name__ == "__main__":
    main()























    #_______________________________save Bad trajectories ___________________________________


    """# get an img  
    img_name = "DSC05587.jpg"
    

    # Load gaussians and init-pose from colmap data
    gaussians = load_gaussians_from_ply(gs1_path)
    recon = pycolmap.Reconstruction(gs1_col_sparse_path)
    poses_dict = get_sfm_poses(recon)
    pose_gs1 = np.linalg.inv(poses_dict[img_name])

    # Render init pic 
    img_gs1, depth_gs1 = render_gs_pic(*gaussians, pose_gs1, intrins_gs1, CAM_W, CAM_H)

    # saving bad trajectories
    for i in range(50) :
        pose_gs1 = pose_gs1 @ get_homog([0,0,0.1,0,0,0])
        img, _ = render_gs_pic(*gaussians, pose_gs1, intrins_gs1, CAM_W, CAM_H)
        save_img(img, i, result_test_path)
        print(f"saved {i}")


    return"""

