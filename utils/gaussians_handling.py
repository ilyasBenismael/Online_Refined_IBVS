import torch
from plyfile import PlyData
from gsplat import rasterization
import numpy as np
import open3d as o3d
from moge.model.v2 import MoGeModel
import utils3d 






class GaussiansHandling :



    @staticmethod
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





    @staticmethod
    def get_gs_viewmat(T, device="cuda"):
        # Taking normal pose (cam relative to wrld) and turn it to (wrld telative to cam)
        T = torch.tensor(T, dtype=torch.float32, device=device)
        viewmat = torch.linalg.inv(T)
        return viewmat.unsqueeze(0)




    @staticmethod
    def render_gs_pic(means, quats, scales, opacities, sh, T, K, W, H):

        T = GaussiansHandling.get_gs_viewmat(T)

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
            sh_degree=0,
            rasterize_mode="antialiased",
            render_mode="RGB+ED",
        )

        img   = image[0, ..., :-1].detach().cpu().numpy() 
        depth = image[0, ..., -1].detach().cpu().numpy() 

        img = np.clip(img, 0, 1)
        return img, depth





    def get_moge_points(img, threshold=0.005):
        
        # Loading MoGe
        device = "cuda"
        model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
        model.eval()

        # Making sure img is in 0-1
        if img.mean() > 1.0:
            img = img / 255.0

        # turn img to torch and apply moge
        image = torch.from_numpy(img).float().to(device).permute(2, 0, 1)
        output = model.infer(image)


        points = output["points"].cpu().numpy()  # (H, W, 3)
        depth = output["depth"].cpu().numpy()
        mask = output["mask"].cpu().numpy()

        all_points = points.copy()

        #check the edges (big depth diffs) and add it to the mask area to remove
        edge_mask = utils3d.np.depth_map_edge(depth, rtol=threshold)
        mask_cleaned = mask & (~edge_mask)

    
        
        # get colors frm img and flatten all (colors, points, masks)
        colors = img.astype(np.float64)
        all_colors = colors.copy()
        colors_flat = colors.reshape(-1, 3)
        pts_flat = points.reshape(-1, 3).astype(np.float64)
        valid_flat = mask_cleaned.reshape(-1)

        # save clean points and colors
        final_points = pts_flat[valid_flat]
        final_colors = colors_flat[valid_flat]

        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(final_points)
        o3d_points.colors = o3d.utility.Vector3dVector(final_colors)

        # Return FULL arrays (H, W, 3) for indexing by pixel coordinates
        return o3d_points, final_points, final_colors , all_points, all_colors 







    """turn_imgs_folder_to_moges("my_results/playroom/tests/main_test/00")
    return"""

    """# ______________________ Visualizing Gaussians _______________________

    # Paths
    plys_path = "my_results/playroom/mvs_test/mvs_low_resolution/plys" # gs original ouput format
    sfm_path = "my_results/playroom/mvs_test/mvs_low_resolution/sfm_aligned/sparse/1"

    # images names
    img1_name = "72.png"
    img2_name = "view2.jpg"
    img3_name = "view3.jpg"
    img4_name = "view4.jpg"
    img5_name = "view5.jpg"

  
    # getting img poses from colmap
    recon = pycolmap.Reconstruction(sfm_path)
    poses_dict2 = get_sfm_poses(recon)
    img1_pose = np.linalg.inv(poses_dict2[img1_name])
    img2_pose = np.linalg.inv(poses_dict2[img2_name])
    img3_pose = np.linalg.inv(poses_dict2[img3_name])
    img4_pose = np.linalg.inv(poses_dict2[img4_name])
        
    # The load_gaussian_frm_folder takes the gs ouput path and we get a dict ([itr] = gaussians..), then load and make dense nd sparse gaussians dict
    print("Loading Gaussian models...")
    ply_dict = load_gaussians_from_folder(plys_path)

    print("Rendering all iterations...")
    results = render_all_iterations(ply_dict, img1_pose, img2_pose, img3_pose, img4_pose)

    print("Launching visualization...")
    visualize_iterations(
        results)

    return"""