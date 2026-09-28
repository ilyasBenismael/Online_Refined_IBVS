
import torch
from plyfile import PlyData, PlyElement
from gsplat import rasterization
import numpy as np
import open3d as o3d
from moge.model.v2 import MoGeModel
import utils3d 
import cv2
import subprocess
import time





class GaussiansHandling :



    @staticmethod
    def run_gs_training(sfm_path, output_path, gs_reso = 1, gs_nbr_itrs = 15000, curr_case_gs2_path = None):
        
        import sys, os
        # gaussian_splatting/ is a sibling of scripts/
        gs_inria_path = "gaussian_splatting2"
        sys.path.insert(0, gs_inria_path)
        
        from argparse import ArgumentParser
        from arguments import ModelParams, OptimizationParams, PipelineParams
        from train import training
        from utils.general_utils import safe_state

        # Step 1 - Fresh parser + register all GS arguments with their defaults
        parser = ArgumentParser()
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)

        # Step 2 - Parse only your overrides, everything else gets its default
        gs_args = [
            "-s", os.path.abspath(sfm_path),
            "-m", os.path.abspath(output_path),
            "-r", str(gs_reso),
            "--iterations", str(gs_nbr_itrs)           
        ]

        # we pass to gs inria repo (init function) this initplypath holdng the gs2 path of the case
        if curr_case_gs2_path:
            gs_args.extend(["--initial_ply_path", os.path.abspath(curr_case_gs2_path)])

        args = parser.parse_args(gs_args)

        # Step 3 - Manually set args that belong to __main__ only (not in any ParamGroup)
        args.save_iterations    = [gs_nbr_itrs]
        args.test_iterations    = []
        args.checkpoint_iterations = []
        args.start_checkpoint   = None
        args.debug_from         = -1
        args.disable_viewer     = True
        args.detect_anomaly     = False

        # Step 4 - Extract param groups and run training
        training(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.debug_from
        )


        




    @staticmethod
    def turn_points_to_gaussians(
        xyz,
        rgb,
        scale=0.025,
        alpha=0.09,
        device="cuda",
    ):

        C0 = 0.28209479177387814

        # =========================
        # Input prep
        # =========================

        # flatten to (H*W,3) if (H,W,3)
        xyz = np.asarray(xyz).reshape(-1, 3)
        rgb = np.asarray(rgb).reshape(-1, 3)
        N = xyz.shape[0]

        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)

        # =========================
        # SH degree = 3
        # =========================
        sh_degree = 3
        num_sh = (sh_degree + 1) ** 2  # = 16

        # =========================
        # RGB → SH DC
        # =========================
        f_dc = (rgb - 0.5) / C0  # (N,3)

        # =========================
        # Means
        # =========================
        means = torch.from_numpy(xyz).float().to(device)

        # =========================
        # Scales (REAL, not log)
        # =========================
        scales = torch.full((N, 3), scale, device=device)

        # =========================
        # Rotation (identity quaternion)
        # =========================
        quats = torch.zeros((N, 4), device=device)
        quats[:, 0] = 1.0

        # =========================
        # Opacity (already alpha)
        # =========================
        opacities = torch.full((N,), alpha, device=device)

        # =========================
        # SH tensor
        # =========================
        sh = torch.zeros((N, num_sh, 3), device=device)
        sh[:, 0, :] = torch.from_numpy(f_dc).to(device)


        return [means, quats, scales, opacities, sh]




    @staticmethod
    def merge_2_gaussians(g1, g2):

        means1, quats1, scales1, opacities1, sh1 = g1
        means2, quats2, scales2, opacities2, sh2 = g2

        # =========================
        # Safety check
        # =========================
        assert sh1.shape[1] == sh2.shape[1], "SH degrees mismatch"

        # =========================
        # Concatenate
        # =========================
        means = torch.cat([means1, means2], dim=0)
        quats = torch.cat([quats1, quats2], dim=0)
        scales = torch.cat([scales1, scales2], dim=0)
        opacities = torch.cat([opacities1, opacities2], dim=0)
        sh = torch.cat([sh1, sh2], dim=0)

        return [means, quats, scales, opacities, sh]



    @staticmethod
    def turn_gaussians_to_ply(gaussians, ply_path):

        C0 = 0.28209479177387814

        means, quats, scales, opacities, sh = gaussians

        # Move to CPU numpy
        xyz = means.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        quats = quats.detach().cpu().numpy()
        opacities = opacities.detach().cpu().numpy()
        sh = sh.detach().cpu().numpy()

        N = xyz.shape[0]
        num_sh = sh.shape[1]  # should be 16 for degree=3
        n_rest = 3 * (num_sh - 1)

        # =========================
        # SH → PLY format
        # =========================
        f_dc = sh[:, 0, :]  # (N,3)

        # flatten rest correctly
        f_rest = sh[:, 1:, :].reshape(N, -1)  # (N, 45)

        # =========================
        # Opacity → logit
        # =========================
        eps = 1e-6
        opacities = np.clip(opacities, eps, 1 - eps)
        opacity_logit = np.log(opacities / (1 - opacities))

        # =========================
        # Scale → log
        # =========================
        scale_log = np.log(scales)

        # =========================
        # Build dtype
        # =========================
        dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ]

        for i in range(n_rest):
            dtype.append((f"f_rest_{i}", "f4"))

        dtype.append(("opacity", "f4"))

        for i in range(3):
            dtype.append((f"scale_{i}", "f4"))

        for i in range(4):
            dtype.append((f"rot_{i}", "f4"))

        data = np.empty(N, dtype=dtype)

        # =========================
        # Fill data
        # =========================
        data["x"], data["y"], data["z"] = xyz.T
        data["nx"] = data["ny"] = data["nz"] = 0.0

        data["f_dc_0"] = f_dc[:, 0]
        data["f_dc_1"] = f_dc[:, 1]
        data["f_dc_2"] = f_dc[:, 2]

        for i in range(n_rest):
            data[f"f_rest_{i}"] = f_rest[:, i]

        data["opacity"] = opacity_logit

        data["scale_0"] = scale_log[:, 0]
        data["scale_1"] = scale_log[:, 1]
        data["scale_2"] = scale_log[:, 2]

        data["rot_0"] = quats[:, 0]
        data["rot_1"] = quats[:, 1]
        data["rot_2"] = quats[:, 2]
        data["rot_3"] = quats[:, 3]

        # =========================
        # Save
        # =========================
        PlyData([PlyElement.describe(data, "vertex")]).write(ply_path)

        print(f"PLY of {N} gaussians {N} saved to : {ply_path}")


    


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
    def turn_cam_matrix_to_gsplat_format(K) :

        intrins_gs = (
        torch.from_numpy(K)          
        .float()                     
        .to("cuda")                  
        .unsqueeze(0)                
        )

        return intrins_gs





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




    @staticmethod
    def get_moge_points(img, fov_x = None, reso_div = None, model_reso_lvl = 8, use_fp16_bool = True, depth_edge_threshold=0.005, moge_model = None):
        
        t0 = time.time()
        if reso_div is not None :
            h, w = img.shape[:2]
            new_w = max(1, w // reso_div)
            new_h = max(1, h // reso_div)
            img = cv2.resize(img, (new_w, new_h), cv2.INTER_AREA)



        # create nd put MoGe to eval
        if moge_model is None :
            moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda")
        moge_model.eval()

        # Making sure img is in 0-1
        if img.mean() > 1.0:
            img = img / 255.0

        # turn img to torch and apply moge
        image = torch.from_numpy(img).float().to("cuda").permute(2, 0, 1)
        t0 = time.time()

        if fov_x is None : 
            output = moge_model.infer(image, resolution_level=model_reso_lvl, use_fp16=use_fp16_bool)
        else : 
            output = moge_model.infer(image, resolution_level=model_reso_lvl, use_fp16=use_fp16_bool, fov_x = fov_x)    
        print(f"[infer] Took {time.time() - t0:.3f} sec")

        points = output["points"].cpu().numpy()  # (H, W, 3)
        depth = output["depth"].cpu().numpy()
        mask = output["mask"].cpu().numpy()

        all_points = points.copy()

        t0 = time.time()
        # check the edges (big depth diffs) and add it to the mask area to remove
        edge_mask = utils3d.np.depth_map_edge(depth, rtol=depth_edge_threshold)
        mask_cleaned = mask & (~edge_mask)
        print(f"[edge_mask] Took {time.time() - t0:.3f} sec")
        
        # get colors frm img and flatten all (colors, points, masks)
        colors = img.astype(np.float64)
        all_colors = colors.copy()
        colors_flat = colors.reshape(-1, 3)
        pts_flat = points.reshape(-1, 3).astype(np.float64)
        valid_flat = mask_cleaned.reshape(-1)

        # save clean points and colors
        masked_points = pts_flat[valid_flat]
        masked_colors = colors_flat[valid_flat]

        intrinsics = output["intrinsics"]  # (3, 3) numpy/torch array, normalized

        return masked_points, masked_colors , all_points, all_colors , mask_cleaned, intrinsics




    @staticmethod
    def get_gsinria_tensor_from_img(img):
        """
        Convert a NumPy image (BGR, as returned by cv2.imread or OpenCV)
        to the tensor format expected by Gaussian Splatting.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img).float() / 255.0

        gs_img_tensor = (
            img.permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
        )

        return gs_img_tensor









# # ______________________ Visualizing Gaussians _______________________

#     # Paths
#     plys_path = "my_results/playroom/mvs_test/mvs_low_resolution/plys" # gs original ouput format
#     sfm_path = "my_results/playroom/mvs_test/mvs_low_resolution/sfm_aligned/sparse/1"

#     # images names
#     img1_name = "72.png"
#     img2_name = "view2.jpg"
#     img3_name = "view3.jpg"
#     img4_name = "view4.jpg"
#     img5_name = "view5.jpg"

  
#     # getting img poses from colmap
#     recon = pycolmap.Reconstruction(sfm_path)
#     poses_dict2 = get_sfm_poses(recon)
#     img1_pose = np.linalg.inv(poses_dict2[img1_name])
#     img2_pose = np.linalg.inv(poses_dict2[img2_name])
#     img3_pose = np.linalg.inv(poses_dict2[img3_name])
#     img4_pose = np.linalg.inv(poses_dict2[img4_name])
        
#     # The load_gaussian_frm_folder takes the gs ouput path and we get a dict ([itr] = gaussians..), then load and make dense nd sparse gaussians dict
#     print("Loading Gaussian models...")
#     ply_dict = load_gaussians_from_folder(plys_path)

#     print("Rendering all iterations...")
#     results = render_all_iterations(ply_dict, img1_pose, img2_pose, img3_pose, img4_pose)

#     print("Launching visualization...")
#     visualize_iterations(
#         results)

#     return


