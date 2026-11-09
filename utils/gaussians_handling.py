import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from gsplat import rasterization
import numpy as np
import open3d as o3d
from moge.model.v2 import MoGeModel
import utils3d 
import cv2
import subprocess
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
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
        alpha=0.9,
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
        T = T.to(dtype=torch.float32, device=device) if torch.is_tensor(T) else torch.as_tensor(T, dtype=torch.float32, device=device)
        #T = torch.tensor(T, dtype=torch.float32, device=device)
        viewmat = torch.linalg.inv(T)
        return viewmat.unsqueeze(0)




    @staticmethod
    def render_gs_pic(means, quats, scales, opacities, sh, T, K, W, H, rndr_alpha=False, raster_mode="antialiased"):
        """Render RGB and depth, and optionally transfer the alpha map to CPU."""

        T = GaussiansHandling.get_gs_viewmat(T)

        with torch.no_grad():
            image, alpha, _ = rasterization(
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
                rasterize_mode=raster_mode,
                render_mode="RGB+ED")

        # Transfer RGB and depth together
        output = image[0].detach().cpu().numpy()
        img = output[..., :-1]
        depth = output[..., -1]
        np.clip(img, 0, 1, out=img)

        if rndr_alpha:
            alpha_map = alpha[0, ..., 0].detach().cpu().numpy()
            return img, depth, alpha_map

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

        normals = output["normal"].float().cpu().numpy()

        return masked_points, masked_colors , all_points, all_colors , mask_cleaned, intrinsics, normals




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
















    @staticmethod
    def turn_surfaces_to_gaussians(points, colors, organized_subsurfaces, texture_mask, textured_pixels_per_gaussian=10, max_gaussians_per_subsurface=100, min_points_per_gaussian=5, normal_scale=0.003, scale_multiplier=1.0, alpha=0.9, device="cuda"):
        """
        Create an adaptive number of Gaussians for every planar sub-surface.

        The number of Gaussians increases with the number of textured pixels.
        The sub-surface is spatially divided in image space, then one PCA
        Gaussian is fitted to all 3D points of every resulting region.
        (The PCA fitting is done for all the regions at once at the end, not one by one.)
        """

        C0 = 0.28209479177387814

        if textured_pixels_per_gaussian <= 0:
            raise ValueError("textured_pixels_per_gaussian must be greater than zero")

        texture_mask = np.asarray(texture_mask, dtype=bool)
        h, w = texture_mask.shape

        points_flat = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        colors_flat = np.asarray(colors, dtype=np.float64).reshape(-1, 3)

        if len(points_flat) != h * w or len(colors_flat) != h * w:
            raise ValueError("points, colors and texture_mask must represent the same HxW image")

        # checked once for the whole image (which points are valid, which pixels are textured)
        finite_flat = np.isfinite(points_flat).all(axis=1)
        tex_flat = texture_mask.ravel()

        min_pts = min_points_per_gaussian
        final_indices = []   # one array of point indices per final group

        for surface in organized_subsurfaces:
            for subsurface in surface["subsurfaces"]:
                indices = np.asarray(subsurface["indices"], dtype=np.int64)
                pixel_yx = np.asarray(subsurface["pixel_yx"], dtype=np.int32)

                # Remove invalid 3D points before subdivision
                valid = finite_flat[indices]
                if not valid.all():
                    indices = indices[valid]
                    pixel_yx = pixel_yx[valid]

                if len(indices) < min_pts:
                    continue

                textured_count = int(np.count_nonzero(tex_flat[indices]))

                # One base Gaussian plus another for every X textured pixels
                target = min(1 + textured_count // textured_pixels_per_gaussian, max_gaussians_per_subsurface)

                # each group keeps: [its indices, its pixel_yx, its textured count]
                # (so we never have to recount the textured pixels again)
                groups = [[indices, pixel_yx, textured_count]]

                # Repeatedly divide the group containing the most textured pixels
                while len(groups) < target:
                    best_id = None
                    best_score = -1
                    for gid, g in enumerate(groups):
                        if len(g[1]) < 2 * min_pts:
                            continue
                        if g[2] > best_score:
                            best_score = g[2]
                            best_id = gid

                    if best_id is None:
                        break

                    g = groups.pop(best_id)
                    g_idx, g_yx = g[0], g[1]

                    # Split along the longest image-space axis: Y or X
                    split_axis = int(np.argmax(np.ptp(g_yx, axis=0)))
                    order = np.argsort(g_yx[:, split_axis], kind="stable")
                    middle = len(order) // 2
                    first, second = order[:middle], order[middle:]

                    if len(first) < min_pts or len(second) < min_pts:
                        groups.append(g)
                        break

                    i1, i2 = g_idx[first], g_idx[second]
                    groups.append([i1, g_yx[first], int(np.count_nonzero(tex_flat[i1]))])
                    groups.append([i2, g_yx[second], int(np.count_nonzero(tex_flat[i2]))])

                # every final spatial group will become one Gaussian
                for g in groups:
                    if len(g[0]) >= min_pts:
                        final_indices.append(g[0])

        if not final_indices:
            raise ValueError("No valid Gaussians could be created")

        # ---------- Fit one PCA Gaussian to every final group, all at once ----------

        # glue all the groups together in one big array, and remember where each one starts
        counts = np.fromiter((len(a) for a in final_indices), dtype=np.int64, count=len(final_indices))
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        all_idx = np.concatenate(final_indices)
        gid = np.repeat(np.arange(len(counts)), counts)

        pts = points_flat[all_idx]
        cols = colors_flat[all_idx]

        # Gaussian center and mean color of every group
        centers = np.add.reduceat(pts, starts, axis=0) / counts[:, None]
        rgb = np.add.reduceat(cols, starts, axis=0) / counts[:, None]

        # Covariance of every group
        centered = pts - centers[gid]
        outer = (centered[:, :, None] * centered[:, None, :]).reshape(-1, 9)
        cov = (np.add.reduceat(outer, starts, axis=0) / counts[:, None]).reshape(-1, 3, 3)

        # PCA: eigh gives the smallest eigenvalue first, so we flip to get
        # Longest axis, second planar axis, normal axis
        _, eigvecs = np.linalg.eigh(cov)
        R = eigvecs[:, :, ::-1].copy()

        # Make the PCA axes a proper rotation matrix
        neg = np.linalg.det(R) < 0
        R[neg, :, 2] *= -1.0

        # Project points onto their PCA axes, and take the biggest distance on the 2 planar axes
        projected = np.einsum("ni,nij->nj", centered, R[gid])
        half_ext = np.maximum.reduceat(np.abs(projected[:, :2]), starts, axis=0)

        # Planar coverage and fixed surface thickness
        scales_np = np.empty((len(counts), 3), dtype=np.float64)
        scales_np[:, :2] = np.maximum(scale_multiplier * half_ext, normal_scale)
        scales_np[:, 2] = normal_scale

        # SciPy: [x,y,z,w] → Inria: [w,x,y,z]
        quats_np = Rotation.from_matrix(R).as_quat()[:, [3, 0, 1, 2]]

        xyz = centers

        if rgb.max() > 1.0:
            rgb = rgb / 255.0

        rgb = np.clip(rgb, 0.0, 1.0)
        f_dc = (rgb - 0.5) / C0
        number_gaussians = len(xyz)

        # send everything to the GPU
        means = torch.as_tensor(xyz, dtype=torch.float32, device=device)
        scales = torch.as_tensor(scales_np, dtype=torch.float32, device=device)
        quats = torch.as_tensor(quats_np, dtype=torch.float32, device=device)
        opacities = torch.full((number_gaussians,), alpha, dtype=torch.float32, device=device)

        sh_degree = 3
        number_sh = (sh_degree + 1) ** 2
        sh = torch.zeros((number_gaussians, number_sh, 3), dtype=torch.float32, device=device)
        sh[:, 0, :] = torch.as_tensor(f_dc, dtype=torch.float32, device=device)

        return [means, quats, scales, opacities, sh]






    @staticmethod
    def get_low_alpha_mask(alpha_map, alpha_threshold=0.3):
        """Return True for pixels with low or invalid accumulated opacity."""
        alpha_map = np.asarray(alpha_map, dtype=np.float64)
        low_alpha_mask = (~np.isfinite(alpha_map) | (alpha_map < alpha_threshold))
        return low_alpha_mask















    @staticmethod
    def turn_image_to_gaussians(image, points, valid_mask, texture_mask, moge_colors, patch_size=50, textured_pixels_per_gaussian=20, max_gaussians_per_patch=20, default_scale=0.01, scale_multiplier=0.3, min_scale=1e-4, max_scale=0.05, alpha=0.99, device="cuda"):
        """
        Initialize one Gaussian per selected image pixel.

        Each patch gets at least one Gaussian at the valid pixel nearest its
        center. Additional Gaussians are selected according to the number and
        spatial distribution of textured valid pixels. Gaussian scales are
        initialized from the average distance to the three nearest selected
        Gaussian points.
        """

        C0 = 0.28209479177387814

        image = np.asarray(image)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        texture_mask = np.asarray(texture_mask, dtype=bool)
        moge_colors = np.asarray(moge_colors)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (H, W, 3)")

        h, w = image.shape[:2]

        if valid_mask.shape != (h, w) or texture_mask.shape != (h, w):
            raise ValueError("valid_mask and texture_mask must have shape (H, W)")

        if moge_colors.shape != (h, w, 3):
            raise ValueError("moge_colors must have shape (H, W, 3)")

        if textured_pixels_per_gaussian <= 0 or max_gaussians_per_patch <= 0:
            raise ValueError("Gaussian count parameters must be greater than zero")

        if scale_multiplier <= 0 or min_scale <= 0 or default_scale <= 0:
            raise ValueError("scale parameters must be greater than zero")

        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)

        if len(points) != h * w:
            raise ValueError("points must contain one 3D point per image pixel")

        colors_flat = moge_colors.reshape(-1, 3).astype(np.float64)
        finite_points = np.isfinite(points).all(axis=1).reshape(h, w)
        finite_colors = np.isfinite(colors_flat).all(axis=1).reshape(h, w)
        valid_mask &= finite_points & finite_colors

        selected_indices = []

        for y0 in range(0, h, patch_size):
            for x0 in range(0, w, patch_size):
                y1 = min(y0 + patch_size, h)
                x1 = min(x0 + patch_size, w)

                local_y, local_x = np.where(valid_mask[y0:y1, x0:x1])

                if len(local_y) == 0:
                    continue

                global_y = local_y + y0
                global_x = local_x + x0
                coords = np.column_stack((global_y, global_x))
                textured = texture_mask[global_y, global_x]

                textured_count = int(textured.sum())
                target = max(1, int(np.ceil(textured_count / textured_pixels_per_gaussian)))
                target = min(target, max_gaussians_per_patch, len(coords))

                # Always select the valid pixel nearest the patch center.
                patch_center = np.array([(y0 + y1 - 1) / 2, (x0 + x1 - 1) / 2])
                center_id = np.argmin(np.sum((coords - patch_center) ** 2, axis=1))
                selected_indices.append(global_y[center_id] * w + global_x[center_id])

                if target == 1:
                    continue

                remaining = np.arange(len(coords)) != center_id
                groups = [(coords[remaining], textured[remaining])]

                # Split image regions with the most textured pixels first.
                while len(groups) < target - 1:
                    eligible = [i for i, (group_coords, _) in enumerate(groups) if len(group_coords) >= 2]

                    if not eligible:
                        break

                    split_id = max(eligible, key=lambda i: (groups[i][1].sum(), len(groups[i][0])))
                    group_coords, group_textured = groups.pop(split_id)

                    axis = int(np.argmax(np.ptp(group_coords, axis=0)))
                    order = np.argsort(group_coords[:, axis], kind="stable")
                    middle = len(order) // 2
                    first_ids, second_ids = order[:middle], order[middle:]

                    groups.append((group_coords[first_ids], group_textured[first_ids]))
                    groups.append((group_coords[second_ids], group_textured[second_ids]))

                # Select one pixel per region, preferring textured pixels.
                for group_coords, group_textured in groups:
                    candidates = np.flatnonzero(group_textured)

                    if len(candidates) == 0:
                        candidates = np.arange(len(group_coords))

                    region_center = group_coords.mean(axis=0)
                    candidate_coords = group_coords[candidates]
                    chosen = candidates[np.argmin(np.sum((candidate_coords - region_center) ** 2, axis=1))]
                    py, px = group_coords[chosen]
                    selected_indices.append(py * w + px)

        if not selected_indices:
            print("No valid pixels were found; returning one invisible default Gaussian")
            means = torch.zeros((1, 3), dtype=torch.float32, device=device)
            quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
            scales = torch.full((1, 3), default_scale, dtype=torch.float32, device=device)
            opacities = torch.zeros((1,), dtype=torch.float32, device=device)
            sh = torch.zeros((1, 16, 3), dtype=torch.float32, device=device)
            return [means, quats, scales, opacities, sh]

        selected_indices = np.asarray(selected_indices, dtype=np.int64)
        xyz = points[selected_indices]
        rgb = colors_flat[selected_indices]

        if rgb.max() > 1.0:
            rgb = rgb / 255.0

        rgb = np.clip(rgb, 0.0, 1.0)
        f_dc = (rgb - 0.5) / C0
        number_gaussians = len(selected_indices)

        # Use the average distance to up to three other selected points.
        if number_gaussians > 1:
            tree = cKDTree(xyz)
            neighbor_count = min(7, number_gaussians)
            distances, _ = tree.query(xyz, k=neighbor_count)
            average_neighbor_distance = distances[:, 1:].mean(axis=1)
            average_neighbor_distance = np.maximum(average_neighbor_distance * scale_multiplier, min_scale)
            average_neighbor_distance = np.minimum(average_neighbor_distance, max_scale)

        else:
            average_neighbor_distance = np.full(number_gaussians, default_scale * scale_multiplier)

        # Use the same scale along X, Y, and Z.
        scales_np = np.repeat(average_neighbor_distance[:, None], 3, axis=1)

        # Identity rotations in [w, x, y, z] order.
        quats_np = np.zeros((number_gaussians, 4), dtype=np.float32)
        quats_np[:, 0] = 1.0

        means = torch.as_tensor(xyz, dtype=torch.float32, device=device)
        scales = torch.as_tensor(scales_np, dtype=torch.float32, device=device)
        quats = torch.as_tensor(quats_np, dtype=torch.float32, device=device)
        opacities = torch.full((number_gaussians,), alpha, dtype=torch.float32, device=device)

        sh_degree = 3
        number_sh = (sh_degree + 1) ** 2
        sh = torch.zeros((number_gaussians, number_sh, 3), dtype=torch.float32, device=device)
        sh[:, 0, :] = torch.as_tensor(f_dc, dtype=torch.float32, device=device)

        return [means, quats, scales, opacities, sh]















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




    @staticmethod
    def optimize_poses_gaussians(
        img_names_to_optimize_poses,
        poses_by_img_name,
        img_folder_path,
        intrinsics,
        gaussians,
        number_iters,
        raster_mode="classic",
        device=None,
        last_2_kfs_weight=1,
        pose_learning_rate=1e-4,
    ):
        import os
        import numpy as np
        import torch
        import torch.nn.functional as F
        from utils.image_handling import ImageHandling

        img_names = list(poses_by_img_name)
        init_name = img_names[0]
        optimize_names = set(img_names_to_optimize_poses)

        unknown_names = optimize_names.difference(img_names)
        if unknown_names:
            raise ValueError(f"Unknown pose-optimization names: {sorted(unknown_names)}")

        if device is None:
            device = gaussians[0].device if torch.is_tensor(gaussians[0]) else ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        dtype = torch.float32

        def fixed_tensor(value):
            return torch.as_tensor(value, device=device, dtype=dtype).detach().clone().contiguous()

        images = {}
        for name in img_names:
            image = np.ascontiguousarray(ImageHandling.load_np_img(os.path.join(img_folder_path, name)))
            images[name] = fixed_tensor(image).permute(2, 0, 1).unsqueeze(0).contiguous() / 255.0

        _, _, H, W = images[init_name].shape
        K = fixed_tensor(intrinsics).reshape(1, 3, 3)

        means_init, quats_init, scales_init, opacities_init, sh_init = [fixed_tensor(value) for value in gaussians]
        if sh_init.ndim == 2:
            sh_init = sh_init.unsqueeze(1)
        elif sh_init.shape[-1] != 3 and sh_init.shape[1] == 3:
            sh_init = sh_init.transpose(1, 2)
        sh_init = sh_init.contiguous()

        means = torch.nn.Parameter(means_init)
        quats = torch.nn.Parameter(quats_init)
        log_scales = torch.nn.Parameter(scales_init.log())
        opacity_logits = torch.nn.Parameter(torch.logit(opacities_init.reshape(-1).clamp(1e-6, 1.0 - 1e-6)))
        sh = torch.nn.Parameter(sh_init)

        fixed_poses = {name: fixed_tensor(pose) for name, pose in poses_by_img_name.items()}
        fixed_poses[init_name] = torch.eye(4, device=device, dtype=dtype)

        pose_deltas = {
            name: torch.nn.Parameter(torch.zeros(6, device=device, dtype=dtype))
            for name in img_names[1:]
            if name in optimize_names
        }

        parameter_groups = [
            {"params": [means], "lr": 1.6e-4, "eps": 1e-15},
            {"params": [log_scales], "lr": 5e-3, "eps": 1e-15},
            {"params": [opacity_logits], "lr": 5e-2, "eps": 1e-15},
            {"params": [quats], "lr": 1e-3, "eps": 1e-15},
            {"params": [sh], "lr": 2.5e-3, "eps": 1e-15},
        ]
        if pose_deltas:
            parameter_groups.append({"params": list(pose_deltas.values()), "lr": pose_learning_rate, "weight_decay": 1e-6})

        optimizer = torch.optim.Adam(parameter_groups, foreach=False, fused=False)

        # Same SSIM implementation as before.
        window_size, sigma = 11, 1.5
        pad = window_size // 2
        coords = torch.arange(window_size, device=device, dtype=dtype)
        kernel = torch.exp(-((coords - pad) ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / kernel.sum()
        window = torch.outer(kernel, kernel).reshape(1, 1, window_size, window_size).expand(3, 1, -1, -1).contiguous()

        def ssim_loss(predicted, target):
            mu1 = F.conv2d(predicted, window, padding=pad, groups=3)
            mu2 = F.conv2d(target, window, padding=pad, groups=3)
            mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2

            var1 = F.conv2d(predicted * predicted, window, padding=pad, groups=3) - mu1_sq
            var2 = F.conv2d(target * target, window, padding=pad, groups=3) - mu2_sq
            cov = F.conv2d(predicted * target, window, padding=pad, groups=3) - mu12

            c1, c2 = 0.01 ** 2, 0.03 ** 2
            similarity = ((2.0 * mu12 + c1) * (2.0 * cov + c2)) / ((mu1_sq + mu2_sq + c1) * (var1 + var2 + c2))
            return 1.0 - similarity.mean()

        def get_pose(name):
            if name not in pose_deltas:
                return fixed_poses[name].unsqueeze(0)

            delta = pose_deltas[name]
            x, y, z = delta[3:].unbind()
            zero = torch.zeros_like(x)
            skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero]).reshape(3, 3)
            rotation = torch.matrix_exp(skew)

            top = torch.cat([rotation, delta[:3, None]], dim=1)
            bottom = delta.new_tensor([[0.0, 0.0, 0.0, 1.0]])
            correction = torch.cat([top, bottom], dim=0)
            return (fixed_poses[name] @ correction).unsqueeze(0)

        # Preserve the exact weighted iteration allocation.
        weights = np.ones(len(img_names), dtype=np.float64)
        if len(img_names) > 1:
            weights[max(1, len(img_names) - 2):] = last_2_kfs_weight

        raw_counts = number_iters * weights / weights.sum()
        counts = np.floor(raw_counts).astype(int)
        leftover = number_iters - int(counts.sum())
        counts[np.argsort(-(raw_counts - counts))[:leftover]] += 1

        schedule = [name for name, count in zip(img_names, counts) for _ in range(int(count))]
        np.random.default_rng(0).shuffle(schedule)

        print(f"\nPose + GS optimization | Device: {device} | Image: {W}x{H} | Gaussians: {means.shape[0]}")
        print(f"Trainable poses: {list(pose_deltas)} | Pose LR: {pose_learning_rate:.3e}")
        print(f"Iteration allocation: {dict(zip(img_names, counts.tolist()))}")
        if init_name in optimize_names:
            print(f"{init_name} is the initial image: its pose stays fixed at identity.")
        if not pose_deltas:
            print("No trainable keyframe poses: only Gaussians will change.")

        def gradient_text(gradient):
            return "None" if gradient is None else f"{gradient.detach().norm().item():.3e}"

        diagnosed_names = set()
        pose_updates = dict.fromkeys(pose_deltas, 0)
        loss_history = []

        for step, name in enumerate(schedule):
            optimizer.zero_grad(set_to_none=True)
            delta = pose_deltas.get(name)
            candidate_c2w = get_pose(name)

            raw_viewmat = GaussiansHandling.get_gs_viewmat(candidate_c2w[0])
            viewmat = torch.as_tensor(raw_viewmat, device=device, dtype=dtype).reshape(1, 4, 4)

            if delta is not None:
                candidate_c2w.retain_grad()
                if viewmat.requires_grad:
                    viewmat.retain_grad()
                delta_before = delta.detach().clone()
                pose_before = candidate_c2w.detach().clone()

            rendered, alpha, _ = rasterization(
                means=means,
                quats=quats,
                scales=log_scales.exp(),
                opacities=opacity_logits.sigmoid(),
                colors=sh,
                viewmats=viewmat,
                Ks=K,
                width=W,
                height=H,
                sh_degree=0,
                rasterize_mode=raster_mode,
                render_mode="RGB",
            )

            if delta is not None and name not in diagnosed_names:
                print(
                    f"[pose path: {name}] c2w.requires_grad={candidate_c2w.requires_grad} | "
                    f"viewmat source={type(raw_viewmat).__name__} | viewmat.requires_grad={viewmat.requires_grad} | "
                    f"render.requires_grad={rendered.requires_grad} | mean alpha={alpha.detach().mean().item():.3e}"
                )
                diagnosed_names.add(name)

            rendered = rendered.permute(0, 3, 1, 2).contiguous()
            target = images[name]
            l1 = F.l1_loss(rendered, target)
            ssim = ssim_loss(rendered, target)
            loss = torch.lerp(l1, ssim, 0.2)
            loss.backward()

            if delta is None:
                pose_log = "pose=fixed"
            else:
                grad = delta.grad
                translation_grad = None if grad is None else grad[:3]
                rotation_grad = None if grad is None else grad[3:]
                view_grad = viewmat.grad if viewmat.requires_grad else None

                pose_log = (
                    f"pose_c2w_grad={gradient_text(candidate_c2w.grad)}"
                )

                if grad is None:
                    pose_log += " | NO GRADIENT RECEIVED"
                elif not torch.isfinite(grad).all().item():
                    pose_log += " | NON-FINITE POSE GRADIENT"
                elif torch.count_nonzero(grad).item() == 0:
                    pose_log += f" | ZERO GRADIENT, mean alpha={alpha.detach().mean().item():.3e}"

            means_grad = gradient_text(means.grad)
            optimizer.step()

            if delta is not None:
                pose_updates[name] += 1

            loss_value, l1_value, ssim_value = torch.stack([loss.detach(), l1.detach(), ssim.detach()]).cpu().tolist()
            loss_history.append({"iteration": step, "image": name, "loss": loss_value, "l1": l1_value, "ssim": ssim_value})

            print(
                f"Iteration {step + 1:03d}/{number_iters} | {name} | loss={loss_value:.6f} | "
                f"L1={l1_value:.6f} | SSIM_loss={ssim_value:.6f} | means_grad={means_grad} | {pose_log}"
            )

        with torch.no_grad():
            refined_poses = {name: get_pose(name)[0].cpu().numpy() for name in img_names}
            optimized_gaussians = [means.detach(), quats.detach(), log_scales.exp().detach(), opacity_logits.sigmoid().detach(), sh.detach()]

            for name, delta in pose_deltas.items():
                position_change = (get_pose(name)[0, :3, 3] - fixed_poses[name][:3, 3]).norm().item()
                rotation_change = torch.rad2deg(delta[3:].norm()).item()
                print(
                    f"[final pose: {name}] selected_iterations={pose_updates[name]} | "
                    f"position_change={position_change:.3e} | rotation_change={rotation_change:.3e} degrees"
                )

        return refined_poses, optimized_gaussians, loss_history





