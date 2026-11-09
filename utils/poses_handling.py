from utils.lin_algeb import LinAlgeb
from utils.my_utils import MyUtils
import numpy as np
import subprocess
import os
import pycolmap
import shutil
import heapq
import cv2
from pathlib import Path
import torch
import torch.nn.functional as F
from gsplat import rasterization
from utils.gaussians_handling import GaussiansHandling




class PosesHandling :


    @staticmethod
    def get_recons(sfm_path) :
        return pycolmap.Reconstruction(f"{sfm_path}/sparse/0")



    @staticmethod
    def create_sfm_structure(root_path, sfm_name):
        sfm_path = os.path.join(root_path, sfm_name)

        images_path = os.path.join(sfm_path, "images")
        sparse_path = os.path.join(sfm_path, "sparse")
        txt_path = os.path.join(sfm_path, "new_imgs.txt")

        # Create folders
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(sparse_path, exist_ok=True)

        # Create empty txt file
        with open(txt_path, "w") as f:
            pass

        print(f"Created sfm structure at: {sfm_path}")
    




    @staticmethod
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



    @staticmethod
    def get_sparse_points(recon) :
        return np.array([p.xyz for p in recon.points3D.values()]) 



    @staticmethod
    def get_2d_3d_points_of_img(recon, img_name) :
        
        img_sfm = recon.find_image_with_name(img_name)
        img_sfm_points_2d = []
        img_sfm_points_3d = []
        for p2D in img_sfm.points2D:
            if p2D.has_point3D():  # only keep triangulated points
                xy = p2D.xy                    # (x,y) pixel
                p3D = recon.point3D(p2D.point3D_id)
                xyz = p3D.xyz                  # 3D coordinate
                img_sfm_points_2d.append(xy)
                img_sfm_points_3d.append(xyz)
        img_sfm_points_2d = np.array(img_sfm_points_2d).astype(int)   # shape (N,2)
        img_sfm_points_3d = np.array(img_sfm_points_3d)   # shape (N,3)
        
        return img_sfm_points_2d, img_sfm_points_3d



    @staticmethod
    def get_cam_matrix(recon) :
        camera_id = list(recon.cameras.keys())[0]
        camera = recon.cameras[camera_id]
        K = camera.calibration_matrix()
        return K
    



    @staticmethod
    def get_img_sfm_pose(recon, img_name) :
        homog_poses = PosesHandling.get_sfm_poses(recon)
        img_pose = homog_poses[img_name]
        return LinAlgeb.inverse_mat(img_pose)



    @staticmethod
    def apply_sfm_reconstruction(sfm_path, sequential = True) :

        images_path = f"{sfm_path}/images"
        sparse_path = f"{sfm_path}/sparse"
        database_path = f"{sfm_path}/database.db" 
        
        # 1. Create database
        MyUtils.run_cmd(f"colmap database_creator --database_path {database_path}")

        # 2. Feature extraction
        MyUtils.run_cmd(
            f"colmap feature_extractor "
            f"--database_path {database_path} "
            f"--image_path {images_path} "
            f"--ImageReader.camera_model PINHOLE "
            f"--ImageReader.single_camera 1 "
            f"--FeatureExtraction.use_gpu 1 "
            f"--SiftExtraction.max_image_size 1500 "
        )

        # 3. Sequential matching
        matcher = "sequential"
        if not sequential :
            matcher = "exhaustive"
        
        MyUtils.run_cmd(
            f"colmap {matcher}_matcher "
            f"--database_path {database_path} "
            f"--FeatureMatching.use_gpu 1"
        )
        
        # 4. Mapping
        MyUtils.run_cmd(
            f"colmap mapper "
            f"--database_path {database_path} "
            f"--image_path {images_path} "
            f"--output_path {sparse_path}"
        )






    @staticmethod
    def align_new_image(new_image_path, sfm_path, sequential = False):
                
        # the img path 
        new_image_name = os.path.basename(new_image_path)

        # Copy image into images/
        images_dir = os.path.join(sfm_path, "images")
        destina_new_img_path = os.path.join(images_dir, new_image_name)

        if os.path.exists(destina_new_img_path):
            raise Exception(f"img to align already exist")
        
        shutil.copy(new_image_path, destina_new_img_path)

        # Overwrite new_imgs.txt
        new_imgs_txt = os.path.join(sfm_path, "new_imgs.txt")
        with open(new_imgs_txt, "w") as f: f.write(new_image_name + "\n")

        # Run COLMAP commands
        db_path = os.path.join(sfm_path, "database.db")
        sparse_path = os.path.join(sfm_path, "sparse/0")

        # Feature extraction (ONLY new image)
        cmd = (
            f"colmap feature_extractor "
            f"--database_path {db_path} "
            f"--image_path {images_dir} "
            f"--image_list_path {new_imgs_txt} "
            f"--ImageReader.existing_camera_id 1 "
            f"--FeatureExtraction.use_gpu 1 "
            f"--SiftExtraction.max_image_size 1500")
        MyUtils.run_cmd(cmd)

        # Matching
        matcher = "sequential"
        if not sequential :
            matcher = "exhaustive"
        
        MyUtils.run_cmd(
            f"colmap {matcher}_matcher "
            f"--database_path {db_path} "
            f"--FeatureMatching.use_gpu 1")

        # Registration
        cmd = (
            f"colmap image_registrator "
            f"--database_path {db_path} "
            f"--input_path {sparse_path} "
            f"--output_path {sparse_path}")
        MyUtils.run_cmd(cmd)






    @staticmethod
    def align_new_images(new_image_paths, sfm_path, sequential=False):

        # Convert paths to strings and extract image names.
        new_image_paths = [os.fspath(path) for path in new_image_paths]
        new_image_names = [
            os.path.basename(path)
            for path in new_image_paths
        ]

        if not new_image_paths:
            raise ValueError("new_image_paths is empty.")

        # Prevent different source images from having the same filename.
        if len(new_image_names) != len(set(new_image_names)):
            raise ValueError(
                "Some input images have the same filename."
            )

        images_dir = os.path.join(sfm_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Check everything before copying any image.
        for image_path, image_name in zip(
            new_image_paths,
            new_image_names
        ):
            if not os.path.isfile(image_path):
                raise FileNotFoundError(
                    f"Image does not exist: {image_path}"
                )

            destination = os.path.join(images_dir, image_name)

            if os.path.exists(destination):
                raise FileExistsError(
                    f"Image to align already exists: {destination}"
                )

        # Copy all new images into the COLMAP images folder.
        for image_path, image_name in zip(
            new_image_paths,
            new_image_names
        ):
            destination = os.path.join(images_dir, image_name)
            shutil.copy2(image_path, destination)

        # Write all image names, one per line.
        new_imgs_txt = os.path.join(sfm_path, "new_imgs.txt")

        with open(new_imgs_txt, "w") as file:
            for image_name in new_image_names:
                file.write(image_name + "\n")

        db_path = os.path.join(sfm_path, "database.db")
        sparse_path = os.path.join(sfm_path, "sparse/0")

        # Extract features only for the new images.
        cmd = (
            f'colmap feature_extractor '
            f'--database_path "{db_path}" '
            f'--image_path "{images_dir}" '
            f'--image_list_path "{new_imgs_txt}" '
            f'--ImageReader.existing_camera_id 1 '
            f'--FeatureExtraction.use_gpu 1 '
            f'--SiftExtraction.max_image_size 1500'
        )
        MyUtils.run_cmd(cmd)

        # Match the images.
        matcher = "sequential" if sequential else "exhaustive"

        cmd = (
            f'colmap {matcher}_matcher '
            f'--database_path "{db_path}" '
            f'--FeatureMatching.use_gpu 1'
        )
        MyUtils.run_cmd(cmd)

        # Register all new images.
        cmd = (
            f'colmap image_registrator '
            f'--database_path "{db_path}" '
            f'--input_path "{sparse_path}" '
            f'--output_path "{sparse_path}"'
        )
        MyUtils.run_cmd(cmd)

        print(f"Aligned {len(new_image_names)} new images:")
        for image_name in new_image_names:
            print(f"  - {image_name}")










    @staticmethod
    def astar_from_voxel_centers(all_centers, occupied_centers, start, goal, voxel_size):

        all_centers = np.asarray(all_centers, dtype=float)
        occupied_centers = np.asarray(occupied_centers, dtype=float).reshape(-1, 3)

        # 1. Recover grid origin
        min_corner = all_centers.min(axis=0) - voxel_size / 2

        # 2. Convert center → index
        def to_idx(p):
            return tuple(np.floor((p - min_corner) / voxel_size).astype(int))

        start_idx = to_idx(start)
        goal_idx  = to_idx(goal)

        # Convert occupied centers → indices
        occupied_idx = set(map(tuple, np.floor((occupied_centers - min_corner) / voxel_size).astype(int)))

        # 3. A* setup
        neighbors = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        open_set = []
        heapq.heappush(open_set, (0, start_idx))

        came_from = {}
        g_score = {start_idx: 0}

        # 4. A* loop
        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_idx:
                # reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path = path[::-1]

                # convert back to centers
                path_world = min_corner + (np.array(path) + 0.5) * voxel_size
                return path_world

            for dx, dy, dz in neighbors:
                neighbor = (current[0]+dx, current[1]+dy, current[2]+dz)

                if neighbor in occupied_idx:
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal_idx)
                    heapq.heappush(open_set, (f, neighbor))

        return None


    @staticmethod
    def get_K(fx, fy, cx, cy):
        return np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)




    @staticmethod
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




    @staticmethod
    def get_intrins_gs(gs_sfm_path):
        recons = PosesHandling.get_recons(gs_sfm_path)
        K = PosesHandling.get_cam_matrix(recons)
        intrins_gs = GaussiansHandling.turn_cam_matrix_to_gsplat_format(K)
        return intrins_gs




    @staticmethod
    def make_colmap_data(sparse_path, img_ids, img_names, poses, W, H, fx, fy, cx=None, cy=None) :

        if cx is None :
            cx = W / 2.0
            cy =  H / 2.0

        # ---------------- cameras.txt ----------------
        with open(os.path.join(sparse_path, "cameras.txt"), "w") as f_cam:
            f_cam.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, W, H, fx, fy, cx, cy\n")
            f_cam.write("1 PINHOLE {} {} {} {} {} {}\n".format(W, H, fx, fy, cx, cy))

        # ---------------- point3d.txt ---------------
        Path(f"{sparse_path}/points3D.txt").touch()


        # ---------------- images.txt ----------------
        with open(os.path.join(sparse_path, "images.txt"), "w") as f_img:

            f_img.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f_img.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

            for (img_id, pose, name) in zip(img_ids, poses, img_names):

                # Prepare the qvec (for rotation) and translation
                w2c = np.linalg.inv(pose)
                R = w2c[:3, :3]
                qvec = PosesHandling.rotmat_to_quaternion_hamilton(R)
                t = w2c[:3, 3]

                # line1 of poses
                f_img.write(
                    f"{img_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
                    f"{t[0]} {t[1]} {t[2]} 1 {name}\n")
                # line2 of 2d points keep it blank
                f_img.write(" " + "\n")


    @staticmethod
    def add_imgs_to_colmap(ids, img_names, poses, sparse_path) :

        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        # POINTS2D[] as (X, Y, POINT3D_ID)
        camera_id = 1
        txt_file = f"{sparse_path}/0/images.txt"

        for (pose, img_name, image_id) in zip(poses, img_names, ids):

            # Prepare the qvec (for rotation) and translation
            w2c = np.linalg.inv(pose)
            R = w2c[:3, :3]
            qvec = PosesHandling.rotmat_to_quaternion_hamilton(R)
            t = w2c[:3, 3]

            # line1 of poses
            MyUtils.add_line_to_text(txt_file, f"{image_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {t[0]} {t[1]} {t[2]} {camera_id} {img_name}" )

            # line 2 blank for 2d points
            MyUtils.add_line_to_text(txt_file, "")






    @staticmethod
    def estimate_pose_pnp_ransac(
        points_3d_i0,
        pixels_kf1,
        K,
        dist_coeffs=None,
        iterations=1000,
        reprojection_error_px=3.0,
        confidence=0.99, min_inliers=6):
        """
        Estimate the transform from the initial camera frame to KF1.

        points_3d_i0: (N, 3) 3D points in I0's camera frame
        pixels_kf1:   (N, 2) corresponding matched pixels in KF1
        K:            (3, 3) intrinsics for the KF1 image
        dist_coeffs:  distortion coefficients, or None for undistorted images

        Returns:
            R:             (3, 3) rotation
            t:             (3, 1) translation
            inlier_indices: indices into the original input arrays
        """
        points_3d_i0 = np.asarray(points_3d_i0, dtype=np.float64).reshape(-1, 3)
        pixels_kf1 = np.asarray(pixels_kf1, dtype=np.float64).reshape(-1, 2)
        K = np.asarray(K, dtype=np.float64)

        if len(points_3d_i0) != len(pixels_kf1):
            raise ValueError("Each 3D point must have one corresponding KF1 pixel.")
        if K.shape != (3, 3):
            raise ValueError("K must have shape (3, 3).")

        # Remove invalid pairs, while retaining their original indices.
        valid = (
            np.isfinite(points_3d_i0).all(axis=1)
            & np.isfinite(pixels_kf1).all(axis=1)
        )
        original_indices = np.flatnonzero(valid)
        X = np.ascontiguousarray(points_3d_i0[valid])
        uv = np.ascontiguousarray(pixels_kf1[valid])

        if len(X) < 4:
            raise ValueError("PnP-RANSAC needs at least 4 valid 3D–2D pairs.")

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=X,
            imagePoints=uv,
            cameraMatrix=K,
            distCoeffs=dist_coeffs,
            iterationsCount=iterations,
            reprojectionError=reprojection_error_px,
            confidence=confidence,
            flags=cv2.SOLVEPNP_EPNP,
        )

        if not success or inliers is None:
            raise RuntimeError("PnP-RANSAC could not estimate a pose.")

        inlier_indices = original_indices[inliers.reshape(-1)]
        if len(inlier_indices) < min_inliers:
            raise RuntimeError(
                f"Too few PnP inliers: {len(inlier_indices)} "
                f"(minimum required: {min_inliers})."
            )

        R, _ = cv2.Rodrigues(rvec)

        if inlier_indices is None or len(inlier_indices) < 6:
                        raise ValueError(f"PnP failed ! not enough inliers")

        return R, tvec.reshape(3, 1), inlier_indices


    @staticmethod
    def refine_pose_pnp(
        points_3d_i0,
        pixels_kf1,
        K,
        R,
        t,
        inlier_indices,
        dist_coeffs=None):
        """
        Refine an initial PnP pose using its RANSAC inliers.

        Returns:
            R_refined: (3, 3) refined rotation
            t_refined: (3, 1) refined translation
        """
        points_3d_i0 = np.asarray(points_3d_i0, dtype=np.float64).reshape(-1, 3)
        pixels_kf1 = np.asarray(pixels_kf1, dtype=np.float64).reshape(-1, 2)
        K = np.asarray(K, dtype=np.float64)

        idx = np.asarray(inlier_indices, dtype=np.int64).reshape(-1)
        if len(idx) < 4:
            raise ValueError("At least 4 inliers are required for refinement.")
        if np.any(idx < 0) or np.any(idx >= len(points_3d_i0)):
            raise ValueError("An inlier index is outside the input arrays.")

        X = np.ascontiguousarray(points_3d_i0[idx])
        uv = np.ascontiguousarray(pixels_kf1[idx])

        if not (np.isfinite(X).all() and np.isfinite(uv).all()):
            raise ValueError("The selected inlier pairs contain invalid values.")

        rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
        tvec = np.asarray(t, dtype=np.float64).reshape(3, 1).copy()

        rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
            objectPoints=X,
            imagePoints=uv,
            cameraMatrix=K,
            distCoeffs=dist_coeffs,
            rvec=rvec,
            tvec=tvec)

        R_refined, _ = cv2.Rodrigues(rvec_refined)
        return R_refined, tvec_refined.reshape(3, 1)








    @staticmethod
    def rotation_6d_to_matrix(rotation_6d):
        # Split the six values into two 3D vectors.
        first = rotation_6d[..., :3]
        second = rotation_6d[..., 3:]

        # Normalize the first vector.
        axis_1 = F.normalize(first, dim=-1)

        # Remove the part of the second vector parallel to the first.
        # This makes the vectors perpendicular.
        second = second - (axis_1 * second).sum(dim=-1, keepdim=True) * axis_1

        # Normalize the second vector and create the third axis.
        axis_2 = F.normalize(second, dim=-1)
        axis_3 = torch.cross(axis_1, axis_2, dim=-1)

        # Stack the three axes into a valid 3x3 rotation matrix.
        return torch.stack((axis_1, axis_2, axis_3), dim=-2)





    @staticmethod
    def optimize_keyframe_pose(
        initial_pose_np,
        frame,
        gaussians,
        K,
        W,
        H,
        raster_mode="classic",
        iterations=100,
        learning_rate=1e-4,
        ssim_weight=0.2,
        frame_mask=None,
        downscale_levels=None,
        downscale=1,
    ):
        """Refine a camera-to-world pose against a fixed 3DGS using L1 + SSIM."""
        import torch
        import torch.nn.functional as F

        means, quats, scales, opacities, sh = gaussians
        device = means.device
        dtype = torch.float32

        initial_pose = torch.as_tensor(initial_pose_np, device=device, dtype=dtype).detach().clone().unsqueeze(0)

        frame_tensor = torch.as_tensor(frame, device=device)
        frame_tensor = frame_tensor.float() / 255.0 if frame_tensor.dtype == torch.uint8 else frame_tensor.float()
        frame_tensor = frame_tensor.detach().clone().permute(2, 0, 1).unsqueeze(0).contiguous()

        K = torch.as_tensor(K, device=device, dtype=dtype).detach().clone().reshape(1, 3, 3)

        mask_full = None
        if frame_mask is not None:
            mask_full = torch.as_tensor(frame_mask, device=device, dtype=dtype).detach().clone()[None, None]

        means = means.detach().to(device=device, dtype=dtype).clone()
        quats = quats.detach().to(device=device, dtype=dtype).clone()
        scales = scales.detach().to(device=device, dtype=dtype).clone()
        opacities = opacities.detach().to(device=device, dtype=dtype).clone().reshape(-1)
        sh = sh[:, :1].detach().to(device=device, dtype=dtype).clone().contiguous()

        levels = (
            [(1.0 / float(downscale), int(iterations))]
            if downscale_levels is None
            else [(float(scale), int(n_iters)) for scale, n_iters in downscale_levels]
        )
        total_iterations = sum(n_iters for _, n_iters in levels)

        print("\n========== KEYFRAME POSE OPTIMIZATION ==========")
        print(f"Device: {device} | Gaussians: {means.shape[0]} | Image: {W}x{H}")
        print(f"Iterations: {total_iterations} | Learning rate: {learning_rate} | SSIM weight: {ssim_weight}")
        if mask_full is not None:
            print(f"Valid mask pixels: {100.0 * mask_full.mean().item():.2f}%")

        pose_delta = torch.nn.Parameter(torch.zeros(1, 9, device=device, dtype=dtype))
        identity_rotation_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device, dtype=dtype)
        bottom_row = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        optimizer = torch.optim.Adam([pose_delta], lr=learning_rate, weight_decay=1e-6)

        def current_c2w():
            rotation = PosesHandling.rotation_6d_to_matrix(pose_delta[:, 3:] + identity_rotation_6d)
            top = torch.cat([rotation, pose_delta[:, :3].unsqueeze(-1)], dim=2)
            correction = torch.cat([top, bottom_row], dim=1)
            return initial_pose @ correction

        # Same 2D Gaussian window as in optimize_poses_gaussians.
        window_size, sigma = 11, 1.5
        pad = window_size // 2
        coords = torch.arange(window_size, device=device, dtype=dtype)
        kernel_1d = torch.exp(-((coords - pad) ** 2) / (2.0 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        ssim_window = kernel_2d.reshape(1, 1, window_size, window_size).expand(3, 1, window_size, window_size).contiguous()

        def prepare_level(scale):
            height = max(1, int(round(H * scale)))
            width = max(1, int(round(W * scale)))
            target, mask = frame_tensor, mask_full

            if (height, width) != (H, W):
                target = F.interpolate(frame_tensor, size=(height, width), mode="area")
                if mask_full is not None:
                    mask = (F.interpolate(mask_full, size=(height, width), mode="area") >= 0.5).to(dtype)

            K_level = K.clone()
            K_level[:, 0, :] *= width / W
            K_level[:, 1, :] *= height / H

            mask_sum = valid_weights = None
            if mask is not None:
                if not mask.any().item():
                    raise ValueError(f"frame_mask has no valid pixels at scale {scale}.")
                mask_sum = mask.sum()
                valid_weights = F.conv2d(mask, ssim_window[:1], padding=pad).clamp_min(1e-8)

            return {
                "H": height,
                "W": width,
                "K": K_level,
                "target": target,
                "mask": mask,
                "mask_sum": mask_sum,
                "valid_weights": valid_weights,
            }

        def calculate_ssim_loss(predicted, target, level):
            mask = level["mask"]

            if mask is None:
                mu1 = F.conv2d(predicted, ssim_window, padding=pad, groups=3)
                mu2 = F.conv2d(target, ssim_window, padding=pad, groups=3)

                mu1_sq = mu1 * mu1
                mu2_sq = mu2 * mu2
                mu1_mu2 = mu1 * mu2

                sigma1_sq = F.conv2d(predicted * predicted, ssim_window, padding=pad, groups=3) - mu1_sq
                sigma2_sq = F.conv2d(target * target, ssim_window, padding=pad, groups=3) - mu2_sq
                sigma12 = F.conv2d(predicted * target, ssim_window, padding=pad, groups=3) - mu1_mu2
            else:
                # Exclude invalid pixels from the local SSIM statistics.
                valid_weights = level["valid_weights"]
                mu1 = F.conv2d(predicted * mask, ssim_window, padding=pad, groups=3) / valid_weights
                mu2 = F.conv2d(target * mask, ssim_window, padding=pad, groups=3) / valid_weights

                mu1_sq = mu1 * mu1
                mu2_sq = mu2 * mu2
                mu1_mu2 = mu1 * mu2

                sigma1_sq = F.conv2d(predicted * predicted * mask, ssim_window, padding=pad, groups=3) / valid_weights - mu1_sq
                sigma2_sq = F.conv2d(target * target * mask, ssim_window, padding=pad, groups=3) / valid_weights - mu2_sq
                sigma12 = F.conv2d(predicted * target * mask, ssim_window, padding=pad, groups=3) / valid_weights - mu1_mu2

            c1, c2 = 0.01 ** 2, 0.03 ** 2
            ssim_map = (
                ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2))
                / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
            )

            if mask is None:
                return 1.0 - ssim_map.mean()

            return 1.0 - (ssim_map * mask).sum() / (3.0 * level["mask_sum"])

        loss_tensors = []
        global_step = 0

        for level_index, (scale, n_iters) in enumerate(levels):
            level = prepare_level(scale)
            target, mask = level["target"], level["mask"]

            print(f"\nLevel {level_index + 1}/{len(levels)} | Resolution: {level['W']}x{level['H']} | Iterations: {n_iters}")

            for _ in range(n_iters):
                optimizer.zero_grad(set_to_none=True)

                candidate_c2w = current_c2w()
                viewmat = GaussiansHandling.get_gs_viewmat(candidate_c2w[0])
                viewmat = viewmat.to(device=device, dtype=dtype).reshape(1, 4, 4)

                rendered, _, _ = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=sh,
                    viewmats=viewmat,
                    Ks=level["K"],
                    width=level["W"],
                    height=level["H"],
                    sh_degree=0,
                    rasterize_mode=raster_mode,
                    render_mode="RGB",
                )

                rendered = rendered.permute(0, 3, 1, 2).contiguous()

                if mask is None:
                    l1 = F.l1_loss(rendered, target)
                else:
                    l1 = ((rendered - target).abs() * mask).sum() / (3.0 * level["mask_sum"])

                ssim_loss = calculate_ssim_loss(rendered, target, level)
                loss = torch.lerp(l1, ssim_loss, ssim_weight)

                loss.backward()
                optimizer.step()
                loss_tensors.append(loss.detach())
                global_step += 1

                # Read all three loss values together for logging.
                loss_value, l1_value, ssim_value = torch.stack([loss.detach(), l1.detach(), ssim_loss.detach()]).cpu().tolist()
                print(f"Iteration {global_step:03d}/{total_iterations} | Loss: {loss_value:.6f} | L1: {l1_value:.6f} | SSIM loss: {ssim_value:.6f}")

        with torch.no_grad():
            final_c2w = current_c2w()
            refined_pose = final_c2w[0].cpu().numpy()
            loss_history = torch.stack(loss_tensors).cpu().tolist() if loss_tensors else []

            rotation = PosesHandling.rotation_6d_to_matrix(pose_delta[:, 3:] + identity_rotation_6d)[0]
            rotation_cos = ((rotation.trace() - 1.0) / 2.0).clamp(-1.0, 1.0)
            rotation_change = torch.rad2deg(torch.acos(rotation_cos)).item()
            translation_change = (final_c2w[0, :3, 3] - initial_pose[0, :3, 3]).norm().item()

        print("\n========== POSE OPTIMIZATION FINISHED ==========")
        print(f"Pose change: translation={translation_change:.6f} scene units | rotation={rotation_change:.4f} degrees")
        print(f"Iterations completed: {len(loss_history)}")

        return refined_pose, loss_history

    






