import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from collections import deque
import time
import cv2
from utils.lin_algeb import LinAlgeb
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from numba import njit





@njit(cache=True)
def _grow_regions_numba(normals, valid, cos_thr, min_group_size):
    """Same BFS region growing as before, compiled. Returns label_map (H, W) int32."""
    h, w = valid.shape
    label_map = np.full((h, w), -1, np.int32)
    visited = np.zeros((h, w), np.bool_)
    queue = np.empty(h * w, np.int32)  # flat pixel indices, reused for each group
    dys = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
    dxs = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    n_groups = 0
    for sy in range(h):
        for sx in range(w):
            if (not valid[sy, sx]) or visited[sy, sx]:
                continue
            visited[sy, sx] = True
            queue[0] = sy * w + sx
            head = 0
            tail = 1
            sum_x = normals[sy, sx, 0]
            sum_y = normals[sy, sx, 1]
            sum_z = normals[sy, sx, 2]
            while head < tail:
                p = queue[head]
                head += 1
                y = p // w
                x = p - y * w
                gn = np.sqrt(sum_x * sum_x + sum_y * sum_y + sum_z * sum_z)
                if gn < 1e-12:
                    gn = 1e-12
                gx = sum_x / gn
                gy = sum_y / gn
                gz = sum_z / gn
                cx = normals[y, x, 0]
                cy = normals[y, x, 1]
                cz = normals[y, x, 2]
                for k in range(8):
                    ny = y + dys[k]
                    nx = x + dxs[k]
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if (not valid[ny, nx]) or visited[ny, nx]:
                        continue
                    nnx = normals[ny, nx, 0]
                    nny = normals[ny, nx, 1]
                    nnz = normals[ny, nx, 2]
                    local_sim = cx * nnx + cy * nny + cz * nnz
                    if local_sim < cos_thr:
                        continue
                    group_sim = gx * nnx + gy * nny + gz * nnz
                    if group_sim < cos_thr:
                        continue
                    visited[ny, nx] = True
                    queue[tail] = ny * w + nx
                    tail += 1
                    sum_x += nnx
                    sum_y += nny
                    sum_z += nnz
            if tail >= min_group_size:
                for q in range(tail):
                    pp = queue[q]
                    label_map[pp // w, pp % w] = n_groups
                n_groups += 1
    return label_map




class MeshHandling :


    
    def __init__(self, CAM_W, CAM_H, fx = None, fy = None, cx = None, cy = None):
        
        self.CAM_W = CAM_W
        self.CAM_H = CAM_H

        if fx is None : 
            self.FX = self.FY = 0.8 * max(CAM_W, CAM_H)
        else :
            self.FX = fx
            self.FY = fy

        if cx is None :
            self.CX = CAM_W / 2.0
            self.CY = CAM_H / 2.0
        else :    
            self.CX = cx
            self.CY = cy
        
        self.intrins_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=self.CAM_W,
            height=self.CAM_H,
            fx=self.FX,
            fy=self.FY,
            cx=self.CX,
            cy=self.CY)
          



    @staticmethod
    def load_o3dpcd(path) :
        pcd = o3d.io.read_point_cloud(path)
        return pcd



    @staticmethod
    def save_o3dpcd(pcd, path):
        o3d.io.write_point_cloud(path, pcd)




    @staticmethod 
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





    @staticmethod
    def turn_points_to_spheres(points, radius=0.1, random=True, color=[0, 0, 1]):

        n = len(points)
        if random:
            colors = plt.cm.hsv(np.linspace(0, 1, n))[:, :3]
        else:
            colors = [color for _ in range(n)]

        combined_spheres = o3d.geometry.TriangleMesh()

        for point, c in zip(points, colors):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.translate(point)
            sphere.paint_uniform_color(c)
            combined_spheres += sphere

        return combined_spheres



    @staticmethod
    def turn_points_to_o3d(points, colors=None):
        points = np.array(points, dtype=np.float64)
        
        if points.ndim == 1:
            points = points.reshape(1, 3)   # (3,) → (1, 3)
        elif points.ndim > 2:
            points = points.reshape(-1, 3)  # (H, W, 3) → (N, 3)

        if colors is None:
            colors = np.tile([1.0, 0.0, 0.0], (len(points), 1))
        colors = np.array(colors, dtype=np.float64)
        if colors.ndim == 1 and len(colors) == 3:
            colors = np.tile(colors, (len(points), 1))  # broadcast single color to all points
        if colors.ndim > 2:
            colors = colors.reshape(-1, 3)

        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(points)
        o3d_points.colors = o3d.utility.Vector3dVector(colors)
        return o3d_points




        
    def render_mesh_pic(self, scene_compos, extrins) :

        # defining extrins and T_cw_final params 
        extrins = np.linalg.inv(extrins)

        # making our pincamparams objct
        cam_params = o3d.camera.PinholeCameraParameters()
        cam_params.extrinsic = extrins
        cam_params.intrinsic = self.intrins_o3d
    
        # ---------- Create visualizer ----------
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.CAM_W, height=self.CAM_H, visible=False)
        opt = vis.get_render_option()
        opt.background_color = np.array([0.0, 0.0, 0.0])  # black
        for geom in scene_compos:
            vis.add_geometry(geom)
    

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






    def render_mesh_pic_withnoshad(self, scene_compos, camera_to_world):

        # Open3D expects world-to-camera extrinsic
        world_to_camera = np.linalg.inv(camera_to_world)

        cam_params = o3d.camera.PinholeCameraParameters()
        cam_params.extrinsic = world_to_camera
        cam_params.intrinsic = self.intrins_o3d

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="Mesh render",
            width=self.CAM_W,
            height=self.CAM_H,
            visible=False,
        )

        opt = vis.get_render_option()
        opt.background_color = np.array([0.0, 0.0, 0.0])

        # Equivalent to MeshLab's "Shading: None"
        opt.light_on = False

        # Explicitly use the mesh's vertex colors
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Color

        opt.mesh_show_wireframe = False
        opt.mesh_show_back_face = True

        for geom in scene_compos:
            if isinstance(geom, o3d.geometry.TriangleMesh):
                print(
                    f"Mesh: vertices={len(geom.vertices)}, "
                    f"triangles={len(geom.triangles)}, "
                    f"has colors={geom.has_vertex_colors()}, "
                    f"has normals={geom.has_vertex_normals()}"
                )

            vis.add_geometry(geom, reset_bounding_box=True)

        ctr = vis.get_view_control()
        success = ctr.convert_from_pinhole_camera_parameters(
            cam_params,
            allow_arbitrary=True,
        )

        if not success:
            print("WARNING: camera parameters were not applied correctly")

        vis.poll_events()
        vis.update_renderer()

        image = np.asarray(
            vis.capture_screen_float_buffer(do_render=True)
        )

        depth = np.asarray(
            vis.capture_depth_float_buffer(do_render=True)
        )

        vis.destroy_window()

        return image, depth




    def visualize_scene(scene_compos):

        def move_forward(vis):
            vis.get_view_control().camera_local_translate(0.0, 0.0, -0.5)
            return False

        def move_backward(vis):
            vis.get_view_control().camera_local_translate(0.0, 0.0, 0.5)
            return False

        def move_left(vis):
            vis.get_view_control().camera_local_translate(0.0, -0.5, 0.0)
            return False

        def move_right(vis):
            vis.get_view_control().camera_local_translate(0.0, 0.5, 0.0)
            return False

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("Scene", width=800, height=800)

        for geom in scene_compos:
            vis.add_geometry(geom)

        vis.get_render_option().mesh_show_back_face = True

        # Register controls (WASD)
        vis.register_key_callback(ord("W"), move_forward)
        vis.register_key_callback(ord("S"), move_backward)
        vis.register_key_callback(ord("A"), move_left)
        vis.register_key_callback(ord("D"), move_right)

        vis.run()
        vis.destroy_window()





    def get_cam_pose_from_mesh_view(self, scene_compos):

        def move_top(vis):
            ctr = vis.get_view_control()
            ctr.camera_local_translate(0.0, 0.0, -0.1)
            return False
        

        def move_bot(vis):
            ctr = vis.get_view_control()
            ctr.camera_local_translate(0.0, 0.0, 0.1)
            return False


        def move_left(vis):
            ctr = vis.get_view_control()
            ctr.camera_local_translate(0.0, -0.1, 0.0)
            return False


        def move_right(vis):
            ctr = vis.get_view_control()
            ctr.camera_local_translate(0.0, 0.1, 0.0)
            return False



        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Choose ur pose",
                        width=self.CAM_W,
                        height=self.CAM_H)

        for geom in scene_compos:
            vis.add_geometry(geom)

        vis.get_render_option().mesh_show_back_face = True

        # get viewcontrol -> to pinhole -> change intrinsics to same robot cam intrinsics
        ctr = vis.get_view_control()
        cam_params = ctr.convert_to_pinhole_camera_parameters()
        cam_params.intrinsic = self.intrins_o3d  
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
    



    @staticmethod
    def get_pose_axis(pose, size = 1) :
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
        axis.transform(pose)
        return axis







    @staticmethod
    def align_points(points1, points2):

        # Convert to numpy
        points1 = np.asarray(points1, dtype=np.float64)
        points2 = np.asarray(points2, dtype=np.float64)

        # flatten to (H*W,3) if H,W,3)
        if points1.ndim > 2:
            points1 = points1.reshape(-1, 3)
        if points2.ndim > 2:
            points2 = points2.reshape(-1, 3)


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
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # scale
        scale = np.sum(S) / np.sum(X1**2)

        # translation
        t = c2 - scale * R @ c1

        # transform
        points1_aligned = (scale * (R @ points1.T)).T + t

        return points1_aligned, scale, R, t







    @staticmethod
    def align_points_only_z(points1, points2):
        """
        Estimate only the Z translation aligning corresponding 3D points.
        Rotation, scale, X translation and Y translation remain fixed.
        """


        points1 = np.asarray(points1, dtype=np.float64).reshape(-1, 3)
        points2 = np.asarray(points2, dtype=np.float64).reshape(-1, 3)

        if points1.shape != points2.shape:
            raise ValueError("source_points and target_points must have the same shape")

        valid = np.isfinite(points1).all(axis=1) & np.isfinite(points2).all(axis=1)

        if not valid.any():
            raise ValueError("No valid 3D correspondences")

        z_differences = points2[valid, 2] - points1[valid, 2]

        # Robust against incorrect 3D correspondences
        delta_z = np.median(z_differences)

        return delta_z






    @staticmethod
    def denoise_pcd(pcd):

        # estimate average spacing
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        print("Avg NN distance:", avg_dist)

        # voxel size slightly larger than spacing
        voxel_size = avg_dist * 1.5   # tune this
        print("Voxel size:", voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        return pcd_down





    @staticmethod
    def get_o3d_box_frm_corners(corners):
        # Define the 12 edges of a box
        lines = [
            [0,1], [1,3], [3,2], [2,0],  # bottom face
            [4,5], [5,7], [7,6], [6,4],  # top face
            [0,4], [1,5], [2,6], [3,7]   # vertical edges
        ]

        colors = [[0, 1, 0] for _ in lines]  # red lines

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set
        




    @staticmethod
    def get_box_from_two_points(p1, p2, margin=1.0):
        p1 = np.array(p1)
        p2 = np.array(p2)

        # Get min/max corners
        min_corner = np.minimum(p1, p2) - margin
        max_corner = np.maximum(p1, p2) + margin

        # Extract values
        x_min, y_min, z_min = min_corner
        x_max, y_max, z_max = max_corner

        # 8 corners of the box
        corners = np.array([
            [x_min, y_min, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_min],
            [x_min, y_max, z_max],
            [x_max, y_min, z_min],
            [x_max, y_min, z_max],
            [x_max, y_max, z_min],
            [x_max, y_max, z_max],
        ])

        return corners    
    





    @staticmethod
    def get_cube_from_center(center, margin=1.0):
        center = np.array(center, dtype=float)

        # Compute min/max corners
        min_corner = center - margin
        max_corner = center + margin

        # Extract values
        x_min, y_min, z_min = min_corner
        x_max, y_max, z_max = max_corner

        # 8 corners of the cube
        corners = np.array([
            [x_min, y_min, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_min],
            [x_min, y_max, z_max],
            [x_max, y_min, z_min],
            [x_max, y_min, z_max],
            [x_max, y_max, z_min],
            [x_max, y_max, z_max],
        ], dtype=float)

        return corners


    @staticmethod
    def get_voxel_centers_from_box_corners(corners, voxel_size):
        corners = np.asarray(corners, dtype=float).reshape(-1, 3)

        # Extract min/max from corners
        min_corner = corners.min(axis=0)
        max_corner = corners.max(axis=0)

        # Compute box size
        dims = max_corner - min_corner

        # Number of voxels (extend → ceil)
        nx, ny, nz = np.ceil(dims / voxel_size).astype(int)

        # Generate voxel centers
        xs = min_corner[0] + (np.arange(nx) + 0.5) * voxel_size
        ys = min_corner[1] + (np.arange(ny) + 0.5) * voxel_size
        zs = min_corner[2] + (np.arange(nz) + 0.5) * voxel_size

        # Create 3D grid
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        centers = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        return centers
    


    @staticmethod
    def get_occupied_voxel(voxel_centers, moge_points, voxel_size):
        voxel_centers = np.asarray(voxel_centers, dtype=float)
        moge_points = np.asarray(moge_points, dtype=float)

        # Recover grid bounds
        min_corner = voxel_centers.min(axis=0) - voxel_size / 2
        max_corner = voxel_centers.max(axis=0) + voxel_size / 2

        # FILTER points inside the box
        mask = np.all((moge_points >= min_corner) & (moge_points <= max_corner), axis=1)
        moge_points = moge_points[mask]

        # Map to indices
        indices = np.floor((moge_points - min_corner) / voxel_size).astype(int)

        unique_indices = np.unique(indices, axis=0)

        # Back to centers
        centers = min_corner + (unique_indices + 0.5) * voxel_size

        return centers








    @staticmethod
    def align_clouds_icp(
        source_cloud,
        target_cloud,
        max_correspondence_distance=0.05,
        max_iterations=50,
        voxel_size=None):

        """
        Align source_points toward target_points using rigid point-to-plane ICP.
        """

        # Optional downsampling
        if voxel_size is not None:
            source_icp = source_cloud.voxel_down_sample(voxel_size)
            target_icp = target_cloud.voxel_down_sample(voxel_size)
        else:
            source_icp = source_cloud
            target_icp = target_cloud

        # 1. CRITICAL: Estimate normals for Point-to-Plane
        # Set search radius proportional to voxel size (or ~0.1m if no voxelization)
        normal_radius = voxel_size * 3 if voxel_size else 0.1
        
        source_icp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
        
        target_icp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
        

        # Clouds are already approximately aligned, so start from identity
        T_initial = np.eye(4)

        result = o3d.pipelines.registration.registration_icp(
            source_icp,
            target_icp,
            max_correspondence_distance,
            T_initial,
            # 2. SWITCHED FROM PointToPoint TO PointToPlane
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=max_iterations))
        
        T = result.transformation
        R = T[:3, :3]
        t = T[:3, 3]

        return R, t













        
    @staticmethod
    def get_focal_point(cloud_in_cam_frame, mask, area_size=10):
        """
        Return the valid 3D point whose pixel is closest to the image centre.

        Parameters
        ----------
        cloud_in_cam_frame : np.ndarray, shape (H, W, 3)
            Organized point cloud expressed in the camera frame.

        mask : np.ndarray, shape (H, W), dtype bool
            Invalid-point mask:
            True  -> depth edge or low-confidence point.
            False -> valid point.

        area_size : int
            Size of the central search area in pixels [square].

        Returns
        -------
        focal_point : np.ndarray, shape (3,)
            Selected 3D focal point in the camera frame.

        pixel_coordinates : tuple[int, int]
            Selected pixel coordinates in (u, v) format.
        """

        cloud = np.asarray(cloud_in_cam_frame, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)

        if cloud.ndim != 3 or cloud.shape[2] != 3:
            raise ValueError("cloud_in_cam_frame must have shape (H, W, 3).")

        if mask.shape != cloud.shape[:2]:
            raise ValueError("mask must have the same (H, W) shape as the point cloud.")

        height, width = cloud.shape[:2]
        area_height = area_width =  area_size

        if area_height <= 0 or area_width <= 0:
            raise ValueError("area_size values must be positive.")

        area_height = min(area_height, height)
        area_width = min(area_width, width)

        center_v = height // 2
        center_u = width // 2

        v_min = max(0, center_v - area_height // 2)
        v_max = min(height, v_min + area_height)

        u_min = max(0, center_u - area_width // 2)
        u_max = min(width, u_min + area_width)

        # Readjust the beginning when the area touches an image boundary.
        v_min = max(0, v_max - area_height)
        u_min = max(0, u_max - area_width)

        central_cloud = cloud[v_min:v_max, u_min:u_max]
        central_mask = mask[v_min:v_max, u_min:u_max]

        finite = np.all(np.isfinite(central_cloud), axis=2)

        # Assumes that valid points are in front of the camera: Z > 0.
        positive_depth = central_cloud[..., 2] > 0
        valid = central_mask & finite & positive_depth
        valid_v, valid_u = np.nonzero(valid)

        if len(valid_v) == 0:
            raise ValueError("No valid 3D point was found inside the selected central area.")

        # Convert local crop coordinates to complete-image coordinates.
        image_v = valid_v + v_min
        image_u = valid_u + u_min

        # Squared pixel distance from the image centre.
        distances_squared = ((image_u - center_u) ** 2 + (image_v - center_v) ** 2)
        closest_index = np.argmin(distances_squared)

        selected_v = image_v[closest_index]
        selected_u = image_u[closest_index]
        focal_point = cloud[selected_v, selected_u].copy()

        return focal_point











    @staticmethod
    def get_corresp_3d_point_matches(matches_img1, matches_img2, moge_mask1, moge_mask2, all_moge_points1, all_moge_points2) :
        """
        return matched_points3d1, matched_points3d2 np : (H, W, 3)
        """
        # get the indices(x,y) of the xfeat matched 2d points on img1 and img2
        x1 = matches_img1[:, 0]
        y1 = matches_img1[:, 1]
        x2 = matches_img2[:, 0]
        y2 = matches_img2[:, 1]

        # return the moge mask for the xfeat points (True for the valid xfeat points) => 1D [T, F, T..] lngth of the xfeat points
        valid1 = moge_mask1[y1, x1]
        valid2 = moge_mask2[y2, x2]
        valid = valid1 & valid2

        H, W = moge_mask1.shape
        all_moge_points1 = all_moge_points1.reshape(H, W, 3)
        all_moge_points2 = all_moge_points2.reshape(H, W, 3)

        # Corresponding 3D points and colors of the valid 2d xfeat matches
        matched_points3d1 = all_moge_points1[y1, x1][valid]
        matched_points3d2 = all_moge_points2[y2, x2][valid]
        return matched_points3d1, matched_points3d2 



    @staticmethod
    def get_corresp_3d_points(matches_img, moge_mask, all_moge_points) :
        """
        return matched_points3d (H, W, 3)
        """
        # get the indices(x,y) of the xfeat matched 2d points on img1
        x = matches_img[:, 0]
        y = matches_img[:, 1]

        # return the moge mask for the xfeat points (True for the valid xfeat points) => 1D [T, F, T..] lngth of the xfeat points
        valid = moge_mask[y, x]

        H, W = moge_mask.shape
        all_moge_points = all_moge_points.reshape(H, W, 3)

        # Corresponding 3D points and colors of the valid 2d xfeat matches
        matched_points3d = all_moge_points[y, x][valid]
        
        return matched_points3d 












    def render_planar_structure_imgs(self, points, organized_subsurfaces, cam_pose = None, color_variation=0.10, visualize=False):
        """
        Render the planar surfaces and their sub-surfaces.

        Main surfaces receive different colors. Sub-surfaces preserve their
        parent surface color with a slight brightness variation.

        Returns
        -------
        surfaces_img : np.ndarray
            Rendered main planar surfaces.

        subsurfaces_img : np.ndarray
            Rendered planar sub-surfaces.
        """

        points_flat = np.asarray(points).reshape(-1, 3)
        rng = np.random.default_rng(0)

        if cam_pose is None:
            cam_pose = LinAlgeb.get_eye_matrix(4)

        surface_points, surface_colors = [], []
        subsurface_points, subsurface_colors = [], []

        for surface in organized_subsurfaces:
            # Give every main surface a different base color
            base_color = rng.uniform(0.2, 0.9, size=3)
            surface_indices = surface["indices"]

            surface_points.append(points_flat[surface_indices])
            surface_colors.append(np.tile(base_color, (len(surface_indices), 1)))

            # Give sub-surfaces slight variations of their parent color
            for subsurface in surface["subsurfaces"]:
                subsurface_indices = subsurface["indices"]
                subsurface_id = subsurface["subsurface_id"]

                variation = color_variation * np.sin((subsurface_id + 1) * 2.4)
                color = np.clip(base_color * (1.0 + variation), 0.0, 1.0)

                subsurface_points.append(points_flat[subsurface_indices])
                subsurface_colors.append(np.tile(color, (len(subsurface_indices), 1)))

        # Create the main-surface point cloud
        surface_points = np.concatenate(surface_points, axis=0)
        surface_colors = np.concatenate(surface_colors, axis=0)
        surfaces_o3d = self.turn_points_to_o3d(surface_points, surface_colors)

        # Create the sub-surface point cloud
        subsurface_points = np.concatenate(subsurface_points, axis=0)
        subsurface_colors = np.concatenate(subsurface_colors, axis=0)
        subsurfaces_o3d = self.turn_points_to_o3d(subsurface_points, subsurface_colors)

        # Render both point clouds from the original MoGe camera
        surfaces_img, _ = self.render_mesh_pic([surfaces_o3d], cam_pose)
        subsurfaces_img, _ = self.render_mesh_pic([subsurfaces_o3d], cam_pose)

        if(visualize) :
            MeshHandling.visualize_scene([surfaces_o3d])

        if(visualize) :
            MeshHandling.visualize_scene([subsurfaces_o3d])

        return surfaces_img, subsurfaces_img
    









    @staticmethod
    def get_subsurface_centers_and_colors(points, colors, organized_subsurfaces):
        """
        Calculate one XYZ center and one average RGB color for every sub-surface.

        The rows correspond:
            centers_xyz[i] belongs to centers_rgb[i]
        """

        points_flat = np.asarray(points).reshape(-1, 3)
        colors_flat = np.asarray(colors).reshape(-1, 3)

        centers_xyz = []
        centers_rgb = []

        for surface in organized_subsurfaces:
            for subsurface in surface["subsurfaces"]:
                indices = subsurface["indices"]

                # Center and average color of the sub-surface
                center_xyz = np.mean(points_flat[indices], axis=0)
                center_rgb = np.mean(colors_flat[indices], axis=0)

                centers_xyz.append(center_xyz)
                centers_rgb.append(center_rgb)

        return np.asarray(centers_xyz), np.asarray(centers_rgb)
    
















    @staticmethod
    def group_points_by_normals(points, normals, valid_mask, theta_max_deg=12.0, min_group_size=40):
        """
        Group image-organized 3D points into connected regions with similar normals.

        Parameters
        ----------
        all points : np.ndarray, shape (H, W, 3)

        normals : np.ndarray, shape (H, W, 3)

        valid_mask : np.ndarray, shape (H, W)

        theta_max_deg : float
            Maximum accepted normal-angle difference.

        min_group_size : int
            Groups smaller than this are discarded.

        Returns
        -------
        normal_groups : list[np.ndarray]
            a list of sub-lists (each sub-list contains flattened_indices of points belonging to same surface)

        label_map : np.ndarray, shape (H, W)
            H,W where the corresp img indexe gives the group_id it belongs to
        """

        points = np.asarray(points)
        normals = np.asarray(normals, dtype=np.float64)
        valid = np.asarray(valid_mask, dtype=bool).copy()
        h, w = valid.shape

        # Remove invalid 3D points and normals
        normal_norms = np.linalg.norm(normals, axis=2)
        valid &= np.isfinite(points).all(axis=2)
        valid &= np.isfinite(normals).all(axis=2)
        valid &= normal_norms > 1e-8

        # Ensure unit normals (and zero out the bad ones so no NaNs go in the compiled function)
        normals = normals / np.maximum(normal_norms[..., None], 1e-8)
        normals = np.ascontiguousarray(np.where(valid[..., None], normals, 0.0))
        cosine_threshold = float(np.cos(np.deg2rad(theta_max_deg)))

        # The fast compiled (numba) region growing does the old pixel by pixel BFS:
        # for each valid pixel we check its 8 neighbours, their normals' cosine sim with the pixel normal,
        # and with the group's average normal (prevents gradual normal drift across curved surfaces).
        # It gives every pixel a group id (-1 = no group)
        label_map = _grow_regions_numba(normals, np.ascontiguousarray(valid), cosine_threshold, int(min_group_size))

        # Turn the label map into one list of flattened indices per group
        # flat_index = y * w + x  (a single unique number for each pixel), for example with width = 4 :
        # (0,0) -> 0 | (0,1) -> 1 | (1,0) -> 4 | (1,1) -> 5
        # and it can be converted back by : y = flat_index // w , x = flat_index % w
        flat_labels = label_map.ravel()
        idx = np.flatnonzero(flat_labels >= 0)
        if idx.size == 0:
            return [], label_map

        lab = flat_labels[idx]
        order = np.argsort(lab, kind="stable")      # sort the pixels by group id
        idx, lab = idx[order], lab[order]
        cuts = np.cumsum(np.bincount(lab))[:-1]     # where each group ends
        groups = np.split(idx, cuts)                # first group = 0, second = 1, third = 2 ...

        return groups, label_map









    @staticmethod
    def organize_planar_subsurfaces(points, label_map, max_side_pixels=50, min_subsurface_pixels=10):
        """
        -takes all 3d points + the label_map (H,W each value is the group id it belogs to)
        -find the 2d bbx of that planar surface divide it to 2d tiles of max_side_pixels(square)
        -organize these points to the ones belonging to same tile on same planar srface => sub-surface points
        -discard a subsurface if it's too small (less than min_subsurface_pixels)

        Returns
        -------
        sub_surfaces : list[dict]
            --list of dict where each dict got a :
            "surface_id" (the mother one), its "bounding_box", "pixel_yx" (a np list of (y,x) indices of pixels of that surface), then the "subsurfaces"
            --surface["subsurfaces"] got a "subsurface_id", "indices" (the flattened indices), "pixel_yx" (np list of (y,x) indices of pxls of that subsrfce), "tiles" (its tile bbx)
        """

        points = np.asarray(points)
        label_map = np.asarray(label_map)
        h, w = label_map.shape

        if points.shape[:2] != (h, w):
            raise ValueError("points and label_map must have the same height and width")

        S = int(max_side_pixels)

        # all the pixels that belong to some surface (raster order)
        ys, xs = np.nonzero(label_map >= 0)
        if ys.size == 0:
            return []
        ys = ys.astype(np.int64)
        xs = xs.astype(np.int64)
        flat = ys * w + xs
        lab = label_map[ys, xs]

        # sort the pixels by surface, and get each surface's bounding box
        uniq, inv = np.unique(lab, return_inverse=True)
        n_surf = len(uniq)
        order = np.argsort(inv, kind="stable")
        counts = np.bincount(inv, minlength=n_surf)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))

        ys_s, xs_s = ys[order], xs[order]
        min_y = np.minimum.reduceat(ys_s, starts)
        max_y = np.maximum.reduceat(ys_s, starts)
        min_x = np.minimum.reduceat(xs_s, starts)
        max_x = np.maximum.reduceat(xs_s, starts)

        # cut each surface's box into square tiles of max_side_pixels, and find the tile of every pixel
        n_tx = (max_x - min_x) // S + 1
        n_ty = (max_y - min_y) // S + 1
        ty = (ys - min_y[inv]) // S
        tx = (xs - min_x[inv]) // S
        tile_id = ty * n_tx[inv] + tx
        tiles_per_surface = int((n_tx * n_ty).max())

        # one key per pixel = (which surface, which tile)
        key = np.full((h, w), -1, dtype=np.int64)
        key[ys, xs] = inv.astype(np.int64) * tiles_per_surface + tile_id

        # A tile might contain several disconnected pieces,
        # so we link neighbor pixels (left-right and up-down) only if they have the same key
        node = np.arange(h * w, dtype=np.int64).reshape(h, w)
        rows, cols = [], []

        a, b = key[:, :-1], key[:, 1:]
        m = (a >= 0) & (a == b)
        rows.append(node[:, :-1][m]); cols.append(node[:, 1:][m])

        a, b = key[:-1, :], key[1:, :]
        m = (a >= 0) & (a == b)
        rows.append(node[:-1, :][m]); cols.append(node[1:, :][m])

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        graph = coo_matrix((np.ones(rows.size, dtype=np.uint8), (rows, cols)), shape=(h * w, h * w))

        # connected pieces = the sub-surfaces
        _, comp = connected_components(graph, directed=False)

        # sort the pixels by piece
        comp_v = comp[flat]
        o2 = np.argsort(comp_v, kind="stable")
        comp_sorted = comp_v[o2]
        flat_sorted = flat[o2]
        ys2, xs2 = ys[o2], xs[o2]
        inv2 = inv[o2]
        tile2 = tile_id[o2]

        # find where each piece starts and ends
        c_starts = np.flatnonzero(np.concatenate(([True], comp_sorted[1:] != comp_sorted[:-1])))
        c_ends = np.concatenate((c_starts[1:], [comp_sorted.size]))
        c_sizes = c_ends - c_starts

        # discard a subsurface if it's too small
        keep = c_sizes >= min_subsurface_pixels
        c_starts, c_ends = c_starts[keep], c_ends[keep]

        # order the pieces by surface, then tile, then first pixel
        c_surf = inv2[c_starts]
        c_tile = tile2[c_starts]
        c_first = flat_sorted[c_starts]
        c_order = np.lexsort((c_first, c_tile, c_surf))
        c_starts, c_ends, c_surf, c_tile = c_starts[c_order], c_ends[c_order], c_surf[c_order], c_tile[c_order]

        # build the surfaces list (the mother surfaces)
        surfaces = []
        for s in range(n_surf):
            sl = slice(starts[s], starts[s] + counts[s])
            sy, sx = ys_s[sl], xs_s[sl]
            surfaces.append({
                "surface_id": int(uniq[s]),
                "indices": sy * w + sx,
                "pixel_yx": np.column_stack((sy, sx)),
                "bounding_box": (int(min_y[s]), int(min_x[s]), int(max_y[s]), int(max_x[s])),
                "subsurfaces": []
            })

        # put each piece inside its mother surface
        for st, en, s, t in zip(c_starts, c_ends, c_surf, c_tile):
            s = int(s)
            t = int(t)
            ty_i, tx_i = divmod(t, int(n_tx[s]))
            y0 = int(min_y[s]) + ty_i * S
            x0 = int(min_x[s]) + tx_i * S
            y1 = min(y0 + S, int(max_y[s]) + 1)
            x1 = min(x0 + S, int(max_x[s]) + 1)

            py, px = ys2[st:en], xs2[st:en]
            subs = surfaces[s]["subsurfaces"]
            subs.append({
                "subsurface_id": len(subs),
                "indices": py * w + px,
                "pixel_yx": np.column_stack((py, px)),
                "tile": (y0, x0, y1, x1)
            })

        return surfaces












    @staticmethod
    def get_valid_match_indices(coords_2d, mask):
        """
        takes a 2d coords list and a 2d boolean mask of the img and filters and returns just the indices of the valid 2d coords list
        """
        coords = np.asarray(coords_2d, dtype=float).reshape(-1, 2)
        mask = np.asarray(mask, dtype=bool)

        kernel = np.ones((3, 3), dtype=np.uint8)
        safe_mask = cv2.erode(
            mask.astype(np.uint8),
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)

        finite = np.isfinite(coords).all(axis=1)
        xy = np.zeros(coords.shape, dtype=int)
        xy[finite] = np.rint(coords[finite]).astype(int)
        x, y = xy[:, 0], xy[:, 1]

        h, w = mask.shape
        inside = finite & (x >= 0) & (x < w) & (y >= 0) & (y < h)

        valid = np.zeros(len(coords), dtype=bool)
        valid[inside] = safe_mask[y[inside], x[inside]]

        return np.flatnonzero(valid).tolist()
    







    @staticmethod
    def estimate_and_apply_z_offset(cloud1, cloud2, cloud2_full, pose2_ofthecamcloud):
        """
        -Takes cloud1 cloud2 corresp 3d points in world frame
        -get the delta_Z to apply to cloud2 to best fit cloud1
        -Return delta_z_in in camframe of img2 and return also the full updated cloud2 in world frame
        """
        
        cloud1 = np.asarray(cloud1, dtype=np.float64).reshape(-1, 3)
        cloud2 = np.asarray(cloud2, dtype=np.float64).reshape(-1, 3)
        cloud2_full = np.asarray(cloud2_full, dtype=np.float64)
        full_shape = cloud2_full.shape
        cloud2_full_flat = cloud2_full.reshape(-1, 3).copy()
        pose2_ofthecamcloud = np.asarray(pose2_ofthecamcloud, dtype=np.float64)
        T_cw_2 = np.linalg.inv(pose2_ofthecamcloud)

        if len(cloud1) != len(cloud2):
            raise ValueError("cloud1 and cloud2 must contain corresponding 3D points.")

        # Compare the matched world points in KF2's camera frame.
        cloud1_cam2 = LinAlgeb.transform_points_to_world(cloud1, T_cw_2)
        cloud2_cam2 = LinAlgeb.transform_points_to_world(cloud2, T_cw_2)
        z1 = cloud1_cam2[:, 2]
        z2 = cloud2_cam2[:, 2]
        valid_matches = np.isfinite(z1) & np.isfinite(z2)
        if np.count_nonzero(valid_matches) < 8:
            raise ValueError(f"Not enough valid 3D matches for Z correction: {np.count_nonzero(valid_matches)}")
        delta_z = float(np.median(z1[valid_matches] - z2[valid_matches]))

        # Apply the offset to the full cloud in KF2's camera frame, along each camera ray.
        valid_cloud = np.isfinite(cloud2_full_flat).all(axis=1)     
        valid_indices = np.flatnonzero(valid_cloud)
        cloud2_full_cam2 = LinAlgeb.transform_points_to_world(cloud2_full_flat[valid_indices], T_cw_2)
        z_old = cloud2_full_cam2[:, 2]
        z_new = z_old + delta_z
        can_correct = np.isfinite(z_new) & (np.abs(z_old) > 1e-6) & (z_old * z_new > 0)
        ray_scale = np.ones_like(z_old)
        ray_scale[can_correct] = z_new[can_correct] / z_old[can_correct]
        cloud2_full_cam2[can_correct] *= ray_scale[can_correct, None]

        # Return the corrected dense cloud in world coordinates.
        corrected_world = LinAlgeb.transform_points_to_world(cloud2_full_cam2, pose2_ofthecamcloud)
        cloud2_after_offset = cloud2_full_flat.copy()
        cloud2_after_offset[valid_indices] = corrected_world
        return cloud2_after_offset.reshape(full_shape), delta_z
    












