import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement







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
        voxel_size=None) :

        """
        Align source_points toward target_points using rigid point-to-point ICP.

        Parameters
        ----------
        o3d_source_points : np.ndarray
        o3d_target_points : np.ndarray
        max_correspondence_distance : float
            Maximum distance allowed between corresponding points.
            Uses the same unit as the point clouds.
        max_iterations : int
            Maximum number of ICP iterations.
        voxel_size : float or None
            Downsampling voxel size. If None, no downsampling is applied.

        Returns
        -------
        R : np.ndarray
            Estimated rotation, shape (3, 3).
        t : np.ndarray
            Estimated translation, shape (3,).
        T : np.ndarray
            Complete rigid transformation, shape (4, 4).
        """

        # optio downsampling
        if voxel_size is not None:
            source_icp = source_cloud.voxel_down_sample(voxel_size)
            target_icp = target_cloud.voxel_down_sample(voxel_size)
        else:
            source_icp = source_cloud
            target_icp = target_cloud

        # Clouds are already approximately aligned, so start from identity
        T_initial = np.eye(4)

        result = o3d.pipelines.registration.registration_icp(
            source_icp,
            target_icp,
            max_correspondence_distance,
            T_initial,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
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





    





    




