import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement







class MeshHandling :


    
    def __init__(self, CAM_W, CAM_H):
        
        self.CAM_W = CAM_W
        self.CAM_H = CAM_H
        self.CX = CAM_W / 2.0
        self.CY = CAM_H / 2.0
        self.FX = self.FY = 0.8 * max(CAM_W, CAM_H)

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
    def turn_points_to_spheres(points) :
        
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


        
    def render_mesh_pic(self, scene_compos, extrins):

        # defining extrins and T_cw_final params 
        extrins = np.linalg.inv(extrins)

        # making our pincamparams objct
        cam_params = o3d.camera.PinholeCameraParameters()
        cam_params.intrinsic = self.intrins_o3d
        cam_params.extrinsic = extrins

        # ---------- Create visualizer ----------
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.CAM_W, height=self.CAM_H, visible=False)
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
    def get_pose_axis(pose) :
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
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



    