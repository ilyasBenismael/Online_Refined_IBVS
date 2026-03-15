import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt





# robot camera
CAM_W, CAM_H = 1264, 832
FX = FY = 0.8 * max(CAM_W, CAM_H)
f=1
CX, CY = CAM_W / 2.0, CAM_H / 2.0




class MeshHandling :


    
    def __init__(self, CAM_W, CAM_H):
        
        self.CAM_W = CAM_W
        self.CAM_H = CAM_H
        self.FX = self.FY = 0.8 * max(CAM_W, CAM_H)
        self.CX = self.CY = CAM_W / 2.0, CAM_H / 2.0

        self.intrins_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=self.CAM_W,
            height=self.CAM_H,
            fx=self.FX,
            fy=self.FY,
            cx=self.CX,
            cy=self.CY)
          



    @staticmethod
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




    @staticmethod
    def turn_points_to_o3d2(points, color = [0,0,0]) :
        
        n = len(points)
        colors = np.tile(color, (n, 1))

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
    def visualize_scene(scene_compos) :
            o3d.visualization.draw_geometries(
            scene_compos,
            window_name="The scene",
            width=800,
            height=800,
            mesh_show_back_face=True)





        
    def render_mesh_pic(self, mesh, extrins):

        # defining extrins and T_cw_final params 
        extrins = np.linalg.inv(extrins)

        # making our pincamparams objct
        cam_params = o3d.camera.PinholeCameraParameters()
        cam_params.intrinsic = self.intrins_o3d
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





    def get_cam_pose_from_mesh_view(self, mesh):

        def move_top(vis):
            ctr = vis.get_view_control()
            ctr.camera_local_translate(0.0, 0.0, -0.01)
            return False
        

        def move_bot(vis):
            ctr = vis.get_view_control()
            ctr.camera_local_translate(0.0, 0.0, 0.01)
            return False


        def move_left(vis):
            ctr = vis.get_view_control()
            ctr.camera_local_translate(0.0, -0.01, 0.0)
            return False


        def move_right(vis):
            ctr = vis.get_view_control()
            ctr.camera_local_translate(0.0, 0.01, 0.0)
            return False


        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Choose ur pose",
                        width=CAM_W,
                        height=CAM_H)

        vis.add_geometry(mesh)
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


