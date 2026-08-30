from utils.lin_algeb import LinAlgeb
from utils.my_utils import MyUtils
import numpy as np
import subprocess
import os
import pycolmap
import shutil
import heapq
from pathlib import Path




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
            f"--SiftExtraction.max_image_size 1500"
        )
        MyUtils.run_cmd(cmd)


        # Matching
        matcher = "sequential"
        if not sequential :
            matcher = "exhaustive"
        
        MyUtils.run_cmd(
            f"colmap {matcher}_matcher "
            f"--database_path {db_path} "
            f"--FeatureMatching.use_gpu 1"
        )


        # Registration
        cmd = (
            f"colmap image_registrator "
            f"--database_path {db_path} "
            f"--input_path {sparse_path} "
            f"--output_path {sparse_path}"
        )
        MyUtils.run_cmd(cmd)






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

