"""
from utils.image_handling import ImageHandling
from utils.mesh_handling import MeshHandling
from utils.semantics_handling import SemanticsHadling
from utils.gaussians_handling import GaussiansHandling
import numpy as np



INDOOR_TEXT_PROMPT = (
"stroller, chair . sofa . table . bed . tv . window . door . lamp . light . carpet . rug . painting . mirror . bookshelf . cabinet . "
"shelf . plant . curtain . blinds . pillow . blanket . towel . dish . cup . glass . fork . spoon . knife . refrigerator . oven . microwave . "
"sink . faucet . toilet . shower . bathtub . trash can . clock . fan . heater . air conditioner . pillow . book . phone . laptop . computer . "
"keyboard . mouse . speaker . camera . remote control") 

OBJCT_TEXT_PROMPT = ("chair")

home_path = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/occlusions_avoider_test/test1"
des_img_path = f"{home_path}/initial_segmentation/initial.jpg"


# loading img
des_img = ImageHandling.load_np_img(des_img_path)

# getting gd bbxs
boxes, labels, scores = SemanticsHadling.get_gd_bbxs(des_img, OBJCT_TEXT_PROMPT, max_detections=1)
img_with_bbxs = SemanticsHadling.draw_gd_bbxs(des_img, boxes, labels, scores)
ImageHandling.save_img(img_with_bbxs, "img_with_bbxs", home_path)

# getting sam masks
sam_masks = SemanticsHadling.get_sam_masks(des_img, boxes, labels, scores)
img_with_masks = SemanticsHadling.draw_sam_masks(des_img, sam_masks)
ImageHandling.save_img(img_with_masks, "img_with_masks", home_path)

# Save
np.save("sam_masks.npy", sam_masks)


"""







import torch
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
import torch
from utils.lin_algeb import LinAlgeb
from utils.gaussians_handling import GaussiansHandling
from utils.mesh_handling import MeshHandling
from utils.image_handling import ImageHandling
from utils.poses_handling import PosesHandling
from utils.semantics_handling import SemanticsHadling
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from moge.model.v2 import MoGeModel









def look_at_pose(C, P):
    z = P - C
    z = z / np.linalg.norm(z)

    world_up = np.array([0, 1, 0])
    if abs(np.dot(z, world_up)) > 0.99:
        world_up = np.array([1, 0, 0])

    # swap order: z cross up instead of up cross z
    x = np.cross(z, world_up)
    x = x / np.linalg.norm(x)

    y = np.cross(z, x)
    y = y / np.linalg.norm(y)

    R = np.stack([x, y, z], axis=0)
    t = -R @ C

    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t

    return T



init_img_name_sfm = "inita.jpg"
des_img_name_sfm = "des.jpg"
home_path = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/occlusions_avoider_test/test1"
voxel_size = 0.5

sfm1_gs2_path = f"{home_path}/sfm1"
init_img_path = f"{home_path}/initial_segmentation/initial.jpg"
des_img_path = f"{home_path}/desired_segmentation/desired.jpg"



# loading init img 
init_img = ImageHandling.load_np_img(init_img_path)


# getting sfm infos
recons2 = PosesHandling.get_recons(sfm1_gs2_path)
init_img_pose = PosesHandling.get_img_sfm_pose(recons2, "inita.jpg")
des_img_pose = PosesHandling.get_img_sfm_pose(recons2, "des.jpg")
sparse_sfm_points = PosesHandling.get_sparse_points(recons2)
o3d_sparse_sfm_points = MeshHandling.turn_points_to_o3d(sparse_sfm_points)


# getting init des poses from sfm nd turn to o3d
r_init, t_init = LinAlgeb.get_Rt_from_homog_matrix(init_img_pose)
r_des, t_des = LinAlgeb.get_Rt_from_homog_matrix(des_img_pose)
o3d_init_pose = MeshHandling.turn_points_to_o3d(t_init)
o3d_des_pose = MeshHandling.turn_points_to_o3d(t_des)
spheres_init_des_o3d = MeshHandling.turn_points_to_spheres([t_init, t_des])


# making the 3d cube
corners = MeshHandling.get_box_from_two_points(t_init, t_des, margin=2)
o3d_box = MeshHandling.get_o3d_box_frm_corners(corners)


# making the voxel centers
voxel_centers = MeshHandling.get_voxel_centers_from_box_corners(corners, voxel_size)
voxel_centers_o3d = MeshHandling.turn_points_to_o3d(voxel_centers)

# making the voxel 3d_cells
all_voxels_o3d = []
for center in voxel_centers :
    voxel_corners = MeshHandling.get_cube_from_center(center, (voxel_size/2))
    voxel_o3d = MeshHandling.get_o3d_box_frm_corners(voxel_corners)
    all_voxels_o3d.append(voxel_o3d)




# get moge points and mask
moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda")
masked_points, masked_colors, all_moge_points, all_moge_colors, moge_mask = GaussiansHandling.get_moge_points(init_img, depth_edge_threshold=0.05)


# Get init_img sfm's 2Ds 3Ds
img_sfm_points_2d, img_sfm_points_3d = PosesHandling.get_2d_3d_points_of_img(recons2, init_img_name_sfm) 


# Keep only moge points corresp to sfm (& removing masked points (inf values..))
x = img_sfm_points_2d[:,0].astype(int)
y = img_sfm_points_2d[:,1].astype(int)
sfm_moge_points = all_moge_points[y, x]
sfm_mask = moge_mask[y, x]
sfm_moge_points = sfm_moge_points[sfm_mask]
img_sfm_points_3d = img_sfm_points_3d[sfm_mask]


# aligning moge points with sfm points
_, s, R, t = MeshHandling.align_points(sfm_moge_points, img_sfm_points_3d)


# aligning moge points with sfm points
mskd_colors_flat = masked_colors.reshape(-1, 3) #turning H,W,3 to H*W,3
mskd_points_flat = masked_points.reshape(-1, 3)
final_moge_points = (s * (R @ mskd_points_flat.T)).T + t
final_moge_points_o3d = MeshHandling.turn_points_to_o3d(final_moge_points, mskd_colors_flat)


# now that moges are in their right place, we can recognize occupied voxels
occupied_voxel_centers = MeshHandling.get_occupied_voxel(voxel_centers, final_moge_points, voxel_size)
occupied_voxel_centers_o3d = MeshHandling.turn_points_to_o3d(occupied_voxel_centers)

# make the 3d cells of these occupied voxels
occupied_voxels_o3d = []
for center in occupied_voxel_centers :
    voxel_corners = MeshHandling.get_cube_from_center(center, (voxel_size/2))
    voxel_o3d = MeshHandling.get_o3d_box_frm_corners(voxel_corners)
    occupied_voxels_o3d.append(voxel_o3d)    


# getting the A* path as a list of 3d points
path_points = PosesHandling.astar_from_voxel_centers(voxel_centers, occupied_voxel_centers, t_init, t_des, voxel_size)
path_points_o3d = MeshHandling.turn_points_to_o3d(path_points)

# visualizing segmented moge points + path + voxels
sam_masks = np.load("sam_masks.npy")
sam_masks_flat = sam_masks[0].reshape(-1)
masked_moge_points_o3d = SemanticsHadling.get_masked_moge_points(masked_points, masked_colors, [sam_masks_flat], moge_mask)

"""
MeshHandling.visualize_scene([masked_moge_points_o3d])
MeshHandling.visualize_scene([*all_voxels_o3d, masked_moge_points_o3d, spheres_init_des_o3d, o3d_box])
"""
MeshHandling.visualize_scene([*occupied_voxels_o3d, masked_moge_points_o3d, spheres_init_des_o3d, o3d_box, path_points_o3d])



# picking a random 3d moge point corresponding to the sam_mask (point we will be looking at)
mask = sam_masks[0]
ys, xs = np.where(mask)
idx = np.random.randint(len(xs))
x = xs[idx]
y = ys[idx]
semantic_point_gs2 = all_moge_points[y, x]
semantic_point_gs2_o3d = MeshHandling.turn_points_to_spheres([semantic_point_gs2], raduis=0.1)
MeshHandling.visualize_scene([semantic_point_gs2_o3d, masked_moge_points_o3d, spheres_init_des_o3d])

semantic_point_world = (s * (R @ semantic_point_gs2)) + t

CAM_H, CAM_W = init_img.shape[:2]

i=0

for i, path_point in enumerate(path_points):
    print(f"path point {i}: {path_point}")


for path_point in path_points :
    i+=1
    pose = look_at_pose(path_point, semantic_point_world)
    mesh_handling = MeshHandling(CAM_W, CAM_H)
    render, _ =mesh_handling.render_mesh_pic([masked_moge_points_o3d], pose)
    ImageHandling.save_img(render, i, home_path)


