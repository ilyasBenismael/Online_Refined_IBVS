import torch
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
import torch
from utils.lin_algeb import LinAlgeb
from utils.gaussians_handling import GaussiansHandling
from utils.mesh_handling import MeshHandling
from utils.image_handling import ImageHandling
import numpy as np
import open3d as o3d
import torch
from PIL import Image

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.utils import get_phrases_from_posmap



"""TEXT_PROMPT = (
   stroller, "chair . sofa . table . bed . tv . window . door . lamp . light . carpet . rug . painting . mirror . bookshelf . cabinet . "
   "shelf . plant . curtain . blinds . pillow . blanket . towel . dish . cup . glass . fork . spoon . knife . refrigerator . oven . microwave . "
   "sink . faucet . toilet . shower . bathtub . trash can . clock . fan . heater . air conditioner . pillow . book . phone . laptop . computer . "
   "keyboard . mouse . speaker . camera . remote control"
)"""



# CONFIGs 
DEVICE = "cpu"
CONFIG_PATH  = "/home/user/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "/home/user/GroundingDINO/weights/groundingdino_swint_ogc.pth"
IMAGE_PATH   = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test/playroom/real_scene/sfm_playroom/images/DSC05626.jpg"
BOX_THRESHOLD  = 0.35 
TEXT_THRESHOLD = 0.25  
max_detections = 1
cx, cy, w, h, cx_px, cy_px = None,None,None,None,None,None
TEXT_PROMPT = ("stroller")
Z_desired = 2



# Load model
args = SLConfig.fromfile(CONFIG_PATH)
args.device = DEVICE
model = build_model(args)
checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
model.eval().to(DEVICE)


# Load img infos
image_pil = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(image_pil) 
CAM_W, CAM_H = image_pil.size

# Transform img to tensor
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),])
image, _ = transform(image_pil, None)
image = image.to(DEVICE)





# GD INFERENCE
caption = TEXT_PROMPT.lower().strip()
if not caption.endswith("."):
    caption += "."
with torch.no_grad():
    outputs = model(image[None], captions=[caption])
logits = outputs["pred_logits"].sigmoid()[0]  # (N, 256)
boxes  = outputs["pred_boxes"][0]             # (N, 4)



# Keeping above treshhold results
scores = logits.max(dim=1)[0]
keep   = scores > BOX_THRESHOLD
boxes_filt  = boxes[keep].cpu()
scores_filt = scores[keep].cpu()
logits_filt = logits[keep].cpu()



# Keep only number of detections I want (with best scores)
if len(boxes_filt) > max_detections:
    top_scores, top_indices = scores_filt.topk(max_detections)
    boxes_filt = boxes_filt[top_indices]
    scores_filt = scores_filt[top_indices]
    logits_filt = logits_filt[top_indices]


# Get text labels of the detected boxes 
tokenizer  = model.tokenizer
tokenized  = tokenizer(caption)
labels = []
for logit in logits_filt:
    phrase = get_phrases_from_posmap(logit > TEXT_THRESHOLD, tokenized, tokenizer)
    labels.append(phrase)

print(f"\nDetected {len(boxes_filt)} objects (threshold={BOX_THRESHOLD}):")
for label, score in zip(labels, scores_filt):
    print(f"  {label:30s}  score: {score:.3f}")



# Draw all boxes on the img and save it and get the cx,cy,w,h (box coordinates)
draw = ImageDraw.Draw(image_pil)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except:
    font = ImageFont.load_default()

for box, label, score in zip(boxes_filt, labels, scores_filt):
    # normalized xywh → pixel xyxy
    cx, cy, w, h = box * torch.tensor([CAM_W, CAM_H, CAM_W, CAM_H])
    cx_px = int(cx.item())
    cy_px = int(cy.item())
    x0 = (cx - w / 2).item()
    y0 = (cy - h / 2).item()
    x1 = (cx + w / 2).item()
    y1 = (cy + h / 2).item()

    # random color per box
    color = tuple(random.randint(50, 255) for _ in range(3))

    # draw box
    draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

    # draw label background + text
    text = f"{label} {score:.2f}"
    bbox = draw.textbbox((x0, y0), text, font=font)
    draw.rectangle(bbox, fill=color)
    draw.text((x0, y0), text, fill="white", font=font)

    # draw center dot
    r = 10
    draw.ellipse([cx_px - r, cy_px - r, cx_px + r, cy_px + r], fill="red", outline="white", width=2)

ImageHandling.save_img(image_pil, "bbx_img", "my_results/sam_test")
print(f"Box center (pixels): ({cx_px}, {cy_px})")
print(f"\nSaved img with bbx")




#____________________________________ Now we got box center we need the moge img with this box center in the middle


# Get MoGe points and corresponding bbx center point 
masked_points, masked_colors, all_moge_points, all_moge_colors, moge_mask = GaussiansHandling.get_moge_points(image_np, model_reso_lvl = 1, use_fp16_bool = False) 
moge_points_o3d = MeshHandling.turn_points_to_o3d(all_moge_points, all_moge_colors) 
objct_point_3d = all_moge_points[cy_px, cx_px]  # (x, y, z) in camera frame
print(f"3D point in camera frame: {objct_point_3d}")


# create a red sphere at the point's position
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
sphere.translate(objct_point_3d)
sphere.paint_uniform_color([1, 0, 0])
MeshHandling.visualize_scene([moge_points_o3d, sphere])



# By moge points are in WF and we will considere cam is in WF, then I can calculate the cam new translation (instead of the old "0,0,0") that will make the point on cam center
# Cam supposed initial pose
R = np.eye(3)        
t = np.zeros(3)  

# Compute new translation and new cam pose
t_new = np.array([
    objct_point_3d[0],               # shift camera X to align with object X
    objct_point_3d[1],               # shift camera Y to align with object Y
    objct_point_3d[2] - Z_desired    # shift camera Z so object is Z_desired away
])
new_cam_pose = LinAlgeb.get_homog_frm_rt(R, t_new)


# render 3d points from new pose and save our des_img
mesh_handling = MeshHandling(CAM_W, CAM_H) 
all_points_o3d = MeshHandling.turn_points_to_o3d(all_moge_points, all_moge_colors)
des_img, _ = mesh_handling.render_mesh_pic([all_points_o3d], new_cam_pose)
ImageHandling.plot_img(des_img, "des_img")
ImageHandling.save_img(des_img, "des_img", "my_results/sam_test")