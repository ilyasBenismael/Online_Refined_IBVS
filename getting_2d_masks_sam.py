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



# SAM
import sys
sys.path.append("/home/user/sam2/")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor








TEXT_PROMPT = (
   "stroller, chair . sofa . table . bed . tv . window . door . lamp . light . carpet . rug . painting . mirror . bookshelf . cabinet . "
   "shelf . plant . curtain . blinds . pillow . blanket . towel . dish . cup . glass . fork . spoon . knife . refrigerator . oven . microwave . "
   "sink . faucet . toilet . shower . bathtub . trash can . clock . fan . heater . air conditioner . pillow . book . phone . laptop . computer . "
   "keyboard . mouse . speaker . camera . remote control"
)


# CONFIGs 
DEVICE = "cpu"
CONFIG_PATH  = "/home/user/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "/home/user/GroundingDINO/weights/groundingdino_swint_ogc.pth"
IMAGE_PATH   = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test/playroom/real_scene/sfm_playroom/images/DSC05598.jpg"
BOX_THRESHOLD  = 0.35 
TEXT_THRESHOLD = 0.25  
max_detections = 100




# Load GD model
args = SLConfig.fromfile(CONFIG_PATH)
args.device = DEVICE
model = build_model(args)
checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
model.eval().to(DEVICE)


# Load Sam model
checkpoint_path = "/home/user/sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
sam2_model = build_sam2(model_cfg, checkpoint_path, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)




# Load img infos
image_pil = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(image_pil) 
CAM_W, CAM_H = image_pil.size

# get its moge point cloud 
masked_points, masked_colors, all_moge_points, all_moge_colors, moge_mask = GaussiansHandling.get_moge_points(image_np, model_reso_lvl = 1, use_fp16_bool = False) 



# Transform img to tensor for GD
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),])
image, _ = transform(image_pil, None)
image = image.to(DEVICE)


# put img in SAM
predictor.set_image(image_np)



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
# convert PIL image to numpy for overlay
image_overlay = np.array(image_pil).copy()
results = []
image_overlay = np.array(image_pil).copy()
image_boxes = image_pil.copy()
draw_boxes = ImageDraw.Draw(image_boxes)

for box, label, score in zip(boxes_filt, labels, scores_filt):

    # normalized xywh → pixel xyxy
    cx, cy, w, h = box * torch.tensor([CAM_W, CAM_H, CAM_W, CAM_H])
    cx_px = int(cx.item())
    cy_px = int(cy.item())
    x0 = int((cx - w / 2).item())
    y0 = int((cy - h / 2).item())
    x1 = int((cx + w / 2).item())
    y1 = int((cy + h / 2).item())
    input_box = np.array([x0, y0, x1, y1])

    # random color per object (reused for both outputs)
    color_np    = np.array([random.randint(50, 255) for _ in range(3)])
    color_tuple = tuple(color_np.tolist())

    # -------------------------
    # SAM2 MASK
    # -------------------------
    masks, scores_sam, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False
    )
    mask = masks[0].astype(bool)  # (H, W) bool

    # -------------------------
    # PIXEL INDICES
    # -------------------------
    ys, xs = np.where(mask)

    # -------------------------
    # STORE RESULT
    # -------------------------
    results.append({
        "label": label,
        "score": float(score),
        "box":   input_box,
        "mask":  mask,
        "pixels": np.stack([xs, ys], axis=1)  # (N, 2) as [[x,y], ...]
    })

    print(f"\nObject : {label}")
    print(f"Score  : {score:.3f}")
    print(f"Box    : {input_box}")
    print(f"#pixels: {len(xs)}")

    # -------------------------
    # IMAGE 1: colored mask overlay
    # -------------------------
    alpha = 0.5
    image_overlay[mask] = (
        alpha * color_np + (1 - alpha) * image_overlay[mask]
    ).astype(np.uint8)

    # -------------------------
    # IMAGE 2: bounding boxes + label names only
    # -------------------------
    draw_boxes.rectangle([x0, y0, x1, y1], outline=color_tuple, width=3)
    bbox_text = draw_boxes.textbbox((x0, y0 - 22), label, font=font)
    draw_boxes.rectangle(bbox_text, fill=color_tuple)
    draw_boxes.text((x0, y0 - 22), label, fill="white", font=font)

    color = np.array([
        random.random(),
        random.random(),
        random.random()
    ])  # random color per object

    all_moge_colors[mask] = color



moge_points_o3d = MeshHandling.turn_points_to_o3d(all_moge_points, all_moge_colors) 
MeshHandling.visualize_scene([moge_points_o3d])


# -------------------------
# SAVE BOTH IMAGES
# -------------------------
Image.fromarray(image_overlay).save("segmented_masks.png")
print("\nSaved: segmented_masks.png")

image_boxes.save("bounding_boxes.png")
print("Saved: bounding_boxes.png")


