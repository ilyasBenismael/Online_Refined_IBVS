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




class SemanticsHadling :


    @staticmethod
    def get_gd_bbxs(image_np, txt_prompt, BOX_THRESHOLD  = 0.35, TEXT_THRESHOLD = 0.25, max_detections = 100, device = "cpu") :
        
        # Load GD model
        CONFIG_PATH  = "/home/user/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        WEIGHTS_PATH = "/home/user/GroundingDINO/weights/groundingdino_swint_ogc.pth"
        args = SLConfig.fromfile(CONFIG_PATH)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval().to(device)

       
        # Load img & Transform it to tensor for GD
        image_pil = Image.fromarray(image_np)
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),])
        image, _ = transform(image_pil, None)
        image = image.to(device)

        # GD INFERENCE
        caption = txt_prompt.lower().strip()
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

        # printing all detected objcts
        print(f"\nDetected {len(boxes_filt)} objects (threshold={BOX_THRESHOLD}):")
        for label, score in zip(labels, scores_filt):
            print(f"  {label:30s}  score: {score:.3f}")

        return boxes_filt, labels, scores_filt





    @staticmethod
    def get_sam_masks(image_np, boxes_filt, labels, scores_filt) : 

        CAM_H, CAM_W = image_np.shape[:2]
        masks = []

        # Load Sam model
        checkpoint_path = "/home/user/sam2/checkpoints/sam2.1_hiera_tiny.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        sam2_model = build_sam2(model_cfg, checkpoint_path, device="cuda")
        predictor = SAM2ImagePredictor(sam2_model)

        # put img in SAM
        predictor.set_image(image_np)


        for box, label, score in zip(boxes_filt, labels, scores_filt):

            # normalized xywh → pixel xyxy
            cx, cy, w, h = box * torch.tensor([CAM_W, CAM_H, CAM_W, CAM_H])
            x0 = int((cx - w / 2).item())
            y0 = int((cy - h / 2).item())
            x1 = int((cx + w / 2).item())
            y1 = int((cy + h / 2).item())
            input_box = np.array([x0, y0, x1, y1])

            # SAM inference
            mask, score_sam, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False
            )
            mask = mask[0].astype(bool)  # (H, W) bool
            masks.append(mask)


            print(f"\nObject : {label}")
            print(f"Score  : {score:.3f}")

        
        return masks
            





    @staticmethod
    def draw_gd_bbxs(image_np, boxes_filt, labels, scores_filt):
    
        image_pil = Image.fromarray(image_np.copy())
        draw = ImageDraw.Draw(image_pil)

        CAM_W, CAM_H = image_pil.size

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
            )
        except:
            font = ImageFont.load_default()

        for box, label, score in zip(boxes_filt, labels, scores_filt):

            cx, cy, w, h = box * torch.tensor([CAM_W, CAM_H, CAM_W, CAM_H])
            x0 = int((cx - w / 2).item())
            y0 = int((cy - h / 2).item())
            x1 = int((cx + w / 2).item())
            y1 = int((cy + h / 2).item())

            color = tuple(random.randint(50, 255) for _ in range(3))

            # box
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

            # label
            text = f"{label} {score:.2f}"
            bbox = draw.textbbox((x0, y0 - 22), text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0 - 22), text, fill="white", font=font)

        return np.array(image_pil)




    @staticmethod
    def draw_sam_masks(image_np, masks, alpha=0.5):
        """
        image_np: (H,W,3) uint8
        masks: list of (H,W) bool OR array (N,H,W)
        return: image with colored masks
        """

        image_overlay = image_np.copy()

        # handle both list and numpy array
        if isinstance(masks, np.ndarray):
            masks_iter = masks
        else:
            masks_iter = list(masks)

        for mask in masks_iter:
            mask = mask.astype(bool)

            color = np.array([random.randint(50, 255) for _ in range(3)])

            image_overlay[mask] = (
                alpha * color + (1 - alpha) * image_overlay[mask]
            ).astype(np.uint8)

        return image_overlay


    @staticmethod
    def get_masked_moge_points(moge_points, moge_colors, masks, moge_mask):

        # Flatten to 1D if 2D
        if moge_points.ndim == 3:
            moge_points = moge_points.reshape(-1, 3)
            moge_colors = moge_colors.reshape(-1, 3)
            masks = [m.reshape(-1) for m in masks]

        moge_mask = moge_mask.reshape(-1)  # ensure 1D regardless of input shape

        for mask in masks:
            mask = mask.astype(bool)          # (H*W,) full image mask
            object_mask = mask[moge_mask]     # (N,) filtered to valid moge points only
            color = np.array([random.random(), random.random(), random.random()])
            moge_colors[object_mask] = color

        moge_points_o3d = MeshHandling.turn_points_to_o3d(moge_points, moge_colors)
        return moge_points_o3d

