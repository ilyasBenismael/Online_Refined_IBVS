import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os, io
import sys
from pathlib import Path
from utils.gaussians_handling import GaussiansHandling


# Path to IBVS_CODE
project_root = Path(__file__).resolve().parents[2]
# Add gaussian_splatting2 to Python path
sys.path.insert(0, str(project_root / "gaussian_splatting2"))

from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr





class ImageHandling :



    @staticmethod
    def return_crrct_extns_img_path(img_path):
        # If the path already exists → return it
        if os.path.exists(img_path):
            return img_path

        # If not , Split path into base + extension
        base, _ = os.path.splitext(img_path)
        # Try possible extensions
        for ext in [".png", ".jpg", ".jpeg"]:
            new_path = base + ext
            if os.path.exists(new_path):
                return new_path
            
            

    @staticmethod
    def load_np_img(img_path) :
        img_path = ImageHandling.return_crrct_extns_img_path(img_path)
        img = Image.open(img_path).convert("RGB")
        return np.array(img)


    @staticmethod
    def plot_img(img, title="") :
        plt.imshow(img)
        plt.axis("off")
        plt.suptitle(title)
        plt.show()


    @staticmethod
    def plot_2_imgs(img1, img2, title1="", title2="", suptitle=None):

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].imshow(img1)
        axs[0].axis("off")
        axs[0].set_title(title1)

        axs[1].imshow(img2)
        axs[1].axis("off")
        axs[1].set_title(title2)

        if suptitle is not None:
            fig.suptitle(suptitle)

        plt.tight_layout()
        plt.show()





    @staticmethod
    def plot_4_imgs(img1, img2, img3, img4,
                    title1="", title2="", title3="", title4="",
                    show=True, save=False):

        numpy_img = None
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        axs[0, 0].imshow(img1); axs[0, 0].axis("off"); axs[0, 0].set_title(title1)
        axs[0, 1].imshow(img2); axs[0, 1].axis("off"); axs[0, 1].set_title(title2)
        axs[1, 0].imshow(img3); axs[1, 0].axis("off"); axs[1, 0].set_title(title3)
        axs[1, 1].imshow(img4); axs[1, 1].axis("off"); axs[1, 1].set_title(title4)

        plt.tight_layout()

        if save:
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            numpy_img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            numpy_img = cv2.imdecode(numpy_img, cv2.IMREAD_COLOR)
            numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)

        if show:
            plt.show()
        else:
            plt.close(fig) # (preventing auto-display)

        return numpy_img





    @staticmethod
    def save_img(img, title, folder_path, assume_rgb=True):
        os.makedirs(folder_path, exist_ok=True)
        img = np.asarray(img, dtype=np.float32)
        img_u8 = (img * 255 if img.mean() <= 1.0 else img).clip(0, 255).astype(np.uint8)
        if assume_rgb:
            img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(folder_path, f"{title}.png"), img_u8)



    @staticmethod
    def white_to_black(img, threshold=250):
        """
        Replace white (or near-white) pixels with black.
        threshold allows tolerance.
        """
        img = img.copy()

        # Only check RGB channels
        rgb = img[:, :, :3]

        # Detect white pixels (near white with threshold)
        mask = np.all(rgb >= threshold, axis=-1)

        # Set them to black
        img[mask, :3] = 0

        return img
    




    @staticmethod
    def turn_img_to_gray(img) :
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]



    @staticmethod
    def compute_texturemap_and_mask(image_rgb, threshold=0.05):
        """
        bigger the treshhold fewer the details kept

        Args:
            image_rgb: (H, W, 3)
            threshold: float in [0,1]

        Returns:
            texture_map: (H, W) in [0,1]
            masked_rgb: (H, W, 3) with low-texture pixels set to black
        """

        # --- Texture map
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        texture_map = magnitude / (np.max(magnitude) + 1e-8)

        # --- Mask
        mask = texture_map > threshold  # (H, W)

        return texture_map, mask





    @staticmethod
    def img_255_to_01(img):
        img = np.asarray(img)
        # If already in [0,1], just return float version
        if img.dtype != np.uint8 and img.max() <= 1.0:
            return img.astype(np.float32)
        # Otherwise assume [0,255]
        return img.astype(np.float32) / 255.0



    @staticmethod
    def img_01_to_255(img):
        # Ensure float
        img = np.asarray(img, dtype=np.float32)
        # Clip to [0,1] to avoid overflow
        img = np.clip(img, 0.0, 1.0)
        # Scale and convert to uint8
        img_255 = (img * 255.0).astype(np.uint8)

        return img_255




    @staticmethod
    def compute_grayscale_difference(img1, img2, normalize=True):
        img1_float = img1.astype(np.float32)
        img2_float = img2.astype(np.float32)
        
        diff = np.abs(img1_float - img2_float)
        
        if normalize:
            if diff.max() > 0:
                diff = (diff / diff.max() * 255).astype(np.uint8)
            else:
                diff = diff.astype(np.uint8)
        else:
            diff = diff.astype(np.uint8)
        
        if diff.ndim == 2:
            diff = diff[:, :, np.newaxis]
        
        return diff


    @staticmethod
    def turn_img_to_gray(img) :
        gray_img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        return gray_img


    @staticmethod
    def draw_matches(matches_1, matches_2, img1, img2):
        
        # Copy images
        im1_vis = img1.copy()
        im2_vis = img2.copy()

        # Ensure uint8
        if im1_vis.dtype != np.uint8:
            im1_vis = (im1_vis * 255).astype(np.uint8)
        if im2_vis.dtype != np.uint8:
            im2_vis = (im2_vis * 255).astype(np.uint8)

        # Stack images side by side
        h1, w1 = im1_vis.shape[:2]
        h2, w2 = im2_vis.shape[:2]
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = im1_vis
        canvas[:h2, w1:] = im2_vis

        # Draw matches
        for pt1, pt2 in zip(matches_1, matches_2):
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]) + w1, int(pt2[1])  # shift x for second image
            color = (0, 255, 0)  # green
            cv2.circle(canvas, (x1, y1), 5, color, -1)
            cv2.circle(canvas, (x2, y2), 5, color, -1)
            cv2.line(canvas, (x1, y1), (x2, y2), color, 2)

        return canvas






    @staticmethod
    def get_grads_visp(I, FX, FY):
        # coefficients
        c1 = 2047.0
        c2 = 913.0
        c3 = 112.0
        norm = 8418.0

        H, W = I.shape
        I = I.astype(np.float64)

        # adding edges length 3 in all sides (they get the value of their neighbours)
        I_padded = np.pad(I, pad_width=3, mode='edge')

        # X derivative (horizontal)
        # we can divide the width indinces of the padded-img as follows: 0(1st elmnt of pad_img)-3(crspnd to 1st elmnt of orig-img)-H+3(crspnd to last elmnt of orig-img)-H+6(lst elmnt of pd-img)
        dI_du = (
            c1 * (I_padded[3:H+3, 4:W+4] - I_padded[3:H+3, 2:W+2]) +  # j+1 vs j-1
            c2 * (I_padded[3:H+3, 5:W+5] - I_padded[3:H+3, 1:W+1]) +  # j+2 vs j-2
            c3 * (I_padded[3:H+3, 6:W+6] - I_padded[3:H+3, 0:W+0])    # j+3 vs j-3
        ) / norm

        # Y derivative (vertical)
        dI_dv = (
            c1 * (I_padded[4:H+4, 3:W+3] - I_padded[2:H+2, 3:W+3]) +  # i+1 vs i-1
            c2 * (I_padded[5:H+5, 3:W+3] - I_padded[1:H+1, 3:W+3]) +  # i+2 vs i-2
            c3 * (I_padded[6:H+6, 3:W+3] - I_padded[0:H+0, 3:W+3])    # i+3 vs i-3
        ) / norm

        # turning grad values to frm pxls to cam units
        Ix = FX * dI_du
        Iy = FY * dI_dv

        return Ix, Iy






    @staticmethod
    def calc_imgs_sim_metrics(img1, img2, i, mask = False) :
        if mask :      
            # apply mask on both img1 and img2 and 
            pass
        img1 = GaussiansHandling.get_gsinria_tensor_from_img(img1)
        img2 = GaussiansHandling.get_gsinria_tensor_from_img(img2)
        ssim_score = ssim(img1, img2)
        psnr_score = psnr(img1, img2)
        lpips_score = lpips(img1, img2, net_type='vgg')
        # save as .npy : to [i] : [ssim, psnr, lpip] 
        print(i)
        print(f"SSIM : {ssim_score.item():.4f}")
        print(f"PSNR : {psnr_score.item():.4f}")
        print(f"LPIPS: {lpips_score.item():.4f}")

        return ssim_score.item(), psnr_score.item(), lpips_score.item()




    @staticmethod
    def get_mask_frm_depth(depth):
        depth = np.asarray(depth)
        mask = np.isfinite(depth) & (depth > 0) 
        return mask

    
    
    @staticmethod
    def apply_mask_on_img(img, mask):
        out = img.copy()
        out[~mask] = 0
        return out
    




    @staticmethod
    def resize_imgs_folder(folder_path, max_size=1500):
        folder_path = Path(folder_path)

        valid_extensions = {
            ".jpg", ".jpeg", ".png", ".bmp",
            ".tif", ".tiff", ".webp"
        }

        for image_path in folder_path.iterdir():
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in valid_extensions:
                continue

            with Image.open(image_path) as img:
                # Correct orientation using the image's EXIF information
                img = ImageOps.exif_transpose(img)

                width, height = img.size
                largest_dimension = max(width, height)

                # Do not enlarge images smaller than 1500 pixels
                if largest_dimension <= max_size:
                    print(f"Skipped: {image_path.name} ({width}x{height})")
                    continue

                scale = max_size / largest_dimension

                new_width = round(width * scale)
                new_height = round(height * scale)

                resized_img = img.resize(
                    (new_width, new_height),
                    Image.Resampling.LANCZOS
                )

                # JPEG cannot store RGBA images
                if image_path.suffix.lower() in {".jpg", ".jpeg"}:
                    if resized_img.mode not in {"RGB", "L"}:
                        resized_img = resized_img.convert("RGB")

                    resized_img.save(
                        image_path,
                        quality=95,
                        subsampling=0
                    )
                else:
                    resized_img.save(image_path)

                print(
                    f"Resized: {image_path.name} "
                    f"({width}x{height} -> {new_width}x{new_height})"
                )












    @staticmethod
    def get_hw_of_img(img) :
        """
        expect a 3d np array, rtrn h,w
        """
        height, width = img.shape[:2]
        return height, width

