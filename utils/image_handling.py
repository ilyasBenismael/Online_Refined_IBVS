import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os





class ImageHandling :

    @staticmethod
    def load_np_img(img_path) :
        img = Image.open(img_path).convert("RGB")
        return np.array(img)

    @staticmethod
    def plot_img(des_img, title) :
        plt.imshow(des_img)
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
    def draw_matches(matches_1, matches_2, init_img, desired_img):
        
        # Copy images
        im1_vis = init_img.copy()
        im2_vis = desired_img.copy()

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










