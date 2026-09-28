import torch
import gc
import subprocess
import matplotlib.pyplot as plt
import os, shutil
import numpy as np
from pathlib import Path
import uuid





class MyUtils :



    @staticmethod
    def run_cmd(cmd, cwd=None):
        print("Running:", cmd)
        subprocess.run(cmd, shell=True, check=True, cwd=cwd)
        


    @staticmethod
    def cleanup():

        # Delete GPU cache (PyTorch)
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except:
            pass

        # Python garbage collector
        gc.collect()

        # Close all matplotlib figures
        plt.close('all')







    @staticmethod
    def copy_any(src_path, dst_path, overwrite=False):

        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source not found: {src_path}")

        # If it's a file
        if os.path.isfile(src_path):
            # ensure parent folder exists
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            if os.path.exists(dst_path) and not overwrite:
                raise FileExistsError(f"Destination exists: {dst_path}") 
            shutil.copy2(src_path, dst_path)
            return

        # If it's a folder
        if os.path.exists(dst_path):
            if overwrite:
                shutil.rmtree(dst_path)
            else:
                raise FileExistsError(f"Destination exists: {dst_path}")

        shutil.copytree(src_path, dst_path)



    @staticmethod
    def save_arrays_to_npy(npy_path, iteration, array):

        data = {}

        if os.path.exists(npy_path):
            try:
                data = np.load(npy_path, allow_pickle=True).item()
            except (EOFError, ValueError, OSError):
                print(f"Corrupted NPY file replaced: {npy_path}")

        data[iteration] = array
        np.save(npy_path, data, allow_pickle=True)



    @staticmethod
    def get_files_with_suffix(folder_path, suffix):
        """
        Return a sorted list of all files paths -str) in folder_path ending with suffix.

        Parameters
        ----------
        folder_path : str or Path
            Path to the folder.
        suffix : str
            File extension, e.g. ".png", ".ply", ".txt".

        Returns
        -------
        list[str]
            List of matching file paths.
        """
        folder = Path(folder_path)
        return sorted(str(f) for f in folder.iterdir()
                    if f.is_file() and f.name.endswith(suffix))



    @staticmethod
    def get_last_file_with_suffix(folder_path, suffix):
        """
        Return the last sorted file path in folder_path ending with suffix.
        Returns None if no matching file is found.
        """
        folder = Path(folder_path)
        files = sorted(
            f for f in folder.iterdir()
            if f.is_file() and f.name.endswith(suffix))
        return str(files[-1]) if files else None


    @staticmethod
    def add_line_to_text(txt_file: str, line: str):
        """Append one line to a text file."""
        with open(txt_file, "a") as f:
            f.write(line + "\n")



    @staticmethod
    def rename_imgs_folder(folder_path):
        folder_path = Path(folder_path)  # Convert str to Path

        valid_extensions = {
            ".jpg", ".jpeg", ".png", ".bmp",
            ".tif", ".tiff", ".webp"}

        images = sorted(
            (
                file for file in folder_path.iterdir()
                if file.is_file() and file.suffix.lower() in valid_extensions),
            key=lambda file: (file.stat().st_mtime_ns, file.name))

        # First use temporary names to prevent filename conflicts
        temporary_files = []

        for index, image in enumerate(images):
            temporary_path = folder_path / f"__temporary_{index}{image.suffix.lower()}"
            image.rename(temporary_path)
            temporary_files.append(temporary_path)

        # Then rename them from 1 to x
        for index, temporary_path in enumerate(temporary_files, start=1):
            final_path = folder_path / f"{index}{temporary_path.suffix}"
            temporary_path.rename(final_path)



    @staticmethod
    def get_dummy_mask(width, height, all_true = True) :
        """
        return a dummy 2d mask (W, H) all values are True 
        """
        if all_true : 
            mask = np.ones((height, width), dtype=bool)
        else :
            mask = np.zeros((height, width), dtype=bool)
        return mask


