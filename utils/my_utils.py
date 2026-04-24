import torch
import gc
import subprocess
import matplotlib.pyplot as plt
import os, shutil



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