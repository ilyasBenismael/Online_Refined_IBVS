import os
import numpy as np
import matplotlib.pyplot as plt
from utils.lin_algeb import LinAlgeb
from utils.image_handling import ImageHandling
from utils.gaussians_handling import GaussiansHandling
from utils.poses_handling import PosesHandling
from utils.my_utils import MyUtils
from utils.mesh_handling import MeshHandling
import os
import glob
import re
import cv2
from datetime import datetime



# Scenarios
BASE_DIR = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test"
SCENE_NAMES = ["kitchen", "playroom", "thehouse"]
CASE_NBRS = [1, 2, 3]


# Some results files infos 
CONFIGS_FILENAME = "configs.txt"
IBVS_INFOS_FILENAME = "ibvs_infos.npy"
RESULTS_FOLDER_NAME = "results"
RESULTS_TXT_NAME = "results.txt"
INIT_IMG_KEY = "init_img_gs1_name"
gt_img_name = "des0.png"

# Indices of ibvs_infos elements
PXL_ERROR_IDX = 0
POSE_ERROR_IDX = 1
COND_NBR_IDX = 2
V_IDX = 3
MATCHES_IDX = 4
POSE_GS1_IDX = 5

DVS_INFOS_FILENAME = "dvs_infos.npy"
DVS_COST_IDX = 0
DVS_CUR_POSE_IDX = 1
DVS_TIME_IDX = 2

DES0_IBVS_INFOS_FILENAME = "des0_ibvs_infos.npy"

# des0_ibvs_infos indices — [norm_of_error, condit_nbr, V, [matches_cur, matches_des], cur_pose_gs1]
DES0_NORM_ERROR_IDX = 0
DES0_COND_NBR_IDX = 1
DES0_V_IDX = 2
DES0_MATCHES_IDX = 3
DES0_POSE_GS1_IDX = 4





FINAL_RESULTS_PATH = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test/final_results.txt"

METRICS = [
    "total_nbr_itrs",
    "avrg_condit_nbr",
    "R_diff_last_itr_deg",
    "t_diff_last_itr_norm",
    "R_diff_dvs_last_itr_deg",
    "t_diff_dvs_last_itr_norm",
    "ibvs_last_des_PSNR",
    "ibvs_last_des_LPIPS",
    "ibvs_last_des_SSIM",
    "ibvs_gt_des_PSNR",
    "ibvs_gt_des_LPIPS",
    "ibvs_gt_des_SSIM",
    "dvs_last_des_PSNR",
    "dvs_last_des_LPIPS",
    "dvs_last_des_SSIM",
    "dvs_gt_des_PSNR",
    "dvs_gt_des_LPIPS",
    "dvs_gt_des_SSIM",
    "avrg_condit_nbr_des0"
]





def load_results(txt_path):
    metrics = {}

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()

            # Simple "key = value"
            if "=" in line and ":" not in line:
                key, value = line.split("=", 1)
                key = key.strip()
                if key in METRICS:
                    metrics[key] = float(value)

            # Metrics lines
            elif ":" in line:
                title, rest = line.split(":", 1)
                title = title.strip()

                m = re.search(
                    r"PSNR=([-\d.]+),\s*LPIPS=([-\d.]+),\s*SSIM=([-\d.]+)",
                    rest,
                )

                if m:
                    metrics[f"{title}_PSNR"] = float(m.group(1))
                    metrics[f"{title}_LPIPS"] = float(m.group(2))
                    metrics[f"{title}_SSIM"] = float(m.group(3))

    return metrics







# Handling text files __________________________________________________________________

def add_line_to_text(line: str, txt_file: str):
    """Append one line to a text file."""
    with open(txt_file, "a") as f:
        f.write(line + "\n")


def make_results_fldr_nd_txt_file(case_path: str):
    """Create <case_path>/results/ and an empty results.txt inside it."""
    results_dir = os.path.join(case_path, RESULTS_FOLDER_NAME)
    os.makedirs(results_dir, exist_ok=True)
    txt_path = os.path.join(results_dir, RESULTS_TXT_NAME)
    open(txt_path, "w").close()  # start fresh each run
    return results_dir, txt_path



def get_init_img_name_frm_txt(txt_path: str) -> str:
    with open(txt_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.split("=")[0].strip() == INIT_IMG_KEY:
                value = line.split("=", 1)[1].strip()
                value = value.strip("'\"")
                return value
    raise ValueError(f"Key '{INIT_IMG_KEY}' not found in {txt_path}")









# Getting infos from IBVS_INFOS file  ===========================================

def _sorted_iters(ibvs_infos: dict, until_iter=None):
    iters = sorted(ibvs_infos.keys())
    if until_iter is not None:
        iters = [it for it in iters if it <= until_iter]
    return iters


def get_total_nbr_of_itrs(ibvs_infos: dict, until_iter=None) -> int:
    return len(_sorted_iters(ibvs_infos, until_iter))


def get_avrg_condit_nbr(ibvs_infos: dict, until_iter=None, cond_idx=COND_NBR_IDX) -> float:
    iters = _sorted_iters(ibvs_infos, until_iter)
    cond_nbrs = [ibvs_infos[it][cond_idx] for it in iters]
    return float(np.mean(cond_nbrs))


def compute_pose_errors_per_iter(ibvs_infos: dict, R_gt, t_gt, t_norm, until_iter=None, pose_idx=POSE_GS1_IDX):
    iters = _sorted_iters(ibvs_infos, until_iter)
    R_diffs, t_diffs = {}, {}
    for it in iters:
        pose = ibvs_infos[it][pose_idx]
        R_it, t_it = LinAlgeb.get_Rt_from_homog_matrix(pose)
        r_d = LinAlgeb.rotation_diff(R_gt, R_it)
        t_d = LinAlgeb.pose_distance(t_gt, t_it)
        t_diffs[it] = LinAlgeb.normalize_t(t_d, t_norm)
        R_diffs[it] = r_d
    return iters, R_diffs, t_diffs


def get_values_per_iter(infos: dict, idx: int, until_iter=None) -> dict:
    iters = _sorted_iters(infos, until_iter)
    return {it: infos[it][idx] for it in iters}



# Plots ____________________________________________________________

def plot_error_vs_iter(iters, values_dict, ylabel, title, save_path):
    values = [values_dict[it] for it in iters]
    plt.figure(figsize=(10, 6))
    plt.plot(iters, values, marker="o", markersize=2, linewidth=0.6)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_comparison_vs_iter(series, ylabel, title, save_path):
    """series: list of (iters, values_dict, label, color) tuples."""
    plt.figure(figsize=(10, 6))
    for iters, values_dict, label, color in series:
        values = [values_dict[it] for it in iters]
        plt.plot(iters, values, marker="o", markersize=2, linewidth=0.6,
                  label=label, color=color)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()






#_________________________________________




def get_img_pose_gs1(img_name, gs1_sfm_path):
    recons1 = PosesHandling.get_recons(gs1_sfm_path)
    return PosesHandling.get_img_sfm_pose(recons1, img_name)




def get_itr_X(ibvs_infos, threshold = 0.1):

    data = ibvs_infos
    iterations = sorted(data.keys())

    for k in range(len(iterations) - 2):
        i1, i2, i3 = iterations[k:k+3]

        e1 = data[i1][0]
        e2 = data[i2][0]
        e3 = data[i3][0]

        if e1 < threshold and e2 < threshold and e3 < threshold:
            print("Selected iteration:", i1)
            print(f"for Pixel errors: {e1:.3f}, {e2:.3f}, {e3:.3f}")
            return i1








def load_last_des_img(desimg_folder):
    candidates = []

    for f in glob.glob(os.path.join(desimg_folder, "des*.png")):
        name = os.path.basename(f)
        m = re.fullmatch(r"des(\d+)\.png", name)
        if m:
            candidates.append((int(m.group(1)), f))

    if not candidates:
        raise FileNotFoundError("No des<number>.png images found.")

    _, last_img_path = max(candidates)
    return cv2.imread(last_img_path)





def load_last_image(folder_path):
    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")

    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not files:
        raise FileNotFoundError(f"No images found in {folder_path}")

    # Last image alphabetically
    last_img_path = sorted(files)[-1]

    return cv2.imread(last_img_path)












# Main PIPELINE  =======================================================================================================


def process_case(case_path: str, scene_name: str, case_nbr: int):

    print(f"--- Processing {scene_name} case {case_nbr} ---")

    # making the results.txt _______________________________
    results_dir, txt_path = make_results_fldr_nd_txt_file(case_path)
    add_line_to_text(f"scene_name = {scene_name}", txt_path)
    add_line_to_text(f"case_nbr = {case_nbr}", txt_path)


    # get GT / init poses
    configs_path = os.path.join(case_path, CONFIGS_FILENAME)
    init_img_name = get_init_img_name_frm_txt(configs_path)
    T_init = get_img_pose_gs1(init_img_name, f"{case_path}/gs1_sfm_aligned")
    T_gt = get_img_pose_gs1(gt_img_name, f"{case_path}/gs1_sfm_aligned")
    add_line_to_text(f"init_img_gs1_name = {init_img_name}", txt_path)

    # calc GT_init R and t diff
    R_init, t_init = LinAlgeb.get_Rt_from_homog_matrix(T_init)
    R_gt, t_gt = LinAlgeb.get_Rt_from_homog_matrix(T_gt)
    R_diff_gt_init = LinAlgeb.rotation_diff(R_gt, R_init)
    t_diff_gt_init = LinAlgeb.pose_distance(t_gt, t_init)  # normalization vector
    add_line_to_text(f"R_diff_gt_init_deg = {R_diff_gt_init}", txt_path)
    add_line_to_text(f"t_diff_gt_init = {t_diff_gt_init}", txt_path)



    # Load ibvs_infos____________________________________________________________________________________
    
    ibvs_infos_path = os.path.join(case_path, IBVS_INFOS_FILENAME)
    ibvs_infos = np.load(ibvs_infos_path, allow_pickle=True).item()

    # Get IBVS infos until last iters ________________________________________
    add_line_to_text("--- infos on all iters ---", txt_path)

    # save ttle itrs nd condit nbr to txt
    total_nbr_itrs = get_total_nbr_of_itrs(ibvs_infos)
    add_line_to_text(f"total_nbr_itrs = {total_nbr_itrs}", txt_path)
    avrg_condit_nbr = get_avrg_condit_nbr(ibvs_infos)
    add_line_to_text(f"avrg_condit_nbr = {avrg_condit_nbr}", txt_path)

    # save all R diffs and t diffs
    iters_full, R_diffs_full, t_diffs_full = compute_pose_errors_per_iter(
        ibvs_infos, R_gt, t_gt, t_diff_gt_init)
    R_diff_full_path = os.path.join(results_dir, "R_diff_all.npy")
    t_diff_full_path = os.path.join(results_dir, "t_diff_all.npy")
    np.save(R_diff_full_path, R_diffs_full)
    np.save(t_diff_full_path, t_diffs_full)

    # save last R nd t diffs to txt
    last_iter_full = iters_full[-1]
    add_line_to_text(f"last_itr = {last_iter_full}", txt_path)
    add_line_to_text(f"R_diff_last_itr_deg = {R_diffs_full[last_iter_full]}", txt_path)
    add_line_to_text(f"t_diff_last_itr_norm = {t_diffs_full[last_iter_full]}", txt_path)


    # Get IBVS infos until iter_X (errs<0.1)________________________________________
    itr_X = get_itr_X(ibvs_infos)
    add_line_to_text("--- infos until_itrX ---", txt_path)
    add_line_to_text(f"itr_X = {itr_X}", txt_path)

    # save ttle itrs nd condit nbr to txt until itr_X
    total_nbr_itrs_X = get_total_nbr_of_itrs(ibvs_infos, until_iter=itr_X)
    add_line_to_text(f"total_nbr_itrs_until_X = {total_nbr_itrs_X}", txt_path)
    avrg_condit_nbr_X = get_avrg_condit_nbr(ibvs_infos, until_iter=itr_X)
    add_line_to_text(f"avrg_condit_nbr_until_X = {avrg_condit_nbr_X}", txt_path)

    # save all R diffs and t diffs until itr_X
    iters_X, R_diffs_X, t_diffs_X = compute_pose_errors_per_iter(
        ibvs_infos, R_gt, t_gt, t_diff_gt_init, until_iter=itr_X)
    R_diff_X_path = os.path.join(results_dir, "R_diff_until_itrX.npy")
    t_diff_X_path = os.path.join(results_dir, "t_diff_until_itrX.npy")
    np.save(R_diff_X_path, R_diffs_X)
    np.save(t_diff_X_path, t_diffs_X)

    #save irs_x's R nd t diff to txt
    add_line_to_text(f"R_diff_itrX_deg = {R_diffs_X[itr_X]}", txt_path)
    add_line_to_text(f"t_diff_itrX_norm = {t_diffs_X[itr_X]}", txt_path)



    # Plot & save the 4 errors R,t R,t ________________________________________________
    
    plot_error_vs_iter(
        iters_full, R_diffs_full,
        ylabel="R diff (deg)",
        title=f"{scene_name} case{case_nbr} - R diff vs iter (full run)",
        save_path=os.path.join(results_dir, "R_diff_full.png"),)
    
    plot_error_vs_iter(
        iters_full, t_diffs_full,
        ylabel="Normalized t diff",
        title=f"{scene_name} case{case_nbr} - t diff vs iter (full run)",
        save_path=os.path.join(results_dir, "t_diff_full.png"),)
    
    plot_error_vs_iter(
        iters_X, R_diffs_X,
        ylabel="R diff (deg)",
        title=f"{scene_name} case{case_nbr} - R diff vs iter (until itr_X={itr_X})",
        save_path=os.path.join(results_dir, "R_diff_until_itrX.png"),)
    
    plot_error_vs_iter(
        iters_X, t_diffs_X,
        ylabel="Normalized t diff",
        title=f"{scene_name} case{case_nbr} - t diff vs iter (until itr_X={itr_X})",
        save_path=os.path.join(results_dir, "t_diff_until_itrX.png"),)

    print(f"    -> done. results saved to {results_dir}")



    # ___________________________________ DVS ________________________________________________

    dvs_infos_path = os.path.join(case_path, DVS_INFOS_FILENAME)
    dvs_infos = np.load(dvs_infos_path, allow_pickle=True).item()

    iters_dvs, R_diffs_dvs, t_diffs_dvs = compute_pose_errors_per_iter(
        dvs_infos, R_gt, t_gt, t_diff_gt_init, pose_idx=DVS_CUR_POSE_IDX)

    R_diff_dvs_path = os.path.join(results_dir, "R_diff_dvs_all.npy")
    t_diff_dvs_path = os.path.join(results_dir, "t_diff_dvs_all.npy")
    np.save(R_diff_dvs_path, R_diffs_dvs)
    np.save(t_diff_dvs_path, t_diffs_dvs)

    last_iter_dvs = iters_dvs[-1]
    add_line_to_text(f"last_itr_dvs = {last_iter_dvs}", txt_path)
    add_line_to_text(f"R_diff_dvs_last_itr_deg = {R_diffs_dvs[last_iter_dvs]}", txt_path)
    add_line_to_text(f"t_diff_dvs_last_itr_norm = {t_diffs_dvs[last_iter_dvs]}", txt_path)



    # Combined IBVS vs DVS plots
    iters_dvs = [it + itr_X for it in iters_dvs]

    R_diffs_dvs = {it + itr_X: val for it, val in R_diffs_dvs.items()}
    t_diffs_dvs = {it + itr_X: val for it, val in t_diffs_dvs.items()}

    plot_comparison_vs_iter(
        series=[
            (iters_full, R_diffs_full, "IBVS", "tab:blue"),
            (iters_dvs, R_diffs_dvs, "DVS", "tab:orange"),
        ],
        ylabel="R diff (deg)",
        title=f"{scene_name} case{case_nbr} - R diff vs iter (IBVS vs DVS)",
        save_path=os.path.join(results_dir, "R_diff_ibvs_vs_dvs.png"),
    )
    plot_comparison_vs_iter(
        series=[
            (iters_full, t_diffs_full, "IBVS", "tab:blue"),
            (iters_dvs, t_diffs_dvs, "DVS", "tab:orange"),
        ],
        ylabel="Normalized t diff",
        title=f"{scene_name} case{case_nbr} - t diff vs iter (IBVS vs DVS)",
        save_path=os.path.join(results_dir, "t_diff_ibvs_vs_dvs.png"),
    )


    # des0_ibvs_infos vs ibvs_infos pixel-error comparison _______________
    des0_ibvs_infos_path = os.path.join(case_path, DES0_IBVS_INFOS_FILENAME)
    des0_ibvs_infos = np.load(des0_ibvs_infos_path, allow_pickle=True).item()

    pxl_err_ibvs = get_values_per_iter(ibvs_infos, PXL_ERROR_IDX)
    pxl_err_des0 = get_values_per_iter(des0_ibvs_infos, DES0_NORM_ERROR_IDX)

    np.save(os.path.join(results_dir, "pxl_error_ibvs.npy"), pxl_err_ibvs)
    np.save(os.path.join(results_dir, "pxl_error_des0_ibvs.npy"), pxl_err_des0)

    add_line_to_text("--- des0_ibvs_infos ---", txt_path)
    avrg_condit_nbr_des0 = get_avrg_condit_nbr(des0_ibvs_infos, cond_idx=DES0_COND_NBR_IDX)
    add_line_to_text(f"avrg_condit_nbr_des0 = {avrg_condit_nbr_des0}", txt_path)

    iters_pxl_ibvs = sorted(pxl_err_ibvs.keys())
    iters_pxl_des0 = sorted(pxl_err_des0.keys())

    plot_comparison_vs_iter(
        series=[
            (iters_pxl_ibvs, pxl_err_ibvs, "IBVS", "tab:blue"),
            (iters_pxl_des0, pxl_err_des0, "des0_IBVS", "tab:green"),
        ],
        ylabel="Pixel error",
        title=f"{scene_name} case{case_nbr} - pixel error vs iter (IBVS vs des0_IBVS)",
        save_path=os.path.join(results_dir, "pxl_error_ibvs_vs_des0.png"),
    )

    """
    #________________________________________

    desimg_folder = f"{case_path}/desired_imgs"
    ibvs_frames_path = f"{case_path}/ibvs_frames/real_frames"
    dvs_frames_path = f"{case_path}/ibvs_frames/real_frames/DVS"
    txt_path = f"{case_path}/results/results.txt"

    last_des_img = np.asarray(load_last_des_img(desimg_folder))
    gt_des_img = ImageHandling.load_np_img(f"{desimg_folder}/GT_des.png")
    last_frame_IBVS = np.asarray(load_last_image(ibvs_frames_path))
    last_frame_DVS = np.asarray(load_last_image(dvs_frames_path))

    ssim, psnr, lpips = ImageHandling.calc_imgs_sim_metrics(last_frame_IBVS, last_des_img, 1)
    add_line_to_text(f"ibvs_last_des : PSNR={psnr:.3f}, LPIPS={lpips:.3f}, SSIM={ssim:.3f}",txt_path)

    ssim, psnr, lpips = ImageHandling.calc_imgs_sim_metrics(last_frame_IBVS, gt_des_img, 1)
    add_line_to_text(f"ibvs_gt_des : PSNR={psnr:.3f}, LPIPS={lpips:.3f}, SSIM={ssim:.3f}", txt_path)

    ssim, psnr, lpips = ImageHandling.calc_imgs_sim_metrics(last_frame_DVS, last_des_img, 1)
    add_line_to_text(f"dvs_last_des : PSNR={psnr:.3f}, LPIPS={lpips:.3f}, SSIM={ssim:.3f}", txt_path)

    ssim, psnr, lpips = ImageHandling.calc_imgs_sim_metrics(last_frame_DVS, gt_des_img, 1)
    add_line_to_text(f"dvs_gt_des : PSNR={psnr:.3f}, LPIPS={lpips:.3f}, SSIM={ssim:.3f}", txt_path)        

    print("saving similarities to txt !")
    
    """








# =========================================================================
# MAIN LOOP
# =========================================================================

def main():

    all_metrics = {k: [] for k in METRICS}

    for scene_name in SCENE_NAMES:
        for case_nbr in CASE_NBRS:

            case_folder = f"{scene_name}_case{case_nbr}"
            case_path = os.path.join(BASE_DIR, scene_name, case_folder)

            txt_path = os.path.join(case_path, "results", "results.txt")

            metrics = load_results(txt_path)

            for k in METRICS:
                all_metrics[k].append(metrics[k])


    with open(FINAL_RESULTS_PATH, "w") as f:

        f.write("============== FINAL RESULTS ==============\n\n")

        for k in METRICS:
            vals = np.array(all_metrics[k])

            f.write(
                f"{k}: "
                f"{vals.mean():.4f} ± {vals.std():.4f}\n"
            )

    return




    for scene_name in SCENE_NAMES:
        for case_nbr in CASE_NBRS:
            case_folder_name = f"{scene_name}_case{case_nbr}"
            case_path = os.path.join(BASE_DIR, scene_name, case_folder_name)
            process_case(case_path, scene_name, case_nbr)

    

if __name__ == "__main__":
    main()