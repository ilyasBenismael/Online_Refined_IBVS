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
from pathlib import Path
import open3d as o3d



# Scenarios
BASE_DIR = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test"
SCENE_NAMES = ["playroom", "thehouse", "woodroom", "livingroom"]
CASE_NBRS = [1, 2, 3]

# Some results files infos 
CONFIGS_FILENAME = "configs.txt"
IBVS_INFOS_FILENAME = "ibvs_infos.npy"
RESULTS_FOLDER_NAME = "results"
RESULTS_TXT_NAME = "results.txt"
INIT_IMG_KEY = "init_img_gs1_name"

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
    # Desired-image quality
    "initdes_GTdes_ssim",
    "initdes_GTdes_psnr",
    "initdes_GTdes_lpips",
    "finaldes_GTdes_ssim",
    "finaldes_GTdes_psnr",
    "finaldes_GTdes_lpips",

    # IBVS results
    "nbr_des",
    "nbr_kfs",
    "total_time",
    "R_diff_gt_init_deg",
    "t_diff_gt_init",
    "total_nbr_itrs",
    "avrg_condit_nbr",
    "R_diff_last_itr_deg",
    "t_diff_last_itr_norm",
]

DES_QUALITY_METRICS = [
    "initdes_GTdes_ssim",
    "initdes_GTdes_psnr",
    "initdes_GTdes_lpips",
    "finaldes_GTdes_ssim",
    "finaldes_GTdes_psnr",
    "finaldes_GTdes_lpips",
]

scene_name = "livingroom"
case_nbr = 1
CAM_W, CAM_H = 1557, 1038 
main_test_path = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test"
parent_case_test_path = f"{main_test_path}/{scene_name}/{scene_name}_case{case_nbr}"







def load_results(txt_path, metrics_list=METRICS):
    metrics = {}

    with open(txt_path, "r") as f:
        for line in f:
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()

            if key not in metrics_list:
                continue

            try:
                number = float(value.strip())
            except ValueError:
                continue

            if np.isfinite(number):
                metrics[key] = number

    return metrics






# Handling text files __________________________________________________________________

def add_line_to_text(txt_file: str, line: str):
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





def get_last_des(folder_path):
    pattern = re.compile(r"des(\d+)\.png$")

    des_files = []

    for filename in os.listdir(folder_path):
        m = pattern.match(filename)
        if m:
            des_files.append((int(m.group(1)), filename))

    if not des_files:
        raise FileNotFoundError(f"No des_<nbr>.png images found in {folder_path}")

    _, last_filename = max(des_files, key=lambda x: x[0])

    img = cv2.imread(os.path.join(folder_path, last_filename), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read {last_filename}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



"""
def get_mid_des(des_path) -> np.ndarray:
    pattern = re.compile(r"des(\d+)_plus\.png$")

    plus_files = []

    for filename in os.listdir(des_path):
        m = pattern.match(filename)
        if m:
            plus_files.append((int(m.group(1)), filename))

    if not plus_files:
        raise FileNotFoundError(f"No des*_plus.png files found in {des_path}")

    # Sort by the image number to recover the sequence
    plus_files.sort(key=lambda x: x[0])

    # Middle element (for even count, take the second middle)
    mid_idx = len(plus_files) // 2

    mid_number = plus_files[mid_idx][0]

    target_path = os.path.join(des_path, f"des{mid_number + 2}.png")

    img = cv2.imread(target_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {target_path}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
"""


def get_mid_des(gs2_path, des_path) -> np.ndarray:
    # Match only gs2_<number>_pre.ply
    pattern = re.compile(r"gs2_(\d+)_pre\.ply$")

    pre_files = []

    for filename in os.listdir(gs2_path):
        m = pattern.match(filename)
        if m:
            pre_files.append((int(m.group(1)), filename))

    if not pre_files:
        raise FileNotFoundError(
            f"No gs2_<nbr>_pre.ply files found in {gs2_path}"
        )

    # Sort according to gs2 number
    pre_files.sort(key=lambda x: x[0])

    # Middle element (for even count, take the second middle)
    mid_idx = len(pre_files) // 2

    # Get corresponding gs2 number
    mid_number = pre_files[mid_idx][0]

    # Load des image having the SAME number
    target_path = os.path.join(des_path, f"des{mid_number}.png")
    print("mid nbr", mid_number)
    img = cv2.imread(target_path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(
            f"Corresponding desired image not found: {target_path}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)







def get_last_kf_nbr(keyframes_path):

    pattern = os.path.join(keyframes_path, "keyframe*.png")
    max_nbr = -1

    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        match = re.fullmatch(r"keyframe(\d+)\.png", filename)
        if match:
            max_nbr = max(max_nbr, int(match.group(1)))

    return max_nbr






def get_last_des_nbr(desiredimgs_path):

    pattern = os.path.join(desiredimgs_path, "des*.png")
    max_nbr = -1

    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        match = re.fullmatch(r"des(\d+)\.png", filename)
        if match:
            max_nbr = max(max_nbr, int(match.group(1)))

    return max_nbr




def get_nbr_lnchd(txt_path):

    count = 0

    with open(txt_path, "r") as f:
        for line in f:
            if "launched" in line.lower():
                count += 1

    return count



def get_total_time(txt_path):

    time_pattern = re.compile(r"\b(\d{2}:\d{2}:\d{2}\.\d{3})\b")

    with open(txt_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    first_time = None
    last_time = None

    # Find first timestamp
    for line in lines:
        match = time_pattern.search(line)
        if match:
            first_time = match.group(1)
            break

    # Find last timestamp
    for line in reversed(lines):
        match = time_pattern.search(line)
        if match:
            last_time = match.group(1)
            break

    if first_time is None or last_time is None:
        return None

    t0 = datetime.strptime(first_time, "%H:%M:%S.%f")
    t1 = datetime.strptime(last_time, "%H:%M:%S.%f")

    return (t1 - t0).total_seconds()








def extract_2d_errors_and_velocities(ibvs_infos):
    """
    Extract the pixel error and 6D velocity at every IBVS iteration.

    Assumed velocity order:
        [vx, vy, vz, wx, wy, wz]

    Returns
    -------
    iterations : np.ndarray, shape (K,)
    errors_2d : np.ndarray, shape (K,)
    velocities : np.ndarray, shape (K, 6)
    """

    iterations = sorted(ibvs_infos.keys())

    valid_iterations = []
    errors_2d = []
    velocities = []

    for iteration in iterations:
        iteration_info = ibvs_infos[iteration]

        pxl_error = iteration_info[0]
        velocity = iteration_info[3]

        # Convert pixel error to a scalar
        pxl_error = np.asarray(
            pxl_error,
            dtype=np.float64
        ).squeeze()

        if pxl_error.ndim == 0:
            error_value = float(pxl_error)
        else:
            # If pxl_error is an error vector, plot its L2 norm
            error_value = float(
                np.linalg.norm(pxl_error.reshape(-1))
            )

        # Convert V from (6,), (6,1), etc. into (6,)
        velocity = np.asarray(
            velocity,
            dtype=np.float64
        ).reshape(-1)

        if velocity.size != 6:
            print(
                f"Skipping iteration {iteration}: "
                f"expected velocity with 6 elements, "
                f"but got shape {np.asarray(iteration_info[3]).shape}"
            )
            continue

        if (
            not np.isfinite(error_value)
            or not np.isfinite(velocity).all()
        ):
            print(
                f"Skipping iteration {iteration}: "
                "non-finite error or velocity"
            )
            continue

        valid_iterations.append(iteration)
        errors_2d.append(error_value)
        velocities.append(velocity)

    if not valid_iterations:
        raise ValueError(
            "No valid IBVS errors and velocities were found."
        )

    return (
        np.asarray(valid_iterations),
        np.asarray(errors_2d),
        np.asarray(velocities)
    )






def plot_velocities_vs_iter(
    iterations,
    velocities,
    title,
    save_path
):
    """
    Plot vx, vy, vz, wx, wy and wz in one figure.
    """

    velocities = np.asarray(
        velocities,
        dtype=np.float64
    )

    if velocities.ndim != 2 or velocities.shape[1] != 6:
        raise ValueError(
            "velocities must have shape (N, 6), "
            f"but got {velocities.shape}"
        )

    velocity_names = [
        r"$v_x$",
        r"$v_y$",
        r"$v_z$",
        r"$\omega_x$",
        r"$\omega_y$",
        r"$\omega_z$"
    ]

    colors = [
        "tab:red",
        "tab:green",
        "tab:blue",
        "tab:orange",
        "tab:purple",
        "tab:brown"
    ]

    linestyles = [
        "-",
        "-",
        "-",
        "--",
        "--",
        "--"
    ]

    plt.figure(figsize=(12, 6))

    for component_index in range(6):
        plt.plot(
            iterations,
            velocities[:, component_index],
            label=velocity_names[component_index],
            color=colors[component_index],
            linestyle=linestyles[component_index],
            linewidth=1.3,
            alpha=0.9
        )

    # Horizontal zero line helps reveal sign changes and oscillations
    plt.axhline(
        y=0.0,
        color="black",
        linestyle=":",
        linewidth=1.0,
        alpha=0.7
    )

    plt.xlabel("IBVS iteration")
    plt.ylabel("Camera velocity")
    plt.title(title)

    plt.grid(
        True,
        linestyle="--",
        alpha=0.35
    )

    plt.legend(
        ncol=3,
        loc="best"
    )

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()






def find_first_curve_difference(
    iterations1,
    values1,
    iterations2,
    values2,
    tolerance=1e-10
):
    values1_by_iter = dict(
        zip(iterations1, values1)
    )

    values2_by_iter = dict(
        zip(iterations2, values2)
    )

    common_iterations = sorted(
        set(values1_by_iter)
        & set(values2_by_iter)
    )

    for iteration in common_iterations:
        difference = abs(
            values1_by_iter[iteration]
            - values2_by_iter[iteration]
        )

        if difference > tolerance:
            return iteration, difference

    return None, 0.0









def save_representation_comparison_plots(
    base_dir, scene_name, parent_nbr, lvl, case_nbr
):
    """
    Compare Cloud, Mesh, 3DGS, and Init for one test case.

    Saves plots into:
        <base_dir>/<scene>/<scene>_case<parent>/represenations_comparison/
    """
    representation_settings = {
        "Cloud": {
            "folder_suffix": "cloud",
            "filename": IBVS_INFOS_FILENAME,
            "color": "tab:orange",
        },
        "Mesh": {
            "folder_suffix": "mesh",
            "filename": IBVS_INFOS_FILENAME,
            "color": "tab:green",
        },
        "3DGS": {
            "folder_suffix": "gs",
            "filename": IBVS_INFOS_FILENAME,
            "color": "tab:blue",
        },
        "Init": {
            "folder_suffix": "init",
            "filename": "des0_ibvs_infos.npy",
            "color": "tab:purple",
        },
    }

    parent_case_path = os.path.join(
        base_dir, scene_name, f"{scene_name}_case{parent_nbr}"
    )
    tag = f"{scene_name.capitalize()} L{lvl} case {case_nbr}"
    file_prefix = f"L{lvl}_case{case_nbr}"
    comparison_data = {}

    for representation, settings in representation_settings.items():
        ibvs_infos_path = os.path.join(
            parent_case_path,
            f"level_{lvl}",
            f"case_{case_nbr}_{settings['folder_suffix']}",
            settings["filename"],
        )

        if not os.path.isfile(ibvs_infos_path):
            print(f"[skip] {tag}: {representation} IBVS file not found: {ibvs_infos_path}")
            return

        ibvs_infos = np.load(ibvs_infos_path, allow_pickle=True).item()

        # Init entries are [error, condition, velocity, matches, pose].
        # Add the missing pose_error slot to match the regular six-field layout.
        if representation == "Init":
            normalized_infos = {}

            for iteration, info in ibvs_infos.items():
                info = list(info)

                if len(info) == 5:
                    error, cond_nbr, velocity, matches, pose = info
                    info = [error, None, cond_nbr, velocity, matches, pose]
                elif len(info) != 6:
                    print(
                        f"[skip] {tag}: unexpected Init entry length "
                        f"at iteration {iteration}: {len(info)}"
                    )
                    return

                normalized_infos[iteration] = info

            ibvs_infos = normalized_infos

        iterations, errors_2d, velocities = extract_2d_errors_and_velocities(
            ibvs_infos
        )

        comparison_data[representation] = {
            "iterations": iterations,
            "errors_2d": errors_2d,
            "velocities": velocities,
            "color": settings["color"],
        }

    comparison_dir = os.path.join(
        parent_case_path, "represenations_comparison"
    )
    os.makedirs(comparison_dir, exist_ok=True)

    # Check whether each non-3DGS curve first differs from the 3DGS curve.
    gs_data = comparison_data["3DGS"]

    for representation, representation_data in comparison_data.items():
        if representation == "3DGS":
            continue

        first_difference, difference_value = find_first_curve_difference(
            gs_data["iterations"],
            gs_data["errors_2d"],
            representation_data["iterations"],
            representation_data["errors_2d"],
            tolerance=1e-8,
        )

        if first_difference is None:
            print(
                f"{tag}: {representation} and 3DGS 2D-error curves "
                "are identical over all common iterations."
            )
        else:
            print(
                f"{tag}: {representation} first differs from 3DGS at "
                f"iteration {first_difference}; "
                f"absolute difference = {difference_value:.12e}"
            )

    # 2D feature error comparison
    plt.figure(figsize=(10, 6))

    for representation, data in comparison_data.items():
        plt.plot(
            data["iterations"],
            data["errors_2d"],
            label=representation,
            color=data["color"],
            linewidth=1.1,
            alpha=0.9,
        )

    plt.xlabel("IBVS iteration")
    plt.ylabel("2D feature error")
    plt.title(f"{tag}: 2D feature error")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(comparison_dir, f"{file_prefix}_2d_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    velocity_names = [
        r"$v_x$", r"$v_y$", r"$v_z$",
        r"$\omega_x$", r"$\omega_y$", r"$\omega_z$",
    ]
    velocity_filenames = ["vx", "vy", "vz", "wx", "wy", "wz"]

    # Save each velocity component separately.
    for component_index in range(6):
        plt.figure(figsize=(10, 5))

        for representation, data in comparison_data.items():
            plt.plot(
                data["iterations"],
                data["velocities"][:, component_index],
                label=representation,
                color=data["color"],
                linewidth=1.0,
                alpha=0.9,
            )

        plt.axhline(
            y=0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.7
        )
        plt.xlabel("IBVS iteration")
        plt.ylabel(
            "Linear velocity" if component_index < 3 else "Angular velocity"
        )
        plt.title(f"{tag}: {velocity_names[component_index]}")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                comparison_dir,
                f"{file_prefix}_{velocity_filenames[component_index]}_comparison.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Combined 2x3 velocity figure.
    figure, axes = plt.subplots(
        nrows=2, ncols=3, figsize=(15, 7), sharex=True
    )
    axes = axes.flatten()

    for component_index, axis in enumerate(axes):
        for representation, data in comparison_data.items():
            axis.plot(
                data["iterations"],
                data["velocities"][:, component_index],
                label=representation,
                color=data["color"],
                linewidth=0.9,
                alpha=0.9,
            )

        axis.axhline(
            y=0.0, color="black", linestyle=":", linewidth=0.7, alpha=0.7
        )
        axis.set_title(velocity_names[component_index])
        axis.set_xlabel("IBVS iteration")
        axis.set_ylabel(
            "Linear velocity" if component_index < 3 else "Angular velocity"
        )
        axis.grid(True, linestyle="--", alpha=0.3)

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.suptitle(f"{tag}: camera velocity comparison", y=1.04)
    figure.tight_layout()
    figure.savefig(
        os.path.join(comparison_dir, f"{file_prefix}_6d_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Representation comparison plots saved to: {comparison_dir}")











# Main PIPELINE  =======================================================================================================

def process_case(case_path: str, scene_name: str, case_nbr: int, variant: str = "", level=None, parent_case=1):

    case_path = Path(case_path)
    results_dir = case_path / "results"
    output_txt = results_dir / "results.txt"

    is_init = variant.lower() == "init"
    ibvs_filename = "des0_ibvs_infos.npy" if is_init else IBVS_INFOS_FILENAME
    ibvs_infos_path = case_path / ibvs_filename

    tag = f"{scene_name} case{parent_case} L{level} test{case_nbr}_{variant}"
    print(f"--- Processing {tag} ---")

    # A case marked with error.txt or errors.txt did not converge.
    error_files = [
        path
        for path in case_path.glob("*.txt")
        if path.stem.lower() in {"error", "errors"}
    ]

    results_dir.mkdir(parents=True, exist_ok=True)

    if error_files:
        output_txt.write_text("Error : didn't converge\n")
        print(f"    -> marked as error: {error_files[0].name}")
        return

    if not ibvs_infos_path.is_file():
        print(f"[skip] missing IBVS file: {ibvs_infos_path}")
        return

    # Reset the output so reruns do not append duplicate results.
    output_txt.write_text("")

    gs1_sfm_path = case_path.parent.parent
    gs1_sfm_aligned_path = gs1_sfm_path / "gs1_sfm_aligned"
    desired_dir = case_path / "desired_imgs"

    # The run log is in the case folder; output_txt is in case/results/.
    source_results_txt = case_path / "results.txt"
    total_time = (
        get_total_time(str(source_results_txt))
        if source_results_txt.is_file()
        else None
    )

    add_line_to_text(str(output_txt), "----------- time -------------")
    add_line_to_text(str(output_txt), f"total_time = {total_time}")

    # Desired-image counts and similarity metrics are not available for init.
    if not is_init:
        nbr_des = get_last_des_nbr(str(desired_dir))
        nbr_kfs = get_last_kf_nbr(str(case_path / "sfm" / "images"))

        add_line_to_text(str(output_txt), f"nbr_des = {nbr_des}")
        add_line_to_text(str(output_txt), f"nbr_kfs = {nbr_kfs}")

        # des0 vs masked GT desired image
        gt_des = ImageHandling.load_np_img(str(desired_dir / "gt_des.png"))
        des0 = ImageHandling.load_np_img(str(desired_dir / "des0.png"))
        gt_des_mask_npy = MyUtils.load_frst_dict_np_elmnt(str(desired_dir / "des_masks.npy"))
        gt_des_masked = ImageHandling.apply_mask_on_img(gt_des, gt_des_mask_npy)
        ssim_val, psnr_val, lpips_val = ImageHandling.calc_imgs_sim_metrics(des0, gt_des_masked, 1)

        add_line_to_text(str(output_txt), "----------- imgs similarity -------------")
        add_line_to_text(str(output_txt), f"des0_GTmasked_ssim = {ssim_val}")
        add_line_to_text(str(output_txt), f"des0_GTmasked_psnr = {psnr_val}")
        add_line_to_text(str(output_txt), f"des0_GTmasked_lpips = {lpips_val}")

        # Initial desired image vs GT desired image
        des0 = ImageHandling.load_np_img(str(desired_dir / "des0.png"))
        ssim_val, psnr_val, lpips_val = ImageHandling.calc_imgs_sim_metrics(
            des0, gt_des, 1
        )

        add_line_to_text(str(output_txt), f"initdes_GTdes_ssim = {ssim_val}")
        add_line_to_text(str(output_txt), f"initdes_GTdes_psnr = {psnr_val}")
        add_line_to_text(str(output_txt), f"initdes_GTdes_lpips = {lpips_val}")

        # Final desired image vs GT desired image
        last_des = get_last_des(str(desired_dir))
        gt_des = ImageHandling.load_np_img(str(desired_dir / "gt_des.png"))
        ssim_val, psnr_val, lpips_val = ImageHandling.calc_imgs_sim_metrics(
            last_des, gt_des, 3
        )

        add_line_to_text(str(output_txt), f"finaldes_GTdes_ssim = {ssim_val}")
        add_line_to_text(str(output_txt), f"finaldes_GTdes_psnr = {psnr_val}")
        add_line_to_text(str(output_txt), f"finaldes_GTdes_lpips = {lpips_val}")

    add_line_to_text(str(output_txt), "----------- ibvs infos -------------")

    # Load the IBVS data. Init uses des0_ibvs_infos.npy.
    ibvs_infos = np.load(ibvs_infos_path, allow_pickle=True).item()

    if is_init:
        normalized_infos = {}

        for iteration, info in ibvs_infos.items():
            info = list(info)

            if len(info) == 5:
                error, cond_nbr, velocity, matches, pose = info
                # Insert a placeholder for the missing pose_error field.
                info = [error, None, cond_nbr, velocity, matches, pose]
            elif len(info) != 6:
                raise ValueError(
                    f"Unexpected init IBVS entry at iteration {iteration}: "
                    f"expected 5 fields, got {len(info)}"
                )

            normalized_infos[iteration] = info

        ibvs_infos = normalized_infos

    # Get the initial and target poses from the shared GS1 reconstruction.
    configs_path = case_path / CONFIGS_FILENAME

    if is_init:
        configs_candidates = [
            case_path.parent / f"case_{case_nbr}_{other_variant}" / CONFIGS_FILENAME
            for other_variant in ("gs", "mesh", "cloud")]
        configs_path = next(
            (path for path in configs_candidates if path.is_file()), None,)
        if configs_path is None:
            print(f"[skip] no sibling config found for init case: {case_path}")
            return
    else:
        configs_path = case_path / CONFIGS_FILENAME

    init_img_name = get_init_img_name_frm_txt(str(configs_path))
    gt_img_name = f"level_{level}_{case_nbr}_des0.png"
    gt_img_name = f"level_{level}_12_des0.png"

    T_init = get_img_pose_gs1(init_img_name, str(gs1_sfm_aligned_path))
    T_gt = get_img_pose_gs1(gt_img_name, str(gs1_sfm_aligned_path))

    add_line_to_text(str(output_txt), f"init_img_gs1_name = {init_img_name}")

    R_init, t_init = LinAlgeb.get_Rt_from_homog_matrix(T_init)
    R_gt, t_gt = LinAlgeb.get_Rt_from_homog_matrix(T_gt)

    R_diff_gt_init = LinAlgeb.rotation_diff(R_gt, R_init)
    t_diff_gt_init = LinAlgeb.pose_distance(t_gt, t_init)

    add_line_to_text(str(output_txt), f"R_diff_gt_init_deg = {R_diff_gt_init}")
    add_line_to_text(str(output_txt), f"t_diff_gt_init = {t_diff_gt_init}")

    # IBVS iteration count and average condition number
    total_nbr_itrs = get_total_nbr_of_itrs(ibvs_infos)
    avrg_condit_nbr = get_avrg_condit_nbr(ibvs_infos)

    add_line_to_text(str(output_txt), f"total_nbr_itrs = {total_nbr_itrs}")
    add_line_to_text(str(output_txt), f"avrg_condit_nbr = {avrg_condit_nbr}")

    # Rotation and translation errors for every IBVS iteration
    iters_full, R_diffs_full, t_diffs_full = compute_pose_errors_per_iter(
        ibvs_infos,
        R_gt,
        t_gt,
        t_diff_gt_init,
    )

    np.save(results_dir / "R_diff_all.npy", R_diffs_full)
    np.save(results_dir / "t_diff_all.npy", t_diffs_full)

    last_iter_full = iters_full[-1]
    add_line_to_text(str(output_txt), f"last_itr = {last_iter_full}")
    add_line_to_text(
        str(output_txt),
        f"R_diff_last_itr_deg = {R_diffs_full[last_iter_full]}",
    )
    add_line_to_text(
        str(output_txt),
        f"t_diff_last_itr_norm = {t_diffs_full[last_iter_full]}",
    )

    # 2D feature errors and six camera-velocity components
    iters_ibvs, errors_2d, velocities_6d = extract_2d_errors_and_velocities(
        ibvs_infos
    )

    errors_2d_dict = dict(zip(iters_ibvs.tolist(), errors_2d.tolist()))

    plot_error_vs_iter(
        iters_ibvs,
        errors_2d_dict,
        ylabel="2D feature error",
        title=f"{tag} - 2D feature error vs iteration",
        save_path=str(results_dir / "error_2d_vs_iter.png"),
    )

    plot_velocities_vs_iter(
        iters_ibvs,
        velocities_6d,
        title=f"{tag} - 6D camera velocity vs iteration",
        save_path=str(results_dir / "velocity_6d_vs_iter.png"),
    )

    # Pose-error plots
    plot_error_vs_iter(
        iters_full,
        R_diffs_full,
        ylabel="R diff (deg)",
        title=f"{tag} - R diff vs iter (full run)",
        save_path=str(results_dir / "R_diff_full.png"),
    )

    plot_error_vs_iter(
        iters_full,
        t_diffs_full,
        ylabel="Normalized t diff",
        title=f"{tag} - t diff vs iter (full run)",
        save_path=str(results_dir / "t_diff_full.png"),
    )

    print(f"    -> done. Results saved to {results_dir}")



















def get_lvl_des_paths(case_path, levels=None, tst_nbrs=None, des_type="des0"):
    
    """
    Return selected desired-image paths.

    Examples
    --------
    levels=1, tst_nbrs=3, des_type="gt_des"
    levels=[1, 2], tst_nbrs=[1, 4, 7], des_type="des0"
    levels=None, tst_nbrs=None -> get everything from levels 1, 2 and 3
    """
    if des_type not in ("des0", "gt_des"):
        raise ValueError("des_type must be 'des0' or 'gt_des'.")

    if levels is None:
        levels = [1, 2, 3]
    elif isinstance(levels, int):
        levels = [levels]

    if isinstance(tst_nbrs, int):
        tst_nbrs = [tst_nbrs]

    des_paths = []

    for diff_lvl in levels:
        level_path = Path(f"{case_path}/all_lvl_des") / f"level_{diff_lvl}"

        if tst_nbrs is None:
            level_paths = sorted(
                level_path.glob(f"*_{des_type}.png"),
                key=lambda path: int(path.stem.split("_")[0])
            )
        else:
            level_paths = [
                level_path / f"{tst_nbr}_{des_type}.png"
                for tst_nbr in tst_nbrs
            ]

        for path in level_paths:
            if path.exists():
                des_paths.append(str(path))
            else:
                print(f"Image not found: {path}")

    return des_paths














def get_level_and_case(img_path):
    img_path = Path(img_path)

    diff_lvl = int(img_path.parent.name.split("_")[-1])
    case_nbr = int(img_path.stem.split("_")[0])

    return diff_lvl, case_nbr













# ----------------------------------------------------------------------
# find the 3 plys in gs2s: '0' -> gs2_0_pre.ply, '150' -> gs2_1.ply, last -> highest gs2_<nbr>.ply
# ----------------------------------------------------------------------
def find_gs2_plys(gs2s_dir):
    gs2s_dir = Path(gs2s_dir)

    ply_0 = gs2s_dir / "gs2_0_pre.ply"
    ply_150 = gs2s_dir / "gs2_1.ply"

    numbered = {}
    for p in gs2s_dir.glob("gs2_*.ply"):
        m = re.fullmatch(r"gs2_(\d+)\.ply", p.name)   # only 'nbr.ply', excludes *_pre.ply
        if m:
            numbered[int(m.group(1))] = p
    ply_last = numbered[max(numbered)] if numbered else None

    return {"0": ply_0, "150": ply_150, "last": ply_last}


# ----------------------------------------------------------------------
# save helper (handles torch tensors / float imgs). Set rgb=True if your imgs are RGB
# ----------------------------------------------------------------------
def save_img(img, path, rgb=True):
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if rgb and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img)




# ----------------------------------------------------------------------
# Everything one variant needs, loaded only from its own case folder
# ----------------------------------------------------------------------

def load_variant_data(case_path):
    case_path = Path(case_path)

    # getting intrins_gs2
    gs2_recon = PosesHandling.get_recons(f"{case_path}/sfm/")
    gs2_sfm_K= PosesHandling.get_cam_matrix(gs2_recon)
    gs2_intrins = GaussiansHandling.turn_cam_matrix_to_gsplat_format(gs2_sfm_K)

    # getting init and des0 pose
    init_pose_gs2 = PosesHandling.get_img_sfm_pose(gs2_recon, "init_img.png")
    init_des_trans = MyUtils.load_frst_dict_np_elmnt(f"{case_path}/init_des_trans.npy")
    des_pose_gs2 = LinAlgeb.transform_pose(init_des_trans, init_pose_gs2)
    # load the transf, load the init pose from sfm and apply the transfo on it

    plys = find_gs2_plys(case_path / "gs2s")
    for stage, p in plys.items():
        if p is None or not Path(p).is_file():
            print(f"[skip] missing ply ({stage}) in {case_path}: {p}")
            return None

    return {
        "gaussians": {s: GaussiansHandling.load_gaussians_from_ply(str(plys[s]))
                      for s in ("0", "150", "last")},
        "intrins": gs2_intrins,
        "views": {
            "des":  {"pose": des_pose_gs2,
                     "gt": ImageHandling.load_np_img(f"{case_path}/desired_imgs/gt_des.png")},
            "init": {"pose": init_pose_gs2 ,
                     "gt": ImageHandling.load_np_img(f"{case_path}/sfm/images/init_img.png")}}}






# ----------------------------------------------------------------------
# render + save for one (scene, parent, level, test case)
# ----------------------------------------------------------------------

def save_gs2_renders_comparison(base_dir, out_dir, scene_name, parent_nbr, lvl, case_nbr,
                                variants=("old", "gs")):
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parent_case_path = base_dir / scene_name / f"{scene_name}_case{parent_nbr}"
    tag = f"{scene_name}_{lvl}_{case_nbr}"

    # load anyimg from the case to get H,W
    img = ImageHandling.load_np_img(f"{parent_case_path}/level_1/case_1_gs/desired_imgs/des0.png")
    CAM_H, CAM_W = ImageHandling.get_hw_of_img(img)

    # each variant loads its own data, nothing is shared
    data = {}
    for v in variants:
        case_path = parent_case_path / f"level_{lvl}" / f"case_{case_nbr}_{v}"
        d = load_variant_data(case_path)
        if d is None:
            print(f"[skip] {tag}: variant '{v}' incomplete")
            return
        data[v] = d

    # order: [des | init] -> [0 | 150 | last] -> renders (old, gs) then GTs (old, gs)
    idx = 1
    for view_name in ("des", "init"):
        for stage in ("0", "150", "last"):

            for v in variants:
                d = data[v]
                img, _ = GaussiansHandling.render_gs_pic(
                    *d["gaussians"][stage], d["views"][view_name]["pose"],
                    d["intrins"], CAM_W, CAM_H)
                save_img(img, out_dir / f"{tag}_{idx:02d}_{view_name}_render{stage}_{v}.png")
                idx += 1

            for v in variants:
                save_img(data[v]["views"][view_name]["gt"],
                         out_dir / f"{tag}_{idx:02d}_{view_name}_GT_render{stage}_{v}.png")
                idx += 1

    print(f"[ok] {tag} -> {out_dir}")






def compute_renders_metrics(out_dir, scene_names, parent_nbrs, levels, test_nbrs,
                            variants=("old", "gs"), out_txt=None):
    """
    Reads the saved renders + GT imgs from out_dir, computes ssim/psnr/lpips per case,
    averages over all cases and scenes, and writes one txt.
    Must use the same variants order as save_gs2_renders_comparison (it defines the file indices).
    """
    out_dir = Path(out_dir)
    out_txt = Path(out_txt) if out_txt else out_dir / "renders_metrics.txt"
    names = ["ssim", "psnr", "lpips"]

    # rebuild the index of each file: same loop order as save_gs2_renders_comparison
    def file_index(view_name, stage, kind, v):
        idx = 1
        for vw in ("des", "init"):
            for st in ("0", "150", "last"):
                for vv in variants:               # renders
                    if (vw, st, "render", vv) == (view_name, stage, kind, v):
                        return idx
                    idx += 1
                for vv in variants:               # GTs
                    if (vw, st, "GT", vv) == (view_name, stage, kind, v):
                        return idx
                    idx += 1

    per_case = []   # one dict per case: {(view, stage, variant): (ssim, psnr, lpips)}

    for scene_name in scene_names:
        for parent_nbr in parent_nbrs:
            for lvl in levels:
                for case_nbr in test_nbrs:
                    tag = f"{scene_name}_{lvl}_{case_nbr}"
                    case_metrics = {}
                    try:
                        for view_name in ("des", "init"):
                            for stage in ("0", "150", "last"):
                                for v in variants:
                                    r_idx = file_index(view_name, stage, "render", v)
                                    g_idx = file_index(view_name, stage, "GT", v)
                                    r_path = out_dir / f"{tag}_{r_idx:02d}_{view_name}_render{stage}_{v}.png"
                                    g_path = out_dir / f"{tag}_{g_idx:02d}_{view_name}_GT_render{stage}_{v}.png"

                                    render_np = ImageHandling.load_np_img(str(r_path))
                                    gt_np = ImageHandling.load_np_img(str(g_path))
                                    s, p, l = ImageHandling.calc_imgs_sim_metrics(render_np, gt_np, 1)
                                    case_metrics[(view_name, stage, v)] = (float(s), float(p), float(l))
                        per_case.append(case_metrics)
                    except Exception as e:
                        print(f"[skip] {tag}: {e}")

    with open(out_txt, "w") as f:
        f.write(f"nbr of cases used: {len(per_case)}\n\n")
        for view_name in ("init", "des"):
            for stage in ("0", "150", "last"):
                for i, name in enumerate(names):
                    cells = []
                    for v in ("gs", "old"):
                        arr = [m[(view_name, stage, v)][i] for m in per_case
                               if (view_name, stage, v) in m]
                        cells.append(f"{np.mean(arr):.4f} ({v})" if arr else f"nan ({v})")
                    f.write(f"{name} render {view_name} {stage} : " + " | ".join(cells) + "\n")
                f.write("\n")

    print(f"metrics saved to {out_txt}")




LEVEL_COLORS = {
    1: {"ibvs": (0.10, 0.35, 0.90), "gt": (0.55, 0.70, 1.00)},
    2: {"ibvs": (0.10, 0.65, 0.25), "gt": (0.55, 0.85, 0.60)},
    3: {"ibvs": (0.90, 0.25, 0.10), "gt": (1.00, 0.60, 0.45)}}








def show_all_scenes_trajectories(
    base_dir,
    scene_names,
    parent_nbr=1,
    levels=(1, 2, 3, 4),
    test_nbrs=(1, 2),
    align="translate",
    normalize=False,
):
    """
    Show up to two converged GS trajectories per level, per scene.

    align="translate": move each initial pose to the origin; keep its orientation.
    align="relative": express each trajectory in its initial camera frame.
    normalize=True: scale each case so its initial-to-GT distance is 1.
    """
    from pathlib import Path
    import numpy as np
    import open3d as o3d

    colors = {
        1: {"ibvs": (0.10, 0.35, 0.90), "gt": (0.55, 0.70, 1.00)},
        2: {"ibvs": (0.10, 0.65, 0.25), "gt": (0.55, 0.85, 0.60)},
        3: {"ibvs": (0.90, 0.25, 0.10), "gt": (1.00, 0.60, 0.45)},
        4: {"ibvs": (0.60, 0.20, 0.75), "gt": (0.82, 0.60, 0.92)},
    }

    GT_CASE_IDS = {
    ("livingroom", 4, 1): 5,
    ("livingroom", 4, 2): 21,
    ("playroom", 4, 1): 3,
    ("playroom", 4, 2): 4,
    ("thehouse", 4, 1): 1,
    ("thehouse", 4, 2): 12,
}

    geometries = []
    all_positions = [np.zeros(3)]
    gt_markers = []
    gt_poses = []
    last_poses = []
    miss_segments = []

    # At most two test cases per level, per scene.
    selected_test_nbrs = list(test_nbrs)[:2]

    for scene_name in scene_names:
        parent_case_path = (
            Path(base_dir) / scene_name / f"{scene_name}_case{parent_nbr}"
        )

        # Try the primary aligned reconstruction, then the fallback.
        aligned_paths = [
            str(parent_case_path / "gs1_sfm_aligned"),
            str(parent_case_path / "gs1_sfm_aligned_2"),
        ]

        for lvl in levels:
            if lvl not in colors:
                print(f"[skip] unsupported level {lvl}")
                continue

            for case_nbr in selected_test_nbrs:
                case_path = (
                    parent_case_path
                    / f"level_{lvl}"
                    / f"case_{case_nbr}_gs"
                )
                tag = f"{scene_name} L{lvl} case{case_nbr} (gs)"
                npy_path = case_path / IBVS_INFOS_FILENAME
                results_dir = case_path / "results"

                # Skip cases with missing or empty results folders.
                if not results_dir.is_dir():
                    print(f"[skip] {tag}: missing results folder")
                    continue

                result_entries = list(results_dir.iterdir())

                if not result_entries:
                    print(f"[skip] {tag}: empty results folder")
                    continue

                # A results folder containing only results.txt means failure.
                if (
                    len(result_entries) == 1
                    and result_entries[0].is_file()
                    and result_entries[0].name == "results.txt"
                ):
                    print(f"[skip] {tag}: did not converge")
                    continue

                if not npy_path.is_file():
                    print(f"[skip] {tag}: missing {npy_path}")
                    continue

                try:
                    data = np.load(npy_path, allow_pickle=True).item()
                except Exception as exc:
                    print(f"[skip] {tag}: could not load IBVS data: {exc}")
                    continue

                # Get the GT pose from the first reconstruction containing it.
                gt_pose = None
                pose_errors = []

                for aligned_path in aligned_paths:
                    gt_case_id = GT_CASE_IDS.get((scene_name, lvl, case_nbr), case_nbr)
                    gt_img_name = f"level_{lvl}_{gt_case_id}_des0.png"
                    try:
                        candidate_pose = np.asarray(
                            get_img_pose_gs1(
                                gt_img_name,
                                aligned_path,
                            ),
                            dtype=float,
                        )

                        if candidate_pose.shape == (4, 4):
                            gt_pose = candidate_pose
                            break

                        pose_errors.append(
                            f"{aligned_path}: returned shape "
                            f"{candidate_pose.shape}"
                        )

                    except Exception as exc:
                        pose_errors.append(f"{aligned_path}: {exc}")

                if gt_pose is None:
                    print(
                        f"[skip] {tag}: GT pose not found; "
                        + " | ".join(pose_errors)
                    )
                    continue

                poses = []

                for iteration in sorted(data.keys(), key=lambda key: int(key)):
                    try:
                        pose = np.asarray(
                            data[iteration][POSE_GS1_IDX],
                            dtype=float,
                        )
                    except (IndexError, TypeError, ValueError) as exc:
                        print(
                            f"[skip] {tag}: invalid pose at iteration "
                            f"{iteration}: {exc}"
                        )
                        continue

                    if pose.shape == (4, 4):
                        poses.append(pose)

                if not poses:
                    print(f"[skip] {tag}: no valid IBVS poses")
                    continue

                # Move the initial camera pose to the origin.
                T0 = poses[0]

                if align == "relative":
                    T_align = np.linalg.inv(T0)
                elif align == "translate":
                    T_align = np.eye(4)
                    T_align[:3, 3] = -T0[:3, 3]
                else:
                    raise ValueError(
                        "align must be either 'translate' or 'relative'"
                    )

                poses = [T_align @ pose for pose in poses]
                gt_pose = T_align @ gt_pose

                if normalize:
                    scale = np.linalg.norm(gt_pose[:3, 3])
                    if scale > 1e-12:
                        for pose in poses:
                            pose[:3, 3] /= scale
                        gt_pose[:3, 3] /= scale

                positions = np.array([pose[:3, 3] for pose in poses])
                gt_position = gt_pose[:3, 3]

                all_positions.extend(positions)
                all_positions.append(gt_position)

                color = colors[lvl]["ibvs"]

                # Draw the connected trajectory.
                if len(positions) >= 2:
                    lines = [
                        [i, i + 1]
                        for i in range(len(positions) - 1)
                    ]

                    trajectory = o3d.geometry.LineSet()
                    trajectory.points = o3d.utility.Vector3dVector(positions)
                    trajectory.lines = o3d.utility.Vector2iVector(lines)
                    trajectory.colors = o3d.utility.Vector3dVector(
                        np.tile(color, (len(lines), 1))
                    )
                    geometries.append(trajectory)

                # Draw the individual camera positions.
                trajectory_points = o3d.geometry.PointCloud()
                trajectory_points.points = o3d.utility.Vector3dVector(
                    positions
                )
                trajectory_points.paint_uniform_color(color)
                geometries.append(trajectory_points)

                last_poses.append(poses[-1])
                gt_poses.append(gt_pose)
                gt_markers.append((gt_position, colors[lvl]["gt"]))
                miss_segments.append((positions[-1], gt_position))

                init_to_gt = np.linalg.norm(gt_position)
                end_to_gt = np.linalg.norm(positions[-1] - gt_position)
                remaining = 100 * end_to_gt / max(init_to_gt, 1e-12)

                print(
                    f"{tag}: init->GT {init_to_gt:.4f} | "
                    f"end->GT {end_to_gt:.4f} | "
                    f"remaining {remaining:.1f}%"
                )

    if not geometries:
        print("No converged GS trajectories to display.")
        return

    # Scale markers and axes based on the full scene extent.
    points = np.asarray(all_positions)
    span = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    radius = max(span * 0.008, 0.001)
    axis_size = max(span * 0.03, 0.02)

    def make_sphere(position, color, sphere_radius):
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=sphere_radius
        )
        sphere.translate(position)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        return sphere

    # Shared initial point at the origin.
    geometries.append(
        make_sphere(np.zeros(3), (0.0, 0.0, 0.0), radius * 2)
    )

    # GT positions use lighter shades of their level colors.
    for position, color in gt_markers:
        geometries.append(make_sphere(position, color, radius))

    # Connect each final pose to its GT position.
    if miss_segments:
        segment_points = np.array(
            [point for segment in miss_segments for point in segment]
        )
        segment_lines = [
            [2 * i, 2 * i + 1]
            for i in range(len(miss_segments))
        ]

        missing_segments = o3d.geometry.LineSet()
        missing_segments.points = o3d.utility.Vector3dVector(segment_points)
        missing_segments.lines = o3d.utility.Vector2iVector(segment_lines)
        missing_segments.colors = o3d.utility.Vector3dVector(
            np.tile((0.5, 0.5, 0.5), (len(segment_lines), 1))
        )
        geometries.append(missing_segments)

    # Show world, GT, and final camera pose axes.
    geometries.append(
        MeshHandling.get_pose_axis(np.eye(4), size=axis_size)
    )

    for gt_pose in gt_poses:
        geometries.append(
            MeshHandling.get_pose_axis(gt_pose, size=axis_size)
        )

    for last_pose in last_poses:
        geometries.append(
            MeshHandling.get_pose_axis(last_pose, size=axis_size * 0.8)
        )

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Converged GS trajectories by level",
        width=1400,
        height=900,
    )













def compute_semi_final_results(
    base_dir,
    scene_names,
    parent_case_nbrs,
    levels,
    test_nbrs,
    variants,
    metrics_list=METRICS,
    out_dir=None,
):
    base_dir = Path(base_dir)
    out_dir = Path(out_dir) if out_dir else base_dir

    for scene_name in scene_names:
        for variant in variants:
            # Keep counts and metric values separately for each level.
            level_data = {
                lvl: {
                    "converged": 0,
                    "failed": 0,
                    "metrics": {key: [] for key in metrics_list},
                }
                for lvl in levels
            }

            for parent_nbr in parent_case_nbrs:
                parent_case_path = (
                    base_dir / scene_name / f"{scene_name}_case{parent_nbr}"
                )

                for lvl in levels:
                    for case_nbr in test_nbrs:
                        case_path = (
                            parent_case_path
                            / f"level_{lvl}"
                            / f"case_{case_nbr}_{variant}"
                        )

                        # Ignore cases that do not exist.
                        if not case_path.is_dir():
                            print(f"[ignore] missing case: {case_path}")
                            continue

                        results_dir = case_path / "results"

                        # Ignore cases without a results folder.
                        if not results_dir.is_dir():
                            print(f"[ignore] missing results folder: {results_dir}")
                            continue

                        entries = list(results_dir.iterdir())

                        # Ignore an existing but empty results folder.
                        if not entries:
                            print(f"[ignore] empty results folder: {results_dir}")
                            continue

                        results_txt = results_dir / "results.txt"

                        # A results folder containing only results.txt means failure.
                        if (
                            len(entries) == 1
                            and entries[0].is_file()
                            and entries[0].name == "results.txt"
                        ):
                            level_data[lvl]["failed"] += 1
                            print(f"[failed] {case_path}")
                            continue

                        # Any other nonempty results folder means convergence.
                        level_data[lvl]["converged"] += 1
                        print(f"[converged] {case_path}")

                        # Count the case as converged even if its metrics file is absent.
                        if not results_txt.is_file():
                            print(f"[skip metrics] missing {results_txt}")
                            continue

                        case_metrics = load_results(
                            results_txt,
                            metrics_list=metrics_list,
                        )

                        for key, value in case_metrics.items():
                            level_data[lvl]["metrics"][key].append(value)

            # Combine the level results to get scene-wide counts and averages.
            overall = {
                "converged": sum(level_data[lvl]["converged"] for lvl in levels),
                "failed": sum(level_data[lvl]["failed"] for lvl in levels),
                "metrics": {key: [] for key in metrics_list},
            }

            for lvl in levels:
                for key in metrics_list:
                    overall["metrics"][key].extend(
                        level_data[lvl]["metrics"][key]
                    )

            # Store each scene's summaries outside its level folders.
            scene_out_dir = out_dir / scene_name
            scene_out_dir.mkdir(parents=True, exist_ok=True)

            out_path = scene_out_dir / f"{variant}_results_{scene_name}.txt"

            with open(out_path, "w") as f:
                f.write(f"============== {variant.upper()} RESULTS: {scene_name} ==============\n")
                self_write_group(f, "all levels", overall, metrics_list)

                for lvl in levels:
                    f.write(f"\n\n============== LEVEL {lvl} ==============\n")
                    self_write_group(
                        f,
                        f"level_{lvl}",
                        level_data[lvl],
                        metrics_list,
                    )

            print(
                f"[{scene_name} / {variant}] "
                f"{overall['converged']} converged, "
                f"{overall['failed']} failed -> {out_path}"
            )


def self_write_group(file_obj, label, group, metrics_list):
    file_obj.write(f"\n----------- {label} -------------\n")
    file_obj.write(f"number_converged_cases = {group['converged']}\n")
    file_obj.write(f"number_failed_cases = {group['failed']}\n")

    file_obj.write("\n----------- des quality -------------\n")
    for key in DES_QUALITY_METRICS:
        if key not in metrics_list:
            continue

        values = group["metrics"][key]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            file_obj.write(f"{key} = {mean:.3f} ± {std:.3f}\n")

    file_obj.write("\n----------- ibvs infos -------------\n")
    for key in metrics_list:
        if key in DES_QUALITY_METRICS:
            continue

        values = group["metrics"][key]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            file_obj.write(f"{key} = {mean:.3f} ± {std:.3f}\n")









# _________________________________________ this only for the final results ______________________________________________

import numpy as np
from pathlib import Path


METRICS = [
    "initdes_GTdes_ssim",
    "initdes_GTdes_psnr",
    "initdes_GTdes_lpips",
    "finaldes_GTdes_ssim",
    "finaldes_GTdes_psnr",
    "finaldes_GTdes_lpips",
    "nbr_des",
    "nbr_kfs",
    "total_time",
    "R_diff_gt_init_deg",
    "t_diff_gt_init",
    "total_nbr_itrs",
    "avrg_condit_nbr",
    "R_diff_last_itr_deg",
    "t_diff_last_itr_norm",
]

DES_QUALITY_METRICS = [
    "initdes_GTdes_ssim",
    "initdes_GTdes_psnr",
    "initdes_GTdes_lpips",
    "finaldes_GTdes_ssim",
    "finaldes_GTdes_psnr",
    "finaldes_GTdes_lpips",
]


def load_results(txt_path, metrics_list=METRICS):
    metrics = {}

    with open(txt_path, "r") as file:
        for line in file:
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()

            if key not in metrics_list:
                continue

            try:
                number = float(value.strip())
            except ValueError:
                continue

            if np.isfinite(number):
                metrics[key] = number

    return metrics


def _new_result_group(metrics_list):
    return {
        "converged": 0,
        "failed": 0,
        "metrics": {key: [] for key in metrics_list},
    }



def _write_result_group(file, title, group, metrics_list):
    file.write(f"\n----------- {title} -------------\n")
    file.write(f"number_converged_cases = {group['converged']}\n")
    file.write(f"number_failed_cases = {group['failed']}\n")

    total_cases = group["converged"] + group["failed"]
    success_percentage = (
        100.0 * group["converged"] / total_cases
        if total_cases > 0
        else 0.0
    )
    file.write(f"success_percentage = {success_percentage:.3f}%\n")

    file.write("\n----------- des quality -------------\n")
    for key in DES_QUALITY_METRICS:
        if key not in metrics_list:
            continue

        values = group["metrics"][key]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            file.write(f"{key} = {mean:.3f} ± {std:.3f}\n")

    file.write("\n----------- ibvs infos -------------\n")
    for key in metrics_list:
        if key in DES_QUALITY_METRICS:
            continue

        values = group["metrics"][key]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            file.write(f"{key} = {mean:.3f} ± {std:.3f}\n")



def compute_final_results(
    base_dir,
    scene_names,
    parent_case_nbrs,
    levels,
    test_nbrs,
    variants,
    metrics_list=METRICS,
    out_dir=None,
):
    base_dir = Path(base_dir)
    out_dir = Path(out_dir) if out_dir else base_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        level_groups = {
            lvl: _new_result_group(metrics_list)
            for lvl in levels
        }

        for scene_name in scene_names:
            for parent_nbr in parent_case_nbrs:
                parent_case_path = (
                    base_dir
                    / scene_name
                    / f"{scene_name}_case{parent_nbr}"
                )

                for lvl in levels:
                    for case_nbr in test_nbrs:
                        case_path = (
                            parent_case_path
                            / f"level_{lvl}"
                            / f"case_{case_nbr}_{variant}"
                        )
                        results_dir = case_path / "results"

                        if not case_path.is_dir():
                            print(f"[ignore] missing case: {case_path}")
                            continue

                        if not results_dir.is_dir():
                            print(f"[ignore] missing results folder: {results_dir}")
                            continue

                        entries = list(results_dir.iterdir())

                        if not entries:
                            print(f"[ignore] empty results folder: {results_dir}")
                            continue

                        results_txt = results_dir / "results.txt"

                        # Only results.txt means the case failed.
                        if (
                            len(entries) == 1
                            and entries[0].is_file()
                            and entries[0].name == "results.txt"
                        ):
                            level_groups[lvl]["failed"] += 1
                            print(f"[failed] {scene_name} L{lvl} case{case_nbr} {variant}")
                            continue

                        # Any other nonempty results folder means the case converged.
                        level_groups[lvl]["converged"] += 1
                        print(f"[converged] {scene_name} L{lvl} case{case_nbr} {variant}")

                        if not results_txt.is_file():
                            print(f"[skip metrics] missing {results_txt}")
                            continue

                        case_metrics = load_results(
                            results_txt,
                            metrics_list=metrics_list,
                        )

                        for key, value in case_metrics.items():
                            level_groups[lvl]["metrics"][key].append(value)

        # Combine level counts and metric values for the all-level summary.
        all_levels = _new_result_group(metrics_list)

        for lvl in levels:
            group = level_groups[lvl]
            all_levels["converged"] += group["converged"]
            all_levels["failed"] += group["failed"]

            for key in metrics_list:
                all_levels["metrics"][key].extend(group["metrics"][key])

        out_path = out_dir / f"{variant}_results.txt"

        with open(out_path, "w") as file:
            file.write(
                f"============== {variant.upper()} RESULTS: ALL SCENES ==============\n"
            )
            _write_result_group(
                file,
                "all levels",
                all_levels,
                metrics_list,
            )

            for lvl in levels:
                file.write(f"\n\n============== LEVEL {lvl} ==============\n")
                _write_result_group(
                    file,
                    f"level_{lvl}",
                    level_groups[lvl],
                    metrics_list,
                )

        print(
            f"[{variant}] "
            f"{all_levels['converged']} converged, "
            f"{all_levels['failed']} failed -> {out_path}"
        )











# =========================================================================
# MAIN LOOP
# =========================================================================


def main():


    from pathlib import Path

    BASE_DIR = Path("/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test")

    SCENE_NAMES = ["playroom", "livingroom", "thehouse"] 
    PARENT_CASE_NBRS = [1]      # the 'scene_name'_case1 / case2 / case3 folders
    LEVELS = [1, 2, 3, 4]
    TEST_NBRS = [1, 2]
    VARIANTS = ["gs", "mesh", "cloud", "init" ] 

    
    # IBVS nd des img results for each case_______________________________________________________
    """
    for scene_name in SCENE_NAMES:
        for parent_nbr in PARENT_CASE_NBRS:
            parent_case_path = BASE_DIR / scene_name / f"{scene_name}_case{parent_nbr}"

            if not parent_case_path.is_dir():
                print(f"[skip] missing parent case: {parent_case_path}")
                continue

            for lvl in LEVELS:
                for case_nbr in TEST_NBRS:
                    for variant in VARIANTS:
                        case_folder = f"case_{case_nbr}_{variant}"
                        case_path = parent_case_path / f"level_{lvl}" / case_folder

                        if not case_path.is_dir():
                            print(f"[skip] missing case: {case_path}")
                            continue

                        process_case(str(case_path), scene_name, case_nbr, variant=variant, level=lvl, parent_case=parent_nbr)    
    #return
    """


    # IBVS nd des img results for all cases_______________________________________________________

    #"""
    # final results, one txt per representation
    compute_semi_final_results(BASE_DIR, SCENE_NAMES, PARENT_CASE_NBRS, LEVELS, TEST_NBRS, VARIANTS)
    
    #return
    #"""


    # Compa results for the 3 representations_________________________________________________________

    """
    # Cloud/Mesh/3DGS comparison plots
    for scene_name in SCENE_NAMES:
        for parent_nbr in PARENT_CASE_NBRS:
            for lvl in LEVELS:
                for case_nbr in TEST_NBRS:
                    save_representation_comparison_plots(
                        base_dir=BASE_DIR,
                        scene_name=scene_name,
                        parent_nbr=parent_nbr,
                        lvl=lvl,
                        case_nbr=case_nbr)
    #return            
    """



    # GS renderings compa_________________________________________________________
    
    """
    OUT_DIR = Path(BASE_DIR) / "gs2_renders_comparison"   # change to your output path
    for scene_name in SCENE_NAMES:
        for parent_nbr in PARENT_CASE_NBRS:
            for lvl in LEVELS:
                for case_nbr in TEST_NBRS:
                    save_gs2_renders_comparison(
                        BASE_DIR, OUT_DIR, scene_name, parent_nbr, lvl, case_nbr)
    """
    
    # step 2: metrics from the saved imgs (fast, rerun as often as you want)
    """
    compute_renders_metrics(OUT_DIR, SCENE_NAMES, PARENT_CASE_NBRS, LEVELS, TEST_NBRS, variants=("old", "gs"))
    return
    """


    # showing the 3D trajectories for GS !! __________________________________________________
    """
    show_all_scenes_trajectories(BASE_DIR, SCENE_NAMES, parent_nbr=1,
                         levels=[1, 2, 3, 4], test_nbrs=[1, 2],
                         align="translate", normalize=False)
    return
    """


    compute_final_results(
    BASE_DIR,
    SCENE_NAMES,
    PARENT_CASE_NBRS,
    LEVELS,
    TEST_NBRS,
    VARIANTS,
    out_dir="/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test")



if __name__ == "__main__":
    main()