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
SCENE_NAMES = ["kitchen", "playroom", "thehouse", "woodroom", "livingroom"]
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
    "des0_GTmasked_ssim",
    "des0_GTmasked_psnr",
    "des0_GTmasked_lpips",
    "initdes_GTdes_ssim",
    "initdes_GTdes_psnr",
    "initdes_GTdes_lpips",
    "middes_GTdes_ssim",
    "middes_GTdes_psnr",
    "middes_GTdes_lpips",
    "finaldes_GTdes_ssim",
    "finaldes_GTdes_psnr",
    "finaldes_GTdes_lpips",
    "nbr_des",
    "nbr_kfs",
    "total_time"]


"""
"avrg_condit_nbr_des0",

"""




def load_results(txt_path):
    metrics = {}

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()

            # I considere when calling this function that the rslts.txt is "key = value"
            if "=" in line and ":" not in line:
                key, value = line.split("=", 1)
                key = key.strip()
                if key in METRICS:
                    metrics[key] = float(value)

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










import os
import numpy as np
import matplotlib.pyplot as plt


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
    base_dir,
    scene_name,
    case_nbr
):
    """
    Compare Cloud, Mesh and 3DGS for one scenario.

    Reads:
        <scene>_case<case>_cloud/ibvs_infos.npy
        <scene>_case<case>_mesh/ibvs_infos.npy
        <scene>_case<case>_gs/ibvs_infos.npy

    Saves into:
        <base_dir>/<scene_name>/represenations_comparison/
    """

    representation_settings = {
        "Cloud": {
            "folder_suffix": "cloud",
            "color": "tab:orange"
        },
        "Mesh": {
            "folder_suffix": "mesh",
            "color": "tab:green"
        },
        "3DGS": {
            "folder_suffix": "gs",
            "color": "tab:blue"
        }
    }

    comparison_data = {}

    # ------------------------------------------------------------------
    # Load IBVS data for the three representations
    # ------------------------------------------------------------------

    for representation, settings in representation_settings.items():
        case_folder_name = (
            f"{scene_name}_case{case_nbr}_"
            f"{settings['folder_suffix']}"
        )

        ibvs_infos_path = os.path.join(
            base_dir,
            scene_name,
            case_folder_name,
            IBVS_INFOS_FILENAME
        )

        if not os.path.exists(ibvs_infos_path):
            raise FileNotFoundError(
                f"IBVS information not found: {ibvs_infos_path}"
            )

        ibvs_infos = np.load(
            ibvs_infos_path,
            allow_pickle=True
        ).item()

        iterations, errors_2d, velocities = (
            extract_2d_errors_and_velocities(
                ibvs_infos
            )
        )

        comparison_data[representation] = {
            "iterations": iterations,
            "errors_2d": errors_2d,
            "velocities": velocities,
            "color": settings["color"]
        }

    # ------------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------------

    comparison_dir = os.path.join(
        base_dir,
        scene_name,
        "represenations_comparison"
    )

    os.makedirs(
        comparison_dir,
        exist_ok=True
    )





    # part to check overlapping curves ______________________________________________
    gs_data = comparison_data["3DGS"]

    for representation in ["Mesh", "Cloud"]:
        representation_data = comparison_data[representation]

        first_difference, difference_value = (
            find_first_curve_difference(
                gs_data["iterations"],
                gs_data["errors_2d"],
                representation_data["iterations"],
                representation_data["errors_2d"],
                tolerance=1e-8
            )
        )

        if first_difference is None:
            print(
                f"{scene_name} case {case_nbr}: "
                f"{representation} and 3DGS 2D-error curves "
                "are identical over all common iterations."
            )
        else:
            print(
                f"{scene_name} case {case_nbr}: "
                f"{representation} first differs from 3DGS at "
                f"iteration {first_difference}; "
                f"absolute difference = {difference_value:.12e}"
            )
    #__________________________________________________________        




    # ------------------------------------------------------------------
    # Plot 1: 2D error comparison
    # ------------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for representation, data in comparison_data.items():
        plt.plot(
            data["iterations"],
            data["errors_2d"],
            label=representation,
            color=data["color"],
            linewidth=1.1,
            alpha=0.9
        )

    plt.xlabel("IBVS iteration")
    plt.ylabel("2D feature error")

    plt.title(
        f"{scene_name.capitalize()} Scenario {case_nbr}: "
        "2D feature error"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.35
    )

    plt.legend()
    plt.tight_layout()

    error_save_path = os.path.join(
        comparison_dir,
        f"case{case_nbr}_2d_comparison.png"
    )

    plt.savefig(
        error_save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ------------------------------------------------------------------
    # Velocity information
    # ------------------------------------------------------------------

    velocity_names = [
        r"$v_x$",
        r"$v_y$",
        r"$v_z$",
        r"$\omega_x$",
        r"$\omega_y$",
        r"$\omega_z$"
    ]

    velocity_filenames = [
        "vx",
        "vy",
        "vz",
        "wx",
        "wy",
        "wz"
    ]

    # ------------------------------------------------------------------
    # Plots 2–7: save every velocity component separately
    # ------------------------------------------------------------------

    for component_index in range(6):
        plt.figure(figsize=(10, 5))

        for representation, data in comparison_data.items():
            plt.plot(
                data["iterations"],
                data["velocities"][:, component_index],
                label=representation,
                color=data["color"],
                linewidth=1.0,
                alpha=0.9
            )

        plt.axhline(
            y=0.0,
            color="black",
            linestyle=":",
            linewidth=0.8,
            alpha=0.7
        )

        plt.xlabel("IBVS iteration")

        if component_index < 3:
            plt.ylabel("Linear velocity")
        else:
            plt.ylabel("Angular velocity")

        plt.title(
            f"{scene_name.capitalize()} Scenario {case_nbr}: "
            f"{velocity_names[component_index]}"
        )

        plt.grid(
            True,
            linestyle="--",
            alpha=0.35
        )

        plt.legend()
        plt.tight_layout()

        component_save_path = os.path.join(
            comparison_dir,
            (
                f"case{case_nbr}_"
                f"{velocity_filenames[component_index]}"
                "_comparison.png"
            )
        )

        plt.savefig(
            component_save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ------------------------------------------------------------------
    # Additional combined 2x3 velocity figure
    # ------------------------------------------------------------------

    figure, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 7),
        sharex=True
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
                alpha=0.9
            )

        axis.axhline(
            y=0.0,
            color="black",
            linestyle=":",
            linewidth=0.7,
            alpha=0.7
        )

        axis.set_title(
            velocity_names[component_index]
        )

        axis.set_xlabel(
            "IBVS iteration"
        )

        if component_index < 3:
            axis.set_ylabel(
                "Linear velocity"
            )
        else:
            axis.set_ylabel(
                "Angular velocity"
            )

        axis.grid(
            True,
            linestyle="--",
            alpha=0.3
        )

    # Use only one common legend
    legend_handles, legend_labels = (
        axes[0].get_legend_handles_labels()
    )

    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.01)
    )

    figure.suptitle(
        f"{scene_name.capitalize()} Scenario {case_nbr}: "
        "camera velocity comparison",
        y=1.04
    )

    figure.tight_layout()

    combined_velocity_save_path = os.path.join(
        comparison_dir,
        f"case{case_nbr}_6d_comparison.png"
    )

    figure.savefig(
        combined_velocity_save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        f"Representation comparison plots saved to: "
        f"{comparison_dir}"
    )








# Main PIPELINE  =======================================================================================================


def process_case(case_path: str, scene_name: str, case_nbr: int):

    results_dir = f"{case_path}/results"
    txt_path = f"{results_dir}/results.txt"
    desired_dir = f"{case_path}/desired_imgs"
    gs2s_dir = f"{case_path}/gs2s"

    # Create results directory and results.txt if it doesn't exist

    os.makedirs(results_dir, exist_ok=True)
    if not os.path.exists(txt_path):
        open(txt_path, "w").close()
    
    

    print(f"--- Processing {scene_name} case {case_nbr} ---")

    
    nbr_des = get_last_des_nbr(f"{case_path}/desired_imgs")
    nbr_kfs = get_last_kf_nbr(f"{case_path}/sfm/images")
    total_time = get_total_time(f"{case_path}/results.txt")

    add_line_to_text(txt_path, "----------- time -------------")
    add_line_to_text(txt_path, f"nbr_des = {nbr_des}")
    add_line_to_text(txt_path, f"nbr_kfs = {nbr_kfs}")
    add_line_to_text(txt_path, f"total_time = {total_time}")


    """
    # ------------------------------------------------------------------
    # des0 vs GT_des_img_masked
    # ------------------------------------------------------------------
    des0 = ImageHandling.load_np_img(f"{desired_dir}/des0.png")
    gt_des_masked = ImageHandling.load_np_img(f"{desired_dir}/gt_des_img_masked.png")
    ssim_val, psnr_val, lpips_val = ImageHandling.calc_imgs_sim_metrics(des0, gt_des_masked, 1)

    add_line_to_text(txt_path, "----------- imgs similarity -------------")

    add_line_to_text(txt_path, f"des0_GTmasked_ssim = {ssim_val}")
    add_line_to_text(txt_path, f"des0_GTmasked_psnr = {psnr_val}")
    add_line_to_text(txt_path, f"des0_GTmasked_lpips = {lpips_val}")


    # ------------------------------------------------------------------
    # des0 vs GT_des
    # ------------------------------------------------------------------
    gt_des = ImageHandling.load_np_img(f"{desired_dir}/GT_des.png")
    ssim_val, psnr_val, lpips_val = ImageHandling.calc_imgs_sim_metrics(des0, gt_des, 1)

    add_line_to_text(txt_path, f"initdes_GTdes_ssim = {ssim_val}")
    add_line_to_text(txt_path, f"initdes_GTdes_psnr = {psnr_val}")
    add_line_to_text(txt_path, f"initdes_GTdes_lpips = {lpips_val}")


    # ------------------------------------------------------------------
    # middle desired vs GT_des
    # ------------------------------------------------------------------
    mid_des = get_mid_des(gs2s_dir, desired_dir)

    ssim_val, psnr_val, lpips_val = ImageHandling.calc_imgs_sim_metrics(mid_des, gt_des, 2)
    add_line_to_text(txt_path, f"middes_GTdes_ssim = {ssim_val}")
    add_line_to_text(txt_path, f"middes_GTdes_psnr = {psnr_val}")
    add_line_to_text(txt_path, f"middes_GTdes_lpips = {lpips_val}")
    """
    
    # ------------------------------------------------------------------
    # final desired vs GT_des
    # ------------------------------------------------------------------
    last_des = get_last_des(desired_dir)
    gt_des = ImageHandling.load_np_img(f"{desired_dir}/GT_des.png")
    ssim_val, psnr_val, lpips_val = ImageHandling.calc_imgs_sim_metrics(last_des, gt_des, 3)
    add_line_to_text(txt_path, f"finaldes_GTdes_ssim = {ssim_val}")
    add_line_to_text(txt_path, f"finaldes_GTdes_psnr = {psnr_val}")
    add_line_to_text(txt_path, f"finaldes_GTdes_lpips = {lpips_val}")

    add_line_to_text(txt_path, "----------- ibvs infos -------------")
    
    

    # get GT / init poses
    configs_path = os.path.join(case_path, CONFIGS_FILENAME)
    init_img_name = get_init_img_name_frm_txt(configs_path)
    T_init = get_img_pose_gs1(init_img_name, f"{case_path}/gs1_sfm_aligned")
    T_gt = get_img_pose_gs1(gt_img_name, f"{case_path}/gs1_sfm_aligned")
    add_line_to_text(txt_path, f"init_img_gs1_name = {init_img_name}")

    # calc GT_init R and t diff
    R_init, t_init = LinAlgeb.get_Rt_from_homog_matrix(T_init)
    R_gt, t_gt = LinAlgeb.get_Rt_from_homog_matrix(T_gt)
    R_diff_gt_init = LinAlgeb.rotation_diff(R_gt, R_init)
    t_diff_gt_init = LinAlgeb.pose_distance(t_gt, t_init)  # normalization vector
    add_line_to_text(txt_path, f"R_diff_gt_init_deg = {R_diff_gt_init}")
    add_line_to_text(txt_path, f"t_diff_gt_init = {t_diff_gt_init}")
    

    # Load ibvs_infos____________________________________________________________________________________

    ibvs_infos_path = os.path.join(case_path, IBVS_INFOS_FILENAME)
    ibvs_infos = np.load(ibvs_infos_path, allow_pickle=True).item()

    # save ttle itrs nd condit nbr to txt
    total_nbr_itrs = get_total_nbr_of_itrs(ibvs_infos)
    add_line_to_text(txt_path, f"total_nbr_itrs = {total_nbr_itrs}")
    avrg_condit_nbr = get_avrg_condit_nbr(ibvs_infos)
    add_line_to_text(txt_path, f"avrg_condit_nbr = {avrg_condit_nbr}")

    # save all R diffs and t diffs
    iters_full, R_diffs_full, t_diffs_full = compute_pose_errors_per_iter(
        ibvs_infos, R_gt, t_gt, t_diff_gt_init)
    R_diff_full_path = os.path.join(results_dir, "R_diff_all.npy")
    t_diff_full_path = os.path.join(results_dir, "t_diff_all.npy")
    np.save(R_diff_full_path, R_diffs_full)
    np.save(t_diff_full_path, t_diffs_full)

    # save last R nd t diffs to txt
    last_iter_full = iters_full[-1]
    add_line_to_text(txt_path, f"last_itr = {last_iter_full}")
    add_line_to_text(txt_path, f"R_diff_last_itr_deg = {R_diffs_full[last_iter_full]}")
    add_line_to_text(txt_path, f"t_diff_last_itr_norm = {t_diffs_full[last_iter_full]}")


    # Extract 2D errors and 6D velocities
    (iters_ibvs, errors_2d, velocities_6d) = extract_2d_errors_and_velocities(ibvs_infos)



    # Plot 2D feature error versus iteration
    errors_2d_dict = dict(
        zip(
            iters_ibvs.tolist(),
            errors_2d.tolist()))

    plot_error_vs_iter(
        iters_ibvs,
        errors_2d_dict,
        ylabel="2D feature error",
        title=(
            f"{scene_name} case{case_nbr} - "
            "2D feature error vs iteration"),
        save_path=os.path.join(
            results_dir,
            "error_2d_vs_iter.png"))


    # Plot all six velocity components versus iteration
    plot_velocities_vs_iter(
        iters_ibvs,
        velocities_6d,
        title=(
            f"{scene_name} case{case_nbr} - "
            "6D camera velocity vs iteration"),
        save_path=os.path.join(
            results_dir,
            "velocity_6d_vs_iter.png"))


    # Plot & save the 2 errors R,t ________________________________________________
    
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

    print(f"    -> done. results saved to {results_dir}")






    # ___________________________________________________________ DVS ________________________________________________
    """
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
    """




    # _________________________________________________  Des0_ibvs_infos vs ibvs_infos pixel-error comparison  __________________________________________
  
    """
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
            (iters_pxl_des0, pxl_err_des0, "des0_IBVS", "tab:green"),],
        ylabel="Pixel error",
        title=f"{scene_name} case{case_nbr} - pixel error vs iter (IBVS vs des0_IBVS)",
        save_path=os.path.join(results_dir, "pxl_error_ibvs_vs_des0.png"),)
    """











# =========================================================================
# MAIN LOOP
# =========================================================================


def main():

    """
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
                f"{vals.mean():.4f} ± {vals.std():.4f}\n")
    return
    



    SCENE_NAMES = ["playroom"]
    CASE_NBRS = [1, 2, 3]
    for scene_name in SCENE_NAMES:
        for case_nbr in CASE_NBRS:
            case_folder_name = f"{scene_name}_case{case_nbr}_mesh"
            case_path = os.path.join(BASE_DIR, scene_name, case_folder_name)
            process_case(case_path, scene_name, case_nbr)
    

    """
    SCENE_NAMES = ["playroom"]
    CASE_NBRS = [1, 2, 3]
    # New Cloud/Mesh/3DGS comparison plots
    for scene_name in SCENE_NAMES:
        for case_nbr in CASE_NBRS:
            save_representation_comparison_plots(
                base_dir=BASE_DIR,
                scene_name=scene_name,
                case_nbr=case_nbr)



if __name__ == "__main__":
    main()