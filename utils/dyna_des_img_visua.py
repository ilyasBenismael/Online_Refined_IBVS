import os
import re
import time
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


# ============================================================
# PARSER
# ============================================================

def parse_log_file(log_path):
    events = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("+"):
                continue

            m = re.search(
                r"🔵(\d+)-IBVS:\s*2d_err:\s*([0-9.]+)",
                line
            )
            if m:
                iter_num = int(m.group(1))
                err_2d = float(m.group(2))
                m_time = re.search(r"(\d{2}:\d{2}:\d{2}\.\d+)", line)
                if m_time:
                    t = datetime.strptime(m_time.group(1), "%H:%M:%S.%f")
                    events.append({"type": "ibvs", "iter": iter_num, "err": err_2d, "time": t})
                continue

            if "Got a new des_img" in line:
                m_num = re.search(r"des_img\((\d+)\)", line)
                m_time = re.search(r"(\d{2}:\d{2}:\d{2}\.\d+)", line)
                if m_num and m_time:
                    t = datetime.strptime(m_time.group(1), "%H:%M:%S.%f")
                    events.append({"type": "des_img", "idx": int(m_num.group(1)), "time": t})
                continue

            if "Received KF" in line:
                m_num = re.search(r"Received KF(\d+)", line)
                m_time = re.search(r"(\d{2}:\d{2}:\d{2}\.\d+)", line)
                if m_num and m_time:
                    t = datetime.strptime(m_time.group(1), "%H:%M:%S.%f")
                    events.append({"type": "kf", "idx": int(m_num.group(1)), "time": t})

    events.sort(key=lambda e: e["time"])
    return events


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_rgb(path):
    if not os.path.exists(path):
        print(f"[WARNING] Missing file: {path}")
        return np.zeros((480, 640, 3), dtype=np.uint8)
    img = cv2.imread(path)
    if img is None:
        print(f"[WARNING] Failed to load: {path}")
        return np.zeros((480, 640, 3), dtype=np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_first_image(folder):
    if not os.path.exists(folder):
        return np.zeros((480, 640, 3), dtype=np.uint8)
    imgs = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not imgs:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    return load_rgb(os.path.join(folder, imgs[0]))


def get_first_numbered_image(folder, prefix):
    if not os.path.exists(folder):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    pattern = re.compile(
        rf"^{re.escape(prefix)}(\d+)\.(png|jpg|jpeg)$",
        re.IGNORECASE,
    )
    imgs = []
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            imgs.append((int(match.group(1)), filename))

    if not imgs:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    imgs.sort(key=lambda item: item[0])
    return load_rgb(os.path.join(folder, imgs[0][1]))


def update_image_keep_ratio(ax, im_obj, img):
    h_img, w_img = img.shape[:2]
    img_ratio = w_img / h_img

    fig = ax.get_figure()
    fig_w, fig_h = fig.get_size_inches()
    pos = ax.get_position()

    panel_w = pos.width  * fig_w
    panel_h = pos.height * fig_h
    panel_ratio = panel_w / panel_h

    if img_ratio >= panel_ratio:
        fit_w = panel_w
        fit_h = panel_w / img_ratio
    else:
        fit_h = panel_h
        fit_w = panel_h * img_ratio

    x0 = (panel_w - fit_w) / 2
    y0 = (panel_h - fit_h) / 2
    x1 = x0 + fit_w
    y1 = y0 + fit_h

    im_obj.set_data(img)
    im_obj.set_extent([x0, x1, y0, y1])

    ax.set_xlim(0, panel_w)
    ax.set_ylim(0, panel_h)


# ============================================================
# MAIN VISUALIZER
# ============================================================
def replay_visualization(
    log_path,
    gt_des_img_path,
    desired_imgs_path,
    keyframes_path,
    current_frames_path,
    matches_path,
    speedup=1.0,
):
    events = parse_log_file(log_path)

    if len(events) == 0:
        print("No events found.")
        return

    # ========================================================
    # FIGURE
    # ========================================================

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")

    outer = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.15)

    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)

    def make_image_ax(subplot_spec, title):
        ax = fig.add_subplot(subplot_spec)
        ax.set_facecolor("white")
        ax.set_title(title, color="black", fontsize=10, pad=4)
        ax.axis("off")
        im = ax.imshow(
            blank_img,
            interpolation="nearest",
            aspect="auto",
            origin="upper",
        )
        return ax, im

    # TOP LEFT  –  Initial Image | Current Frame
    gs_tl = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 0], wspace=0.05)
    ax_init, im_init = make_image_ax(gs_tl[0, 0], "Initial Image")
    ax_curr, im_curr = make_image_ax(gs_tl[0, 1], "Current Frame")

    # TOP RIGHT  –  GT des_img | des_img
    gs_tr = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 1], wspace=0.05)
    ax_gt,  im_gt  = make_image_ax(gs_tr[0, 0], "GT des_img")
    ax_des, im_des = make_image_ax(gs_tr[0, 1], "des_img")

    # BOTTOM LEFT  –  Error plot
    ax_error = fig.add_subplot(outer[1, 0])
    ax_error.set_facecolor("white")
    ax_error.set_title("IBVS Error", color="black", fontsize=10)
    ax_error.set_xlabel("IBVS Iteration", color="black", fontsize=8)
    ax_error.set_ylabel("2D Error",       color="black", fontsize=8)
    ax_error.tick_params(colors="black", labelsize=7)
    for spine in ax_error.spines.values():
        spine.set_edgecolor("#444444")
    ax_error.grid(True, color="#cccccc", linewidth=0.6)

    error_iters, error_vals = [], []
    line_error, = ax_error.plot([], [], color="#00aaff", linewidth=1.5)
    # BOTTOM RIGHT  –  Matches
    ax_matches, im_matches = make_image_ax(outer[1, 1], "IBVS Matches")

    # ========================================================
    # INITIAL DRAW  (one plt.pause to commit layout, then switch
    #                to flush_events for all subsequent redraws)
    # ========================================================

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.05)   # layout commit — only plt.pause call in the hot path

    init_img  = load_rgb(os.path.join(keyframes_path, "init_img.png"))
    first_des = get_first_numbered_image(desired_imgs_path, "des")
    gt_img    = load_rgb(gt_des_img_path)

    update_image_keep_ratio(ax_init,    im_init,    init_img)
    update_image_keep_ratio(ax_curr,    im_curr,    blank_img)
    update_image_keep_ratio(ax_gt,      im_gt,      gt_img)
    update_image_keep_ratio(ax_des,     im_des,     first_des)
    update_image_keep_ratio(ax_matches, im_matches, blank_img)

    fig.canvas.draw()
    fig.canvas.flush_events()

    # ========================================================
    # REPLAY EVENTS
    # ========================================================

    print(f"Loaded {len(events)} events.")
    previous_time = events[0]["time"]

    itr = 0
    for event in events:
        itr+=1
        dt = (event["time"] - previous_time).total_seconds() / speedup

        # Timestamp when this iteration started — used to subtract
        # the time already spent on processing from the sleep budget.
        frame_start = time.perf_counter()

        # ----------------------------------------------------
        if event["type"] == "des_img":
            idx  = event["idx"]
            path = os.path.join(desired_imgs_path, f"des{idx}.png")
            print(f"DES_IMG {idx}")
            img = load_rgb(path)
            update_image_keep_ratio(ax_des, im_des, img)

        # ----------------------------------------------------
        elif event["type"] == "kf":
            pass

        # ----------------------------------------------------
        elif event["type"] == "ibvs":

            if itr == error_iters[0] if error_iters else True:
                time.sleep(5)
            itr = event["iter"]
            err = event["err"]

            curr_frame_path  = os.path.join(current_frames_path, f"{itr}.png")
            matches_img_path = os.path.join(matches_path,         f"{itr}.png")

            print(f"IBVS {itr} | err={err:.4f}")

            curr_img = load_rgb(curr_frame_path)
            update_image_keep_ratio(ax_curr, im_curr, curr_img)

            matches_img = load_rgb(matches_img_path)
            update_image_keep_ratio(ax_matches, im_matches, matches_img)

            error_iters.append(itr)
            error_vals.append(err)
            line_error.set_data(error_iters, error_vals)
            ax_error.relim()
            ax_error.autoscale_view()

        # Redraw — flush_events pumps the GUI queue with no forced sleep
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Sleep only for whatever time budget is left after processing
        elapsed  = time.perf_counter() - frame_start
        remaining = dt - elapsed

        if remaining > 0.001:
            time.sleep(remaining)

        previous_time = event["time"]

    print("Replay finished.")
    plt.show()



# ============================================================
# EXAMPLE
# ============================================================

case_path = "/home/user/Bureau/visual_navigation/IBVS_CODE/my_results/online_ibvs_test/playroom/playroom_case1"

replay_visualization(
    log_path=f"{case_path}/results.txt",
    gt_des_img_path=f"{case_path}/desired_imgs/GT_des.png",
    desired_imgs_path=f"{case_path}/desired_imgs",
    keyframes_path=f"{case_path}/keyframes",
    current_frames_path=f"{case_path}/ibvs_frames/real_frames",
    matches_path=f"{case_path}/ibvs_frames/matches_frames",
    speedup=50.0
)