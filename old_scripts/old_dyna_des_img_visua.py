import os
import re
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


def update_image_keep_ratio(ax, im_obj, img):
    """
    Display img inside ax preserving its aspect ratio.

    Strategy:
      - The axes occupies a panel of size (panel_w, panel_h) in inches
        (read from the figure at draw time via ax.get_position + fig size).
      - We compute the largest box that fits inside the panel while keeping
        img's w/h ratio, then set the imshow extent to that box centred in
        the panel.  The axes xlim/ylim are fixed to the panel size so the
        surrounding space stays black / transparent.
    """
    h_img, w_img = img.shape[:2]
    img_ratio = w_img / h_img          # > 1 landscape, < 1 portrait

    fig = ax.get_figure()
    fig_w, fig_h = fig.get_size_inches()
    pos = ax.get_position()            # fraction of figure

    panel_w = pos.width  * fig_w      # panel width  in inches
    panel_h = pos.height * fig_h      # panel height in inches
    panel_ratio = panel_w / panel_h

    # Fit image inside panel (letterbox / pillarbox)
    if img_ratio >= panel_ratio:
        # Image is wider relative to panel → fit width, letterbox top/bottom
        fit_w = panel_w
        fit_h = panel_w / img_ratio
    else:
        # Image is taller relative to panel → fit height, pillarbox left/right
        fit_h = panel_h
        fit_w = panel_h * img_ratio

    # Centre the fitted box inside the panel (use panel coords 0..panel_w, 0..panel_h)
    x0 = (panel_w - fit_w) / 2
    y0 = (panel_h - fit_h) / 2
    x1 = x0 + fit_w
    y1 = y0 + fit_h

    im_obj.set_data(img)
    im_obj.set_extent([x0, x1, y0, y1])   # [left, right, bottom, top]

    # Fix axes limits to the full panel so padding stays consistent
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
    fig.patch.set_facecolor("#111111")

    outer = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.15)

    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)

    # --------------------------------------------------------
    # Helper: create an imshow axes with a title
    # --------------------------------------------------------
    def make_image_ax(subplot_spec, title):
        ax = fig.add_subplot(subplot_spec)
        ax.set_facecolor("#111111")
        ax.set_title(title, color="white", fontsize=10, pad=4)
        ax.axis("off")
        im = ax.imshow(
            blank_img,
            interpolation="nearest",
            aspect="auto",          # let extent control shape
            origin="upper",
        )
        return ax, im

    # ========================================================
    # TOP LEFT  –  Current Frame | Last KeyFrame
    # ========================================================

    gs_tl = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 0], wspace=0.05)

    ax_curr, im_curr = make_image_ax(gs_tl[0, 0], "Current Frame")
    ax_kf,   im_kf   = make_image_ax(gs_tl[0, 1], "Last KeyFrame")

    # ========================================================
    # TOP RIGHT  –  des_img | GT des_img
    # ========================================================

    gs_tr = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 1], wspace=0.05)

    ax_des, im_des = make_image_ax(gs_tr[0, 0], "des_img")
    ax_gt,  im_gt  = make_image_ax(gs_tr[0, 1], "GT des_img")

    # ========================================================
    # BOTTOM LEFT  –  Error plot
    # ========================================================

    ax_error = fig.add_subplot(outer[1, 0])
    ax_error.set_facecolor("#1a1a2e")
    ax_error.set_title("IBVS Error", color="white", fontsize=10)
    ax_error.set_xlabel("IBVS Iteration", color="#aaaaaa", fontsize=8)
    ax_error.set_ylabel("2D Error",       color="#aaaaaa", fontsize=8)
    ax_error.tick_params(colors="#aaaaaa", labelsize=7)
    for spine in ax_error.spines.values():
        spine.set_edgecolor("#444444")
    ax_error.grid(True, color="#333355", linewidth=0.6)

    error_iters, error_vals = [], []
    line_error, = ax_error.plot([], [], color="#00aaff", linewidth=1.5)

    # ========================================================
    # BOTTOM RIGHT  –  Matches
    # ========================================================

    ax_matches, im_matches = make_image_ax(outer[1, 1], "IBVS Matches")

    # ========================================================
    # Initial images  (loaded after tight_layout so positions are set)
    # ========================================================

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.05)   # one small pause so matplotlib commits the layout

    # Now positions are finalised → safe to call update_image_keep_ratio
    first_kf  = get_first_image(keyframes_path)
    first_des = get_first_image(desired_imgs_path)
    gt_img    = load_rgb(gt_des_img_path)

    update_image_keep_ratio(ax_curr,    im_curr,    blank_img)
    update_image_keep_ratio(ax_kf,      im_kf,      first_kf)
    update_image_keep_ratio(ax_des,     im_des,     first_des)
    update_image_keep_ratio(ax_gt,      im_gt,      gt_img)
    update_image_keep_ratio(ax_matches, im_matches, blank_img)

    fig.canvas.draw_idle()

    # ========================================================
    # REPLAY EVENTS
    # ========================================================

    previous_time = events[0]["time"]
    print(f"Loaded {len(events)} events.")

    for event in events:

        dt = (event["time"] - previous_time).total_seconds()
        if dt > 0:
            plt.pause(dt / speedup)

        # ----------------------------------------------------
        if event["type"] == "des_img":
            idx  = event["idx"]
            path = os.path.join(desired_imgs_path, f"des{idx}.png")
            print(f"DES_IMG {idx}")
            img = load_rgb(path)
            update_image_keep_ratio(ax_des, im_des, img)

        # ----------------------------------------------------
        elif event["type"] == "kf":
            idx  = event["idx"]
            path = os.path.join(keyframes_path, f"keyframe{idx}.png")
            print(f"KF {idx}")
            img = load_rgb(path)
            update_image_keep_ratio(ax_kf, im_kf, img)

        # ----------------------------------------------------
        elif event["type"] == "ibvs":
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

        fig.canvas.draw_idle()
        plt.pause(0.001)
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