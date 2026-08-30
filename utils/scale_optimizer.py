import math
import numpy as np
from utils.image_handling import ImageHandling
from utils.mesh_handling import MeshHandling
import os
import matplotlib.pyplot as plt




class ScaleOptimizer :




    def __init__(self, case_test_path, S_MIN=0.01, S_MAX=3.0, N_INITIAL=20, N_CANDIDATES=3, NEIGHBOR_RADIUS=1.5, REFINEMENT_FACTOR=10.0, SCALE_THRESHOLD=0.005) :

        # Initial scale range
        self.S_MIN = S_MIN
        self.S_MAX = S_MAX

        # Dividing initial range to N_INITIAL(20) samples and picking N_candidates(3)
        self.N_INITIAL = N_INITIAL
        self.N_CANDIDATES = N_CANDIDATES

        self.NEIGHBOR_RADIUS = NEIGHBOR_RADIUS
        self.REFINEMENT_FACTOR = REFINEMENT_FACTOR
        self.SCALE_THRESHOLD = SCALE_THRESHOLD

        self.render_save_path = f"{case_test_path}/scales_test/renders"
        self.diff_save_path = f"{case_test_path}/scales_test/diffs"
        self.plot_save_path = f"{case_test_path}/scales_test/plots"
        os.makedirs(self.render_save_path, exist_ok=True)
        os.makedirs(self.diff_save_path, exist_ok=True)
        os.makedirs(self.plot_save_path, exist_ok=True)


        # Cache: scale -> photometric error
        self.cache = {}



        # Fixed axis limits, set on first plot call and only ever widened -------------------------------------------------
        self._y_min = None
        self._y_max = None






    # Evaluate one scale ==========================================================

    def evaluate_scale(self, s, pose2, pose1, mesh_handling : MeshHandling, moge_points1_o3d, img2):

        # If the evaluated scale is already in the cache just skip it (checking 4 decimals after
        s = float(s)
        key = round(s, 4)
        if key in self.cache:
            return

        # Scale relative translation
        pose2_scaled = pose2.copy()
        pose2_scaled[:3, 3] = (pose1[:3, 3] + s * (pose2[:3, 3] - pose1[:3, 3]))

        # Render
        img2_render, depth2 = mesh_handling.render_mesh_pic([moge_points1_o3d], pose2_scaled)
        depth2 = np.asarray(depth2)
        mask2 = np.isfinite(depth2) & (depth2 > 0)
    
        # Check coverage if less than 10% then it's a false scale skip
        coverage = np.mean(mask2)
        if coverage <  0.10:
            return
        
        else:

            # check photo loss on masked areas only (pixels where img2 captured some mogepoints_1)
            img2_render_g = ImageHandling.turn_img_to_gray(img2_render)
            img2_g = ImageHandling.turn_img_to_gray(img2)
            diff_img = ImageHandling.compute_grayscale_difference(img2_g, img2_render_g)
            photo_err = float(diff_img[mask2].mean())

            LAMBDA_COVERAGE = 10.0 # the loss on the mask is good but we need also to care about the % of cvrge (if cvrg is bigger the scale is safer)
            err = photo_err + LAMBDA_COVERAGE * (1.0 - coverage)


            # Save the scale X error in the cache
            self.cache[key] = err
   
        # Save images
        ImageHandling.save_img(img2_render, f"{s:.6f}_render", self.render_save_path)
        ImageHandling.save_img(diff_img, f"{s:.6f}_diff", self.diff_save_path)








    # Find best DISTINCT candidates ==========================================================

    def get_best_candidates(self, delta_s):
        """
        Select the best N_CANDIDATES scales from the COMPLETE cache.

        Candidates closer than delta_s are considered part of
        the same region, so only one candidate is kept.
        """
        sorted_samples = sorted(self.cache.items(), key=lambda x: x[1])
        candidates = []
        min_candidate_distance = delta_s

        for s, err in sorted_samples:

            # Check whether this candidate belongs to an already
            # selected region
            too_close = any(
                abs(s - selected_s) < min_candidate_distance
                for selected_s, _ in candidates)

            if too_close:
                continue

            candidates.append((s, err))
            if len(candidates) >= self.N_CANDIDATES:
                break

        return candidates







    # Merge overlapping ranges ==========================================================

    @staticmethod
    def merge_ranges(ranges):

        if len(ranges) == 0:
            return []

        ranges = sorted(ranges, key=lambda x: x[0])
        merged = [list(ranges[0])]

        for start, end in ranges[1:]:
            previous_start, previous_end = merged[-1]

            # Overlap
            if start <= previous_end:
                merged[-1][1] = max(previous_end, end)
            else:
                merged.append([start, end])

        return [ (start, end) for start, end in merged ]





    # Sample one range at a requested resolution =========================================================

    @staticmethod
    def sample_range(start, end, delta_s):
        """
        Generate approximately equally spaced samples with spacing
        <= delta_s.

        Existing cache entries are automatically skipped later.
        """

        width = end - start

        if width <= 0:
            return np.array([start])

        n_intervals = max(1, int(math.ceil(width / delta_s)))
        return np.linspace(start, end, n_intervals + 1)




    # Plot current cache state with ranges highlighted ==========================================================
    def _plot_level(self, level, delta_s, candidates=None, ranges=None, merged_ranges=None, tag="level"):

        scales = sorted(self.cache.keys())
        errors = [self.cache[s] for s in scales]

        fig, ax = plt.subplots(figsize=(10, 6))

        # All evaluated points -------------------------------------------------
        ax.scatter(scales, errors, s=15, color="steelblue", alpha=0.6, label="Evaluated scales")

        # Raw candidate ranges (before merge) -------------------------------------------------
        if ranges:
            for range_min, range_max in ranges:
                ax.axvspan(range_min, range_max, color="orange", alpha=0.10)

        # Merged search regions -------------------------------------------------
        if merged_ranges:
            for range_min, range_max in merged_ranges:
                ax.axvspan(range_min, range_max, color="green", alpha=0.15)

        # Candidates picked this level -------------------------------------------------
        if candidates:
            cand_s = [s for s, _ in candidates]
            cand_e = [e for _, e in candidates]
            ax.scatter(cand_s, cand_e, s=90, color="red", marker="x", label="Candidates", zorder=5)

        ax.set_xlabel("Scale (s)")
        ax.set_ylabel("Photometric error")
        ax.set_title(f"Scale search — {tag} {level} | delta_s={delta_s:.6f} | n={len(self.cache)}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(self.plot_save_path, f"{tag}_{level:02d}.png"), dpi=150)
        plt.close(fig)











    # Full coarse-to-fine optimization ==========================================================

    def optimize(self, pose2, pose1, mesh_handling : MeshHandling, moge_points1_o3d, img2):

        print("Start Scale Optimization ========================================")

        # Sampling N_initial over the initial range smin_smax and getting its delta_s
        initial_scales = np.linspace(self.S_MIN, self.S_MAX, self.N_INITIAL)
        delta_s = initial_scales[1] - initial_scales[0]

        print(
            f"\nInitial search: [{self.S_MIN:.3f}, {self.S_MAX:.3f}]"
            f" | samples={self.N_INITIAL}"
            f" | delta_s={delta_s:.6f}")

        # Evaluating each scale and adding all to the cache
        for s in initial_scales:
            self.evaluate_scale(s, pose2, pose1, mesh_handling, moge_points1_o3d, img2)

        # Save initial search plot -------------------------------------------------
        self._plot_level(0, delta_s, tag="initial")

        best_s0, best_err0 = min(self.cache.items(), key=lambda x: x[1])
        print(f"[DEBUG] initial cache size={len(self.cache)} | global best s={best_s0:.6f} err={best_err0:.6f}")


        # Coarse-to-fine search -------------------------------------------------
        level = 0
        # keep searching until the delta_s > treshhold (until we testing a very small range where we wont have some diffrnce)
        while delta_s > self.SCALE_THRESHOLD :
            level += 1

            # Get N_CANDIDATES(3) best distinct candidates from ALL evaluations -------------------------------------------------
            candidates = self.get_best_candidates(delta_s)
            candidates = self.get_best_candidates(delta_s)

            # --- DEBUG ---
            print(f"[DEBUG] level {level} cache size at candidate time={len(self.cache)}")
            print(f"[DEBUG] level {level} candidates full precision: {[(round(s,6), round(e,6)) for s, e in candidates]}")
            print(f"\n--- Refinement level {level} ---")
            print(f"Current delta_s: {delta_s:.6f}")
            print("Candidates:")

            for s, err in candidates:
                print(
                    f"  s={s:.6f}"
                    f" | error={err:.4f}")

            # Making a list of all ranges (+-radius around each s is a single range /the min max to avoid getting out of the initial range scales)-------------------------------------------------
            ranges = []
            radius = self.NEIGHBOR_RADIUS * delta_s
            for s, err in candidates:
                range_min = max(self.S_MIN, s - radius)
                range_max = min(self.S_MAX, s + radius)
                ranges.append((range_min, range_max))

            # Merge overlapping candidate ranges
            merged_ranges = self.merge_ranges(ranges)

            # Now we get the ranges to check over best previous candidates, we will check 10 samples in each range, so we got the new_delta_s
            new_delta_s = delta_s / self.REFINEMENT_FACTOR

            print(
                f"Searching {len(merged_ranges)} region(s)"
                f" with new delta_s={new_delta_s:.6f}")

            # Sampling the new ranges (of past best 3 candidates)
            n_new_evaluations = 0
            for range_min, range_max in merged_ranges:
                new_scales = self.sample_range(range_min, range_max, new_delta_s)

                for s in new_scales:
                    key = round(float(s), 4)

                    # Skip scales already evaluated -------------------------------------------------
                    if key in self.cache:
                        continue

                    # Adding the error of each one to the cache
                    self.evaluate_scale(s, pose2, pose1, mesh_handling, moge_points1_o3d, img2)
                    n_new_evaluations += 1

            print(f"New evaluations: {n_new_evaluations} | total cached: {len(self.cache)}")

            # Save this level's plot (candidates + ranges considered) -------------------------------------------------
            self._plot_level(level, delta_s, candidates=candidates, ranges=ranges, merged_ranges=merged_ranges, tag="level")

            # Update resolution -------------------------------------------------
            delta_s = new_delta_s

        # Final result -------------------------------------------------
        best_s, best_error = min(self.cache.items(), key=lambda x: x[1])

        print("\n========================================")
        print(" FINAL RESULT")
        print("========================================")

        print(f"Best scale : {best_s:.6f}")
        print(f"Best error : {best_error:.4f}")
        print(f"Final delta_s : {delta_s:.6f}")
        print(f"Total renders : {len(self.cache)}")
        print("========================================\n")

        # Save final overview plot (all points, no ranges) -------------------------------------------------
        self._plot_level(level, delta_s, tag="final")

        return best_s, best_error