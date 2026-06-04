import json
import matplotlib
import matplotlib.pyplot as plt
import traceback
import numpy as np
import os
import pickle
import matplotlib.animation as animation

from sklearn.cluster import DBSCAN

# ---- same paths as your main script ----
json_path = r"C:/Users/shikh/Desktop/Master Thesis/Pointclouds_New/pointclouds_v1.tar/pointclouds_v1/pointclouds/index.json"
LRS_PCL_FOLDER    = "C:/Users/shikh/Desktop/Master Thesis/Pointclouds_New/pointclouds_v1.tar/pointclouds_v1/pointclouds/lrs"
ROBOT_PCL_FOLDER  = "C:/Users/shikh/Desktop/Master Thesis/Pointclouds_New/pointclouds_v1.tar/pointclouds_v1/pointclouds/robot_laser"

cache_dir_dbscan = r"C:/Users/shikh/Desktop/Master Thesis/Detection_methods_/cache/cache_dbscan_clusters"
os.makedirs(cache_dir_dbscan, exist_ok=True)

with open(json_path, "r") as f:
    data = json.load(f)

dbscan_stats         = {}
thresholds_distances = [0.05, 0.075, 0.10]


# ---------------------------------------------------------------
# records_dbscan  — mirrors records_() exactly, DBSCAN in place
#                   of MeanShift, eps_list in place of bandwidths
# ---------------------------------------------------------------
def records_dbscan(record, trial_index, eps_list, min_samples=5):
    global dbscan_stats

    samples = record["Samples"]

    filtered_samples = []
    valid_lx, valid_ly   = [], []
    valid_rx, valid_ry   = [], []
    valid_lcx, valid_lcy = [], []
    valid_rcx, valid_rcy = [], []
    valid_t              = []

    clusters_per_frame = [[] for _ in eps_list]

    for s in samples:
        Lx,  Ly  = s.get("LeftLegX",    np.nan), s.get("LeftLegY",    np.nan)
        Rx,  Ry  = s.get("RightLegX",   np.nan), s.get("RightLegY",   np.nan)
        LCx, LCy = s.get("LeftCrutchX", np.nan), s.get("LeftCrutchY", np.nan)
        RCx, RCy = s.get("RightCrutchX",np.nan), s.get("RightCrutchY",np.nan)

        if all(np.isfinite(x) and abs(x) < 10 for x in [Lx,Ly,Rx,Ry,LCx,LCy,RCx,RCy]):
            valid_lx.append(Lx);  valid_ly.append(Ly)
            valid_rx.append(Rx);  valid_ry.append(Ry)
            valid_lcx.append(LCx); valid_lcy.append(LCy)
            valid_rcx.append(RCx); valid_rcy.append(RCy)
            valid_t.append(s.get("Time", 0.0))
            filtered_samples.append(s)

    valid_lx  = np.array(valid_lx);  valid_ly  = np.array(valid_ly)
    valid_rx  = np.array(valid_rx);  valid_ry  = np.array(valid_ry)
    valid_lcx = np.array(valid_lcx); valid_lcy = np.array(valid_lcy)
    valid_rcx = np.array(valid_rcx); valid_rcy = np.array(valid_rcy)
    valid_t   = np.array(valid_t)
    num_frames = len(valid_lx)

    hip_x = (valid_lx + valid_rx) / 2
    hip_y = (valid_ly + valid_ry) / 2

    lrs_frame_cache = [
        np.loadtxt(os.path.join(LRS_PCL_FOLDER, s["PCLFile"]), skiprows=11)[:, :2] / 1000.0
        for s in filtered_samples
    ]
    robot_frame_cache = [
        np.loadtxt(os.path.join(ROBOT_PCL_FOLDER, s["LaserFile"]), skiprows=11)[:, :2]
        for s in filtered_samples
    ]

    for thresh in thresholds_distances:
        dbscan_stats.setdefault(thresh, {})
        for eps in eps_list:
            dbscan_stats[thresh].setdefault(eps, {})

    for i in range(num_frames):
        try:
            lrs_points = lrs_frame_cache[i]

            # guard: loadtxt returns 1-D array when the file has exactly one row
            if lrs_points.ndim == 1:
                lrs_points = lrs_points[np.newaxis, :]

            hip_pos     = np.array([hip_x[i], hip_y[i]])
            dist_to_hip = np.linalg.norm(lrs_points - hip_pos, axis=-1)
            cropped_points = lrs_points[dist_to_hip < 1.0]

            for eps in eps_list:
                # DBSCAN requires at least 1 sample; skip clustering when the
                # cropped window is empty or has fewer points than min_samples
                if len(cropped_points) < min_samples:
                    labels       = np.full(len(cropped_points), -1, dtype=int)
                    valid_labels = []
                else:
                    db     = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = db.fit_predict(cropped_points.copy())
                    unique_labels, counts = np.unique(labels, return_counts=True)

                    # label == -1 is noise in DBSCAN — exclude it
                    valid_labels = [
                        lbl for lbl, cnt in zip(unique_labels, counts)
                        if lbl != -1
                    ]

                if len(valid_labels) > 0:
                    clusters_points = np.array([
                        cropped_points[labels == lbl].mean(axis=0)
                        for lbl in valid_labels
                    ])
                else:
                    clusters_points = np.empty((0, 2))

                eps_idx = eps_list.index(eps)
                clusters_per_frame[eps_idx].append(clusters_points)

                num_clusters = clusters_points.shape[0]

                gt = np.array([
                    [valid_lx[i],  valid_ly[i]],
                    [valid_rx[i],  valid_ry[i]],
                    [valid_lcx[i], valid_lcy[i]],
                    [valid_rcx[i], valid_rcy[i]],
                ])

                if num_clusters > 0:
                    gt_cluster_dist      = np.linalg.norm(
                        gt[:, None, :] - clusters_points[None, :, :], axis=-1
                    )
                    print(gt_cluster_dist)
                    gt_bestmatches       = np.argmin(gt_cluster_dist, axis=1)
                    clusters_bestmatches = np.argmin(gt_cluster_dist, axis=0)
                else:
                    gt_cluster_dist      = None
                    gt_bestmatches       = np.array([-1, -1, -1, -1])
                    clusters_bestmatches = np.array([])

                used_clusters   = set(gt_bestmatches) if num_clusters > 0 else set()
                unused_clusters = [idx for idx in range(num_clusters) if idx not in used_clusters]

                average_points_per_cluster = (
                    np.mean([np.sum(labels == lbl) for lbl in valid_labels])
                    if len(valid_labels) > 0 else 0
                )

                for thresh in thresholds_distances:
                    dbscan_stats[thresh][eps].setdefault(trial_index, {
                        "hits":                    np.zeros(4),
                        "misses":                  np.zeros(4),
                        "frame_count":             0,
                        "unused_clusters_count":   0,
                        "total_cluster_points":    0.0,
                        "total_cluster_distances": np.zeros(4),
                    })

                    s_ = dbscan_stats[thresh][eps][trial_index]
                    s_["frame_count"]           += 1
                    s_["total_cluster_points"]  += average_points_per_cluster
                    s_["unused_clusters_count"] += len(unused_clusters)

                    for k in range(4):
                        if (num_clusters > 0
                                and clusters_bestmatches[gt_bestmatches[k]] == k
                                and gt_cluster_dist[k, gt_bestmatches[k]] < thresh):
                            s_["hits"][k] += 1
                            s_["total_cluster_distances"][k] += gt_cluster_dist[k, gt_bestmatches[k]]
                        else:
                            s_["misses"][k] += 1

        except Exception as e:
            print(f"ERROR in frame {i}, eps {eps}: {e}")
            traceback.print_exc()

    # per-trial summary (same as records_() logic)
    for thresh in thresholds_distances:
        for eps in eps_list:
            if trial_index not in dbscan_stats[thresh][eps]:
                continue
            s_     = dbscan_stats[thresh][eps][trial_index]
            frames = s_["frame_count"]

            s_["average_cluster_points_per_trial"] = np.array([
                s_["total_cluster_points"] / frames if frames > 0 else 0.0
            ] * 4)

            avg_dist = np.zeros(4)
            for l in range(4):
                if s_["hits"][l] != 0:
                    avg_dist[l] = s_["total_cluster_distances"][l] / s_["hits"][l]
            s_["average_cluster_distances_points_per_trial"] = avg_dist

            print(f"total cluster dists: {s_['total_cluster_distances']}")
            print(f"hits:                {s_['hits']}")

    print("DEBUG DBSCAN clusters_per_frame:")
    print(f"  number of eps values : {len(clusters_per_frame)}")
    print(f"  frames per eps       : {[len(x) for x in clusters_per_frame]}")

    return (valid_lx, valid_ly, valid_rx, valid_ry,
            valid_lcx, valid_lcy, valid_rcx, valid_rcy,
            filtered_samples, lrs_frame_cache, clusters_per_frame)


# ---------------------------------------------------------------
# _dbscan_global_stats  — mirrors _meanshift_global_stats()
# ---------------------------------------------------------------
def _dbscan_global_stats(dbscan_stats):
    stats_global = {}

    for thresh, thresh_stats in dbscan_stats.items():
        for eps, eps_stats in thresh_stats.items():

            key = (thresh, eps)
            if key not in stats_global:
                stats_global[key] = {
                    "hits":                                       np.zeros(4),
                    "misses":                                     np.zeros(4),
                    "frame_count":                                0,
                    "unused":                                     0,
                    "average_cluster_points_per_trial":           [],
                    "average_cluster_distances_points_per_trial": [],
                    "total_cluster_distances":                    np.zeros(4),
                }

            for trial_index, trial_stats in dbscan_stats[thresh][eps].items():
                trial_stats.setdefault("hits",                                   np.zeros(4))
                trial_stats.setdefault("misses",                                 np.zeros(4))
                trial_stats.setdefault("frame_count",                            0)
                trial_stats.setdefault("unused_clusters_count",                  0)
                trial_stats.setdefault("average_cluster_points_per_trial",       np.zeros(4))
                trial_stats.setdefault("average_cluster_distances_points_per_trial", np.zeros(4))

                stats_global[key]["hits"]        += trial_stats["hits"]
                stats_global[key]["misses"]      += trial_stats["misses"]
                stats_global[key]["frame_count"] += trial_stats["frame_count"]
                stats_global[key]["unused"]      += trial_stats["unused_clusters_count"]

                pts_vec = trial_stats.get("average_cluster_points_per_trial", None)
                if pts_vec is not None:
                    stats_global[key]["average_cluster_points_per_trial"].append(
                        np.array(pts_vec, dtype=float)
                    )

                stats_global[key]["total_cluster_distances"] += trial_stats["total_cluster_distances"]

    return stats_global


# ---------------------------------------------------------------
# hit_miss_rates_SBS_dbscan  — mirrors hit_miss_rates_SBS()
# ---------------------------------------------------------------
def hit_miss_rates_SBS_dbscan(stats_global):
    threshold_list = [0.05, 0.075, 0.10]
    eps_list       = [0.05, 0.07, 0.1, 0.15, 0.2]

    plt.ioff()
    if not stats_global:
        print("No DBSCAN stats to plot!")
        return

    labels   = ["Left Leg", "Right Leg", "Left Crutch", "Right Crutch"]
    n_labels = len(labels)
    n_eps    = len(eps_list)
    width    = 0.8 / n_eps
    x        = np.arange(n_labels)

    for thresh in threshold_list:
        plt.figure(figsize=(14, 6))

        for i, eps in enumerate(eps_list):
            stats = stats_global.get((thresh, eps), None)
            if stats is None:
                continue

            hits   = stats["hits"]
            misses = stats["misses"]
            hit_rates  = hits   / (hits + misses + 1e-9)
            miss_rates = misses / (hits + misses + 1e-9)

            print(f"hit rate  eps={eps:.2f}: {hit_rates}")
            print(f"miss rate eps={eps:.2f}: {miss_rates}")

            pos = x + i * width - (width * (n_eps - 1)) / 2
            plt.bar(pos, hit_rates,  width, label=f"eps={eps:.2f}", alpha=0.7)
            plt.bar(pos, miss_rates, width, bottom=hit_rates, alpha=0.4, color="gray")

        plt.xticks(x, labels)
        plt.ylim(0, 1)
        plt.ylabel("Hit or Miss Rates")
        plt.xlabel("Leg or Crutch")
        plt.grid(alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.title(f"Hit & Miss Rates - DBSCAN | Threshold = {thresh:.3f} m")
        plt.savefig(f"DBSCAN_bars_thresh_{thresh:.3f}.png", dpi=300)


# ---------------------------------------------------------------
# trade_off_curves_dbscan  — mirrors trade_off_curves()
# ---------------------------------------------------------------
def trade_off_curves_dbscan(stats_global):
    threshold_list = [0.05, 0.075, 0.10]
    eps_list       = [0.05, 0.07, 0.1, 0.15, 0.2]

    for thresh in threshold_list:
        hits_curve            = []
        misses_curve          = []
        clusters_points_curve = []
        cluster_dist_curve    = []

        for eps in eps_list:
            stats = stats_global.get((thresh, eps))
            if not stats:
                hits_curve.append(0);            misses_curve.append(0)
                clusters_points_curve.append(0); cluster_dist_curve.append(0)
                continue

            hits   = np.sum(stats["hits"])
            misses = np.sum(stats["misses"])
            hits_rate   = hits   / (hits + misses + 1e-9)
            misses_rate = misses / (hits + misses + 1e-9)

            pts_list    = stats["average_cluster_points_per_trial"]
            average_pts = float(np.mean(pts_list)) if len(pts_list) else 0.0

            total_dists      = stats["total_cluster_distances"]
            total_hits       = stats["hits"]
            average_dist_pts = float(np.sum(total_dists) / (np.sum(total_hits) + 1e-9))

            hits_curve.append(hits_rate)
            misses_curve.append(misses_rate)
            clusters_points_curve.append(average_pts)
            cluster_dist_curve.append(average_dist_pts)

        # plot 1: hit/miss vs cluster density
        plt.figure(figsize=(10, 6))
        plt.plot(eps_list, hits_curve,            marker='o', linewidth=2, alpha=0.5, label="Hit Rate")
        plt.plot(eps_list, misses_curve,           marker='*', linewidth=2, alpha=0.5, label="Miss Rate")
        plt.plot(eps_list, clusters_points_curve,  marker='^', linewidth=2, alpha=0.5, label="Cluster Points")
        plt.xlabel("eps")
        plt.ylabel("Rates")
        plt.title(f"DBSCAN Trade-Off Curve Vs Cluster Points | Threshold = {thresh:.3f}")
        plt.grid(alpha=0.4)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(f"DBSCAN_tradeoff_rates_cluster_points_thresh_{thresh:.3f}.png", dpi=300)

        # plot 2: hit/miss vs localisation error
        plt.figure(figsize=(10, 6))
        plt.plot(eps_list, hits_curve,         marker='o', linewidth=2, alpha=0.5, label="Hit Rate")
        plt.plot(eps_list, misses_curve,        marker='*', linewidth=2, alpha=0.5, label="Miss Rate")
        plt.plot(eps_list, cluster_dist_curve,  marker='s', linewidth=2, alpha=0.5, label="Cluster Distances")
        plt.xlabel("eps")
        plt.ylabel("Rates")
        plt.title(f"DBSCAN Trade-Off Curve Vs Cluster Distances | Threshold = {thresh:.3f}")
        plt.grid(alpha=0.4)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(f"DBSCAN_tradeoff_rates_cluster_distances_thresh_{thresh:.3f}.png", dpi=300)
        plt.show()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == "__main__":
    if len(data) == 0:
        print("No records found in JSON.")

    eps_list_dbscan    = [0.05, 0.07, 0.1, 0.15, 0.2]
    min_samples_dbscan = 5

    for trial_index, record in enumerate(data):

        cache_file = os.path.join(
            cache_dir_dbscan, f"trial_{trial_index:02d}_dbscan_cache.pkl"
        )

        if os.path.exists(cache_file):
            print(f"Loading cached DBSCAN trial {trial_index + 1}")
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
                valid_lx           = cache["valid_lx"]
                valid_ly           = cache["valid_ly"]
                valid_rx           = cache["valid_rx"]
                valid_ry           = cache["valid_ry"]
                valid_lcx          = cache["valid_lcx"]
                valid_lcy          = cache["valid_lcy"]
                valid_rcx          = cache["valid_rcx"]
                valid_rcy          = cache["valid_rcy"]
                filtered_samples   = cache["filtered_samples"]
                lrs_frame_cache    = cache["lrs_frame_cache"]
                clusters_per_frame = cache["clusters_per_frame"]
        else:
            print(f"\nProcessing DBSCAN Trial {trial_index + 1} ...")
            result = records_dbscan(
                record, trial_index,
                eps_list=eps_list_dbscan,
                min_samples=min_samples_dbscan
            )
            (valid_lx, valid_ly, valid_rx, valid_ry,
             valid_lcx, valid_lcy, valid_rcx, valid_rcy,
             filtered_samples, lrs_frame_cache,
             clusters_per_frame) = result

            cache = {
                "valid_lx":           valid_lx,
                "valid_ly":           valid_ly,
                "valid_rx":           valid_rx,
                "valid_ry":           valid_ry,
                "valid_lcx":          valid_lcx,
                "valid_lcy":          valid_lcy,
                "valid_rcx":          valid_rcx,
                "valid_rcy":          valid_rcy,
                "filtered_samples":   filtered_samples,
                "lrs_frame_cache":    lrs_frame_cache,
                "clusters_per_frame": clusters_per_frame,
            }
            with open(cache_file, "wb") as f:
                pickle.dump(cache, f)
            print(f"DBSCAN cache saved: {cache_file}")

        # ---- GIF for trial 0 (same as meanshift_frames.gif) ----
        if trial_index == 0:
            eps_gif_index    = 1        # eps = 0.07  (index 1 in eps_list_dbscan)
            num_video_frames = len(valid_lx)

            hip_x = (valid_lx + valid_rx) / 2
            hip_y = (valid_ly + valid_ry) / 2

            plt.ioff()
            fig, ax = plt.subplots(figsize=(7, 7))

            def update(frame_index):
                ax.clear()

                lrs_points     = lrs_frame_cache[frame_index]
                hip_pos        = np.array([hip_x[frame_index], hip_y[frame_index]])
                distances      = np.linalg.norm(lrs_points - hip_pos, axis=1)
                cropped_points = lrs_points[distances < 1.0]
                clusters       = clusters_per_frame[eps_gif_index][frame_index]

                ax.scatter(cropped_points[:, 0], cropped_points[:, 1],
                           s=2, color="gray", alpha=0.4)
                ax.scatter(valid_lx[frame_index],  valid_ly[frame_index],
                           s=120, facecolor="none", edgecolor="blue")
                ax.scatter(valid_rx[frame_index],  valid_ry[frame_index],
                           s=120, facecolor="none", edgecolor="red")
                ax.scatter(valid_lcx[frame_index], valid_lcy[frame_index],
                           s=120, facecolor="none", edgecolor="green")
                ax.scatter(valid_rcx[frame_index], valid_rcy[frame_index],
                           s=120, facecolor="none", edgecolor="orange")

                if len(clusters) > 0:
                    ax.scatter(clusters[:, 0], clusters[:, 1],
                               s=80, marker="x", color="purple")

                ax.set_xlim(hip_x[frame_index] - 1.0, hip_x[frame_index] + 1.0)
                ax.set_ylim(hip_y[frame_index] - 1.0, hip_y[frame_index] + 1.0)
                ax.set_aspect("equal")
                ax.set_title(f"DBSCAN Clustering | Frame {frame_index + 1}")
                return ax,

            ani = animation.FuncAnimation(fig, update, frames=num_video_frames, blit=False)
            gif_path = "dbscan_frames.gif"
            ani.save(gif_path, writer="pillow", fps=10)
            plt.close(fig)
            print(f"DBSCAN GIF saved: {gif_path}")

# same pattern as your original script — called at module level after the loop
stats_global_dbscan = _dbscan_global_stats(dbscan_stats)
hit_miss_rates_SBS_dbscan(stats_global_dbscan)
trade_off_curves_dbscan(stats_global_dbscan)
