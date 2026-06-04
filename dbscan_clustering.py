import json
import matplotlib.pyplot as plt
import traceback, sys
import pickle
import numpy as np
import os
from sklearn.cluster import DBSCAN
from collections import defaultdict
from matplotlib import animation
from scipy.optimize import linear_sum_assignment


#json_path = r"/home/swmi3514/Schreibtisch/Data/index.json"
json_path = r"C:/Users/shikh/Desktop/Master Thesis/Pointclouds_New/pointclouds_v1.tar/pointclouds_v1/pointclouds/index.json"

#LRS_PCL_FOLDER = "/home/swmi3514/Schreibtisch/Data/lrs"
#LRS_PLY_FOLDER = "/home/swmi3514/Schreibtisch/Data/lrs_ply"
LRS_PCL_FOLDER = "C:/Users/shikh/Desktop/Master Thesis/Pointclouds_New/pointclouds_v1.tar/pointclouds_v1/pointclouds/lrs"

#ROBOT_FOLDER = "/home/swmi3514/Schreibtisch/Data/robot_laser"
#ROBOT_PLY_FOLDER ="/home/swmi3514/Schreibtisch/Data/robot_laser_ply"
ROBOT_PCL_FOLDER = "C:/Users/shikh/Desktop/Master Thesis/Pointclouds_New/pointclouds_v1.tar/pointclouds_v1/pointclouds/robot_laser"

#os.makedirs(LRS_PLY_FOLDER, exist_ok=True)
#os.makedirs(ROBOT_PLY_FOLDER, exist_ok=True)


with open(json_path, "r") as f:
    data = json.load(f)


cache_dir = r"C:/Users/shikh/Desktop/Master Thesis/Detection_methods_/cache_dbscan"
os.makedirs(cache_dir, exist_ok=True)

# Directory to save per-trial pre-tracking data
trial_data_dir = r"C:/Users/shikh/Desktop/Master Thesis/Detection_methods_/trial_data"
os.makedirs(trial_data_dir, exist_ok=True)

#define parameters and thresholds range
parameters = [(0.07, 3),(0.15, 5),(0.03, 4),(0.25, 3),(0.10, 3),
                        (0.20, 5),(0.07, 6), (0.03, 5), (0.03, 6)]
thresholds_distances = [0.05, 0.075, 0.10]

#record = data[0]

clusters_stats = {}


#Kalman Filter
class TrackManager:
    def _init_(self):
        self.tracks = []
        self.next_id = 0
        self.id_switches = 0
        self.assigned_gt = {}  # gt_id -> track_id


class KalmanFilterCA:
    id_counter = 0

    def _init_(self, dt=0.05, q_pos=0.04, q_vel=0.006, q_acc=0.5, meas_noise=0.05):

        self.id = KalmanFilterCA.id_counterl
        KalmanFilterCA.id_counter += 1
        self.dt = dt
        self.missed = 0
        self.hits = 0
        self.age = 0
        self.confirmed = False
        self.last_measurement = None
        self.last_good =None
        self.smoothed_measurement = None
        self.confidence = 1.0
        self.track_id = None
        self.first_frame = None
        self.last_frame = None

        self.q_pos = q_pos
        self.q_vel = q_vel
        self.q_acc = q_acc

        # State: [x, y, vx, vy, ax, ay]
        self.x = np.zeros(6)

        self.P = np.eye(6) * 0.05

        # Measurement: position only
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        r = meas_noise
        self.R = np.eye(2) * (r ** 2)

        self.set_time(dt)

        self.missed = 0
        self.hits = 0

    def set_time(self, dt):
        self.dt = max(dt, 1e-3)

        dt = self.dt
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4
        dt5 = dt**5

        # Transition (unchanged)
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt2, 0],
            [0, 1, 0, dt, 0, 0.5*dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])


        dt = self.dt
        q = self.q_acc  # single scalar noise


        self.Q = q * np.array([
        [dt5/20, 0,       dt4/8,  0,       dt3/6, 0],
        [0,       dt5/20, 0,       dt4/8,  0,      dt3/6],

        [dt4/8,  0,       dt3/3,  0,       dt2/2, 0],
        [0,       dt4/8,  0,       dt3/3,  0,      dt2/2],

        [dt3/6,  0,       dt2/2,  0,       dt,     0],
        [0,       dt3/6,  0,       dt2/2,  0,      dt]
    ])


    def initialize(self, pos):
        self.x[:2] = pos
        self.x[2:] = 0.0
        self.P = np.eye(6) * 0.05

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        vel = self.x[2:4]
        speed = np.linalg.norm(vel)
        max_speed = 0.2

        if speed > max_speed:
            self.x[2:4] *= max_speed / (speed + 1e-9)

        max_acc = 0.15
        acc = self.x[4:6]
        acc_norm = np.linalg.norm(acc)

        if acc_norm > max_acc:
            self.x[4:6] *= max_acc / (acc_norm + 1e-9)

    def update(self, z):

        if z is None:
            self.missed += 1
            self.confidence *= 0.95

            #self.P *= 1.02  # mild uncertainty growth only
            #self.P = 0.5 * (self.P + self.P.T)
            #self.P += 1e-6 * np.eye(len(self.x))
            return

        innovation = z - (self.H @ self.x)

        S = self.H @ self.P @ self.H.T + self.R
        S_inv = np.linalg.inv(S + 1e-6 * np.eye(2))

        maha = innovation.T @ S_inv @ innovation

        # chi-square gating (DOF = 2)
        if maha > 9.21:
            self.missed += 1
            return

        K = self.P @ self.H.T @ S_inv

        self.x = self.x + K @ innovation
        # damp acceleration
        self.x[4:6] *= 0.7

        I = np.eye(len(self.x))
        #self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T

        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T

        # stability only
        #self.P = 0.5 * (self.P + self.P.T)
        #self.P += 1e-6 * np.eye(len(self.x))

        #TRack manage
        self.hits += 1
        self.missed = 0

        self.last_measurement = z.copy()
        self.confidence = min(1.0, self.confidence + 0.05)
        self.last_good = self.x[:2].copy()


    def estimate(self):
        return self.x[:2]


class KalmanFilterCV:

    id_counter = 0

    def _init_(self,
                 dt=0.05,
                 process_noise=0.01,
                 meas_noise=0.03):

        self.id = KalmanFilterCV.id_counter
        KalmanFilterCV.id_counter += 1

        self.dt = dt
        self.q = process_noise

        self.hits = 0
        self.missed = 0
        self.age = 0
        self.confirmed = False
        self.confidence = 1.0
        self.track_id = None
        self.last_measurement = None
        self.last_good =None
        self.smoothed_measurement = None
        self.first_frame = None
        self.last_frame = None

        # state = [x, y, vx, vy]
        self.x = np.zeros(4)

        self.P = np.eye(4) * 0.5

        self.H = np.array([
            [1,0,0,0],
            [0,1,0,0]
        ])

        r = meas_noise
        self.R = np.eye(2) * (r**2)

        self.set_time(dt)

    def set_time(self, dt):

        self.dt = max(dt, 1e-3)

        dt = self.dt

        self.F = np.array([
            [1,0,dt,0],
            [0,1,0,dt],
            [0,0,1,0],
            [0,0,0,1]
        ])

        # proper white acceleration model
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4

        q = self.q

        self.Q = q * np.array([
            [dt4/4, 0,      dt3/2, 0],
            [0,      dt4/4, 0,      dt3/2],
            [dt3/2, 0,      dt2,   0],
            [0,      dt3/2, 0,      dt2]
        ])

    def initialize(self, pos):

        self.x[:2] = pos
        self.x[2:] = 0

        self.P = np.eye(4) * 0.3

    def predict(self):

        self.x = self.F @ self.x

        self.P = self.F @ self.P @ self.F.T + self.Q

        # numerical stability
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z):

        if z is None:

            self.missed += 1

            # uncertainty growth
            #self.P *= 1.05

            #self.P = np.clip(self.P, 0, 5.0)

            max_cov = 2.0

            diag = np.diag(self.P)

            diag = np.minimum(diag, max_cov)

            np.fill_diagonal(self.P, diag)

            return

        innovation = z - self.H @ self.x

        S = self.H @ self.P @ self.H.T + self.R

        S_inv = np.linalg.inv(S)

        maha = innovation.T @ S_inv @ innovation

        # chi-square gate
        if maha > 9.21:
            self.missed += 1
            return

        K = self.P @ self.H.T @ S_inv

        self.x = self.x + K @ innovation

        I = np.eye(4)

        # Joseph stabilized covariance update
        self.P = (
            (I - K @ self.H) @ self.P @ (I - K @ self.H).T
            + K @ self.R @ K.T
        )

        self.P = 0.5 * (self.P + self.P.T)

        self.hits += 1
        self.missed = 0

        self.last_measurement = z.copy()
        self.confidence = min(1.0, self.confidence + 0.05)
        self.last_good = self.x[:2].copy()

    def estimate(self):

        return self.x[:2].copy()



def records_(record, trial_index):
    global clusters_stats

    plt.close('all')

    samples = record["Samples"]

    filtered_samples= []

    clusters_per_frame = {param: [] for param in parameters}

    valid_lx = []
    valid_ly = []
    valid_rx = []
    valid_ry = []
    valid_lcx = []
    valid_lcy = []
    valid_rcx = []
    valid_rcy = []
    valid_t  = []

    for s in samples:
        Lx = s.get("LeftLegX", float("nan"))
        Ly = s.get("LeftLegY", float("nan"))
        Rx = s.get("RightLegX", float("nan"))
        Ry = s.get("RightLegY", float("nan"))
        LCx = s.get("LeftCrutchX", float("nan"))
        LCy = s.get("LeftCrutchY", float("nan"))
        RCx = s.get("RightCrutchX", float("nan"))
        RCy = s.get("RightCrutchY", float("nan"))

        if all(np.isfinite(x) and abs(x) < 10 for x in [Lx, Ly, Rx, Ry, LCx, LCy, RCx, RCy]):
            valid_lx.append(Lx)
            valid_ly.append(Ly)
            valid_rx.append(Rx)
            valid_ry.append(Ry)
            valid_lcx.append(LCx)
            valid_lcy.append(LCy)
            valid_rcx.append(RCx)
            valid_rcy.append(RCy)
            valid_t.append(s.get("Time", 0.0))
            filtered_samples.append(s)

    valid_lx = np.array(valid_lx, dtype=float)
    valid_ly = np.array(valid_ly, dtype=float)
    valid_rx = np.array(valid_rx, dtype=float)
    valid_ry = np.array(valid_ry, dtype=float)
    valid_lcx = np.array(valid_lcx, dtype=float)
    valid_lcy = np.array(valid_lcy, dtype=float)
    valid_rcx = np.array(valid_rcx, dtype=float)
    valid_rcy = np.array(valid_rcy, dtype=float)
    valid_t  = np.array(valid_t, dtype=float)

    num_frames = len(valid_lx)
    #print("Valid frames:", num_frames)

    hip_x = (valid_lx + valid_rx) / 2
    hip_y = (valid_ly + valid_ry) / 2

    dir_x = np.diff(hip_x, prepend=hip_x[0])
    dir_y = np.diff(hip_y, prepend=hip_y[0])
    norm = np.sqrt(dir_x*2 + dir_y*2) + 1e-6
    dir_x /= norm
    dir_y /= norm

    orientation = np.arctan2(valid_ly - valid_ry, valid_lx - valid_rx)

    lrs_frame_cache = [np.loadtxt(os.path.join(LRS_PCL_FOLDER, s["PCLFile"]), skiprows=11)[:, :2]for s in filtered_samples]
    robot_frame_cache = [np.loadtxt(os.path.join(ROBOT_PCL_FOLDER, s["LaserFile"]), skiprows=11)for s in filtered_samples]



    for thresh in thresholds_distances:
        clusters_stats.setdefault(thresh, {})
        for eps, mins in parameters:
            clusters_stats[thresh].setdefault((eps,mins), {})


    for i in range(num_frames):
        try:
            lrs_points = lrs_frame_cache[i]

            hip_pos = np.array([hip_x[i], hip_y[i]])
            distances = np.linalg.norm(lrs_points - hip_pos, axis=-1)
            cropped_points = lrs_points[distances < 1.0]

            for eps, mins in parameters:
                dbscan = DBSCAN(eps=eps, min_samples=mins)
                labels = dbscan.fit_predict(cropped_points)
                unique_labels, counts = np.unique(labels, return_counts=True)

                valid_labels = [label for label, count in zip(unique_labels, counts) if label != -1 and count >= 5]

                #clusters_points = np.array([cropped_points[labels==label].mean(axis=0) for label in valid_labels])

                #centroids
                clusters = []
                for label in valid_labels:
                    pts = cropped_points[labels == label]

                    clusters.append({
                        "centroid": pts.mean(axis=0),
                        "points": pts,
                        "size": len(pts)})

                clusters_points = np.array([c["centroid"] for c in clusters])

                clusters_per_frame[(eps, mins)].append(clusters_points)

                #clusters = np.array([points.mean(axis = 0) for points in clusters_points])
                num_clusters = clusters_points.shape[0]

                #associate clusters to gt
                gt=np.zeros([4,2])
                gt[0,:]=[valid_lx[i],valid_ly[i]]
                gt[1,:]=[valid_rx[i],valid_ry[i]]
                gt[2,:]=[valid_lcx[i],valid_lcy[i]]
                gt[3,:]=[valid_rcx[i],valid_rcy[i]]
                #print(gt)



                if num_clusters > 0:
                    gt_cluster_dist = np.linalg.norm(gt[:,None,:] - clusters_points[None,:,:], axis=-1)
                    print(gt_cluster_dist)
                    gt_bestmatches = np.argmin(gt_cluster_dist, axis=1)
                    clusters_bestmatches = np.argmin(gt_cluster_dist, axis=0)
                else:
                    gt_bestmatches = np.array([-1,-1,-1,-1])
                    clusters_bestmatches = np.array([])

                used_clusters = set(gt_bestmatches)
                unused_clusters = [idx for idx in range(num_clusters) if idx not in used_clusters]

                average_points_per_cluster = (np.mean([np.sum(labels==label) for label in valid_labels])
                                                if len(valid_labels) > 0 else 0)


            for thresh in thresholds_distances:

                clusters_stats[thresh][(eps, mins)].setdefault(trial_index, {
                                            "hits": np.zeros(4),
                                            "misses": np.zeros(4),
                                            "frame_count": 0,

                                            "unused_cluster_counts": 0,
                                            "total_cluster_counts": 0.0,
                                            "total_cluster_distances": np.zeros(4)})

                clusters_stats[thresh][(eps, mins)][trial_index]["frame_count"] += 1
                clusters_stats[thresh][(eps, mins)][trial_index]["total_cluster_counts"]+=average_points_per_cluster
                clusters_stats[thresh][(eps, mins)][trial_index]["unused_cluster_counts"]+=len(unused_clusters)


            for k in range(gt.shape[0]):
                if num_clusters > 0 and clusters_bestmatches[gt_bestmatches[k]]==k and gt_cluster_dist[k,gt_bestmatches[k]] < thresh:
                        #found true positive
                    #plt.plot([gt[k,0],cluster_centers_arr[gt_bestmatches[k],0]], [gt[k,1],cluster_centers_arr[gt_bestmatches[k],1]], linestyle = 'solid',color="black")
                    clusters_stats[thresh][(eps, mins)][trial_index]["hits"][k] +=1
                    #clusters_distances = np.linalg.norm(gt[k,:]-cluster_centers_arr[gt_bestmatches[k],:])
                    clusters_distances = gt_cluster_dist[k, gt_bestmatches[k]]
                    clusters_stats[thresh][(eps, mins)][trial_index]["total_cluster_distances"][k]+=clusters_distances
                    # print(f"cluster dist", clusters_distances)


                else:
                    #no unique cluster for that
                    plt.scatter(gt[k,0], gt[k,1],s=160 ,facecolor="none", color="black", alpha=0.45, zorder=1)
                    clusters_stats[thresh][(eps, mins)][trial_index]["misses"][k] +=1
                    #clusters_stats[num_clusters]["misses"][k]+=1


                            #print("hitrate:",hits/frame_count, "missrate:",misses/frame_count)


                                # hits=gt_cluster_dist[:,gt_bestmatches]<0.05
                                # print("distance_mat ",gt_cluster_dist)
                                # print("bestmatches " ,gt_bestmatches)
                                # print("hits ",hits)
            #good_k=clusters_stats[trial_index][thresh][(eps, mins)]["hits"] != 0
            #print(good_k)
            #clusters_stats[trial_index][thresh][(eps, mins)]["clusters_average_distances"][good_k]=clusters_stats[trial_index][thresh][(eps, mins)]["clusters_average_distances"][good_k]/clusters_stats[trial_index][thresh][(eps, mins)]["hits"][good_k]

        except Exception as e:
            print(f"ERROR in frame {i}, parameters {parameters}: {e}")
            traceback.print_exc()


    for thresh in thresholds_distances:
        for eps, mins in parameters:
            if trial_index in clusters_stats[thresh][(eps, mins)]:

                stats = clusters_stats[thresh][(eps, mins)][trial_index]
                frames = stats["frame_count"]

                # Average cluster size (density)
                stats["average_cluster_points_per_trial"] = np.array([
                    stats["total_cluster_counts"] / frames if frames > 0 else 0
                ] * 4)

                # Average localization error (distance)
                avg_dist = np.zeros(4)
                for l in range(4):
                    if stats["hits"][l] != 0:
                        avg_dist[l] = stats["total_cluster_distances"][l] / stats["hits"][l]

                print(f"total clusters",  stats["total_cluster_distances"])
                print(f"good values", stats["hits"] )

                stats["average_cluster_distances_points_per_trial"] = avg_dist

    print("DEBUG clusters_per_frame:")
    print("number of parameters:", len(clusters_per_frame))
    print("frames per parameters:", [len(x) for x in clusters_per_frame.values()])

    return (valid_lx, valid_ly, valid_rx, valid_ry, valid_lcx, valid_lcy,
            valid_rcx, valid_rcy, hip_x, hip_y,dir_x, dir_y, orientation,
            filtered_samples, lrs_frame_cache, robot_frame_cache, clusters_per_frame)


def DBSCAN_draw_frame(i, valid_lx, valid_ly, valid_rx, valid_ry, valid_lcx, valid_lcy,
                        valid_rcx, valid_rcy, hip_x, hip_y, dir_x, dir_y, orientation,
                        filtered_samples, lrs_frame_cache, robot_frame_cache, clusters_per_frame, eps, mins):


    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_aspect("equal", "box")

    # Load both point clouds
    lrs_points = lrs_frame_cache[i]
    robot_points = robot_frame_cache[i]

    # Plot LRS (magenta) - downsample for speed if large
    if lrs_points.shape[0] > 0:
        pts = lrs_points if lrs_points.shape[0] < 20000 else lrs_points[::5]
        ax.scatter(pts[:,0], pts[:,1], s=1, color="yellow", alpha=0.45, zorder=1)

    #Plot Robot (orange)
    if robot_points.shape[0] > 0:
        rpts = robot_points if robot_points.shape[0] < 20000 else robot_points[::5]
        ax.scatter(rpts[:,0], rpts[:,1], s=1, color="orange", alpha=0.6, zorder=2)


    # Plot legs and skeleton
    ax.scatter(valid_lx[i], valid_ly[i], s=150, color="blue", zorder=10, label="Left Leg",facecolor="none")
    ax.scatter(valid_rx[i], valid_ry[i], s=150, color="red", zorder=11, label="Right Leg",facecolor="none")

    #Plot crutches
    ax.scatter(valid_lcx[i], valid_lcy[i], s=150, color="green", zorder=10, label="Left Crutch",facecolor="none")
    ax.scatter(valid_rcx[i], valid_rcy[i], s=150, color="cyan", zorder=11, label="Right Crutch",facecolor="none")

    # Hip, direction, torso
    ax.scatter(hip_x[i], hip_y[i], s=120, color="green", marker="X", zorder=12)
    dir_scale = 0.3
    ax.arrow(hip_x[i], hip_y[i], dir_x[i]*dir_scale, dir_y[i]*dir_scale,
                                head_width=0.05, head_length=0.1, color="purple", alpha=0.9, zorder=12)
    torso_len = 0.25
    torso_dx = torso_len * np.cos(orientation[i] + np.pi/2)
    torso_dy = torso_len * np.sin(orientation[i] + np.pi/2)
    ax.plot([hip_x[i], hip_x[i]+torso_dx], [hip_y[i], hip_y[i]+torso_dy],
                                color="black", linewidth=3, zorder=12)

    # Dynamic axes centered on hip (adjust window)
    axis_window = 1.0
    ax.set_xlim(hip_x[i]-axis_window, hip_x[i]+axis_window)
    ax.set_ylim(hip_y[i]-axis_window, hip_y[i]+axis_window)


    if i == 0:
        ax.legend(loc="upper right", fontsize=8)

    fig.canvas.flush_events()



#for frame_index in range(num_frames):
    #DBSCAN_draw_frame(frame_index)
    #plt.pause(0.001)

#plt.show()
#plt.close(fig)


def hit_miss_rates_bar_SBS(stats_global, title="Average Across All Trials"):

    plt.ioff()
    if not stats_global:
        print("No stats to plot!")
        return

    labels = ["Left Leg", "Right Leg", "Left Crutch", "Right Crutch"]
    n_labels = len(labels)

    #consistent order
    threshold_list = sorted(set(thresh for thresh, _ in stats_global.keys()))
    params_list = sorted(set(params for _, params in stats_global.keys()),
                    key=lambda x: (x[0], x[1]))


    n_params = len(params_list)
    width = 0.8 / n_params
    x = np.arange(n_labels)



    for thresh_index, thresh in enumerate(threshold_list):

        plt.figure(figsize=(14,6))


        for params_index, params in enumerate(params_list):

                stats = stats_global[(thresh, params)]
                hits = stats["hits"]
                misses = stats["misses"]

                hit_rates = hits / (hits + misses + 1e-9)
                miss_rates = misses / (hits + misses + 1e-9)

                pos = x - 0.4 + params_index * width + width/2

                plt.bar(
                    pos,
                    hit_rates,
                    width,
                    label=f"eps={params[0]:.3f}, min_samples={params[1]}",
                    alpha = 0.6

                )
                plt.bar(
                    pos,
                    miss_rates,
                    width,
                    bottom=hit_rates,
                    alpha=0.4,
                    color="gray"
                )

        plt.xticks(x, labels)
        plt.ylim(0, 1)
        plt.ylabel("Rates")
        plt.xlabel("Leg or Crutch")
        plt.title(f"Hit & Miss Rates | Threshold = {thresh:.3f} m")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8, ncol=2)

        plt.tight_layout()
        plt.savefig(f"hit_miss_rates_thresh_{thresh:.3f}.png", dpi=300)
        plt.show()


#plot trade off curve
def epsilon_performance(stats_global):

    graph_results = {}
    thresholds_list = sorted(set(thresh for thresh, _ in stats_global.keys()))


    for thresh in thresholds_list:
        performance = defaultdict(lambda:
                                {"hits": 0.0,
                                "misses": 0.0,
                                "clusters": [],
                                "clusters_average": [],
                                "clusters_average_distances": []})

        for(t,(eps, mins)),stats in stats_global.items():
            if t != thresh:
                continue

            performance[eps]["hits"] += stats["hits"].sum()
            performance[eps]["misses"] += stats["misses"].sum()

            performance[eps]["clusters"].extend(stats.get("cluster_counts", []))
            performance[eps]["clusters_average"].extend(stats.get("clusters_average", []))
            performance[eps]["clusters_average_distances"].extend(list(stats.get("clusters_average_distances", [])))


        eps_vals = np.array(sorted(performance.keys()))
        hit_rates = np.zeros(len(eps_vals))
        miss_rates = np.zeros(len(eps_vals))
        clusters_average = np.zeros(len(eps_vals))
        clusters_average_dist = np.zeros(len(eps_vals))


        for i, eps in enumerate(eps_vals):
            h = performance[eps]["hits"]
            m = performance[eps]["misses"]
            hit_rates[i] = h / (h + m + 1e-9)
            miss_rates[i] = m / (h + m + 1e-9)

            # Average points per cluster
            cluster_points_list = performance[eps]["clusters_average"]
            clusters_average[i] = np.mean(cluster_points_list) if len(cluster_points_list) > 0 else 0.0

            # Average distance from cluster centers to hits
            cluster_dist_list = performance[eps]["clusters_average_distances"]
            clusters_average_dist[i] = np.nanmean(cluster_dist_list) if len(cluster_dist_list) > 0 else np.nan


        graph_results[thresh] = (eps_vals, hit_rates, miss_rates, clusters_average, clusters_average_dist)

    return graph_results



def _dbscan_global_stats(clusters_stats):
    stats_global = {}

    for thresh, thresh_stats in clusters_stats.items():
        for params, params_stats in thresh_stats.items():

                key = (thresh, params)
                if key not in stats_global:
                    stats_global[key] = {
                        "hits": np.zeros(4),
                        "misses": np.zeros(4),
                        "frame_count": 0,
                        "unused": 0,
                        "average_cluster_points_per_trial": [],
                        "average_cluster_distances_points_per_trial": [],
                        "total_cluster_distances": []
                    }


                for trial_index, trial_stats in params_stats.items():

                    trial_stats.setdefault("hits", np.zeros(4))
                    trial_stats.setdefault("misses", np.zeros(4))
                    trial_stats.setdefault("frame_count", 0)
                    trial_stats.setdefault("unused_cluster_counts", 0)
                    trial_stats.setdefault("average_cluster_points_per_trial", np.zeros(4))
                    trial_stats.setdefault("average_cluster_distances_points_per_trial", np.zeros(4))
                    trial_stats.setdefault("total_cluster_distances", np.zeros(4))


                    stats_global[key]["hits"] += trial_stats["hits"]
                    stats_global[key]["misses"] += trial_stats["misses"]
                    stats_global[key]["frame_count"] += trial_stats["frame_count"]
                    stats_global[key]["unused"] += trial_stats["unused_cluster_counts"]

                    pts_vec = trial_stats.get("average_cluster_points_per_trial", None)
                    dist_vec = trial_stats.get("average_cluster_distances_points_per_trial", None)

                    if pts_vec is not None:
                        stats_global[key]["average_cluster_points_per_trial"].append(np.array(pts_vec, dtype=float))


                    #stats_global[key]["total_cluster_distances"]+=trial_stats["total_cluster_distances"]

                    tcd = trial_stats.get("total_cluster_distances", None)

                    if tcd is None:
                        tcd = np.zeros(4)
                    else:
                        tcd = np.asarray(tcd).reshape(-1)

                    if tcd.size!=4:
                        tcd = np.zeros(4)


                    stats_global[key]["total_cluster_distances"] += tcd

    return stats_global


#ID switches problem
def update_id_switches(track_manager, gt_ids, track_ids):

    for gt_id, trk_id in zip(gt_ids, track_ids):

        if gt_id in track_manager.assigned_gt:

            if track_manager.assigned_gt[gt_id] != trk_id:
                track_manager.id_switches += 1

        track_manager.assigned_gt[gt_id] = trk_id

#Tracking - KF
def associate(tracks, detections, gate=9.21):

    nT, nD = len(tracks), len(detections)
    cost = np.full((nT, nD), 1e6)

    for t, trk in enumerate(tracks):

        pred = trk.x[:2]
        S = trk.P[:2,:2] + trk.R
        S_inv = np.linalg.inv(S)

        for d, det in enumerate(detections):

            diff = det - pred
            maha = diff.T @ S_inv @ diff

            if maha > gate:
                continue

            motion = np.linalg.norm(diff)

            cost[t,d] = maha + 0.5 * motion

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    unmatched_t = list(range(nT))
    unmatched_d = list(range(nD))

    for r,c in zip(row_ind, col_ind):
        if cost[r,c] < 1e5:
            matches.append((r,c))
            unmatched_t.remove(r)
            unmatched_d.remove(c)

    return matches, unmatched_t, unmatched_d

def associate_tracks_nw(tracks, clusters, gate=9.21):

    nT, nC = len(tracks), len(clusters)
    cost = np.full((nT, nC), 1e6)

    for t, trk in enumerate(tracks):

        pred = trk.x[:2]

        # covariance gate (CRITICAL)
        S = trk.P[:2, :2] + trk.R
        S_inv = np.linalg.inv(S)

        for c, cl in enumerate(clusters):

            diff = cl - pred
            maha = diff.T @ S_inv @ diff

            if maha > gate:
                continue

            # motion consistency
            motion = np.linalg.norm(diff)

            # velocity consistency
            vel_cost = 0
            if trk.last_measurement is not None:
                vel_obs = cl - trk.last_measurement
                vel_cost = np.linalg.norm(vel_obs - trk.x[2:4])

            cost[t, c] = maha + 0.5 * motion + 0.3 * vel_cost

    row_ind, col_ind = linear_sum_assignment(cost)

    matches, unmatched_t, unmatched_c = [], list(range(nT)), list(range(nC))

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 1e5:
            matches.append((r, c))
            unmatched_t.remove(r)
            unmatched_c.remove(c)

    return matches, unmatched_t, unmatched_c




def associate_tracksss(trackers, clusters,
                     max_gate=0.35,
                     w_motion=2.5,
                     w_vel=2.0,
                     w_identity=3.0):

    nT = len(trackers)
    nC = len(clusters)

    cost = np.full((nT, nC), 1e6)

    for t, trk in enumerate(trackers):

        pred = trk.x[:2]
        vel_pred = trk.x[2:4]

        for c, cl in enumerate(clusters):

            dist = np.linalg.norm(cl - pred)
            #if dist > max_gate:
                #cost[t,c] = 1e6
                #continue

            diff = cl - pred
            S = trk.P[:2, :2] + trk.R
            maha= diff.T @ np.linalg.inv(S) @ diff
            motion_cost = maha


            vel_cost = 0.0
            if trk.last_measurement is not None:
                obs_vel = cl - trk.last_measurement
                vel_cost = np.linalg.norm(obs_vel - vel_pred)



            temp_cost = 0.0
            if trk.last_good is not None:
                temp_cost = np.linalg.norm(cl - trk.last_good)


            identity_cost = 0.0

            for other in trackers:
                if other is trk:
                    continue

                if other.last_good is None:
                    continue

                # enforce separation between tracks
                d = np.linalg.norm(cl - other.last_good)

                # penalize if too close to another track's role space
                #identity_cost += np.exp(-d)


            total_cost = (
                motion_cost +
                w_vel * vel_cost +
                0.5 * temp_cost +
                w_identity * identity_cost
            )

            #cost[t, c] = total_cost
            cost[t, c] = np.linalg.norm(cl - pred)

    rows, cols = linear_sum_assignment(cost)

    matches = []
    unmatched_t = list(range(nT))
    unmatched_c = list(range(nC))

    for r, c in zip(rows, cols):
        if cost[r, c] < 1e3:  # valid match threshold
            matches.append((r, c))
            if r in unmatched_t:
                unmatched_t.remove(r)
            if c in unmatched_c:
                unmatched_c.remove(c)

    return matches, unmatched_t, unmatched_c


    def associate_tracks(trackers, clusters, gate=16):

        n_tracks = len(trackers)
        n_clusters = len(clusters)

        if n_clusters == 0:
            return [], list(range(n_tracks)), []

        cost_matrix = np.full((n_tracks, n_clusters), 1e6)

        for t, tracker in enumerate(trackers):

            pred = tracker.x[:2]

            S = tracker.P[:2, :2] + tracker.R

            S_inv = np.linalg.inv(S)

            for c in range(n_clusters):

                diff = clusters[c] - pred

                maha = diff.T @ S_inv @ diff

                # temporal consistency
                if tracker.last_measurement is not None:
                    temporal = np.linalg.norm(clusters[c] - tracker.last_measurement)
                    #temporal = np.linalg.norm(clusters[c] - tracker.last_measurement) / (np.linalg.norm(tracker.last_measurement) + 1e-6)
                    maha += 0.1 * temporal

                if maha < gate:
                    cost_matrix[t, c] = maha

        rows, cols = linear_sum_assignment(cost_matrix)

        matches = []

        unmatched_tracks = list(range(n_tracks))
        unmatched_clusters = list(range(n_clusters))

        for r, c in zip(rows, cols):

            if cost_matrix[r, c] < gate:

                matches.append((r, c))

                unmatched_tracks.remove(r)
                unmatched_clusters.remove(c)

        return matches, unmatched_tracks, unmatched_clusters


def kalman_tracking(valid_lx, valid_ly, valid_rx, valid_ry,
                    valid_lcx, valid_lcy, valid_rcx, valid_rcy,
                    trial_index, lrs_frame_cache, clusters_per_frame,
                    filtered_samples, eps=0.07, mins=3):

    num_frames = len(valid_lx)
    times = np.array([s.get("Time", 0.0) for s in filtered_samples])

    gt = np.stack([
        np.stack([valid_lx, valid_ly], axis=1),
        np.stack([valid_rx, valid_ry], axis=1),
        np.stack([valid_lcx, valid_lcy], axis=1),
        np.stack([valid_rcx, valid_rcy], axis=1),
    ], axis=2)

    dt = np.diff(times)
    #print("min dt", np.min(dt))
    #print("max dt", np.max(dt))
    #print(np.diff(times[:20]))


    # Create 4 Kalman filters
    trackers = [KalmanFilterCV(process_noise=0.002, meas_noise=0.03) for _ in range(4)]
    #for track in trackers:
        #track.confidence = 1.0
    #trackers = []
    trackersCA = [KalmanFilterCA(q_pos=0.002, q_vel=0.005, q_acc=0.01, meas_noise=0.03)for _ in range(4)]
    #for tracks in trackersCA:
        #tracks.confidence = 1.0

    # Initialize
    for k in range(4):
        trackers[k].initialize(gt[0, :, k])
        trackersCA[k].initialize(gt[0, :, k])

        #track aware memory
        trackers[k].smoothed_measurement = gt[0, :, k]
        trackersCA[k].smoothed_measurement = gt[0, :, k]

        if num_frames > 1:

            dt0 = max(dt[0], 1e-3)

            vel0 = (gt[1, :, k] - gt[0, :, k]) / dt0

            # CV
            trackers[k].x[2:4] = vel0

            # CA
            trackersCA[k].x[2:4] = vel0
            trackersCA[k].x[4:6] = 0.0

    tracking_error = np.zeros((4, num_frames))
    trajectories = np.zeros((4, num_frames, 2))

    vector = np.zeros((4, num_frames, 4))
    vectorCA = np.zeros((4, num_frames, 6))

    trajectories_CA = np.zeros((4, num_frames, 2))
    tracking_error_CA = np.zeros((4, num_frames))

    #errors check
    errors = [[] for _ in range(4)]
    errorsCA = [[] for _ in range(4)]

    cluster_counts = []

    plt.figure(figsize=(7,7))

    #temporal smoothing
    #previous_clusters = None
    #alpha = 0.7


    for i in range(num_frames):

        #clusters = clusters_per_frame[(eps,mins)][i]
        clusters_ob = clusters_per_frame[(eps,mins)][i]
        clusters = np.asarray(clusters_ob)
        #clusters = np.array(clusters_per_frame[(eps,mins)][i])
        cluster_counts.append(len(clusters))

        dt_ = dt[0] if i == 0 else dt[i - 1]

        #Prediction
        for k in range(4):

            trackers[k].set_time(dt_)
            trackersCA[k].set_time(dt_)

            trackers[k].predict()
            trackersCA[k].predict()



        #Association

        matches_cv, unmatched_tracks_cv, unmatched_clusters_cv = associate(trackers, clusters)
        matches_ca, unmatched_tracks_ca, unmatched_clusters_ca = associate(trackersCA, clusters)

        #matches_cv, unmatched_tracks_cv, unmatched_clusters_cv = associate_tracks_tracker_centric(
            #trackers,
            #clusters,
            #gate=9.21,
            #motion_gate_dist=0.15,
            #memory_weight=0.5)

        #matches_ca, unmatched_tracks_ca, unmatched_clusters_ca = associate_tracks_tracker_centric(
            #trackersCA,
            #clusters,
            #gate=9.21,
            #motion_gate_dist=0.15,
            #memory_weight=0.5)


        # ---------------- CV UPDATE ----------------

        for t_idx, c_idx in matches_cv:

            measurement = clusters[c_idx]
            #trackers[t_idx].update(measurement)
            #trackers[t_idx].last_measurement = measurement

            #alpha = 0.3
            alpha = 0.9 if trackers[t_idx].missed == 0 else 0.6
            prev_smooth = trackers[t_idx].smoothed_measurement

            smoothed_measurement = (
                alpha * prev_smooth
                + (1 - alpha) * measurement
            )

            trackers[t_idx].smoothed_measurement = smoothed_measurement
            trackers[t_idx].update(smoothed_measurement)

        for t_idx in unmatched_tracks_cv:

            #trackers[t_idx].update(None)

            trackers[t_idx].P *= 1.2
            trackers[t_idx].update(None)


        # ---------------- CA UPDATE ----------------

        for t_idx, c_idx in matches_ca:

            measurement = clusters[c_idx]
            #trackersCA[t_idx].update(measurement)
            #trackersCA[t_idx].last_measurement = measurement


            #alpha = 0.3
            alpha = 0.9 if trackersCA[t_idx].missed == 0 else 0.6

            prev_smooth = trackersCA[t_idx].smoothed_measurement

            smoothed_measurement = (
                alpha * prev_smooth
                + (1 - alpha) * measurement
            )

            trackersCA[t_idx].smoothed_measurement = smoothed_measurement
            trackersCA[t_idx].update(smoothed_measurement)

        for t_idx in unmatched_tracks_ca:

            #trackersCA[t_idx].update(None)

            trackersCA[t_idx].P *= 1.2
            trackersCA[t_idx].update(None)

        #Save
        for k in range(4):

            # ---------------- CV ----------------

            est_cv = trackers[k].estimate()
            trajectories[k, i] = est_cv

            gt_pos = gt[i, :, k]
            error_cv = np.linalg.norm(est_cv - gt_pos)

            tracking_error[k, i] = error_cv
            errors[k].append(error_cv)

            vector[k, i, :] = trackers[k].x


            # ---------------- CA ----------------

            est_ca = trackersCA[k].estimate()
            trajectories_CA[k, i] = est_ca

            error_ca = np.linalg.norm(est_ca - gt_pos)

            tracking_error_CA[k, i] = error_ca
            errorsCA[k].append(error_ca)

            vectorCA[k, i, :] = trackersCA[k].x



            # optional logging
            print(
                f"[CV] Frame {i+1} | Track{k} | "
                f"Err={error_cv:.3f}"
            )

            print(
                f"[CA] Frame {i+1} | Track{k} | "
                f"Err={error_ca:.3f}"
            )


            # Track reset if lost
            #if trackers[k].missed > 10:
             #   trackers[k].confidence = 0.0  #dead

            #if trackersCA[k].missed > 10:
             #   trackersCA[k].confidence = 0.0  #dead

        #print("indx", i)

        # once per frame
        if i > 0:
            plt.clf()

            colors = ["blue", "red", "green", "orange"]


            plt.subplot(2, 2, 1)

            for k_plot in range(4):

                plt.plot(vector[k_plot, :i, 0],
                        label=f"T{k_plot} x")

                plt.plot(vector[k_plot, :i, 1],
                        linestyle="--",
                        label=f"T{k_plot} y")

                #plt.scatter(gt[i, 0, k_plot], gt[i, 1, k_plot], facecolors="none",edgecolors=colors[k_plot],linewidths=2, marker="o")

                # cluster count text
                plt.text(
                    0.02, 0.98,
                    f"Clusters: {len(clusters)}",
                    transform=plt.gca().transAxes,
                    fontsize=12,
                    color="black")

            plt.title("KF-CV States")
            plt.grid(True)

            plt.subplot(2, 2, 2)

            for k_plot in range(4):

                plt.plot(vectorCA[k_plot, :i, 0],
                        label=f"T{k_plot} x")

                plt.plot(vectorCA[k_plot, :i, 1],
                        linestyle="--",
                        label=f"T{k_plot} y")

                #plt.scatter(gt[i, 0, k_plot], gt[i, 1, k_plot], facecolors="none",edgecolors=colors[k_plot],linewidths=2, marker="o")

                # cluster count text
                plt.text(
                    0.02, 0.98,
                    f"Clusters: {len(clusters)}",
                    transform=plt.gca().transAxes,
                    fontsize=12,
                    color="black")

            plt.title("KF-CA States")
            plt.grid(True)


            # ---- CV subplot ----
            #print("CV plot")
            plt.subplot(2, 2, 3)

            for k_plot in range(4):

                plt.plot(
                        trajectories[k_plot, :i, 0],
                        trajectories[k_plot, :i, 1],
                        color=colors[k_plot],
                        linewidth=2,
                        label=f"Track {k_plot}")


                # tracker current position
                plt.scatter(
                        trajectories[k_plot, i, 0],
                        trajectories[k_plot, i, 1],
                        color=colors[k_plot],
                        marker="o")


                # GT - legs
                #plt.scatter(gt[i, 0, 0],gt[i, 1, 0],facecolors="none",edgecolors=colors[0],linewidths=2)
                #plt.scatter(gt[i, 0, 1],gt[i, 1, 1],facecolors="none", edgecolors=colors[1],linewidths=2)

                # GT - crutches
                #plt.scatter(gt[i, 0, 2],gt[i, 1, 2],facecolors="none",edgecolors=colors[2],marker="s",linewidths=2)
                #plt.scatter(gt[i, 0, 3],gt[i, 1, 3],facecolors="none",edgecolors=colors[3],marker="s",linewidths=2)

                #CLUSTERS
                #if len(clusters) > 0:
                    #plt.scatter(clusters[:, 0], clusters[:, 1], c="purple",marker="x",s=60, label="DBSCAN clusters")


                plt.title(f"KF-CV (Constant Velocity) - Frame {i+1}")
                plt.axis("equal")
                plt.grid(alpha=0.3)



            # ---- CA subplot ----
            #print("CA plot")
            plt.subplot(2, 2, 4)

            for k_plot in range(4):
                plt.plot(
                        trajectories_CA[k_plot, :i, 0],
                        trajectories_CA[k_plot, :i, 1],
                        color=colors[k_plot],
                        linewidth=2,
                        label=f"Track {k_plot}"
                    )

                # tracker current position
                plt.scatter(
                        trajectories_CA[k_plot, i, 0],
                        trajectories_CA[k_plot, i, 1],
                        color=colors[k_plot],
                        marker="o"
                    )


                # GT - legs
                #plt.scatter(gt[i, 0, 0],gt[i, 1, 0],facecolors="none",edgecolors=colors[0],linewidths=2)
                #plt.scatter(gt[i, 0, 1],gt[i, 1, 1],facecolors="none",edgecolors=colors[1],linewidths=2)

                # GT - crutches
                #plt.scatter(gt[i, 0, 2],gt[i, 1, 2],facecolors="none",edgecolors=colors[2],marker="s",linewidths=2)
                #plt.scatter(gt[i, 0, 3],gt[i, 1, 3],facecolors="none",edgecolors=colors[3],marker="s",linewidths=2)

                #CLUSTERS
                #if len(clusters) > 0:
                    #plt.scatter(clusters[:, 0],clusters[:, 1],c="purple",marker="x",s=60,label="DBSCAN clusters")


            plt.title(f"KF-CA (Constant Acceleration) - Frame {i+1}")
            plt.axis("equal")
            plt.grid(alpha=0.3)

            plt.tight_layout()
            plt.pause(0.01)

    rmse = 100 * np.sqrt(np.mean(tracking_error**2, axis=1))
    print(f"Kalman RMSE (cm): {rmse}")

    rmseCA = 100 * np.sqrt(np.mean(tracking_error_CA**2, axis=1))
    print(f"Kalman RMSE CA (cm): {rmseCA}")


    labels = ["Left Leg","Right Leg","Left Crutch","Right Crutch"]

    #RMSE plots
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10,6))

    plt.bar(x - width/2, rmse, width, label="KF-CV")
    plt.bar(x + width/2, rmseCA, width, label="KF-CA")

    plt.xticks(x, labels)
    plt.ylabel("RMSE (cm)")
    plt.title("Kalman Filter Comparison (CV vs CA)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"trial_{trial_index}_KF_CV_vs_CA_RMSE.png")
    plt.show()

    #histograms
    plt.figure(figsize=(12,10))

    for k in range(4):
        plt.subplot(2,2,k+1)

        plt.hist(errors[k], bins=40, alpha=0.5, label="CV", density=True)
        plt.hist(errorsCA[k], bins=40, alpha=0.5, label="CA", density=True)

        plt.title(f"{labels[k]} Error Distribution")
        plt.xlabel("Error (m)")
        plt.ylabel("Density")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"trial_{trial_index}_error_histograms_CV_CA.png")
    plt.show()

    #errors
    errors_flat_cv = np.concatenate(errors)
    errors_flat_ca = np.concatenate(errorsCA)

    cluster_counts_rep = np.repeat(cluster_counts, 4)

    plt.figure(figsize=(8,6))

    plt.scatter(cluster_counts_rep, errors_flat_cv, alpha=0.3, label="CV")
    plt.scatter(cluster_counts_rep, errors_flat_ca, alpha=0.3, label="CA")

    plt.xlabel("Number of clusters")
    plt.ylabel("Tracking error (m)")
    plt.title("Error vs Cluster Count (CV vs CA)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(f"trial_{trial_index}_error_vs_clusters_CV_CA.png")
    plt.show()

    return rmse, rmseCA, trajectories, trajectories_CA, gt

#Evaluation Metrics
def mot_evaluate(gt, tracks, dist_th=0.05):

    T, N, _ = gt.shape

    FP = 0
    FN = 0
    IDSW = 0
    total_dist = 0
    matches = 0

    prev_map = {}

    for t in range(T):

        gt_t = gt[t]
        trk_t = np.array([trk[t] for trk in tracks if len(trk) > t])

        if len(trk_t) == 0:
            FN += len(gt_t)
            continue

        cost = np.linalg.norm(gt_t[:,None,:] - trk_t[None,:,:], axis=2)

        row, col = linear_sum_assignment(cost)

        used_gt = set()
        used_trk = set()
        prev_map_new=prev_map
        tracker_id = N[c]
        for r,c in zip(row,col):

            if cost[r,c] < dist_th:

                matches += 1
                total_dist += cost[r,c]

                used_gt.add(r)
                used_trk.add(c)

                if r in prev_map and prev_map[r] != c:
                    IDSW += 1

                prev_map_new[r] = tracker_id

        print("Min of cost", np.min(cost))
        print("Max of cost", np.max(cost))
        print("thresh dist:", dist_th)
        print("matches:", matches)

        prev_map = prev_map_new
        FP += len(trk_t) - len(used_trk)
        FN += len(gt_t) - len(used_gt)

    mota = 1 - (FP + FN + IDSW) / (T * N)
    motp = total_dist / max(matches, 1)

    return {
        "MOTA": mota,
        "MOTP": motp,
        "IDSW": IDSW,
        "FP": FP,
        "FN": FN
    }


# Keys stored in each trial_data pickle
TRIAL_DATA_KEYS = [
    "valid_lx", "valid_ly", "valid_rx", "valid_ry",
    "valid_lcx", "valid_lcy", "valid_rcx", "valid_rcy",
    "hip_x", "hip_y", "dir_x", "dir_y", "orientation",
    "filtered_samples", "lrs_frame_cache", "robot_frame_cache",
    "clusters_per_frame",
]


def save_trial_data(trial_index, trial_tuple):
    """Persist the output of records_() for one trial to disk."""
    path = os.path.join(trial_data_dir, f"trial_{trial_index:02d}_data.pkl")
    payload = dict(zip(TRIAL_DATA_KEYS, trial_tuple))
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  Trial {trial_index} pre-tracking data saved -> {path}")


def load_trial_data(trial_index):
    """Load a previously saved trial data dict and return it as a tuple."""
    path = os.path.join(trial_data_dir, f"trial_{trial_index:02d}_data.pkl")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return tuple(payload[k] for k in TRIAL_DATA_KEYS)


def trial_data_exists(trial_index):
    path = os.path.join(trial_data_dir, f"trial_{trial_index:02d}_data.pkl")
    return os.path.exists(path)


MODE = "both"
RUN_TRACKING = True


if __name__ == "__main__":
    if len(data) == 0:
        print("No records found in JSON.")
        sys.exit(0)


    stats_global = {}

    if MODE in ("compute", "both"):


        for trial_index, record in enumerate(data):
            cache_file = os.path.join(cache_dir, f"trial_{trial_index:02d}_stats.pkl")

            # ----------------------------------------------------------------
            # Step 1: load or compute pre-tracking data (records_ output)
            # ----------------------------------------------------------------
            if trial_data_exists(trial_index):
                print(f"\nTrial {trial_index+1}: loading pre-tracking data from disk ...")
                trial_tuple = load_trial_data(trial_index)
                (valid_lx, valid_ly,
                 valid_rx, valid_ry,
                 valid_lcx, valid_lcy,
                 valid_rcx, valid_rcy,
                 hip_x, hip_y,
                 dir_x, dir_y,
                 orientation,
                 filtered_samples,
                 lrs_frame_cache,
                 robot_frame_cache,
                 clusters_per_frame) = trial_tuple
            else:
                print(f"\nProcessing Trial {trial_index+1} ...")
                trial_tuple = records_(record, trial_index)

                # Save immediately after records_() completes, before tracking
                save_trial_data(trial_index, trial_tuple)

                (valid_lx, valid_ly,
                 valid_rx, valid_ry,
                 valid_lcx, valid_lcy,
                 valid_rcx, valid_rcy,
                 hip_x, hip_y,
                 dir_x, dir_y,
                 orientation,
                 filtered_samples,
                 lrs_frame_cache,
                 robot_frame_cache,
                 clusters_per_frame) = trial_tuple

            # ----------------------------------------------------------------
            # Step 2: load or skip DBSCAN stats cache
            # ----------------------------------------------------------------
            if os.path.exists(cache_file):
                print(f"  Skipping DBSCAN stats for Trial {trial_index+1} (already cached).")
                with open(cache_file, "rb") as f:
                    clusters_stats[trial_index] = pickle.load(f)
            else:
                # save after every trial
                with open(cache_file, "wb") as f:
                    pickle.dump(clusters_stats, f)

            # ----------------------------------------------------------------
            # Step 3: tracking (only when requested)
            # ----------------------------------------------------------------
            if RUN_TRACKING:

                #Kalman Filter tracking
                print("Running Kalman Filter...")
                rmse, rmseCA, trajectories, trajectories_CA, gt= kalman_tracking(
                    valid_lx, valid_ly, valid_rx, valid_ry,
                    valid_lcx, valid_lcy, valid_rcx, valid_rcy,
                    trial_index,
                    lrs_frame_cache,
                    clusters_per_frame,
                    filtered_samples,
                    eps=0.07, mins=3
                )


                print(f"Trial {trial_index+1} KF CV RMSE: {rmse}")
                print(f"Trial {trial_index+1} KF CA RMSE: {rmseCA}")

                gt_array = np.transpose(gt, (0, 2, 1))      # (T, 4, 2)

                pred_cv = np.transpose(trajectories, (1, 0, 2))     # (T, 4, 2)
                pred_ca = np.transpose(trajectories_CA, (1, 0, 2))  # (T, 4, 2)

                metrics_cv = mot_evaluate(gt_array, pred_cv)
                metrics_ca = mot_evaluate(gt_array, pred_ca)

                print("CV MOT:", metrics_cv)
                print("CA MOT:", metrics_ca)

            # SAVE DBSCAN GIF
            if trial_index == 0:

                #eps = 0.07
                #mins = 3

                num_video_frames = len(valid_lx)

                for (eps,mins) in parameters:

                    print(f"Creates GIF for eps={eps}, min_samples={mins}")

                    plt.ioff()
                    fig, ax = plt.subplots(figsize=(7, 7))

                    def update(frame_index):

                        ax.clear()

                        lrs_points = lrs_frame_cache[frame_index]

                        hip_pos = np.array([
                            hip_x[frame_index],
                            hip_y[frame_index]])

                        distances = np.linalg.norm(
                            lrs_points - hip_pos,
                            axis=1)

                        cropped_points = lrs_points[distances < 1.0]

                        clusters = clusters_per_frame[(eps, mins)][frame_index]

                        # raw lidar
                        ax.scatter(
                            cropped_points[:, 0],
                            cropped_points[:, 1],
                            s=2,
                            color="gray",
                            alpha=0.4)

                        # GT points
                        ax.scatter(
                            valid_lx[frame_index],
                            valid_ly[frame_index],
                            s=120,
                            facecolor="none",
                            edgecolor="blue")

                        ax.scatter(
                            valid_rx[frame_index],
                            valid_ry[frame_index],
                            s=120,
                            facecolor="none",
                            edgecolor="red")

                        ax.scatter(
                            valid_lcx[frame_index],
                            valid_lcy[frame_index],
                            s=120,
                            facecolor="none",
                            edgecolor="green")

                        ax.scatter(
                            valid_rcx[frame_index],
                            valid_rcy[frame_index],
                            s=120,
                            facecolor="none",
                            edgecolor="orange")

                        # DBSCAN clusters
                        if len(clusters) > 0:

                            ax.scatter(
                                clusters[:, 0],
                                clusters[:, 1],
                                s=80,
                                marker="x",
                                color="purple")

                        ax.set_xlim(
                            hip_x[frame_index] - 1.0,
                            hip_x[frame_index] + 1.0)

                        ax.set_ylim(
                            hip_y[frame_index] - 1.0,
                            hip_y[frame_index] + 1.0)

                        ax.set_aspect("equal")

                        ax.set_title(
                            f"DBSCAN | eps={eps} | min_samples={mins} | "
                            f"Frame {frame_index+1}")

                        return ax,

                ani = animation.FuncAnimation(fig, update, frames=num_video_frames, blit=False)

                gif_path = f"dbscan_eps_{eps}min{mins}.gif"

                ani.save(gif_path,writer="pillow",fps=10)

                plt.close(fig)

                print(f"DBSCAN GIF saved: {gif_path}")

        stats_global = _dbscan_global_stats(clusters_stats)

    if MODE in ("plot", "both"):

        #Epsilon performance curve
        graph_results = epsilon_performance(stats_global)

        threshold_colors = plt.cm.viridis(np.linspace(0,1, len(graph_results)))


        for i, (thresh, (eps_values, hit_rates, miss_rates, clusters_average, clusters_average_dist)) in enumerate(graph_results.items()):


            fig, ax1 = plt.subplots(figsize=(12,6))
            ax2 = ax1.twinx()  #for clusters


            if len(eps_values) == 0 or len(hit_rates) == 0 or len(miss_rates) == 0:
                print(f"Threshold {thresh} has no valid data. Skipping...")
                continue

            score = hit_rates - miss_rates
            best_idx = np.argmax(score)


            ax1.plot(eps_values, hit_rates, marker = "s",
                        color = threshold_colors[i], linestyle="--",  label = f"Hit rate | Threshold = {thresh}", linewidth=2)
            ax1.plot(eps_values, miss_rates, marker = "o",
                        color = threshold_colors[i], linestyle="--", label = f"Miss rate | Threshold = {thresh}", linewidth=2)

            ax1.axvline(
                    eps_values[best_idx],
                    color="black",
                    linestyle=":",
                    label=f"Selected ε = {eps_values[best_idx]:.3f} | Threshold = {thresh}",
                    linewidth = 2
                )

            ax2.plot(
                    eps_values,
                    clusters_average,
                    color=threshold_colors[i],
                    marker = "*",
                    linestyle=":",
                    label=f"Clusters Average | Threshold = {thresh}"
                )

            ax2.plot(
                    eps_values,
                    clusters_average_dist,
                    color=threshold_colors[i],
                    marker = "^",
                    linestyle=":",
                    label=f"Clusters Average Distance | Threshold = {thresh}"
                )


            ax1.set_xlabel("Epsilon (m)")
            ax1.set_ylabel("Rates")
            ax2.set_ylabel("Cluster metrics")
            ax1.set_ylim(0,1)
            ax1.grid(alpha=0.3)

            axis1, labels1 = ax1.get_legend_handles_labels()
            axis2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(axis1 + axis2, labels1 + labels2,  fontsize=8, ncol=2)

            plt.title("Hit & Miss Rate vs Cluster Count")
            plt.savefig(f"Trade_off_curve_thresh_{thresh:.3f}.png", dpi=300)
            plt.show()

            print(
                    "\n DBSCAN epsilon\n"
                    f"Threshold = {thresh:.3f} \n"
                    f"Best epsilon = {eps_values[best_idx]:.3f} m\n"
                    f"Hit rate = {hit_rates[best_idx]:.3f}\n"
                    f"Miss rate = {miss_rates[best_idx]:.3f}"
                )


hit_miss_rates_bar_SBS(stats_global)

print("\n=== Overall Average Across All Trials ===")

for (thresh, (eps, mins)), stats in stats_global.items():
    total_hits = stats["hits"].sum()
    total_misses = stats["misses"].sum()

    hit_rate = total_hits / (total_hits + total_misses + 1e-9)
    miss_rate = total_misses / (total_hits + total_misses + 1e-9)

    print(
            f"Threshold={thresh:.3f} | "
            f"eps={eps:.3f} | min_samples={mins} | "
            f"AvgHitRate={hit_rate:.3f} | "
            f"Frames={stats['frame_count']} | "
            f"AvgMissRate={miss_rate:.4f}")
