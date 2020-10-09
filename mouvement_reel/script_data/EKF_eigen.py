from pyomeca import Markers, Rototrans
import numpy as np
from scipy.interpolate import interp1d
import biorbd as bd
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path
try:
    import BiorbdViz
    biorbd_viz_found = True
except ModuleNotFoundError:
    biorbd_viz_found = False

parent_dir_path = os.path.split(os.path.dirname(__file__))[0]
biorbd = bd
# Data to track
data = "flex" #abd_co #flex_co #"horizon_co"
mat_contents = sio.loadmat(f"{data}.mat")
mat_contents = mat_contents["mov_reel"]
val = mat_contents[0,0]
integ = val['integ']
EMG = val['EMG']
emg_treat = EMG['EMG_norm1'][0,0]
# biorbd_model = bd.Model(parent_dir_path +'/models/arm_Belaise_v2_scaled.bioMod')
biorbd_model = bd.Model(parent_dir_path +'/models/arm_Belaise_v2_EKF_scaled.bioMod')

# -- Markers exp -- #
data_path = f'./sujet_5/{data}.c3d'
markers_names = ["XIPH", "STER","CLAV_SC","CLAV_AC","SCAP_CP","SCAP_AA","SCAP_TS","SCAP_IA","DELT","ARMl","EPICl",
           "EPICm","LARM_ant","LARM_post","LARM_elb","LARM_dist"]
#
markers = Markers.from_c3d(data_path, usecols=markers_names)
marker_treat = np.ndarray((4, markers.shape[1], emg_treat.shape[1]))
for i in range(3):
    marker_treat[i, :, :emg_treat.shape[1]] = markers[i, :, :emg_treat.shape[1]]*1e-3

n_mark = biorbd_model.nbMarkers()
n_shooting_points = marker_treat.shape[2]
n_frames = n_shooting_points
# c = 1
# for i in range(3):
#     for l in range(n_mark):
#         for k in range(n_frames):
#             if k == 0:
#                 if np.isnan(marker_treat[i, l, k]) == True:
#                     while np.isnan(markers[i, l, k + c]) == True:
#                         c += 1
#                     marker_treat[i, l, k] = markers[i, l, k + c]*1e-3
#                     c = 1
#             else:
#                 if np.isnan(marker_treat[i, l, k]) == True:
#                     while np.isnan(markers[i, l, k + c]) == True:
#                         c += 1
#                     yb = markers[i, l, k + c]*1e-3
#                     xb = k + c
#                     ya = marker_treat[i, l, k - 1]
#                     xa = k - 1
#                     c = (yb - ya) / (xb - xa)
#                     d = ya - c * xa
#                     marker_treat[i, l, k] = c * k + d
#                     c = 1
marker_treat[3, :, :] = [1]


# marker_treat = mat_contents['marker_rot'][:-1]
# if reduce:
#     if marker_treat.shape[2] % 2 == 0:
#         marker_reduce = np.ndarray((3, marker_treat.shape[1], int(marker_treat.shape[2] / 2)))
#         emg_reduce = np.ndarray((emg_treat.shape[0], int(emg_treat.shape[1] / 2)))
#         for i in range(int(marker_treat.shape[2] / 2)):
#             marker_reduce[:, :, i] = marker_treat[:, :, i * 2]
#             emg_reduce[:, i] = emg_treat[:, :, i * 2]
#
#     else:
#         marker_treat = marker_treat[:, :, :-1]
#         emg_treat = emg_treat[:, :-1]
#         marker_reduce = np.ndarray((3, marker_treat.shape[1], int(marker_treat.shape[2] / 2)))
#         emg_reduce = np.ndarray((emg_treat.shape[0], int(emg_treat.shape[1] / 2)))
#         for i in range(int(marker_treat.shape[2] / 2)):
#             marker_reduce[:, :, i] = marker_treat[:, :, i * 2]
#             emg_reduce[:, i] = emg_treat[:, i * 2]
#     emg_treat = emg_reduce
#     marker_treat = marker_reduce

final_time = 1
n_shooting_points = marker_treat.shape[2]
t = np.linspace(0, final_time, n_shooting_points)

# Dispatch markers in biorbd structure so EKF can use it
markersOverFrames = []
for i in range(marker_treat.shape[2]):
    markersOverFrames.append([biorbd.NodeSegment(m) for m in marker_treat[:, :, i].T])

# Create a Kalman filter structure
freq = 100 # Hz
params = biorbd.KalmanParam(freq)
kalman = biorbd.KalmanReconsMarkers(biorbd_model, params)

# Perform the kalman filter for each frame (the first frame is much longer than the next)
Q = biorbd.GeneralizedCoordinates(biorbd_model)
Qdot = biorbd.GeneralizedVelocity(biorbd_model)
Qddot = biorbd.GeneralizedAcceleration(biorbd_model)
q_recons = np.ndarray((biorbd_model.nbQ(), len(markersOverFrames)))
q_dot_recons = np.ndarray((biorbd_model.nbQ(), len(markersOverFrames)))
for i, targetMarkers in enumerate(markersOverFrames):
    kalman.reconstructFrame(biorbd_model, targetMarkers, Q, Qdot, Qddot)
    q_recons[:, i] = Q.to_array()
    q_dot_recons[:, i] = Qdot.to_array()

# -- Plot Q -- #
for k in range(q_recons.shape[0]):
    plt.subplot(5, 4, k+1)
    plt.plot(t, q_recons[k, :])
plt.show()

# Save initial states in .mat file

x_init = np.concatenate((q_recons, q_dot_recons, emg_treat))
dic = {"x_init": x_init}
# sio.savemat(f"./Sujet_5/states_ekf_{data}.mat", dic)
dic = {"marker_reduce": marker_treat}
# sio.savemat(f"./Sujet_5/marker_{data}.mat", dic)
# Animate the results if biorbd viz is installed

# if biorbd_viz_found:
#     b = BiorbdViz.BiorbdViz(loaded_model=biorbd_model)
#     b.load_movement(q_recons)
#     b.exec()
