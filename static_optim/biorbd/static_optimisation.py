import biorbd
import numpy as np
from scipy import interpolate
# from casadi import MX, Function
import scipy.io as sio
import matplotlib.pyplot as plt
import os
def reduce(n_frames_expected, data):
    data_reduce = []
    n_shooting = 0
    if len(data.shape) == 2:
        n_shooting = data.shape[1]
    elif len(data.shape) == 3:
        n_shooting = data.shape[2]
    else:
        print('Wrong size of data')

    if n_shooting % n_frames_expected == 0:
        n_frames = n_frames_expected
    else:
        c = 1
        d = 1
        while n_shooting % (n_frames_expected - c) != 0:
            c += 1
        while n_shooting % (n_frames_expected + d) != 0:
            d += 1
        if c > d:
            n_frames = n_frames_expected + d
        else:
            n_frames = n_frames_expected - c

    k = int(n_shooting / n_frames)
    if len(data.shape) == 2:
        data_reduce = np.ndarray((data.shape[0], n_frames))
        for i in range(n_frames):
            data_reduce[:, i] = data[:, k * i]
    elif len(data.shape) == 3:
        data_reduce = np.ndarray((data.shape[0], data.shape[1], n_frames))
        for i in range(n_frames):
            data_reduce[:, :, i] = data[:, :, k * i]
    else:
        print('Wrong size of data')

    return data_reduce, n_frames

data ='flex' #'abd_co' #'horizon_co'  #
parent_dir_path = os.path.split(os.path.dirname(__file__))[0]
model = "arm_Belaise_v2_scaled.bioMod"
biorbd_model = biorbd.Model(parent_dir_path + "/models/" + model)
n_frames_wanted = 31

# Data to track
# --- Markers --- #
mat_contents = sio.loadmat(parent_dir_path + f"/data_real/sujet_5/marker_{data}.mat")
# marker_treat = mat_contents['marker_reduce']
marker_treat = mat_contents['marker_treat'][:-1]
marker_treat = marker_treat[:, -biorbd_model.nbMarkers():, :]
marker_treat = reduce(n_frames_wanted, marker_treat)[0]

# --- X_init --- #
mat_contents = sio.loadmat(parent_dir_path + f"/data_real/sujet_5/states_ekf_wo_RT_{data}.mat")
x_init = mat_contents["x_init"]
x_init = reduce(n_frames_wanted, x_init)[0]

# --- EMG --- #
emg_treat = x_init[-biorbd_model.nbMuscles():, :-1]

n_shooting_points = marker_treat.shape[2] - 1
t = np.linspace(0, 0.24, n_shooting_points + 1) # 0.24s
final_time = t[n_shooting_points]
q_ref = x_init[6:11, :]

# --- joint velocity approximation --- #

# n_shooting_points = marker_treat.shape[2] - 1
#
# # Read data to fit
# t = np.linspace(0, final_time, n_shooting_points + 1)

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
Q_vec = biorbd.VecBiorbdGeneralizedCoordinates()
Qdot_vec = biorbd.VecBiorbdGeneralizedVelocity()
Qddot = biorbd.GeneralizedAcceleration(biorbd_model)
q_recons = np.ndarray((biorbd_model.nbQ(), len(markersOverFrames)))
q_dot_recons = np.ndarray((biorbd_model.nbQ(), len(markersOverFrames)))
q_ddot_recons = np.ndarray((biorbd_model.nbQ(), len(markersOverFrames)))
Tau = biorbd.VecBiorbdGeneralizedTorque()

for i, targetMarkers in enumerate(markersOverFrames):
    kalman.reconstructFrame(biorbd_model, targetMarkers, Q, Qdot, Qddot)
    q_recons[:, i] = Q.to_array()
    q_dot_recons[:, i] = Qdot.to_array()
    q_ddot_recons[:, i] = Qddot.to_array()

# for i in range(q_recons.shape[0]):
#     x = t
#     y = q_recons[i, :]
#     tck = interpolate.splrep(x, y, s=0)
#     xnew = t
#     ynew = interpolate.splev(xnew, tck, der=1)
#     plt.figure()
#     plt.plot(x, y, xnew, ynew)
#     plt.legend(['Linear', 'Cubic Spline', 'True'])

for i, targetMarkers in enumerate(markersOverFrames):
    kalman.reconstructFrame(biorbd_model, targetMarkers, Q, Qdot, Qddot)
    Q_vec.append(Q)
    Qdot_vec.append(Qdot)
    q_recons[:, i] = Q.to_array()
    q_dot_recons[:, i] = Qdot.to_array()
    q_ddot_recons[:, i] = Qddot.to_array()
    Tau.append(biorbd_model.InverseDynamics(Q, Qdot, Qddot))

Tau_temp = []
for i in range(q_recons.shape[1]):
    Tau_temp.append(Tau[i].to_array())
Tau_tot = np.ndarray((q_recons.shape[0], q_recons.shape[1]))
for i in range(q_recons.shape[1]):
    for j in range(q_recons.shape[0]):
        Tau_tot[j, i] = Tau_temp[i][j]

# Proceed with the static optimization
optim = biorbd.StaticOptimization(biorbd_model, Q_vec, Qdot_vec, Tau)
optim.run()
muscleActivationsPerFrame = optim.finalSolution()

# Print them to the console
activation = []
for activations in muscleActivationsPerFrame:
    activation.append(activations.to_array())
activations_so = np.ndarray((biorbd_model.nbMuscles(),q_recons.shape[1]))
for i in range(biorbd_model.nbMuscles()):
    for j in range(q_recons.shape[1]):
        activations_so[i, j] = activation[j][i]

dict = {'activations_so': activations_so, 'tau': Tau_tot}
sio.savemat(f'activation_so_{data}.mat', dict)