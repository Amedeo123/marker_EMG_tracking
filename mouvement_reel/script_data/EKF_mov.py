from pyomeca import Markers, Rototrans
import numpy as np
from scipy.interpolate import interp1d
import biorbd
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path
import scipy.interpolate
try:
    import bioviz
    bioviz_found = True
except ModuleNotFoundError:
    bioviz_found = False


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
        while n_shooting % (n_frames_expected - c) != 0:
            if c > 20:
                if len(data.shape) == 2:
                    n_shooting = n_shooting - 1
                    c = 0
                elif len(data.shape) == 3:
                    n_shooting = n_shooting-1
                    c = 0
            c += 1
        n_frames = n_frames_expected - c
        if len(data.shape) == 2:
            data = data[:, :n_shooting]
        elif len(data.shape) == 3:
            data = data[:, :, :n_shooting]

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


sujet = "5"
data = "flex_co" #abd_co #flex_co #"horizon_co"
data_path = f'/home/amedeo/Documents/programmation/marker_emg_tracking/mouvement_reel/results/'
model_path = '/home/amedeo/Documents/programmation/marker_emg_tracking/models/'
model = 'arm_Belaise_real_v3_scaled.bioMod'
biorbd_model = biorbd.Model(model_path + model)
nb_try = 1
dic = {}
for i in range(nb_try):
    tries = str(i)
    mat_contents = sio.loadmat(data_path + f"sujet_{sujet}/data_{data}_treat.mat")
    marker_treat = mat_contents[f"marker_try_{tries}"][:, 2:, :]
    emg_norm = mat_contents[f"emg_try_{tries}"]
    t_final = float(marker_treat.shape[2]/mat_contents['marker_rate'])
    t = np.linspace(0, t_final, marker_treat.shape[2])
    # Dispatch markers in biorbd structure so EKF can use it
    markersOverFrames = []

    for k in range(marker_treat.shape[2]):
        markersOverFrames.append([biorbd.NodeSegment(m) for m in marker_treat[:, :, k].T])

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
    for k, targetMarkers in enumerate(markersOverFrames):
        kalman.reconstructFrame(biorbd_model, targetMarkers, Q, Qdot, Qddot)
        q_recons[:, k] = Q.to_array()
        q_dot_recons[:, k] = Qdot.to_array()

    # --- Visualize the results --- #
    # -- Plot Q -- #
    q_new = np.ndarray((biorbd_model.nbQ(), len(markersOverFrames)))
    qdot_new = np.ndarray((biorbd_model.nbQ(), len(markersOverFrames)))
    for k in range(q_recons.shape[0]):
        b = np.poly1d(np.polyfit(t, q_recons[k, :], 4))
        c = np.poly1d(np.polyfit(t, q_dot_recons[k, :], 4))
        q_new[k, :] = b(t)
        qdot_new[k, :] = c(t)

    for k in range(q_recons.shape[0]):
        plt.subplot(5, 4, k+1)
        plt.plot(t, q_recons[k, :])
        # plt.plot(t, q_dot_recons[k, :])
        # plt.plot(t, qdot_new[k, :])
        plt.plot(t, q_new[k, :])
    plt.show()
    if bioviz_found:
        b = bioviz.Viz(loaded_model=biorbd_model)
        b.load_movement(q_new)
        b.exec()


    # Save initial states in .mat file
    emg_norm = reduce(marker_treat.shape[2], emg_norm)[0]
    # x_init = np.concatenate((q_recons, q_dot_recons, emg_norm))
    x_init = np.concatenate((q_new, qdot_new, emg_norm))
    dic[f'x_init_{tries}'] = x_init
    dic[f't_final_{tries}'] = t_final
sio.savemat(data_path + f"sujet_5/states_init_{data}.mat", dic)

