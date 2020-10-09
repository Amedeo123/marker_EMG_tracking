from pyomeca import Analogs, Markers
from pyomeca import Rototrans as RT
import numpy as np
import scipy.io as sio
import pandas as pd
import xarray
import os.path
from scipy.interpolate import UnivariateSpline, interp1d, interp2d
import csv
import biorbd as bd
import matplotlib.pyplot as plt

# --- General informations --- #
sujet = "5"
data = "horizon" #abd_co #flex_co #"horizon_co"
data_path = f'./sujet_{sujet}/{data}.c3d'
model = bd.Model(os.path.split(os.path.dirname(__file__))[0]+'/models/arm_Belaise_real_v2.bioMod')

# --- EMG --- #
muscles_names = ["pect.IM EMG7","deltpost.EMG3","triceps.EMG5", "deltant.EMG1", "deltmed.EMG2", "ssp.EMG8","isp.EMG9",
                 "subs.EMG10","biceps.EMG4","uptrap.EMG6", ]

a = Analogs.from_c3d(data_path, usecols=muscles_names)
emg_rate = int(a.rate)
emg = (
    a.meca.band_pass(order=4, cutoff=[10, 425])
    .meca.center()
    .meca.abs()
    .meca.low_pass(order=4, cutoff=5, freq=a.rate)
    # .meca.normalize()
)
# emg.plot(x="time", col="channel", col_wrap=3)
# plt.show()
# emg.meca.to_matlab("EMG_flex3.mat")

# --- Markers --- #
markers_full_names = ["XIPH","STER", "STERback", "CLAV_SC","CLAV_AC","SCAP_CP","SCAP_AA","SCAP_TS","SCAP_IA","DELT","ARMl",
           "EPICl","EPICm","LARM_ant","LARM_post","LARM_elb","LARM_dist"]

# markers_full_names = ["ASISr","PSISr", "PSISl","ASISl","XIPH","STER","STERlat","STERback","XIPHback","ThL",
#            "CLAV_SC","CLAV_AC","SCAP_CP","SCAP_AA","SCAPspine","SCAP_TS","SCAP_IA","DELT","ARMl",
#            "EPICl","EPICm","LARM_ant","LARM_post","LARM_elb","LARM_dist","STYLrad","STYLrad_up","STYLulna_up",
#            "STYLulna","META2dist","META2prox","META5prox","META5dist","MAIN_opp"]
# markers_full_names = ["CLAV_SC","CLAV_AC","SCAP_CP","SCAP_AA","SCAP_TS","SCAP_IA","DELT","ARMl","EPICl",
#            "EPICm","LARM_ant","LARM_post","LARM_elb","LARM_dist"]

markers_full = Markers.from_c3d(data_path, usecols=markers_full_names)
marker_rate = int(markers_full.rate)
marker_exp = markers_full[:, :, :].data * 1e-3
n_mark = marker_exp.shape[1]
# t = 0.24
n_frames = marker_exp.shape[2]
t = markers_full.time.data
marker_treat = np.ndarray((marker_exp.shape[0], marker_exp.shape[1], marker_exp.shape[2]))
for k in range(n_mark):
    a = pd.DataFrame(marker_exp[:, k, :])
    a = np.array(a.interpolate(method='polynomial', order=3, axis=1))
    marker_treat[:, k, :] = a
marker_treat[3, :, :] = [1]

plt.figure("Markers")
for i in range(marker_treat.shape[1]):
    plt.plot(t, marker_treat[:-1, i, :].T, "k")
    plt.plot(t, marker_exp[:-1, i, :].T, "r--")
plt.xlabel("Time")
plt.ylabel("Markers Position")
plt.show()

# --- MVC --- #
mvc_list = ["biceps_1", "biceps_2","deltant_1","deltant_2","deltmed_1","deltmed_2","deltpost_1", "deltpost_2",
            "isp_1", "isp_2", "pect_1", "pect_2", "ssp_1", "ssp_2", "subs_1","subs_2", "subs_3", "subs_4",
            "triceps_1", "triceps_2", "uptrap_1", "uptrap_2"]

mvc_list_max = np.ndarray((len(muscles_names), 2000))
for i in range(len(mvc_list)):
    b = Analogs.from_c3d(f"./sujet_5/mvc/{str(mvc_list[i])}.c3d", usecols=muscles_names)
    mvc_temp = (
    b.meca.band_pass(order=4, cutoff=[10, 425])
    .meca.center()
    .meca.abs()
    .meca.low_pass(order=4, cutoff=5, freq=b.rate)
    # .meca.normalize()
    )
    mvc_temp = -np.sort(-mvc_temp.data, 1)
    if i == 0:
        mvc_list_max = mvc_temp[:, :2000]
    else:
        mvc_list_max = np.concatenate((mvc_list_max, mvc_temp[:, :2000]), 1)

mvc_list_max = -np.sort(-mvc_list_max)[:, :2000]
mvc_list_max = np.median(mvc_list_max, 1)
# sio.savemat(f"./sujet_{sujet}/mvc_sujet_{sujet}.mat", {'mvc_treat': mvc_list_max})

emg_norm_tmp = np.ndarray((10, emg.shape[1]))
emg_norm = np.ndarray((18, emg.shape[1]))
for i in range(len(muscles_names)):
    emg_norm_tmp[i, :] = emg[i, :]/mvc_list_max[i]

# muscles_names = ["pect.IM EMG7","deltpost.EMG3","triceps.EMG5", "deltant.EMG1", "deltmed.EMG2", "ssp.EMG8","isp.EMG9",
#                  "subs.EMG10","biceps.EMG4","uptrap.EMG6", ]

emg_norm[[0, 1, 6], :] = emg_norm_tmp[0, :]
emg_norm[2, :] = emg_norm_tmp[1, :]
emg_norm[[3, 4, 5], :] = emg_norm_tmp[2, :]
emg_norm[7, :] = emg_norm_tmp[3, :]
emg_norm[8, :] = emg_norm_tmp[4, :]
emg_norm[9, :] = emg_norm_tmp[5, :]
emg_norm[10, :] = emg_norm_tmp[6, :]
emg_norm[11, :] = emg_norm_tmp[7, :]
emg_norm[[12, 13], :] = emg_norm_tmp[8, :]
emg_norm[[14, 15, 16, 17], :] = emg_norm_tmp[9, :]

# --- Cut by try --- #
try_time = [1.22, 3.17, 5.0, 6.76, 8.86, 11.05, 13.55]
nb_try = len(try_time)-1
try_marker_frames = [int(i * marker_rate) for i in try_time]
try_emg_frames = [int(i * emg_rate) for i in try_time]
emg_mov = []
marker_mov = []
dict = {"marker_treat": marker_treat, "emg_norm": emg_norm, 'emg_raw': emg, 'mvc': mvc_list_max,
        "emg_rate": emg_rate, "marker_rate": marker_rate}

for i in range(nb_try):
    dict["marker_try_" + str(i)] = marker_treat[:, :, try_marker_frames[i]:try_marker_frames[i+1]]
    dict["emg_try_" + str(i)] = emg_norm[:, try_emg_frames[i]:try_emg_frames[i+1]]

sio.savemat(f"./sujet_{sujet}/data_{data}_treat.mat", dict)

# --- Export .sto --- #
# nb_frame = range(1, marker_treat.shape[2]+1)
# anb_frame = np.ndarray((1, marker_treat.shape[2]))
# for i in range(len(nb_frame)):
#      anb_frame[:,i] = nb_frame[i]
# t = np.linspace(0, 0.24, num=marker_treat.shape[2]).reshape((1,marker_treat.shape[2]))
# marker_mot = np.reshape(marker_treat[:-1, :, :], (marker_treat.shape[1]*3, marker_treat.shape[2]), order="F")
# marker_mot = np.concatenate((anb_frame, t, marker_mot), axis=0)

# with open(os.path.split(os.path.dirname(__file__))[0]+'/model_scaling/marker_full_rot.trc', "w") as markers:
#      writer = csv.writer(markers, delimiter='\t')
#      writer.writerows(marker_treat.T)