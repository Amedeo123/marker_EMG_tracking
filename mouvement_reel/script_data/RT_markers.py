import biorbd
import numpy as np
from pyomeca import Markers
from pyomeca import Rototrans
import scipy.io as sio
import os.path
import matplotlib.pyplot as plt
from casadi import MX, SX, Function, vertcat, nlpsol
import csv

#
# def RT(transX, transY,transZ, angleX, angleY, angleZ):
#     Rototrans = np.array((
#         (np.cos(angleZ)*np.cos(angleY), np.cos(angleZ)*np.sin(angleY)*np.sin(angleX)-np.sin(angleZ)*np.cos(angleX), np.cos(angleZ)*np.sin(angleY)*np.cos(angleX)+np.sin(angleZ)*np.sin(angleX), transX),
#         (np.sin(angleZ)*np.cos(angleY), np.sin(angleZ)*np.sin(angleY)*np.sin(angleX)+np.cos(angleZ)*np.cos(angleX), np.sin(angleZ)*np.sin(angleY)*np.cos(angleX)-np.cos(angleZ)*np.sin(angleX), transY),
#         (-np.sin(angleY), np.cos(angleY)*np.sin(angleX), np.cos(angleY)*np.cos(angleX), transZ),
#         (0, 0, 0, 1)))
#     return Rototrans


J = 0
w = []
lbw = []
ubw = []

transX = SX.sym('transX', 1)
transY = SX.sym('transY', 1)
transZ = SX.sym('transZ', 1)

angleX = SX.sym('angleX', 1)
angleY = SX.sym('angleY', 1)
angleZ = SX.sym('angleZ', 1)


for i in range(3):
    lbw.append(-10)
    ubw.append(10)
for i in range(3):
    lbw.append(-np.pi)
    ubw.append(np.pi)

# Rototrans_sym = RT(transX, transY, transZ, angleX, angleY, angleZ)
parent_dir_path = os.path.split(os.path.dirname(__file__))[0]
# biorbd_model = biorbd.Model(parent_dir_path +'/model_scaling/new_scal/Belaise_scaled_updated.bioMod')
biorbd_model = biorbd.Model(parent_dir_path +'/models/arm_Belaise_real_v2.bioMod')
data = "flex"
data_path = f'./Sujet_5/{data}.c3d'

markers_names = ["CLAV_SC","CLAV_AC","SCAP_CP","SCAP_AA","SCAP_TS","SCAP_IA","DELT","ARMl","EPICl",
           "EPICm","LARM_ant","LARM_post","LARM_elb","LARM_dist"]
# markers_names = ["XIPH", "STER", "CLAV_SC","CLAV_AC","SCAP_CP","SCAP_AA","SCAP_TS","SCAP_IA","DELT","ARMl","EPICl",
#            "EPICm","LARM_ant","LARM_post","LARM_elb","LARM_dist","STYLrad", "STYLulna"]

markers = Markers.from_c3d(data_path, usecols=markers_names)

mat_contents = sio.loadmat(f"{data}.mat")
mat_contents = mat_contents["mov_reel"]
val = mat_contents[0,0]
integ = val['integ']
q = integ['Q_reel'][0,0]
marker_exp = markers[:, :, :q.shape[1]].data*1e-3
n_mark = len(markers_names)
n_q = q.shape[0]
t = integ['temps'][0,0]      # 0.24s
n_frames = q.shape[1]

marker_model = np.ndarray((4, n_mark, n_frames))
symbolic_states = MX.sym("q", n_q, 1)
# markers_func = Function(
#     "ForwardKin", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q"], ["marker_model"]
# ).expand()
# for i in range(n_frames):
#     marker_model[:-1, :, i] = markers_func(q[:, i])
#     marker_model[3,:,i] = [1]
marker_treat = marker_exp[:, :, :n_frames]
c = 1
for i in range(3):
    for l in range(n_mark):
        for k in range(n_frames):
            if k == 0:
                if np.isnan(marker_treat[i, l, k]) == True:
                    while np.isnan(marker_exp[i, l, k + c]) == True:
                        c += 1
                    marker_treat[i, l, k] = marker_exp[i, l, k + c]
                    c = 1
            else:
                if np.isnan(marker_treat[i, l, k]) == True:
                    while np.isnan(marker_exp[i, l, k + c]) == True:
                        c += 1
                    yb = marker_exp[i, l, k + c]
                    xb = k + c
                    ya = marker_treat[i, l, k - 1]
                    xa = k - 1
                    c = (yb - ya) / (xb - xa)
                    d = ya - c * xa
                    marker_treat[i, l, k] = c * k + d
                    c = 1
# for i in range(8):
#     marker_treat[:,3,832+i] = marker_treat[:,3,832]
# marker_treat[:, 11, 837] = marker_treat[:, 11, 836]
marker_treat[3,:,:] = [1]
# dic = {"marker_treat" : marker_treat}
# sio.savemat(f"./Sujet_5/marker_scaling_{data}.mat", dic)
# mat_contents = sio.loadmat(parent_dir_path+'/model_scaling/new_scal/marker_flex.mat')
# # marker_treat = mat_contents['marker_reduce']
# marker_treat = mat_contents['marker_rot']

marker_rotate = marker_treat
# rt = RT((Rototrans(transX, transY,transZ, angleX, angleY, angleZ)))
# marker_rotate = Markers.from_rototrans(marker_exp, rt)
# for i in range(n_mark):
#     if np.isnan(marker_treat[1,i,0]) != True:
#         J = J + sum((marker_model[:, i, 0] - np.sum(Rototrans_sym * marker_treat[:, i, 0], axis=1))**2)
#
# w = vertcat(transX,transY,transZ,angleX,angleY,angleZ)
# prob = {'f':J, 'x':w}
# options = {'ipopt.hessian_approximation':"exact",
#            # 'ipopt.tol':1e-10,'ipopt.dual_inf_tol':1e-15
#            }
# solver = nlpsol('solver', 'ipopt', prob, options)
# w0 = np.zeros((6,1))
# # w0[0]= marker_model[0,0,0] - marker_treat[0,0,0]
# # w0[1]= marker_model[1,0,0] - marker_treat[1,0,0]
# # w0[2]= marker_model[2,0,0] - marker_treat[2,0,0]
#
# solve = solver(x0 = w0, lbx=-1000, ubx=10000)
# w_opt = solve['x']
# print(w_opt)
# rt = Rototrans((RT(w_opt[0],w_opt[1],w_opt[2],w_opt[3],w_opt[4],w_opt[5])))
# marker_rotate = Markers.from_rototrans(marker_treat, rt)
marker_model = np.ndarray((4, n_mark, n_frames))
symbolic_states = MX.sym("q_recons", n_q, 1)
mat_content = sio.loadmat('Sujet_5/states_ekf_wo_RT_flex.mat')
q_recons = mat_content['x_init'][:n_q]
markers_func = Function(
    "ForwardKin", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q_recons"], ["marker_model"]
).expand()
for i in range(n_frames):
    marker_model[:-1, :, i] = markers_func(q_recons[:, i])
    marker_model[3,:,i] = [1]
# Data for three-dimensional scattered points
ax = plt.axes(projection='3d')
zdata = marker_rotate[2,:,0]
xdata = marker_rotate[0,:,0]
ydata = marker_rotate[1,:,0]
ax.scatter3D(xdata, ydata, zdata, cmap='Greens')
zdata = marker_model[2,:,0]
xdata = marker_model[0,:,0]
ydata = marker_model[1,:,0]
ax.scatter3D(xdata, ydata, zdata, edgecolors="red")
ax.view_init(60, 35)
plt.show()

nb_frame = range(1,marker_rotate.shape[2]+1)
anb_frame = np.ndarray((1,marker_rotate.shape[2]))
for i in range(len(nb_frame)):
     anb_frame[:,i] = nb_frame[i]
t = np.linspace(0, 0.24, num=marker_rotate.shape[2]).reshape((1,marker_rotate.shape[2]))
marker_treat = np.reshape(marker_rotate[:-1,:,:], (marker_rotate.shape[1]*3, marker_rotate.shape[2]), order="F")
marker_treat = np.concatenate((anb_frame, t, marker_treat), axis=0)

with open(parent_dir_path+'/model_scaling/marker_scaling.trc', "w") as markers:
     writer = csv.writer(markers, delimiter='\t')
     writer.writerows(marker_treat.T)


#
# rt = Rototrans((RT(w_opt[0],w_opt[1],w_opt[2],w_opt[3],w_opt[4],w_opt[5])))
# marker_rotate = Markers.from_rototrans(marker_treat, rt)
# dic = {"marker_rot" : marker_rotate, "marker_without_rot" : marker_treat}
# sio.savemat("./Sujet_5/marker_flex.mat", dic)
