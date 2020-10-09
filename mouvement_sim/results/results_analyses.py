import numpy as np
import biorbd
import pickle
import scipy.io as sio
from casadi import MX, Function, vertcat, jacobian
from matplotlib import pyplot as plt
import BiorbdViz
from biorbd_optim import (
    OptimalControlProgram,
    Data,
    ShowResult,
)
mat_contents = sio.loadmat("./data/excitations.mat")
excitations_ref = mat_contents["excitations"]
excitations_ref = excitations_ref[:, :]

mat_contents = sio.loadmat("./data/markers.mat")
marker_ref = mat_contents["markers"]
marker_ref = marker_ref[:, 4:, :]

mat_contents = sio.loadmat("./data/x_init.mat")
x_init = mat_contents["x_init"][: 2 * 6, :]

ocp, sol = OptimalControlProgram.load("activation_driven_ipopt.bo")
states, controls = Data.get_data(ocp, sol["x"])
biorbd_model = biorbd.Model("arm_Belaise.bioMod")
q = states["q"]
qdot = states["q_dot"]
x = vertcat(states["q"], states["q_dot"])
u = controls["muscles"]
nlp = ocp.nlp[0]

result = ShowResult(ocp, sol)
# result.animate()
# result.graphs()

with open("sol_marker_activation_tracking_acados.bob", 'rb' ) as file :
    data = pickle.load(file)
states_ac = data['data'][0]
controls_ac = data['data'][1]
q_ac = states_ac['q']
qdot_ac = states_ac['q_dot']
u_ac = controls_ac['muscles']
final_time = 1
n_shooting_points = q_ac.shape[1]
t = np.linspace(0, final_time, n_shooting_points)
for i in range(q_ac.shape[0]):

    if i == 5:
        plt.subplot(2, 3, i + 1)
        plt.plot(t, q_ac[i, :], label='Acados')
        plt.plot(t, q[i, :], label='Ipopt')
    elif i == 1 :
        plt.subplot(2, 3, i + 1)
        plt.plot(t, q_ac[i, :])
        plt.plot(t, q[i, :])
        plt.title("Q")
    else:
        plt.subplot(2, 3, i + 1)
        plt.plot(t, q_ac[i, :])
        plt.plot(t, q[i, :])
plt.legend()
plt.show()
for i in range(q_ac.shape[0]):
    if i == 5:
        plt.subplot(2, 3, i + 1)
        plt.plot(t, qdot_ac[i, :], label='Acados')
        plt.plot(t, qdot[i, :], label='Ipopt')
    elif i == 1 :
        plt.subplot(2, 3, i + 1)
        plt.plot(t, qdot_ac[i, :])
        plt.plot(t, qdot[i, :])
        plt.title("QDot")
    else:
        plt.subplot(2, 3, i + 1)
        plt.plot(t, qdot_ac[i, :])
        plt.plot(t, qdot[i, :])
plt.legend()
plt.show()
for i in range(u.shape[0]):
    if i == 19:
        plt.subplot(4,5,i+1)
        plt.step(t, u_ac[i,:], label='Acados')
        plt.step(t, u[i, :], label='Ipopt')
        plt.step(t, excitations_ref[i,:], label='Target')
    elif i == 2:
        plt.subplot(4, 5, i + 1)
        plt.step(t, u_ac[i, :])
        plt.step(t, u[i, :])
        plt.step(t, excitations_ref[i, :])
        plt.title("U")
    else:
        plt.subplot(4, 5, i + 1)
        plt.step(t, u_ac[i, :])
        plt.step(t, u[i, :])
        plt.step(t, excitations_ref[i, :])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()
q = q_ac
n_mark = 24
n_q = q.shape[0]
n_frames = n_shooting_points
markers = np.ndarray((3, n_mark, q.shape[1]))
symbolic_states = MX.sym("x", n_q, 1)
markers_func = Function(
    "ForwardKin", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q"], ["markers"]
).expand()
for i in range(n_frames):
    markers[:, :, i] = markers_func(q[:, i])

plt.figure("Markers")
for i in range(markers.shape[1]):
    if i == 1 :
        plt.plot(np.linspace(0, 1, n_shooting_points ), marker_ref[:, i, :].T, "k", label='marker_ref')
        plt.plot(np.linspace(0, 1, n_shooting_points ), markers[:, i, :].T, "r--", label='marker_forwardkin')
    else :
        plt.plot(np.linspace(0, 1, n_shooting_points ), marker_ref[:, i, :].T, "k")
        plt.plot(np.linspace(0, 1, n_shooting_points ), markers[:, i, :].T, "r--")
plt.xlabel("Time")
plt.ylabel("Markers Position")
plt.legend()
plt.show()
b = BiorbdViz.BiorbdViz(loaded_model=biorbd_model)
b.load_movement(q_ac)
b.exec()