import numpy as np
import biorbd
import pickle
import scipy.io as sio
from casadi import MX, Function, vertcat, jacobian
from matplotlib import pyplot as plt
# import BiorbdViz
import os
import csv
from biorbd_optim import (
    OptimalControlProgram,
    Data,
    ShowResult,
)


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


def muscles_tau(states, controls, nlp):
    nq = nlp.mapping['q'].reduce.len
    q = nlp.mapping['q'].expand.map(states[:nq])
    qdot = nlp.mapping['q_dot'].expand.map(states[nq:])

    muscles_states = biorbd.VecBiorbdMuscleState(nlp.model.nbMuscles())
    for k in range(nlp.model.nbMuscles()):
        muscles_states[k].setActivation(controls[k])
    muscles_tau = nlp.model.muscularJointTorque(muscles_states, q, qdot).to_mx()

    return muscles_tau


def muscles_force(states, controls, nlp):
    nq = nlp.mapping['q'].reduce.len
    q = nlp.mapping['q'].expand.map(states[:nq])
    qdot = nlp.mapping['q_dot'].expand.map(states[nq:])

    muscles_states = biorbd.VecBiorbdMuscleState(nlp.model.nbMuscles())
    for k in range(nlp.model.nbMuscles()):
        muscles_states[k].setActivation(controls[k])

    return biorbd_model.muscleForces(muscles_states, q, qdot).to_mx()


data = 'flex' #"horizon_co" #flex #abd_co
parent_dir_path = os.path.split(os.path.dirname(__file__))[0]
model = "arm_Belaise_v2_scaled.bioMod"
biorbd_model = biorbd.Model(parent_dir_path + "/models/" + model)
n_frames_wanted = 31

# --- Markers --- #
mat_contents = sio.loadmat(parent_dir_path + f"/data_real/sujet_5/marker_{data}.mat")
marker_treat = mat_contents['marker_treat'][:-1]
marker_treat = reduce(n_frames_wanted, marker_treat)[0]

# --- X_init --- #
mat_contents = sio.loadmat(parent_dir_path + f"/data_real/sujet_5/states_ekf_wo_RT_{data}.mat")
x_init = mat_contents["x_init"]
x_init = reduce(n_frames_wanted, x_init)[0]

# --- EMG --- #
emg_treat = x_init[-biorbd_model.nbMuscles():, :]

n_shooting_points = marker_treat.shape[2] - 1
t = np.linspace(0, 0.24, n_shooting_points + 1) #0.24s
final_time = t[n_shooting_points]
q_ref = x_init[6:11, :]
q_dot_ref = x_init[17:-biorbd_model.nbMuscles(), :]


# --- SO Activation --- #
mat_contents = sio.loadmat(parent_dir_path + f'/static_optimisation/activation_so_{data}.mat')
a_so = mat_contents['activations_so']
tau_so = mat_contents['tau']

# with open("Markers_EMG_tracking_f_iso_flex.bob", 'rb' ) as file:
#     data = pickle.load(file)
# states_ip = data['data'][0]
# controls_ip = data['data'][1]
ocp, sol = OptimalControlProgram.load(f"Markers_EMG_tracking_{data}_ipopt.bo")
states_ip, controls_ip = Data.get_data(ocp, sol["x"])
q_ip = states_ip['q']
qdot_ip = states_ip['q_dot']
u_ip = controls_ip['muscles']
x_ip = vertcat(states_ip["q"], states_ip["q_dot"])
nlp_ip = ocp.nlp[0]

# result = ShowResult(ocp, sol)
# result.animate()
# result.graphs()
ocp, sol = OptimalControlProgram.load(f"Markers_EMG_tracking_{data}_acados.bo")
states, controls = Data.get_data(ocp, sol["x"])
q_ac = states['q']
qdot_ac = states['q_dot']
x_ac = vertcat(states["q"], states["q_dot"])
u_ac = controls['muscles']
n_shooting_points = q_ac.shape[1]
t = np.linspace(0, final_time, n_shooting_points)
nlp_ac = ocp.nlp[0]

# ---- Tau --- #
symbolic_states = MX.sym("x", biorbd_model.nbQ()*2, 1)
symbolic_controls = MX.sym("a", biorbd_model.nbMuscles(), 1)
torque_func = Function(
    "MuscleTau",
    [symbolic_states, symbolic_controls],
    [muscles_tau(symbolic_states, symbolic_controls, nlp_ac)],
    ["x", "a"],
    ["torque"],
).expand()
torques_ac = np.ndarray((biorbd_model.nbQ(), n_shooting_points))
torques_ac[:, :] = torque_func(x_ac[:, :], u_ac[:, :])

symbolic_states = MX.sym("x", biorbd_model.nbQ()*2, 1)
symbolic_controls = MX.sym("a", biorbd_model.nbMuscles(), 1)
torque_func = Function(
    "MuscleTau",
    [symbolic_states, symbolic_controls],
    [muscles_tau(symbolic_states, symbolic_controls, nlp_ip)],
    ["x", "a"],
    ["torque"],
).expand()
torques_ip = np.ndarray((biorbd_model.nbQ(), n_shooting_points))
torques_ip[:, :] = torque_func(x_ip[:, :], u_ip[:, :])

a_osim = np.ndarray((biorbd_model.nbMuscles(), 124))
for j in range(1, biorbd_model.nbMuscles()+1):
    with open(parent_dir_path + "/Opensim/static_optimisation/colombe_real_data_StaticOptimization_activation.sto") as activation:
        reader2 = csv.reader(activation, delimiter="\t")
        counter_lines = 0
        a = 0
        for row in reader2:
            if counter_lines < 9:
                counter_lines += 1
            else:
                a_osim[j-1, a] = float(row[j])
                a += 1
a_osim = reduce(n_frames_wanted, a_osim)[0]

tau_osim = np.ndarray((biorbd_model.nbQ(), 124))
tau_osim_temp = np.ndarray((19, 124))
for j in range(1, 19 + 1):
    with open(parent_dir_path + "/Opensim/static_optimisation/inverse_dynamics.sto") as activation:
        reader2 = csv.reader(activation, delimiter="\t")
        counter_lines = 0
        a = 0
        for row in reader2:
            if counter_lines < 7:
                counter_lines += 1
            else:
                tau_osim_temp[j-1, a] = float(row[j])
                a += 1

tau_osim[[0, 1, 2, 3, 4], :] = tau_osim_temp[[6, 12, 13, 14, 15], :]
tau_osim = reduce(n_frames_wanted, tau_osim)[0]

q_osim_temp = np.ndarray((19, 124))
q_osim = np.ndarray((biorbd_model.nbQ(), 124))
q_dot_osim = np.ndarray((biorbd_model.nbQ(), 124))
q_ddot_osim = np.ndarray((biorbd_model.nbQ(), 124))
for j in range(1, 19 + 1):
    with open(parent_dir_path + "/Opensim/static_optimisation/inverse_kin.mot") as activation:
        reader2 = csv.reader(activation, delimiter="\t")
        counter_lines = 0
        a = 0
        for row in reader2:
            if counter_lines < 11:
                counter_lines += 1
            else:
                q_osim_temp[j - 1, a] = float(row[j])
                # q_dot_osim[j - 1, a] = float(row[biorbd_model.nbQ() + j])
                # q_ddot_osim[j - 1, a] = float(row[(biorbd_model.nbQ() * 2) + j])
                a += 1

q_osim[[0, 1, 2, 3, 4], :] = q_osim_temp[[6, 12, 13, 14, 15], :] * (np.pi/180)

q_osim = reduce(n_frames_wanted, q_osim)[0]
# q_dot_osim = reduce(n_frames_wanted, q_dot_osim)[0]
# q_ddot_osim = reduce(n_frames_wanted, q_ddot_osim)[0]
q_name = ['clav_rx', 'arm_rx', 'arm,ry', 'arm_rz', 'l_arm_rz']

plt.figure("Tau")
for i in range(q_ac.shape[0]):
    plt.subplot(2, 3, i + 1)
    plt.plot(t, torques_ac[i, :])
    plt.plot(t, torques_ip[i, :])
    plt.plot(t, tau_so[i, :])
    plt.plot(t, tau_osim[i, :])
    plt.title(q_name[i])
plt.legend(labels=['Acados', 'Ipopt', 'Biorbd', 'Osim'], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

plt.figure("Q")
for i in range(q_ac.shape[0]):
    plt.subplot(2, 3, i + 1)
    plt.plot(t, q_ac[i, :])
    plt.plot(t, q_ip[i, :])
    plt.plot(t, q_ref[i, :])
    plt.plot(t, q_osim[i, :])
    plt.title(q_name[i])
plt.legend(labels=['Acados', 'Ipopt', 'EKF', 'Osim'],bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

plt.figure("Q_dot")
for i in range(q_ac.shape[0]):
    plt.subplot(2, 3, i + 1)
    plt.plot(t, qdot_ac[i, :])
    plt.plot(t, qdot_ip[i, :])
    plt.plot(t, q_dot_ref[i, :])
    plt.title(q_name[i])
plt.legend(labels=['Acados', 'Ipopt', 'EKF'], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

plt.figure("Muscles activations")
for i in range(u_ac.shape[0]):
    plt.subplot(4, 5, i + 1)
    plt.step(t, u_ac[i, :])
    plt.step(t, u_ip[i, :])
    plt.step(t, a_so[i, :])
    plt.step(t, a_osim[i, :], color='b')
    plt.step(t, emg_treat[i, :], color='r')
    plt.title(nlp_ip.model.muscleNames()[i].to_string())
plt.legend(labels=['Acados', 'Ipopt', 'Biorbd', 'Osim', 'Target'], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

q=q_ac
n_mark = biorbd_model.nbMarkers()
n_q = q.shape[0]
n_frames = n_shooting_points
markers_ac = np.ndarray((3, n_mark, q.shape[1]))
markers_ip = np.ndarray((3, n_mark, q.shape[1]))
symbolic_states = MX.sym("x", n_q, 1)
markers_func = Function(
    "ForwardKin", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q"], ["markers"]
).expand()

for i in range(n_frames):
    markers_ac[:, :, i] = markers_func(q_ac[:, i])

for i in range(n_frames):
    markers_ip[:, :, i] = markers_func(q_ip[:, i])

plt.figure("Markers")
for i in range(markers_ac.shape[1]):
    plt.plot(np.linspace(0, 1, n_shooting_points ), markers_ac[:, i, :].T, color="g")
    plt.plot(np.linspace(0, 1, n_shooting_points), markers_ip[:, i, :].T, 'k--')
    plt.plot(np.linspace(0, 1, n_shooting_points), marker_treat[:, i, :].T, color="r")

plt.xlabel("Time")
plt.ylabel("Markers Position")
plt.legend(labels=['Acados', 'Ipopt', 'Target'])
plt.show()
# b = BiorbdViz.BiorbdViz(loaded_model=biorbd_model)
# b.load_movement(q_ac)
# b.exec()