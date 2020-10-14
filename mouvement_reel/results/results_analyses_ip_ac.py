import numpy as np
import biorbd
import pickle
import scipy.io as sio
from casadi import MX, Function, vertcat, jacobian
from matplotlib import pyplot as plt
# import BiorbdViz
import os
import csv
from bioptim import (
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


mov = 'flex_co' #"horizon_co" #flex #abd_co
sujet = '5'
model_path = '/home/amedeo/Documents/programmation/marker_emg_tracking/models/'
data_path = '/home/amedeo/Documents/programmation/marker_emg_tracking/mouvement_reel/results'
model = "arm_Belaise_real_v3_scaled.bioMod"
biorbd_model = biorbd.Model(model_path + model)
n_frames_wanted = 50
tries = '0'
# --- Data to track for each repetition --- #
# --- x-init --- #
mat_contents = sio.loadmat(data_path + f"/sujet_{sujet}/states_init_{mov}.mat")
x_init = mat_contents[f"x_init_{tries}"]
x_init = reduce(n_frames_wanted, x_init)[0]
# x_init = x_init[:, :]
q_ref = x_init[:biorbd_model.nbQ(), :]
q_dot_ref = x_init[biorbd_model.nbQ():-biorbd_model.nbMuscles(), :]
t_final = float(mat_contents[f"t_final_{tries}"])

# --- EMG --- #
emg_treat = x_init[-biorbd_model.nbMuscles():, :]

# --- Markers --- #
mat_contents = sio.loadmat(data_path + f"/sujet_{sujet}/data_{mov}_treat.mat")
marker_treat = mat_contents[f"marker_try_{tries}"][:-1, 2:, :]
marker_treat = reduce(n_frames_wanted, marker_treat)[0]

n_shooting_points = marker_treat.shape[2] - 1
t = np.linspace(0, t_final, n_shooting_points + 1)


# --- SO Activation --- #
# mat_contents = sio.loadmat(parent_dir_path + f'/static_optimisation/activation_so_{data}.mat')
# a_so = mat_contents['activations_so']
# tau_so = mat_contents['tau']
mat_contents = sio.loadmat(f'./sujet_{sujet}/param_f_iso_{mov}_try_{tries}_ipopt.mat')
f_iso_ip = mat_contents["f_iso"]
with open(f"./sujet_{sujet}/Markers_EMG_tracking_f_iso_{mov}_try_{tries}_ipopt.bob", 'rb') as file:
    data = pickle.load(file)
states_ip = data['data'][0]
controls_ip = data['data'][1]
# ocp, sol = OptimalControlProgram.load(f"Markers_EMG_tracking_{data}_ipopt.bo")
# states_ip, controls_ip = Data.get_data(ocp, sol["x"])
q_ip = states_ip['q']
qdot_ip = states_ip['q_dot']
u_ip = controls_ip['muscles']
tau_ip = controls_ip['tau']
x_ip = vertcat(states_ip["q"], states_ip["q_dot"])
# nlp_ip = ocp.nlp[0]
q_name = [biorbd_model.nameDof()[i].to_string() for i in range(biorbd_model.nbQ())]
# q_name = ['clav_rx', 'shouler_rx', 'shoulder_ry', 'shoulder_rz', 'l_arm_rz']
# result = ShowResult(ocp, sol)
# result.animate()
# result.graphs()
mat_contents = sio.loadmat(f'./sujet_{sujet}/param_f_iso_{mov}_try_{tries}_acados.mat')
f_iso_ac = mat_contents["f_iso"]
with open(f"./sujet_{sujet}/Markers_EMG_tracking_f_iso_{mov}_try_{tries}_acados.bob", 'rb') as file:
    data = pickle.load(file)
states_ac = data['data'][0]
controls_ac = data['data'][1]

# ocp, sol = OptimalControlProgram.load(f"Markers_EMG_tracking_{data}_acados.bo")
# states, controls = Data.get_data(ocp, sol["x"])
q_ac = states_ac['q']
qdot_ac = states_ac['q_dot']
x_ac = vertcat(states_ac["q"], states_ac["q_dot"])
u_ac = controls_ac['muscles']
tau_ac = controls_ac['tau']
n_shooting_points = q_ac.shape[1]
# t = np.linspace(0, final_time, n_shooting_points)
# nlp_ac = ocp.nlp[0]
x1 = range(biorbd_model.nbMuscles())
width = 0.2
# x1 = [i + width for i in x1]
height = []
for i in range(biorbd_model.nbMuscles()):
    # height.append(abs(np.log(1+(RMSE_ip[i, 30]))))
    height.append(f_iso_ac[i, 0])
plt.bar(x1, height, width, color='red')

x2 = [i - width for i in x1]
height = []
for i in range(biorbd_model.nbMuscles()):
    # height.append(abs(np.log(1+(RMSE_ac[i, 30]))))
    height.append(f_iso_ip[i, 0])
    plt.xticks(range(len(x1)), [biorbd_model.muscleNames()[i].to_string() for i in range(18)], rotation = 45)
plt.bar(x2, height, width, color='blue')
plt.xlabel('muscles')
plt.ylabel('weight')
plt.title('weight on force iso max')
plt.legend(["Acados", "Ipopt"], loc=2)
# plt.show()
plt.figure("Tau")
for i in range(q_ac.shape[0]):
    plt.subplot(2, 3, i + 1)
    plt.plot(t, tau_ac[i, :])
    plt.plot(t, tau_ip[i, :])
    plt.title(q_name[i])
plt.legend(labels=['Acados', 'Ipopt'], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.show()

plt.figure("Q")
for i in range(q_ac.shape[0]):
    plt.subplot(2, 3, i + 1)
    plt.plot(t, (q_ac[i, :]*180/np.pi))
    plt.plot(t, q_ip[i, :]*180/np.pi)
    plt.plot(t, q_ref[i, :]*180/np.pi)
    plt.title(q_name[i])
plt.legend(labels=['Acados', 'Ipopt', 'EKF'],bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.show()

plt.figure("Q_dot")
for i in range(q_ac.shape[0]):
    plt.subplot(2, 3, i + 1)
    plt.plot(t, qdot_ac[i, :])
    plt.plot(t, qdot_ip[i, :])
    plt.plot(t, q_dot_ref[i, :])
    plt.title(q_name[i])
plt.legend(labels=['Acados', 'Ipopt', 'EKF'], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.show()

plt.figure("Muscles activations")
for i in range(u_ac.shape[0]):
    plt.subplot(4, 5, i + 1)
    plt.step(t, u_ac[i, :])
    plt.step(t, u_ip[i, :])
    plt.step(t, emg_treat[i, :], color='r')
    plt.title(biorbd_model.muscleNames()[i].to_string())
plt.legend(labels=['Acados', 'Ipopt', 'Target'], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()
import bioviz
b = bioviz.Viz(model_path=model_path + model)
b.load_movement(q_ac)
b.exec()

