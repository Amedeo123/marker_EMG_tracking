import numpy as np
import biorbd
import seaborn as sns ; sns.set()
from casadi import MX, Function, vertcat, jacobian
from matplotlib import pyplot as plt
import csv
from biorbd_optim import (
    OptimalControlProgram,
    Data,
    ShowResult,
    PlotType,
)
# --- Load OCP --- #
biorbd_model = biorbd.Model("/home/amedeo/Documents/programmation/Article_Colombe/arm_Belaise_buchanan_de_groote.bioMod")
ocp, sol , param = OptimalControlProgram.load("/home/amedeo/Documents/programmation/Article_Colombe/results/Q_tracking_with_residual_torque2020-08-06 15:32:25.127113.bo")
states, controls, params = Data.get_data(ocp, sol["x"], get_parameters=True)
print(params["shape_factor"])
q = states["q"]
qdot = states["q_dot"]
u = states["muscles"]
x = vertcat(states["q"], states["q_dot"], states["muscles"])

e = controls["muscles"]
nlp = ocp.nlp[0]

# --- Casadi stuff ---#
symbolic_states = MX.sym("x", nlp["nbQ"] + nlp["nbQdot"] + nlp["nbMuscle"], 1)
symbolic_controls = MX.sym("e", nlp["nbMuscle"], 1)
symbolic_length = MX.sym("l", nlp["nbMuscle"], 1)
symbolic_force = MX.sym("f", nlp["nbMuscle"], 1)
symbolic_tsl = MX.sym("tsl", nlp["nbMuscle"], 1)
symbolic_pa = MX.sym("pa", nlp["nbMuscle"], 1)
symbolic_insx = MX.sym("insx", nlp["nbMuscle"], 1)
symbolic_insy = MX.sym("insy", nlp["nbMuscle"], 1)
symbolic_insz = MX.sym("insz", nlp["nbMuscle"], 1)

# --- Functions --- #
def muscles_moment_arm(states, nlp):
    nq = nlp["q_mapping"].reduce.len
    q = nlp["q_mapping"].expand.map(states[:nq])

    moment_arm = nlp["model"].musclesLengthJacobian(q).to_mx()

    return moment_arm


def muscles_tau(states, controls, nlp):
    nq = nlp["q_mapping"].reduce.len
    q = nlp["q_mapping"].expand.map(states[:nq])
    qdot = nlp["q_dot_mapping"].expand.map(states[nq:])

    activations = states[nlp["nbQ"] + nlp["nbQdot"]:]
    muscles_states = biorbd.VecBiorbdMuscleState(nlp["nbMuscle"])

    for k in range(nlp["nbMuscle"]):
        muscles_states[k].setActivation(activations[k])
        muscles_states[k].setExcitation(controls[k])
    muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()

    return muscles_tau


def muscles_forces(states, controls, nlp, force=None, len=None, tsl=None, pa=None, insx=None, insy=None, insz=None):
    nq = nlp["q_mapping"].reduce.len
    q = nlp["q_mapping"].expand.map(states[:nq])
    qdot = nlp["q_dot_mapping"].expand.map(states[nq:])

    biorbd_model = biorbd.Model("arm_Belaise.bioMod")

    activations = states[nlp["nbQ"] + nlp["nbQdot"]:]
    muscles_states = biorbd.VecBiorbdMuscleState(nlp["nbMuscle"])

    for k in range(nlp["nbMuscle"]):
        muscles_states[k].setActivation(activations[k])
        muscles_states[k].setExcitation(controls[k])
        if force is not None:
            biorbd_model.muscle(k).characteristics().setForceIsoMax(force[k])
        if len is not None:
            biorbd_model.muscle(k).characteristics().setOptimalLength(len[k])
        if tsl is not None:
            biorbd_model.muscle(k).characteristics().setTendonSlackLength(tsl[k])
        if pa is not None:
            biorbd_model.muscle(k).characteristics().setPennationAngle(pa[k])
        if insx is not None:
            biorbd_model.muscle(k).position().setInsertionInLocal(biorbd.Vector3d(insx[k], insy[k], insz[k]))

    return biorbd_model.muscleForces(muscles_states, q, qdot).to_mx()


# --- Calcul Muscular force & muscular torque ---#
force_func = Function(
    "MuscleForce", 
    [symbolic_states, symbolic_controls], 
    [muscles_forces(symbolic_states, symbolic_controls, nlp)], 
    ["x", "e"],
    ["force"], 
).expand()

force = np.ndarray((nlp["nbMuscle"], nlp["ns"]+1))
force[:, :] = force_func(x[:, :], e[:, :])

torque_func = Function(
    "MuscleTau", 
    [symbolic_states, symbolic_controls], 
    [muscles_tau(symbolic_states, symbolic_controls, nlp)], 
    ["x", "e"],
    ["torque"], 
).expand()
torques = np.ndarray((nlp["nbQ"], nlp["ns"]+1))
torques[:, :] = torque_func(x[:, :], e[:, :])

# # --- Calcul des Jacobienne Forces/Paramètres --#
#
# # -- Activation -- #
# func_jac_force_activation = Function(
#     "JacForce",
#     [symbolic_states, symbolic_controls],
#     [jacobian(muscles_forces(symbolic_states, symbolic_controls, nlp), symbolic_controls)],
#     ["x", "u"],
#     ["jacActivation"],
# ).expand()
#
# jac_force_activation = np.ndarray((nlp["nbMuscle"], nlp["nbMuscle"], nlp["ns"]))
# for i in range(nlp["ns"]):
#     jac_force_activation[:, :, i] = func_jac_force_activation(x[:, i], u[:, i])
#
# jacNorm_force_act = np.ndarray((len(u[:, 0]), nlp["nbMuscle"], nlp["ns"]))
# for i in range(nlp["ns"]):
#     jacNorm_force_act[:, :, i] = u[:, i]/force[:, i] * jac_force_activation[:, :, i]
#
# # # -- Insertion -- #
# # insx = [-0.0219,0.00751,0.00751,-0.0219,-0.0219,-0.0032]
# # insy = [0.010460,-0.04839,-0.04839,0.01046,0.01046,-0.0239]
# # insz = [-0.00078,0.02179,0.02179,-0.00078,-0.00078,0.0009]
# # # ins = np.ndarray((nlp["nbMuscle"], 3))
# # # for i in range(nlp["nbMuscle"]):
# # #     ins[i,:] = (x[i], y[i], z[i])
# #
# # func_jac_force_ins = Function(
# #     "JacForceIns",
# #     [symbolic_states, symbolic_controls, symbolic_insx, symbolic_insy, symbolic_insz],
# #     [jacobian(muscles_forces(symbolic_states, symbolic_controls, nlp, insx=symbolic_insx, insy=symbolic_insy, insz=symbolic_insz), symbolic_insx)],
# #     ["x", "u", "insx", "insy", "insz"],
# #     ["jacIns"],
# # ).expand()
# # func_jac_force_insy = Function(
# #     "JacForceIns",
# #     [symbolic_states, symbolic_controls,  symbolic_insx, symbolic_insy, symbolic_insz],
# #     [jacobian(muscles_forces(symbolic_states, symbolic_controls, nlp, insx=symbolic_insx, insy=symbolic_insy, insz=symbolic_insz), symbolic_insy)],
# #     ["x", "u", "insx", "insy", "insz"],
# #     ["jacIns"],
# # ).expand()
# # func_jac_force_insz = Function(
# #     "JacForceIns",
# #     [symbolic_states, symbolic_controls,  symbolic_insx, symbolic_insy, symbolic_insz],
# #     [jacobian(muscles_forces(symbolic_states, symbolic_controls, nlp, insx=symbolic_insx, insy=symbolic_insy, insz=symbolic_insz), symbolic_insz)],
# #     ["x", "u", "insx", "insy", "insz"],
# #     ["jacIns"],
# # ).expand()
# #
# # print(func_jac_force_ins)
# # jaco_force_ins = np.ndarray((3, nlp["nbMuscle"], nlp["ns"]+1))
# # for n in range(nlp["ns"]):
# #     jaco_force_ins[:, :, n] = func_jac_force_ins(x[:, n], u[:, n], ins[:])
#
# # -- Optimal Length --#
# l = [0.134, 0.1157, 0.1321, 0.1138, 0.1138, 0.0858]
# func_jac_force_length = Function(
#     "JacForceLength",
#     [symbolic_states, symbolic_controls, symbolic_length],
#     [jacobian(muscles_forces(symbolic_states, symbolic_controls, nlp, len=symbolic_length), symbolic_length)],
#     ["x", "u", "l"],
#     ["jacLength"],
# ).expand()
#
# jac_force_length = np.ndarray((nlp["nbMuscle"], nlp["nbMuscle"], nlp["ns"]))
# for i in range(nlp["ns"]):
#     jac_force_length[:, :, i] = func_jac_force_length(x[:, i], u[:, i], l[:])
#
# li = np.ndarray((nlp["nbMuscle"], nlp["ns"]))
# for k in range(nlp["nbMuscle"]):
#     for i in range(nlp["ns"]):
#         li[k, i] = l[k]
#
# jacNorm_force_len = np.ndarray((len(l), nlp["nbMuscle"], nlp["ns"]))
# for i in range(nlp["ns"]):
#     jacNorm_force_len[:, :, i] = li[:, i]/force[:, i] * jac_force_length[:, :, i]
#
# # -- ForceIsoMax --#
# f = [798.52, 624.3, 435.56, 624.3, 624.3, 987.26]
# func_jac_force_forceIso = Function(
#     "JacForceForceIso",
#     [symbolic_states, symbolic_controls, symbolic_force],
#     [jacobian(muscles_forces(symbolic_states, symbolic_controls,  nlp, force=symbolic_force), symbolic_force)],
#     ["x", "u", "f"],
#     ["jacForceIso"],
# ).expand()
#
# jac_force_forceIso = np.ndarray((nlp["nbMuscle"], nlp["nbMuscle"], nlp["ns"]))
# for i in range(nlp["ns"]):
#     jac_force_forceIso[:, :, i] = func_jac_force_forceIso(x[:, i], u[:, i], f[:])
#
# fi = np.ndarray((nlp["nbMuscle"], nlp["ns"]))
# for k in range(nlp["nbMuscle"]):
#     for i in range(nlp["ns"]):
#         fi[k, i] = f[k]
#
# jacNorm_force_forceIso = np.ndarray((len(f), nlp["nbMuscle"], nlp["ns"]))
# for i in range(nlp["ns"]):
#     jacNorm_force_forceIso[:, :, i] = fi[:, i]/force[:, i] * jac_force_forceIso[:, :, i]
#
# # --Tendon Slack Length -- #
# tsl = [0.143, 0.2723, 0.1923, 0.098, 0.0908, 0.0535]
# func_jac_force_tsl = Function(
#     "JacForcetsl",
#     [symbolic_states, symbolic_controls, symbolic_tsl],
#     [jacobian(muscles_forces(symbolic_states, symbolic_controls, nlp, tsl=symbolic_tsl), symbolic_tsl)],
#     ["x", "u", "tsl"],
#     ["jactsl"],
# ).expand()
#
# jac_force_tsl = np.ndarray((nlp["nbMuscle"], nlp["nbMuscle"], nlp["ns"]))
# for i in range(nlp["ns"]):
#     jac_force_tsl[:, :, i] = func_jac_force_tsl(x[:, i], u[:, i], tsl[:])
#
# tsli = np.ndarray((nlp["nbMuscle"], nlp["ns"]))
# for k in range(nlp["nbMuscle"]):
#     for i in range(nlp["ns"]):
#         tsli[k, i] = tsl[k]
#
# jacNorm_force_tsl = np.ndarray((len(tsl), nlp["nbMuscle"], nlp["ns"]))
# for i in range(nlp["ns"]):
#     jacNorm_force_tsl[:, :, i] = tsli[:, i]/force[:, i] * jac_force_tsl[:, :, i]
#
# # -- Pennation angle -- #
# pa = [0.209, 0, 0, 0.157, 0.157, 0]
# func_jac_force_pangle = Function(
#     "JacForcePA",
#     [symbolic_states, symbolic_controls, symbolic_pa],
#     [jacobian(muscles_forces(symbolic_states, symbolic_controls, nlp, pa=symbolic_pa), symbolic_pa)],
#     ["x", "u", "pa", ],
#     ["jacPa"],
# ).expand()
#
# jac_force_pa = np.ndarray((nlp["nbMuscle"], nlp["nbMuscle"], nlp["ns"]))
# for i in range(nlp["ns"]):
#     jac_force_pa[:, :, i] = func_jac_force_pangle(x[:, i], u[:, i], pa[:])
# pai = np.ndarray((nlp["nbMuscle"], nlp["ns"]))
# for k in range(nlp["nbMuscle"]):
#     for i in range(nlp["ns"]):
#         pai[k, i] = pa[k]
#
# jacNorm_force_pa = np.ndarray((len(pa), nlp["nbMuscle"], nlp["ns"]))
# for i in range(nlp["ns"]):
#     jacNorm_force_pa[:, :, i] = pai[:, i]/force[:, i] * jac_force_pa[:, :, i]

# --- Calcul bras de levier ---#
muscles_jac_func = Function(
    "MuscleMA",
    [symbolic_states], 
    [muscles_moment_arm(symbolic_states, nlp)],
    ["x"], 
    ["MomentArm"],
).expand()

moment = np.ndarray((nlp["nbMuscle"], nlp["nbQ"], nlp["ns"]))
for i in range(nlp["ns"]):
    moment[:, :, i] = muscles_jac_func(x[:, i])

moment_shoulder = np.ndarray((nlp["nbMuscle"], nlp["ns"]))
moment_elbow = np.ndarray((nlp["nbMuscle"], nlp["ns"]))
for i in range(nlp["nbMuscle"]):
    moment_shoulder[i, :] = moment[i, 0, :]
    moment_elbow[i, :] = moment[i, 1, :]

for i in range(nlp["ns"]):
    moment_shoulder[:, i] = moment_shoulder[:, i] * -1
    moment_elbow[:, i] = moment_elbow[:, i] * -1

# # --- Print force and torque in result files ---#
# t = np.linspace(0, 1, num=nlp["ns"]+1).reshape(nlp["ns"]+1, 1)
# torquelist = torques.T.reshape(nlp["ns"]+1, biorbd_model.nbDof())
# print(torquelist)
# torquelist = np.concatenate((t, torquelist), axis=1)
# with open(
#         "/home/amedeo/Documents/programmation/Article_Colombe/Results/Output/Torque.csv", 'w') as elb_torque:
#     writer = csv.writer(elb_torque, delimiter=',')
#     writer.writerows(torquelist)
#
# forcelist = force.T.reshape(nlp["ns"]+1, biorbd_model.nbMuscles())
# forcelist = np.concatenate((t, forcelist), axis=1)
# with open("/home/amedeo/Documents/programmation/Article_Colombe/Results/Output/force.csv", 'w') as musforce:
#     writer = csv.writer(musforce, delimiter=',')
#     writer.writerows(forcelist)

# # --- Plot force jacobian heatmap ---#
#
# # -- Concatenate force jacobian --#
# jacNorm_force = np.ndarray((nlp["ns"], nlp["nbMuscle"]*5, nlp["nbMuscle"]))
# for k in range(nlp["nbMuscle"]):
#     jacNorm_force[:, 0+k, k] = jacNorm_force_act.T[:, k, k]
#     jacNorm_force[:, 6+k, k] = jacNorm_force_forceIso.T[:, k, k]
#     jacNorm_force[:, 12+k, k] = jacNorm_force_len.T[:, k, k]
#     jacNorm_force[:, 18+k, k] = jacNorm_force_pa.T[:, k, k]
#     jacNorm_force[:, 24+k, k] = jacNorm_force_tsl.T[:, k, k]
#
# # -- Dictionary for heat map "x" axis -- #
# axe = np.linspace(0, 90, nlp["ns"])
# xe = axe.tolist()
# for i in range(nlp["ns"]):
#     if int(xe[i])%20 == 0:
#         xe[i] = int(xe[i])
#         xe[i] = str(xe[i])
#     elif i == 0:
#         xe[i] = int(xe[i])
#         xe[i] = str(xe[i])
#     elif i == nlp["ns"]-1:
#         xe[i] = int(xe[i])
#         xe[i] = str(xe[i])
#     else:
#         xe[i] = str("")
#
# # -- Plot heat map -- #
# plt.subplot(1, 3, 1)
# ax1 = sns.heatmap(abs(jacNorm_force.T[0, [0, 6, 12, 18, 24], :]), xticklabels=xe,
#                   yticklabels=["Activation", "force_iso", "opt_len", "Pen_angle", "Slack_len"])
# plt.title("TRIlong")
#
# plt.subplot(1, 3, 2)
# ax2 = sns.heatmap(abs(jacNorm_force.T[1, [1, 7, 13, 19, 25], :]), yticklabels=False, xticklabels=xe)
# plt.title("BIClong")
#
# plt.subplot(1, 3, 3)
# ax3 = sns.heatmap(abs(jacNorm_force.T[2, [2, 8, 14, 20, 26], :]), yticklabels=False, xticklabels=xe)
# plt.title("BICshort")
# plt.show()
#
# plt.subplot(1, 3, 1)
# ax4 = sns.heatmap(abs(jacNorm_force.T[3, [3, 9, 15, 21, 27], :]), xticklabels=xe,
#                   yticklabels=["Activation", "force_iso", "opt_len", "Pen_angle", "Slack_len", ])
# plt.title("TRIlat")
#
# plt.subplot(1, 3, 2)
# ax5 = sns.heatmap(abs(jacNorm_force.T[4, [4, 10, 16, 22, 28], :]), xticklabels=xe, yticklabels=False)
# plt.title("TRImed")
#
# plt.subplot(1, 3, 3)
# ax6 = sns.heatmap(abs(jacNorm_force.T[5, [5, 11, 17, 23, 29], :]), xticklabels=xe, yticklabels=False)
# plt.title("BRA")
# plt.show()

# --- Add new custom plot--- #
def plot_torque(torque, torques_to_plot):
    return torque[torques_to_plot, :]

def plot_force(forces, forces_to_plot):
    return forces[forces_to_plot, :]


ocp.add_plot("Torque", lambda x, u, p: plot_torque(torques, range(nlp["nbQ"])), PlotType.PLOT)
ocp.add_plot("MusclesForces", lambda x, u, p: plot_force(force, (range(nlp["nbMuscle"]))),PlotType.PLOT, legend=(nlp["muscleNames"]))

# # -- Plot all moment arm on a same graph -- #
# plt.figure("ElbowMomentArm")
# plt.plot(np.linspace(0, 1, nlp["ns"]), moment_elbow[0, :], label=nlp["muscleNames"][0])
# plt.plot(np.linspace(0, 1, nlp["ns"]), moment_elbow[1, :], label=nlp["muscleNames"][1])
# plt.plot(np.linspace(0, 1, nlp["ns"]), moment_elbow[2, :], label=nlp["muscleNames"][2])
# plt.plot(np.linspace(0, 1, nlp["ns"]), moment_elbow[3, :], label=nlp["muscleNames"][3])
# plt.plot(np.linspace(0, 1, nlp["ns"]), moment_elbow[4, :], label=nlp["muscleNames"][4])
# plt.plot(np.linspace(0, 1, nlp["ns"]), moment_elbow[5, :], label=nlp["muscleNames"][5])
# plt.legend()

# --- Show result ---#
result = ShowResult(ocp, sol)
# result.animate()
result.graphs()
plt.show()
