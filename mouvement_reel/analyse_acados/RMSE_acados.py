import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
from pyomeca import Markers, Rototrans
import numpy as np
import datetime
from time import time
import biorbd
import os.path
from casadi import MX, Function, vertcat, jacobian
from matplotlib import pyplot as plt
from biorbd_optim import (
    OptimalControlProgram,
    BidirectionalMapping,
    Mapping,
    ObjectiveOption,
    DynamicsTypeList,
    DynamicsType,
    Data,
    ObjectiveList,
    Objective,
    BoundsList,
    Bounds,
    QAndQDotBounds,
    InitialConditionsList,
    ParameterList,
    InitialConditions,
    ShowResult,
    InterpolationType,
    Solver,
    Simulate
)


tol_init = 2
tol_final = 6
tol_final += 1

tic = time()


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


def prepare_ocp(
        biorbd_model,
        final_time,
        number_shooting_points,
        marker_ref,
        excitations_ref,
        q_ref,
        state_init,
        use_residual_torque,
        activation_driven,
        kin_data_to_track,
        nb_threads,
        use_SX=True,
        ):

    # --- Options --- #
    nb_mus = biorbd_model.nbMuscleTotal()
    activation_min, activation_max, activation_init = 0, 1, 0.3
    excitation_min, excitation_max, excitation_init = 0, 1, 0.1
    torque_min, torque_max, torque_init = -100, 100, 0

    # Add objective functions
    objective_functions = ObjectiveList()

    objective_functions.add(Objective.Lagrange.TRACK_MUSCLES_CONTROL, weight=10, target=excitations_ref)

    objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=0.01)

    if use_residual_torque:
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1)

    if kin_data_to_track == "markers":
        objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=1000,
                                target=marker_ref[:, -biorbd_model.nbMarkers():, :]
                                )

    elif kin_data_to_track == "q":
        objective_functions.add(
            Objective.Lagrange.TRACK_STATE, weight=100,
            # target=q_ref,
            # states_idx=range(biorbd_model.nbQ())
        )
    else:
        raise RuntimeError("Wrong choice of kin_data_to_track")

    # Dynamics
    dynamics = DynamicsTypeList()
    if use_residual_torque:
        dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)
    elif activation_driven:
        dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_DRIVEN)
    else:
        dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_DRIVEN)

    # Constraints
    constraints = ()

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    if use_SX:
        x_bounds[0].min[:, 0] = np.concatenate((state_init[6:biorbd_model.nbQ() + 6, 0],
                                                state_init[biorbd_model.nbQ() + 12:-nb_mus, 0]))
        x_bounds[0].max[:, 0] = np.concatenate((state_init[6:biorbd_model.nbQ() + 6, 0],
                                                state_init[biorbd_model.nbQ() + 12:-nb_mus, 0]))

    # Add muscle to the bounds
    if activation_driven is not True:
        x_bounds[0].concatenate(
            Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
        )

    # Initial guess
    x_init = InitialConditionsList()
    if activation_driven:
        # state_base = np.ndarray((12, n_shooting_points+1))
        # for i in range(n_shooting_points+1):
        #     state_base[:, i] = np.concatenate((state_init[:6, 0], np.zeros((6))))
        x_init.add(np.concatenate((state_init[6:biorbd_model.nbQ() + 6, :],
                                   state_init[biorbd_model.nbQ() + 12:-nb_mus, :])),
                   interpolation=InterpolationType.EACH_FRAME)
        # x_init.add(state_init[:-nb_mus, :], interpolation=InterpolationType.EACH_FRAME)
    else:
        # x_init.add([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()) + [0] * biorbd_model.nbMuscles())
        x_init.add(state_init[biorbd_model.nbQ():, :], interpolation=InterpolationType.EACH_FRAME)

    # Add muscle to the bounds
    u_bounds = BoundsList()
    u_init = InitialConditionsList()
    nb_tau = 6
    # init_residual_torque = np.concatenate((np.ones((biorbd_model.nbGeneralizedTorque(), n_shooting_points))*0.5,
    #                                        excitations_ref))
    if use_residual_torque:
        u_bounds.add(
            [
                [torque_min] * biorbd_model.nbGeneralizedTorque() + [excitation_min] * biorbd_model.nbMuscleTotal(),
                [torque_max] * biorbd_model.nbGeneralizedTorque() + [excitation_max] * biorbd_model.nbMuscleTotal(),
            ]
        )
        u_init.add(
            [torque_init] * biorbd_model.nbGeneralizedTorque() + [excitation_init] * biorbd_model.nbMuscleTotal())
        # u_init.add(init_residual_torque, interpolation=InterpolationType.EACH_FRAME)

    else:
        u_bounds.add(
            [[excitation_min] * biorbd_model.nbMuscleTotal(), [excitation_max] * biorbd_model.nbMuscleTotal()])
        if activation_driven:
            # u_init.add([activation_init] * biorbd_model.nbMuscleTotal())
            u_init.add(excitations_ref, interpolation=InterpolationType.EACH_FRAME)
        else:
            # u_init.add([excitation_init] * biorbd_model.nbMuscleTotal())
            u_init.add(excitations_ref, interpolation=InterpolationType.EACH_FRAME)

    # Get initial isometric forces
    fiso = []
    for k in range(nb_mus):
        fiso.append(biorbd_model.muscle(k).characteristics().forceIsoMax().to_mx())


    # ------------- #
    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        nb_threads=nb_threads,
        use_SX=use_SX,
        # parameters=parameters
    )
data = 'flex'  # "horizon_co" #flex #abd_co
model = "arm_Belaise_v2_scaled.bioMod"

biorbd_model = biorbd.Model("./models/" + model)
n_frames_wanted = 31
nb_mus = biorbd_model.nbMuscles()

# Data to track
# --- Markers --- #
mat_contents = sio.loadmat(f"./data_real/Sujet_5/marker_{data}.mat")
# marker_treat = mat_contents['marker_reduce']
marker_treat = mat_contents['marker_treat'][:-1]
marker_treat = reduce(n_frames_wanted, marker_treat)[0]

# --- X_init --- #
mat_contents = sio.loadmat(f"./data_real/Sujet_5/states_ekf_wo_RT_{data}.mat")
x_init = mat_contents["x_init"]
x_init = reduce(n_frames_wanted, x_init)[0]

# --- EMG --- #
emg_treat = x_init[-biorbd_model.nbMuscles():, :-1]

n_shooting_points = marker_treat.shape[2] - 1
t = np.linspace(0, 0.24, n_shooting_points + 1)  # 0.24s
final_time = t[n_shooting_points]
q_ref = x_init[:5, :]
use_residual_torque = False
activation_driven = True

kin_data_to_track = "markers"
biorbd_model = biorbd.Model("./models/" + model)  # To allow for non free variable, the model must be reloaded
fiso = []
for k in range(nb_mus):
    fiso.append(biorbd_model.muscle(k).characteristics().forceIsoMax().to_mx())
mat_content = sio.loadmat("./results/param_f_iso_flex.mat")
f_iso = mat_content["f_iso"] * mat_content["f_iso_global"]
for k in range(biorbd_model.nbMuscles()):
    biorbd_model.muscle(k).characteristics().setForceIsoMax(
        f_iso[k] * fiso[k]
    )
ocp_ac = prepare_ocp(
    biorbd_model,
    final_time,
    n_shooting_points,
    marker_treat,
    emg_treat,
    q_ref,
    x_init,
    use_residual_torque=use_residual_torque,
    activation_driven=activation_driven,
    kin_data_to_track=kin_data_to_track,
    nb_threads=8,
    use_SX=True,
)
i = 6
x_ac = np.ndarray((biorbd_model.nbQ()*2, n_shooting_points+1))
sol_ac = ocp_ac.solve(solver=Solver.ACADOS, solver_options={
                                "qp_solver": "PARTIAL_CONDENSING_HPIPM", "integrator_type": "IRK",
                                "nlp_solver_max_iter": 150, "sim_method_num_steps": 5,
                                "nlp_solver_tol_ineq": float("1e%d" %-i), "nlp_solver_tol_stat": float("1e%d" %-i),
                                "nlp_solver_tol_comp": float("1e%d" %-i), "nlp_solver_tol_eq": float("1e%d" %-i)
                                 })# FULL_CONDENSING_QPOASES, "PARTIAL_CONDENSING_HPIPM"


n_shooting_points += 1

states_sol, controls_sol = Data.get_data(ocp_ac, sol_ac["x"])
x_ac[:, :] = np.concatenate((states_sol['q'], states_sol['q_dot']))

sol_simulate_single_shooting_ac = Simulate.from_solve(ocp_ac, sol_ac, single_shoot=True)

RMSE_ac = np.ndarray((1, biorbd_model.nbQ()*2))
for k in range(biorbd_model.nbQ()*2):
    sum = 0
    for s in range(4, n_shooting_points):
        sum = sum + ((x_ac[k, s] - sol_simulate_single_shooting_ac['x'][s*28 + k])**2)
    RMSE_ac[0, k] = (np.sqrt((sum/n_shooting_points)))
sio.savemat("RMSE_ac_nX_" + str(i) + ".mat",
            {"RMSE_ac": RMSE_ac})

RMSE_ac = np.ndarray((1, n_shooting_points))
for s in range(n_shooting_points):
    sum = 0
    for k in range(biorbd_model.nbQ()*2):
        sum = sum + ((x_ac[k, s] - sol_simulate_single_shooting_ac['x'][s*28 + k])**2)
    RMSE_ac[0, s] = (np.sqrt((sum/(biorbd_model.nbQ()*2))))

sio.savemat("RMSE_ac_ns_"+str(i)+".mat",
            {"RMSE_ac": RMSE_ac})