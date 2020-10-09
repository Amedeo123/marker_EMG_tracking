import importlib.util
from pathlib import Path
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

# It is not an optimal control, it only apply a Runge Kutta at each nodes
# def reduce(n_frames_expected, data):
#     data_reduce = []
#     n_shooting = 0
#     if len(data.shape) == 2:
#         n_shooting = data.shape[1]
#     elif len(data.shape) == 3:
#         n_shooting = data.shape[2]
#     else:
#         print('Wrong size of data')
#
#     if n_shooting % n_frames_expected == 0:
#         n_frames = n_frames_expected
#     else:
#         c = 1
#         d = 1
#         while n_shooting % (n_frames_expected - c) != 0:
#             c += 1
#         while n_shooting % (n_frames_expected + d) != 0:
#             d += 1
#         if c > d:
#             n_frames = n_frames_expected + d
#         else:
#             n_frames = n_frames_expected - c
#
#     k = int(n_shooting / n_frames)
#     if len(data.shape) == 2:
#         data_reduce = np.ndarray((data.shape[0], n_frames))
#         for i in range(n_frames):
#             data_reduce[:, i] = data[:, k * i]
#     elif len(data.shape) == 3:
#         data_reduce = np.ndarray((data.shape[0], data.shape[1], n_frames))
#         for i in range(n_frames):
#             data_reduce[:, :, i] = data[:, :, k * i]
#     else:
#         print('Wrong size of data')
#
#     return data_reduce, n_frames

# def modify_isometric_force(biorbd_model, value, fiso_init):
#     for k in range(biorbd_model.nbMuscles()):
#         biorbd_model.muscle(k).characteristics().setForceIsoMax(
#             value[biorbd_model.nbMuscles()] * value[k] * fiso_init[k]
#         )
#
#
# def modify_shape_factor(biorbd_model, value):
#     for k in range(biorbd_model.nbMuscles()):
#         biorbd.StateDynamicsBuchananDeGroote(biorbd_model.muscle(k).state()).setShapeFactor(value[k])

# --- Load pendulum --- #
# PROJECT_FOLDER = Path(__file__).parent / "./"
# spec = importlib.util.spec_from_file_location("flex", str(PROJECT_FOLDER) +
#                                               "/muscles_excitations_marker_tracking_real_data.py")
# flex = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(flex)
# def prepare_ocp(
#         biorbd_model,
#         final_time,
#         number_shooting_points,
#         marker_ref,
#         excitations_ref,
#         q_ref,
#         state_init,
#         use_residual_torque,
#         activation_driven,
#         kin_data_to_track,
#         nb_threads,
#         use_SX=True,
#         param = False
#         ):
#
#     # --- Options --- #
#     nb_mus = biorbd_model.nbMuscleTotal()
#     activation_min, activation_max, activation_init = 0, 1, 0.3
#     excitation_min, excitation_max, excitation_init = 0, 1, 0.1
#     torque_min, torque_max, torque_init = -100, 100, 0
#
#     # -- Force iso ipopt pour acados
#     # if param is not True:
#     #     fiso = []
#     #     for k in range(nb_mus):
#     #         fiso.append(biorbd_model.muscle(k).characteristics().forceIsoMax().to_mx())
#     #     mat_content = sio.loadmat("./results/param_f_iso_flex.mat")
#     #     f_iso = mat_content["f_iso"] * mat_content["f_iso_global"]
#     #     for k in range(biorbd_model.nbMuscles()):
#     #         biorbd_model.muscle(k).characteristics().setForceIsoMax(
#     #             f_iso[k] * fiso[k]
#     #         )
#
#     # Add objective functions
#     objective_functions = ObjectiveList()
#
#     objective_functions.add(Objective.Lagrange.TRACK_MUSCLES_CONTROL, weight=10, target=excitations_ref)
#
#     objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=0.01)
#
#     if use_residual_torque:
#         objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1)
#
#     if kin_data_to_track == "markers":
#         objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=1000,
#                                 target=marker_ref[:, -biorbd_model.nbMarkers():, :]
#                                 )
#
#     elif kin_data_to_track == "q":
#         objective_functions.add(
#             Objective.Lagrange.TRACK_STATE, weight=100,
#             # target=q_ref,
#             # states_idx=range(biorbd_model.nbQ())
#         )
#     else:
#         raise RuntimeError("Wrong choice of kin_data_to_track")
#
#     # Dynamics
#     dynamics = DynamicsTypeList()
#     if use_residual_torque:
#         dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)
#     elif activation_driven:
#         dynamics.add(DynamicsType.MUSCLE_ACTIVATIONS_DRIVEN)
#     else:
#         dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_DRIVEN)
#
#     # Constraints
#     constraints = ()
#
#     # Path constraint
#     x_bounds = BoundsList()
#     x_bounds.add(QAndQDotBounds(biorbd_model))
#     if use_SX:
#         x_bounds[0].min[:, 0] = np.concatenate((state_init[6:biorbd_model.nbQ()+6, 0],
#                                            state_init[biorbd_model.nbQ()+12:-nb_mus, 0]))
#         x_bounds[0].max[:, 0] = np.concatenate((state_init[6:biorbd_model.nbQ()+6, 0],
#                                        state_init[biorbd_model.nbQ()+12:-nb_mus, 0]))
#
#     # Add muscle to the bounds
#     if activation_driven is not True:
#         x_bounds[0].concatenate(
#             Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
#         )
#
#     # Initial guess
#     x_init = InitialConditionsList()
#     if activation_driven:
#         # state_base = np.ndarray((12, n_shooting_points+1))
#         # for i in range(n_shooting_points+1):
#         #     state_base[:, i] = np.concatenate((state_init[:6, 0], np.zeros((6))))
#         x_init.add(np.concatenate((state_init[6:biorbd_model.nbQ()+6, :],
#                                    state_init[biorbd_model.nbQ()+12:-nb_mus, :])),
#                                    interpolation=InterpolationType.EACH_FRAME)
#         # x_init.add(state_init[:-nb_mus, :], interpolation=InterpolationType.EACH_FRAME)
#     else:
#         # x_init.add([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()) + [0] * biorbd_model.nbMuscles())
#         x_init.add(state_init[biorbd_model.nbQ():, :], interpolation=InterpolationType.EACH_FRAME)
#
#     # Add muscle to the bounds
#     u_bounds = BoundsList()
#     u_init = InitialConditionsList()
#     nb_tau = 6
#     # init_residual_torque = np.concatenate((np.ones((biorbd_model.nbGeneralizedTorque(), n_shooting_points))*0.5,
#     #                                        excitations_ref))
#     if use_residual_torque:
#         u_bounds.add(
#             [
#                 [torque_min] * biorbd_model.nbGeneralizedTorque() + [excitation_min] * biorbd_model.nbMuscleTotal(),
#                 [torque_max] * biorbd_model.nbGeneralizedTorque() + [excitation_max] * biorbd_model.nbMuscleTotal(),
#             ]
#         )
#         u_init.add([torque_init] * biorbd_model.nbGeneralizedTorque() + [excitation_init] * biorbd_model.nbMuscleTotal())
#         # u_init.add(init_residual_torque, interpolation=InterpolationType.EACH_FRAME)
#
#     else:
#         u_bounds.add([[excitation_min] * biorbd_model.nbMuscleTotal(), [excitation_max] * biorbd_model.nbMuscleTotal()])
#         if activation_driven:
#             # u_init.add([activation_init] * biorbd_model.nbMuscleTotal())
#             u_init.add(excitations_ref, interpolation=InterpolationType.EACH_FRAME)
#         else:
#             # u_init.add([excitation_init] * biorbd_model.nbMuscleTotal())
#             u_init.add(excitations_ref, interpolation=InterpolationType.EACH_FRAME)
#
#     # Get initial isometric forces
#     fiso = []
#     for k in range(nb_mus):
#         fiso.append(biorbd_model.muscle(k).characteristics().forceIsoMax().to_mx())
#
#     # Define the parameter to optimize
#     bound_p_iso = Bounds(
#         # min_bound=np.repeat(0.75, nb_mus), max_bound=np.repeat(3, nb_mus), interpolation=InterpolationType.CONSTANT)
#         min_bound = [0.5] * nb_mus + [0.75], max_bound = [3.5] * nb_mus + [3], interpolation = InterpolationType.CONSTANT)
#     bound_shape_factor = Bounds(
#         min_bound=np.repeat(-3, nb_mus), max_bound=np.repeat(0, nb_mus), interpolation=InterpolationType.CONSTANT)
#
#     p_iso_init = InitialConditions([1] * nb_mus + [2])
#     initial_guess_A = InitialConditions([-3] * nb_mus)
#
#     parameters = ParameterList()
#     parameters.add(
#         "p_iso",  # The name of the parameter
#         modify_isometric_force,  # The function that modifies the biorbd model
#         p_iso_init,
#         bound_p_iso,  # The bounds
#         size=nb_mus+1,  # The number of elements this particular parameter vector has
#         fiso_init=fiso,
#     )
#     # parameters.add(
#     #         "shape_factor",  # The name of the parameter
#     #         modify_shape_factor,
#     #         initial_guess_A,
#     #         bound_shape_factor,  # The bounds
#     #         size=nb_mus,  # The number of elements this particular parameter vector has
#     # )
#
#     # ------------- #
#     return OptimalControlProgram(
#         biorbd_model,
#         dynamics,
#         number_shooting_points,
#         final_time,
#         x_init,
#         u_init,
#         x_bounds,
#         u_bounds,
#         objective_functions,
#         nb_threads=nb_threads,
#         use_SX=use_SX,
#         # parameters=parameters
#     )
#
# data = 'flex' #"horizon_co" #flex #abd_co
# model = "arm_Belaise_v2_scaled.bioMod"
#
# biorbd_model = biorbd.Model("./models/" + model)
# n_frames_wanted = 31
# nb_mus = biorbd_model.nbMuscles()
#
# # Data to track
# # --- Markers --- #
# mat_contents = sio.loadmat(f"./data_real/Sujet_5/marker_{data}.mat")
# # marker_treat = mat_contents['marker_reduce']
# marker_treat = mat_contents['marker_treat'][:-1]
# marker_treat = reduce(n_frames_wanted, marker_treat)[0]
#
# # --- X_init --- #
# mat_contents = sio.loadmat(f"./data_real/Sujet_5/states_ekf_wo_RT_{data}.mat")
# x_init = mat_contents["x_init"]
# x_init = reduce(n_frames_wanted, x_init)[0]
#
# # --- EMG --- #
# emg_treat = x_init[-biorbd_model.nbMuscles():, :-1]
#
# n_shooting_points = marker_treat.shape[2] - 1
# t = np.linspace(0, 0.24, n_shooting_points + 1) # 0.24s
# final_time = t[n_shooting_points]
# q_ref = x_init[:5, :]
#
# # Track these data
# use_residual_torque = False
# activation_driven = True
# acados = False
# if acados:
#     use_SX = True
#     param = False
# else:
#     use_SX = False
#     param = False
#
# # kin_data_to_track = "q"
# kin_data_to_track = "markers"
# biorbd_model = biorbd.Model("./models/" + model)  # To allow for non free variable, the model must be reloaded
# fiso = []
# for k in range(nb_mus):
#     fiso.append(biorbd_model.muscle(k).characteristics().forceIsoMax().to_mx())
# mat_content = sio.loadmat("./results/param_f_iso_flex.mat")
# f_iso = mat_content["f_iso"] * mat_content["f_iso_global"]
# for k in range(biorbd_model.nbMuscles()):
#     biorbd_model.muscle(k).characteristics().setForceIsoMax(
#         f_iso[k] * fiso[k]
#     )
#
# ocp_ip = prepare_ocp(
#         biorbd_model,
#         final_time,
#         n_shooting_points,
#         marker_treat,
#         emg_treat,
#         q_ref,
#         x_init,
#         use_residual_torque=use_residual_torque,
#         activation_driven=activation_driven,
#         kin_data_to_track=kin_data_to_track,
#         nb_threads=8,
#         use_SX=False,
#         param=param
#     )
# ocp_ac = prepare_ocp(
#         biorbd_model,
#         final_time,
#         n_shooting_points,
#         marker_treat,
#         emg_treat,
#         q_ref,
#         x_init,
#         use_residual_torque=use_residual_torque,
#         activation_driven=activation_driven,
#         kin_data_to_track=kin_data_to_track,
#         nb_threads=8,
#         use_SX=True,
#         param=param
#     )
tol_init = 2
tol_final = 6

tol_final += 1
x_ac = np.ndarray((tol_final - tol_init + 1, 850))
x_ip = np.ndarray((tol_final - tol_init + 1, 850))
sol_simulate_single_shooting_ip = {}
sol_simulate_single_shooting_ac = {}
tic = time()
for i in range(tol_init, tol_final):
    idx = i - tol_init
    print(f"\n------------tols : 1e-{i}; Solver : Ipopt-----------\n")
    # biorbd_model = biorbd.Model("./models/" + model)  # To allow for non free variable, the model must be reloaded
    # fiso = []
    # for k in range(nb_mus):
    #     fiso.append(biorbd_model.muscle(k).characteristics().forceIsoMax().to_mx())
    # mat_content = sio.loadmat("./results/param_f_iso_flex.mat")
    # f_iso = mat_content["f_iso"] * mat_content["f_iso_global"]
    # for k in range(biorbd_model.nbMuscles()):
    #     biorbd_model.muscle(k).characteristics().setForceIsoMax(
    #         f_iso[k] * fiso[k]
    #     )
    # ocp_ip = flex.prepare_ocp(
    #     biorbd_model,
    #     final_time,
    #     n_shooting_points,
    #     marker_treat,
    #     emg_treat,
    #     q_ref,
    #     x_init,
    #     use_residual_torque=use_residual_torque,
    #     activation_driven=activation_driven,
    #     kin_data_to_track=kin_data_to_track,
    #     nb_threads=8,
    #     use_SX=False,
    #     param=param
    # )
    # sol_ip = ocp_ip.solve(solver=Solver.IPOPT, show_online_optim=False, solver_options={"tol": float("1e%d" %-i),
    #                                                                          "max_iter": 150,
    #                                                                          # "dual_inf_tol": 1,
    #                                                                          "constr_viol_tol": float("1e%d" %-i),
    #                                                                          "compl_inf_tol": float("1e%d" %-i),
    #                                                                          # "hessian_approximation": "limited-memory",
    #                                                                          "hessian_approximation": "exact",
    #                                                                          "linear_solver": "ma57",  # "ma57", "ma86", "mumps"
    #                                                                          })
    # sol_simulate_single_shooting_ip[idx] = Simulate.from_solve(ocp_ip, sol_ip, single_shoot=True)
    # x_ip[idx, :] = sol_ip['x']
    #
    # print(f"\n---------tols : 1e-{i}; Solver : Acados------------\n")
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


    def modify_isometric_force(biorbd_model, value, fiso_init):
        for k in range(biorbd_model.nbMuscles()):
            biorbd_model.muscle(k).characteristics().setForceIsoMax(
                value[biorbd_model.nbMuscles()] * value[k] * fiso_init[k]
            )


    def modify_shape_factor(biorbd_model, value):
        for k in range(biorbd_model.nbMuscles()):
            biorbd.StateDynamicsBuchananDeGroote(biorbd_model.muscle(k).state()).setShapeFactor(value[k])

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
            param=False
    ):

        # --- Options --- #
        nb_mus = biorbd_model.nbMuscleTotal()
        activation_min, activation_max, activation_init = 0, 1, 0.3
        excitation_min, excitation_max, excitation_init = 0, 1, 0.1
        torque_min, torque_max, torque_init = -100, 100, 0

        # -- Force iso ipopt pour acados
        # if param is not True:
        #     fiso = []
        #     for k in range(nb_mus):
        #         fiso.append(biorbd_model.muscle(k).characteristics().forceIsoMax().to_mx())
        #     mat_content = sio.loadmat("./results/param_f_iso_flex.mat")
        #     f_iso = mat_content["f_iso"] * mat_content["f_iso_global"]
        #     for k in range(biorbd_model.nbMuscles()):
        #         biorbd_model.muscle(k).characteristics().setForceIsoMax(
        #             f_iso[k] * fiso[k]
        #         )

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

        # Define the parameter to optimize
        bound_p_iso = Bounds(
            # min_bound=np.repeat(0.75, nb_mus), max_bound=np.repeat(3, nb_mus), interpolation=InterpolationType.CONSTANT)
            min_bound=[0.5] * nb_mus + [0.75], max_bound=[3.5] * nb_mus + [3], interpolation=InterpolationType.CONSTANT)
        bound_shape_factor = Bounds(
            min_bound=np.repeat(-3, nb_mus), max_bound=np.repeat(0, nb_mus), interpolation=InterpolationType.CONSTANT)

        p_iso_init = InitialConditions([1] * nb_mus + [2])
        initial_guess_A = InitialConditions([-3] * nb_mus)

        parameters = ParameterList()
        parameters.add(
            "p_iso",  # The name of the parameter
            modify_isometric_force,  # The function that modifies the biorbd model
            p_iso_init,
            bound_p_iso,  # The bounds
            size=nb_mus + 1,  # The number of elements this particular parameter vector has
            fiso_init=fiso,
        )
        # parameters.add(
        #         "shape_factor",  # The name of the parameter
        #         modify_shape_factor,
        #         initial_guess_A,
        #         bound_shape_factor,  # The bounds
        #         size=nb_mus,  # The number of elements this particular parameter vector has
        # )

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
    # sol_ac = ocp_ac.solve(solver=Solver.ACADOS, solver_options={
    #                                 "qp_solver": "PARTIAL_CONDENSING_HPIPM", "integrator_type": "IRK",
    #                                 "nlp_solver_max_iter": 150, "sim_method_num_steps": 5,
    #                                 "nlp_solver_tol_ineq": float("1e%d" %-i), "nlp_solver_tol_stat": float("1e%d" %-i),
    #                                 "nlp_solver_tol_comp": float("1e%d" %-i), "nlp_solver_tol_eq": float("1e%d" %-i)
    #                                  })# FULL_CONDENSING_QPOASES, "PARTIAL_CONDENSING_HPIPM"
    # # sol_simulate_single_shooting_ac[idx] = Simulate.from_solve(ocp_ac, sol_ac, single_shoot=True)
    # # x_ac[idx, :] = sol_ac['x']
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


    def modify_isometric_force(biorbd_model, value, fiso_init):
        for k in range(biorbd_model.nbMuscles()):
            biorbd_model.muscle(k).characteristics().setForceIsoMax(
                value[biorbd_model.nbMuscles()] * value[k] * fiso_init[k]
            )


    def modify_shape_factor(biorbd_model, value):
        for k in range(biorbd_model.nbMuscles()):
            biorbd.StateDynamicsBuchananDeGroote(biorbd_model.muscle(k).state()).setShapeFactor(value[k])

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
            param=False
    ):

        # --- Options --- #
        nb_mus = biorbd_model.nbMuscleTotal()
        activation_min, activation_max, activation_init = 0, 1, 0.3
        excitation_min, excitation_max, excitation_init = 0, 1, 0.1
        torque_min, torque_max, torque_init = -100, 100, 0

        # -- Force iso ipopt pour acados
        # if param is not True:
        #     fiso = []
        #     for k in range(nb_mus):
        #         fiso.append(biorbd_model.muscle(k).characteristics().forceIsoMax().to_mx())
        #     mat_content = sio.loadmat("./results/param_f_iso_flex.mat")
        #     f_iso = mat_content["f_iso"] * mat_content["f_iso_global"]
        #     for k in range(biorbd_model.nbMuscles()):
        #         biorbd_model.muscle(k).characteristics().setForceIsoMax(
        #             f_iso[k] * fiso[k]
        #         )

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

        # Define the parameter to optimize
        bound_p_iso = Bounds(
            # min_bound=np.repeat(0.75, nb_mus), max_bound=np.repeat(3, nb_mus), interpolation=InterpolationType.CONSTANT)
            min_bound=[0.5] * nb_mus + [0.75], max_bound=[3.5] * nb_mus + [3], interpolation=InterpolationType.CONSTANT)
        bound_shape_factor = Bounds(
            min_bound=np.repeat(-3, nb_mus), max_bound=np.repeat(0, nb_mus), interpolation=InterpolationType.CONSTANT)

        p_iso_init = InitialConditions([1] * nb_mus + [2])
        initial_guess_A = InitialConditions([-3] * nb_mus)

        parameters = ParameterList()
        parameters.add(
            "p_iso",  # The name of the parameter
            modify_isometric_force,  # The function that modifies the biorbd model
            p_iso_init,
            bound_p_iso,  # The bounds
            size=nb_mus + 1,  # The number of elements this particular parameter vector has
            fiso_init=fiso,
        )
        # parameters.add(
        #         "shape_factor",  # The name of the parameter
        #         modify_shape_factor,
        #         initial_guess_A,
        #         bound_shape_factor,  # The bounds
        #         size=nb_mus,  # The number of elements this particular parameter vector has
        # )

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

    if acados:
        use_SX = True
        param = False
    else:
        use_SX = False
        param = False

    # kin_data_to_track = "q"
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
        param=param
    )
    sol_ac = ocp_ac.solve(solver=Solver.ACADOS, solver_options={
                                    "qp_solver": "PARTIAL_CONDENSING_HPIPM", "integrator_type": "IRK",
                                    "nlp_solver_max_iter": 150, "sim_method_num_steps": 5,
                                    "nlp_solver_tol_ineq": 1e-5, "nlp_solver_tol_stat": 1e-5,
                                    "nlp_solver_tol_comp": 1e-5, "nlp_solver_tol_eq": 1e-5
                                     })# FULL_CONDENSING_QPOASES, "PARTIAL_CONDENSING_HPIPM"
    sol_simulate_single_shooting_ac[idx] = Simulate.from_solve(ocp_ac, sol_ac, single_shoot=True)
    # x_ac[idx, :] = sol_ac['x']

# sio.savemat("sol_single_shoot.mat", {"sol_ip" : sol_simulate_single_shooting_ip[:]['x'],
#                                      "sol_ac" : sol_simulate_single_shooting_ac[:]['x']})

toc = time() - tic
print(toc)
RMSE_ip = np.ndarray((ocp_ip.nlp[0]['ns'], tol_final - tol_init))
RMSE_ac = np.ndarray((ocp_ac.nlp[0]['ns'], tol_final - tol_init))
for i in range(tol_final - tol_init):
    for s in range(int(ocp_ip.nlp[0]['ns'])):
        RMSE_ip[s, i] = np.sqrt((x_ip[i, s*28+1] - sol_simulate_single_shooting_ip[i]['x'][i*28+1])**2)
for i in range(tol_final - tol_init):
    for s in range(int(ocp_ip.nlp[0]['ns'])):
        RMSE_ac[s, i] = np.sqrt((x_ac[i, s*28+1] - sol_simulate_single_shooting_ac[i]['x'][i*28+1])**2)
sio.savemat("RMSE.mat", {"RMSE_ip": RMSE_ip, "RMSE_ac": RMSE_ac})
x1 = range(tol_init, tol_final)
height = []
for i in range(tol_final - tol_init):
    height.append(RMSE_ip[ocp_ip.nlp[0]['ns']-1, i])
width = 0.02
plt.bar(x1, height, width, color='red')
plt.xlabel('1e-tol')
plt.ylabel('RMSE')

x2 = [i + width for i in x1]
height = []
for i in range(tol_final - tol_init):
    height.append(RMSE_ac[ocp_ip.nlp[0]['ns']-1, i])
plt.bar(x2, height, width, color='blue')
plt.xlabel('1e-tol')
plt.ylabel('RMSE')
plt.title('RMSE')
plt.legend(["Ipopt", "Acados"], loc=2)
plt.show()

# ocp_ac, sol_ac = OptimalControlProgram.load(f"./results/Markers_EMG_tracking_{data}_acados.bo")
# states, controls = Data.get_data(ocp_ac, sol_ac["x"])
# q_ac = states['q']
# qdot_ac = states['q_dot']
# # x_ac = vertcat(states["q"], states["q_dot"])
# x_ac = np.concatenate((q_ac, qdot_ac))
# X = InitialConditions(x_ac[:,0])
# u_ac = controls['muscles']
# # u_ac = np.reshape(u_ac, (ocp_ac.nlp[0]['nu']*ocp_ac.nlp[0]['ns']+1,1))
# U = InitialConditions(u_ac[:,0])
#
# ocp_ip, sol_ip = OptimalControlProgram.load(f"./results/Markers_EMG_tracking_{data}_ipopt.bo")
# states, controls = Data.get_data(ocp_ip, sol_ip["x"])
# q_ip = states['q']
# qdot_ip = states['q_dot']
# # x_ip = vertcat(states["q"], states["q_dot"])
# x_ip = np.concatenate((q_ip, qdot_ip))
# X_ip = InitialConditions(x_ip[:, 0])
# u_ip = controls['muscles']
# # u_ip = np.reshape(u_ip, biorbd_model.nbQ(), ocp.nlp[0]['ns'])
# U_ip = InitialConditions(u_ip[:,0])
# # --- Single shooting --- #
#
# sol_simulate_single_shooting_ac = Simulate.from_data(ocp_ac, Data.get_data(ocp_ac, sol_ac["x"]), single_shoot=True)
# sol_simulate_single_shooting_ip = Simulate.from_data(ocp_ip, Data.get_data(ocp_ip, sol_ip["x"]), single_shoot=True)
# result_single_ac = ShowResult(ocp_ac, sol_simulate_single_shooting_ac)
# result_single_ip = ShowResult(ocp_ip, sol_simulate_single_shooting_ip)
# # result_single_ac.graphs()
# # result_multiple_ac.graphs()
# # result_single_ip.graphs()
# # --- Single shooting --- #
# # dif = (sol_simulate_single_shooting_ac_multiple['x'][30] - sol_simulate_single_shooting_ac['x'][30])
#
# RMSE_ip = np.ndarray((ocp_ip.nlp[0]['ns']),)
# for i in range(int(ocp_ip.nlp[0]['ns'])):
#     RMSE_ip[i] = np.sqrt((x_ip[1,i] - sol_simulate_single_shooting_ip['x'][i*28+1])**2)
#
# RMSE_ac = np.ndarray((ocp_ip.nlp[0]['ns']),)
# for i in range(ocp_ip.nlp[0]['ns']):
#     RMSE_ac[i] = np.sqrt((x_ac[1,i] - sol_simulate_single_shooting_ac['x'][i*28+1])**2)

