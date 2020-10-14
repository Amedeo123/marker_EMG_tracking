import scipy.io as sio
from pyomeca import Markers, Rototrans
import numpy as np
import datetime
from time import time
import biorbd
import os.path
from pathlib import Path
from casadi import MX, Function, vertcat, jacobian
from matplotlib import pyplot as plt
from bioptim import (
    OptimalControlProgram,
    BidirectionalMapping,
    Mapping,
    ObjectiveOption,
    DynamicsTypeList,
    DynamicsType,
    BidirectionalMapping,
    Mapping,
    Data,
    ObjectiveList,
    Objective,
    BoundsList,
    Bounds,
    QAndQDotBounds,
    InitialGuessList,
    ParameterList,
    InitialGuess,
    ShowResult,
    InterpolationType,
    DynamicsFunctions,
    Solver,
    NonLinearProgram,
    ConstraintList,
    Constraint,
    Instant
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
        while n_shooting % (n_frames_expected - c) != 0:
            if c > 5:
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


def modify_isometric_force(biorbd_model, value, fiso_init):
    for k in range(biorbd_model.nbMuscles()):
        biorbd_model.muscle(k).characteristics().setForceIsoMax(
            value[k] * fiso_init[k]
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
        f_init,
        use_residual_torque,
        activation_driven,
        kin_data_to_track,
        nb_threads,
        use_SX=True,
        param = False
        ):

    # --- Options --- #
    nb_mus = biorbd_model.nbMuscleTotal()
    activation_min, activation_max, activation_init = 0, 1, 0.3
    excitation_min, excitation_max, excitation_init = 0, 1, 0.1
    torque_min, torque_max, torque_init = -100, 100, 0
    # nb_tau = biorbd_model.segment(0).nbQ()
    # tau_mapping = BidirectionalMapping(Mapping(range(6)), Mapping(range(nb_tau)))

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.TRACK_MUSCLES_CONTROL, weight=10, target=excitations_ref)
    # objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=1000,
    #                         idx_states=(0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16))
    objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=1, idx_states=(6, 7, 8, 9, 10))


    if use_residual_torque:
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1) # controls_idx=[6, 7, 8, 9, 10],

    if kin_data_to_track == "markers":
        objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=1000,
                                target=marker_ref
                                )

    elif kin_data_to_track == "q":
        objective_functions.add(
            Objective.Lagrange.TRACK_STATE, weight=100,
            target=q_ref,
            states_idx=range(biorbd_model.nbQ())
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
    # constraints = ConstraintList()
    # constraints.add(Constraint.TRACK_TORQUE, instant=Instant.ALL, controls_idx=(6, 7, 8, 9, 10),
    #                 target=np.zeros((biorbd_model.nbQ()*2, number_shooting_points)))

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    # if use_SX:
    #     x_bounds[0].min[biorbd_model.nbQ():, 0] = -10
    #     x_bounds[0].max[biorbd_model.nbQ():, 0] = 10


    # Add muscle to the bounds
    if activation_driven is not True:
        x_bounds[0].concatenate(
            Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
        )

    # Initial guess
    x_init = InitialGuessList()
    if activation_driven:
        # state_base = np.ndarray((12, n_shooting_points+1))
        # for i in range(n_shooting_points+1):
        #     state_base[:, i] = np.concatenate((state_init[:6, 0], np.zeros((6))))
        x_init.add(state_init[:-nb_mus, :], interpolation=InterpolationType.EACH_FRAME)
        # x_init.add(state_init[:-nb_mus, :], interpolation=InterpolationType.EACH_FRAME)
    else:
        x_init.add([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()) + [0] * biorbd_model.nbMuscles())
        # x_init.add(state_init[biorbd_model.nbQ():, :], interpolation=InterpolationType.EACH_FRAME)

    # Add muscle to the bounds
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    init_residual_torque = np.concatenate((np.ones((nb_tau, n_shooting_points))*0.5,
                                           excitations_ref))
    if use_residual_torque:
        u_bounds.add(
            [
                [torque_min] * nb_tau + [excitation_min] * biorbd_model.nbMuscleTotal(),
                [torque_max] * nb_tau + [excitation_max] * biorbd_model.nbMuscleTotal(),
            ]
        )
        # u_init.add([torque_init] * tau_mapping.reduce.len + [excitation_init] * biorbd_model.nbMuscleTotal())
        u_init.add(init_residual_torque, interpolation=InterpolationType.EACH_FRAME)

    else:
        u_bounds.add([[excitation_min] * biorbd_model.nbMuscleTotal(), [excitation_max] * biorbd_model.nbMuscleTotal()])
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
        # min_bound=np.repeat(0.01, nb_mus+1), max_bound=np.repeat(7, nb_mus+1), interpolation=InterpolationType.CONSTANT)
        min_bound=[0.5] * nb_mus, max_bound=[3] * nb_mus, interpolation=InterpolationType.CONSTANT)
    bound_shape_factor = Bounds(
        min_bound=np.repeat(-3, nb_mus), max_bound=np.repeat(0, nb_mus), interpolation=InterpolationType.CONSTANT)

    p_iso_init = InitialGuess(f_init)
    initial_guess_A = InitialGuess([-3] * nb_mus)

    parameters = ParameterList()
    parameters.add(
        "p_iso",  # The name of the parameter
        modify_isometric_force,  # The function that modifies the biorbd model
        p_iso_init,
        bound_p_iso,  # The bounds
        size=nb_mus,  # The number of elements this particular parameter vector has
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
        parameters=parameters,
        # tau_mapping=tau_mapping,
    )


if __name__ == "__main__":
    data ='flex_co' #'abd_co' #'horizon_co'  #
    n_frames_wanted = 50

    # Track these data
    use_residual_torque = True
    activation_driven = True
    acados = True
    sujet = "5"
    nb_try = 1
    if acados:
        use_SX = True
        param = True
    else:
        use_SX = False
        param = True

    # kin_data_to_track = "q"
    for i in range(nb_try):
        tries = str(i)
        print(f"------------ Try_{tries} -----------------------\n")

        model_path = '/home/amedeo/Documents/programmation/marker_emg_tracking/models/'
        data_path = '/home/amedeo/Documents/programmation/marker_emg_tracking/mouvement_reel/results'

        model = "arm_Belaise_real_v3_scaled.bioMod" #"arm_Belaise_real_v2.bioMod"  #
        biorbd_model = biorbd.Model(model_path + model)

        # --- Data to track for each repetition --- #
        # --- x-init --- #
        mat_contents = sio.loadmat(data_path + f"/sujet_{sujet}/states_init_{data}.mat")
        x_init = mat_contents[f"x_init_{tries}"]
        x_init = reduce(n_frames_wanted, x_init)[0]
        # x_init = x_init[:, :]
        q_ref = x_init[:biorbd_model.nbQ(), :]
        t_final = float(mat_contents[f"t_final_{tries}"])

        # --- EMG --- #
        emg_treat = x_init[-biorbd_model.nbMuscles():, :-1]

        # --- Markers --- #
        mat_contents = sio.loadmat(data_path + f"/sujet_{sujet}/data_{data}_treat.mat")
        marker_treat = mat_contents[f"marker_try_{tries}"][:-1, 2:, :]
        marker_treat = reduce(n_frames_wanted, marker_treat)[0]

        n_shooting_points = marker_treat.shape[2] - 1
        t = np.linspace(0, t_final, n_shooting_points + 1)
        # mat_content = sio.loadmat(f'./results/sujet_5/param_f_iso_{data}_try_0.mat')
        # f_init = np.concatenate((mat_content["f_iso"], mat_content["f_iso_global"]))
        f_init = [1] * biorbd_model.nbMuscles() #+ [1]
        print(f'n_shooting : {n_shooting_points}')
        kin_data_to_track = "markers"
        biorbd_model = biorbd.Model(model_path + model)  # To allow for non free variable, the model must be reloaded
        if param is not True:
            fiso = []
            for k in range(biorbd_model.nbMuscles()):
                fiso.append(biorbd_model.muscle(k).characteristics().forceIsoMax().to_mx())
            mat_content = sio.loadmat(f"./results/sujet_5/param_f_iso_{data}_try_{tries}.mat")
            f_iso = mat_content["f_iso"] * mat_content["f_iso_global"]
            for k in range(biorbd_model.nbMuscles()):
                biorbd_model.muscle(k).characteristics().setForceIsoMax(
                    f_iso[k] * fiso[k]
                )
        ocp = prepare_ocp(
            biorbd_model,
            t_final,
            n_shooting_points,
            marker_treat,
            emg_treat,
            q_ref,
            x_init,
            f_init,
            use_residual_torque=use_residual_torque,
            activation_driven=activation_driven,
            kin_data_to_track=kin_data_to_track,
            nb_threads=8,
            use_SX=use_SX,
            param=param
        )

        # --- Solve and save the program --- #
        tic = time()
        if acados is not True:
            i = 5
            sol = ocp.solve(solver=Solver.IPOPT, show_online_optim=False,
                            solver_options={"tol": float("1e%d" % -i),
                                            "max_iter": 150,
                                            # "dual_inf_tol": 1,
                                            "constr_viol_tol": float("1e%d" % -i),
                                            "compl_inf_tol": float("1e%d" % -i),
                                            # "hessian_approximation": "limited-memory",
                                            "hessian_approximation": "exact",
                                            "linear_solver": "ma57"  # "ma57", "ma86", "mumps"
                                            })

        else:
            i = 5
            sol = ocp.solve(solver=Solver.ACADOS,  # FULL_CONDENSING_QPOASES, "PARTIAL_CONDENSING_HPIPM"
                            solver_options={"qp_solver": "PARTIAL_CONDENSING_HPIPM", "integrator_type": "IRK",
                                            "nlp_solver_max_iter": 50, "sim_method_num_steps": 1,
                                            "nlp_solver_tol_ineq": float("1e%d" % -i),
                                            "nlp_solver_tol_stat": float("1e%d" % -i),
                                            "nlp_solver_tol_comp": float("1e%d" % -i),
                                            "nlp_solver_tol_eq": float("1e%d" % -i)})
        toc = time() - tic
        # print(f"Time to solve : {toc}sec")
        p_f_iso = 0
        p_global_iso = 0
        print(toc)
        if param:
            states_sol, controls_sol, params = Data.get_data(ocp, sol["x"], get_parameters=True)
            p_f_iso = params["p_iso"]
            # p_global_iso = params["p_iso"][ocp.nlp[0].model.nbMuscles()]
            print(p_f_iso)
            # print(p_global_iso)
        else:
            states_sol, controls_sol = Data.get_data(ocp, sol["x"])

        q = states_sol["q"]
        q_dot = states_sol["q_dot"]
        if activation_driven:
            activations = controls_sol["muscles"]
        else:
            activations = states_sol["muscles"]
            excitations = controls_sol["muscles"]

        if use_residual_torque:
            tau = controls_sol["tau"]

        n_q = ocp.nlp[0].model.nbQ()
        n_qdot = ocp.nlp[0].model.nbQdot()
        n_mark = ocp.nlp[0].model.nbMarkers()
        n_frames = q.shape[1]

        markers = np.ndarray((3, n_mark, q.shape[1]))
        symbolic_states = MX.sym("x", n_q, 1)
        markers_func = Function(
            "ForwardKin", [symbolic_states], [biorbd_model.markers(symbolic_states)], ["q"], ["markers"]
        ).expand()
        for i in range(n_frames):
            markers[:, :, i] = markers_func(q[:, i])

        # plt.figure("Markers")
        # for i in range(markers.shape[1]):
        #     plt.plot(np.linspace(0, 1, n_shooting_points + 1), marker_treat[:, i, :].T, "k")
        #     plt.plot(np.linspace(0, 1, n_shooting_points + 1), markers[:, i, :].T, "r--")
        # plt.xlabel("Time")
        # plt.ylabel("Markers Position")
        # plt.show()

        # --- Show result --- #
        result = ShowResult(ocp, sol)
        result.graphs()
        result.animate()

        dic = {"f_iso": p_f_iso}

        if acados:
            sio.savemat(f"./results/sujet_{sujet}/param_f_iso_{data}_try_{tries}_acados.mat", dic)
            ocp.save_get_data(sol, f"./results/sujet_{sujet}/Markers_EMG_tracking_f_iso_{data}_try_{tries}_acados.bob")
        else:
            sio.savemat(f"./results/sujet_{sujet}/param_f_iso_{data}_try_{tries}_ipopt.mat", dic)
            ocp.save_get_data(sol, f"./results/sujet_{sujet}/Markers_EMG_tracking_f_iso_{data}_try_{tries}_ipopt.bob")
        if param is not True:
            if acados:
                ocp.save(sol, f"./results/sujet_{sujet}/Markers_EMG_tracking_{data}_try_{tries}_acados.bo")
            else:
                ocp.save(sol, f"./results/sujet_{sujet}/Markers_EMG_tracking_{data}_try_{tries}_ipopt.bo")
