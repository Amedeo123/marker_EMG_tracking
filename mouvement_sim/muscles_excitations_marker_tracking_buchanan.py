import scipy.io as sio
import numpy as np
import datetime
import biorbd
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


def modify_isometric_force(biorbd_model, value, fiso_init):
    for k in range(biorbd_model.nbMuscles()):
        biorbd_model.muscle(k).characteristics().setForceIsoMax(
            value[biorbd_model.nbMuscles()] * value[k] * fiso_init[k]
            # value[k] * fiso_init[k]
        )

def modify_shape_factor(biorbd_model, value):
    for k in range(biorbd_model.nbMuscles()):
        # biorbd.StateDynamicsBuchananDeGroote(biorbd_model.muscle(k).state()).setShapeFactor(value[k])
        # biorbd.StateDynamicsBuchanan(biorbd_model.muscle(k).state()).shapeFactor(value[k])
        biorbd.StateDynamicsBuchananDeGroote(biorbd_model.muscle(k).state()).setA1(value[k])

def modify_A1(biorbd_model, value):
    for k in range(biorbd_model.nbMuscles()):
        biorbd.StateDynamicsBuchananDeGroote(biorbd_model.muscle(k).state()).setA1(value[k])

def modify_A2(biorbd_model, value):
    for k in range(biorbd_model.nbMuscles()):
        biorbd.StateDynamicsBuchananDeGroote(biorbd_model.muscle(k).state()).setA2(value[k])

def prepare_ocp(
        biorbd_model,
        final_time,
        number_shooting_points,
        marker_ref,
        excitations_ref,
        q_ref,
        state_ekf,
        use_residual_torque,
        kin_data_to_track,
        nb_threads,
        use_SX=False,
        ):

    # --- Options --- #
    nb_mus = biorbd_model.nbMuscleTotal()
    activation_min, activation_max, activation_init = 0, 1, 0.5
    excitation_min, excitation_max, excitation_init = 0, 1, 0.1
    torque_min, torque_max, torque_init = -100, 100, 0
    # shapeFactor = [-1.17807827e-05, -1.12345339e-04, -1.06298720e-05, -4.85475442e-02, -1.90456361e-03, -2.60736365e-01,
    #                -7.85131950e-03, -4.15346233e-02, -2.12925976e-05,
    #                -1.48848665e-01,
    #                -1.38652787e-01,
    #                -2.28073508e-05,
    #                -1.36886153e-05,
    #                -1.75216867e-05,
    #                -1.08256972e-01,
    #                -1.95064804e-01,
    #                -1.37723691e-05,
    #                -2.42525695e-02,
    #                -3.90772988e-04,
    #                -3.13160043e-05]
    # for k in range(biorbd_model.nbMuscles()):
    #      biorbd.StateDynamicsBuchananDeGroote(biorbd_model.muscle(k).state()).setShapeFactor(shapeFactor[k])

    # Add objective functions
    objective_functions = ObjectiveList()

    objective_functions.add(Objective.Lagrange.TRACK_MUSCLES_CONTROL, weight=1, target=excitations_ref)

    if use_residual_torque:
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=10)

    if kin_data_to_track == "markers":
        objective_functions.add(Objective.Lagrange.TRACK_MARKERS, weight=100, target=marker_ref)

    elif kin_data_to_track == "q":
        objective_functions.add(
            Objective.Lagrange.TRACK_STATE, weight=100, target=q_ref,
            states_idx=range(biorbd_model.nbQ()))
    else:
        raise RuntimeError("Wrong choice of kin_data_to_track")

    # Dynamics
    dynamics = DynamicsTypeList()
    if use_residual_torque:
        dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN)
    else:
        dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_DRIVEN)


    # Constraints
    constraints = ()

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))
    # x_bounds[0].min[:, :] = - 2 * np.pi
    # x_bounds[0].max[:, :] = 20 * np.pi

    # Add muscle to the bounds
    x_bounds[0].concatenate(
        Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
    )

    # Initial guess
    x_init = InitialConditionsList()
    # x_init.add([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()) + [0] * biorbd_model.nbMuscles())
    x_init.add(state_ekf, interpolation=InterpolationType.EACH_FRAME)

    # Add muscle to the bounds
    u_bounds = BoundsList()
    u_init = InitialConditionsList()
    init_residual_torque = np.concatenate((np.ones((biorbd_model.nbGeneralizedTorque(), n_shooting_points))*0.5,
                                           excitations_ref))
    if use_residual_torque:
        u_bounds.add(
            [
                [torque_min] * biorbd_model.nbGeneralizedTorque() + [excitation_min] * biorbd_model.nbMuscleTotal(),
                [torque_max] * biorbd_model.nbGeneralizedTorque() + [excitation_max] * biorbd_model.nbMuscleTotal(),
            ]
        )
        # u_init.add([torque_init] * biorbd_model.nbGeneralizedTorque() + [excitation_init] * biorbd_model.nbMuscleTotal())
        u_init.add(init_residual_torque, interpolation=InterpolationType.EACH_FRAME)
    else:
        u_bounds.add([[excitation_min] * biorbd_model.nbMuscleTotal(), [excitation_max] * biorbd_model.nbMuscleTotal()])
        # u_init.add([excitation_init] * biorbd_model.nbMuscleTotal())
        u_init.add(excitations_ref, interpolation=InterpolationType.EACH_FRAME)

    # Get initial isometric forces
    fiso = []
    for k in range(nb_mus):
        fiso.append(biorbd_model.muscle(k).characteristics().forceIsoMax().to_mx())

    # Define the parameter to optimize
    bound_p_iso = Bounds(
        min_bound = [0.5] * nb_mus + [0.75], max_bound = [3.5] * nb_mus + [3], interpolation = InterpolationType.CONSTANT)
    bound_shape_factor = Bounds(
        min_bound=np.repeat(-3, nb_mus), max_bound=np.repeat(0, nb_mus), interpolation=InterpolationType.CONSTANT)

    bound_A1 = Bounds(
        min_bound=np.repeat(0.0001, nb_mus), max_bound=np.repeat(0.12, nb_mus), interpolation=InterpolationType.CONSTANT)

    bound_A2 = Bounds(
        min_bound=np.repeat(0.01, nb_mus), max_bound=np.repeat(1e11, nb_mus),
        interpolation=InterpolationType.CONSTANT)

    p_iso_init = InitialConditions([1] * nb_mus + [2])
    initial_guess_A = InitialConditions([-2] * nb_mus)
    initial_guess_A1 = InitialConditions([0.05] * nb_mus)
    initial_guess_A2 = InitialConditions([700] * nb_mus)

    parameters = ParameterList()
    # parameters.add(
    #     "p_iso",  # The name of the parameter
    #     modify_isometric_force,  # The function that modifies the biorbd model
    #     p_iso_init,
    #     bound_p_iso,  # The bounds
    #     size=nb_mus+1,  # The number of elements this particular parameter vector has
    #     fiso_init=fiso,
    # )
    # parameters.add(
    #         "shape_factor",  # The name of the parameter
    #         modify_shape_factor,
    #         initial_guess_A,
    #         bound_shape_factor,  # The bounds
    #         size=nb_mus,  # The number of elements this particular parameter vector has
    # )
    parameters.add(
            "A1",  # The name of the parameter
            modify_A1,
            initial_guess_A1,
            bound_A1,  # The bounds
            size=nb_mus,  # The number of elements this particular parameter vector has
    )
    parameters.add(
            "A2",  # The name of the parameter
            modify_A2,
            initial_guess_A2,
            bound_A2,  # The bounds
            size=nb_mus,  # The number of elements this particular parameter vector has
    )
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
    )


if __name__ == "__main__":

    # Data to track
    mat_contents = sio.loadmat("./simulate_data/markers.mat")
    # n_wanted = 31
    marker_ref = mat_contents["markers"]
    marker_ref = marker_ref[:, 4:, :]
    # marker_ref = reduce(n_wanted, marker_ref)
    # print(mat_contents)

    mat_contents = sio.loadmat("./simulate_data/excitations.mat")
    excitations_ref = mat_contents["excitations"]
    excitations_ref = excitations_ref[:, :-1]
    # print(mat_contents)

    mat_contents = sio.loadmat("./simulate_data/x_init.mat")
    x_init = mat_contents["x_init"]
    mat_contents = sio.loadmat("./simulate_data/q_ref.mat")
    q_ref = mat_contents["q_ref"]
    final_time = 1
    n_shooting_points = x_init.shape[1] - 1

    # Read data to fit
    t = np.linspace(0, final_time, n_shooting_points + 1)


    # Track these data
    use_residual_torque = False
    # kin_data_to_track = "q"
    kin_data_to_track = "markers"
    biorbd_model = biorbd.Model("arm_Belaise_buchanan_de_groote.bioMod")
    # biorbd_model = biorbd.Model("arm_Belaise_buchanan.bioMod")
    # biorbd_model = biorbd.Model("arm_Belaise.bioMod")# To allow for non free variable, the model must be reloaded
    ocp = prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting_points,
        marker_ref,
        excitations_ref,
        q_ref,
        x_init,
        use_residual_torque=use_residual_torque,
        kin_data_to_track=kin_data_to_track,
        nb_threads=8,
        use_SX=False
    )

    # --- Solve and save the program --- #
    sol = ocp.solve(solver=Solver.IPOPT, show_online_optim=False, solver_options={"tol": 1e-5,
                                                             "max_iter": 30,
                                                             # "dual_inf_tol": 1,
                                                             "constr_viol_tol": 1e-5,
                                                             "compl_inf_tol": 1e-6,
                                                             # "hessian_approximation": "limited-memory",
                                                             "hessian_approximation": "exact",
                                                             "linear_solver": "ma57",  # "ma57", "ma86", "mumps"
                                                             })
    # sol = ocp.solve(solver=Solver.ACADOS)

    # states_sol, controls_sol= Data.get_data(ocp, sol["x"])
    states_sol, controls_sol, params = Data.get_data(ocp, sol["x"], get_parameters=True)
    # shape_factors = {"shape_factor": params["shape_factor"]}
    # print(shape_factors)
    A2 = {"A2": params["A2"]}
    A1 = {"A1": params["A1"]}
    print(A1, A2)
    # p_f_iso = params["p_iso"][:-1]
    # p_global_iso = params["p_iso"][ocp.nlp[0]["model"].nbMuscles()]
    # print(p_f_iso)
    # print(p_global_iso)
    q = states_sol["q"]
    q_dot = states_sol["q_dot"]
    activations = states_sol["muscles"]
    if use_residual_torque:
        tau = controls_sol["tau"]
    excitations = controls_sol["muscles"]

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

    plt.figure("Markers")
    for i in range(markers.shape[1]):
        plt.plot(np.linspace(0, 1, n_shooting_points + 1), marker_ref[:, i, :].T, "k")
        plt.plot(np.linspace(0, 1, n_shooting_points + 1), markers[:, i, :].T, "r--")
    plt.xlabel("Time")
    plt.ylabel("Markers Position")
    plt.show()



    # --- Show result --- #
    result = ShowResult(ocp, sol)
    result.graphs()
    #result.animate()

    date = datetime.datetime.now()
    #ocp.save(sol, f"./results/Q_tracking_with_residual_torque{date}.bo", shape_factors)
    #ocp.save_get_data(sol, f"./results/Q_tracking_with_residual_torque{date}.bob", shape_factors)
