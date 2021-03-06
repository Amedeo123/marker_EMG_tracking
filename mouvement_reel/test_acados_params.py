import biorbd
import numpy as np
from pathlib import Path
from bioptim import (
    OptimalControlProgram,
    DynamicsTypeOption,
    DynamicsType,
    BoundsOption,
    Bounds,
    QAndQDotBounds,
    InitialGuessOption,
    InitialGuess,
    ShowResult,
    ObjectiveOption,
    Objective,
    InterpolationType,
    Data,
    ParameterList,
    ObjectiveList,
    Solver
)


def my_parameter_function(biorbd_model, value, extra_value):
    # The pre dynamics function is called right before defining the dynamics of the system. If one wants to
    # modify the dynamics (e.g. optimize the gravity in this case), then this function is the proper way to do it
    # `biorbd_model` and `value` are mandatory. The former is the actual model to modify, the latter is the casadi.MX
    # used to modify it,  the size of which described by the value `size` in the parameter definition.
    # The rest of the parameter are defined by the user in the parameter
    biorbd_model.setGravity(biorbd.Vector3d(0, 0, value * extra_value))


def my_target_function(ocp, value, target_value):
    # The target function is a penalty function.
    # `ocp` and `value` are mandatory. The rest is defined in the
    # parameter by the user
    return value - target_value


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, min_g, max_g, target_g, use_SX=True):
    # --- Options --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    tau_min, tau_max, tau_init = -300, 300, 0
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=10)
    objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=100)

    objective_functions.add(Objective.Mayer.TRACK_STATE,
                            weight=1000000, target=np.tile(3.14, (n_q, number_shooting_points+1)), states_idx=1)
    # Dynamics
    dynamics = DynamicsTypeOption(DynamicsType.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    x_bounds[:, 0] = 0
    # x_bounds.min[:, 0] = 0
    # x_bounds[:, [0, -1]] = 0
    # x_bounds[1, -1] = 3.14

    # Initial guess
    x_init = InitialGuessOption([0] * (n_q + n_qdot), interpolation=InterpolationType.CONSTANT)
    # u_bounds[1, :] = 0
    u_bounds = BoundsOption([[tau_min] * n_tau, [tau_max] * n_tau])
    
    u_init = InitialGuessOption([tau_init] * n_tau,  interpolation=InterpolationType.CONSTANT)

    # Define the parameter to optimize
    # Give the parameter some min and max bounds
    parameters = ParameterList()
    bound_gravity = Bounds(min_bound=min_g, max_bound=max_g, interpolation=InterpolationType.CONSTANT)
    # and an initial condition
    initial_gravity = InitialGuess((min_g + max_g) / 2,  interpolation=InterpolationType.CONSTANT)
    parameter_objective_functions = ObjectiveOption(
        my_target_function, weight=100, quadratic=True, custom_type=Objective.Parameter, target_value=target_g
    )
    parameters.add(
        "gravity_z",  # The name of the parameter
        my_parameter_function,  # The function that modifies the biorbd model
        initial_gravity,  # The initial guess
        bound_gravity,  # The bounds
        size=1,  # The number of elements this particular parameter vector has
        penalty_list=parameter_objective_functions,  # Objective of constraint for this particular parameter
        extra_value=1,  # You can define as many extra arguments as you want
    )

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
        parameters=parameters,
        use_SX=use_SX
    )


if __name__ == "__main__":
    ocp = prepare_ocp(
        biorbd_model_path="/home/amedeo/Documents/programmation/marker_emg_tracking/models/pendulum.bioMod", final_time=3, number_shooting_points=100, min_g=-8, max_g=8, target_g=-7
        , use_SX=True
    )

    # --- Solve the program --- #
    # sol = ocp.solve(show_online_optim=True)
    i = 5
    sol = ocp.solve(solver=Solver.ACADOS,  # FULL_CONDENSING_QPOASES, "PARTIAL_CONDENSING_HPIPM"
                    solver_options={"qp_solver": "PARTIAL_CONDENSING_HPIPM", "integrator_type": "IRK",
                                    "nlp_solver_max_iter": 50, "sim_method_num_steps": 2,
                                    "nlp_solver_tol_ineq": float("1e%d" % -i),
                                    "nlp_solver_tol_stat": float("1e%d" % -i),
                                    "nlp_solver_tol_comp": float("1e%d" % -i),
                                    "nlp_solver_tol_eq": float("1e%d" % -i)})
    # --- Get the results --- #
    states, controls, params = Data.get_data(ocp, sol, get_parameters=True)

    # --- show reults --- #
    # result = ShowResult(ocp, sol)
    # result.graphs()
    # result.animate()
    q = sol['qqdot'][1:3, :]
    length = params["gravity_z"][0, 0]
    print(length)
    import bioviz
    b = bioviz.Viz(model_path="/home/amedeo/Documents/programmation/marker_emg_tracking/models/pendulum.bioMod")
    b.load_movement(q)
    b.exec()
    

