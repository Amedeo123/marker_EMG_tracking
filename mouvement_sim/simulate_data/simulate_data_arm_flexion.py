import biorbd
from time import time
import numpy as np

from biorbd_optim import (
    OptimalControlProgram,
    ObjectiveList,
    Objective,
    DynamicsTypeList,
    DynamicsType,
    BoundsList,
    QAndQDotBounds,
    InitialConditionsList,
    Bounds,
    ShowResult,
    Solver,
)


def prepare_ocp(biorbd_model_path, final_time, number_shooting_points, use_SX=False, nb_threads=1):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    tau_min, tau_max, tau_init = -5, 5, 0
    activation_min, activation_max, activation_init = 0, 1, 0.5
    excitation_min, excitation_max, excitation_init = 0, 1, 0.5

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=100)
    objective_functions.add(Objective.Lagrange.MINIMIZE_STATE, weight=100)
    objective_functions.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10)

    # Dynamics
    dynamics = DynamicsTypeList()
    dynamics.add(DynamicsType.MUSCLE_EXCITATIONS_AND_TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(QAndQDotBounds(biorbd_model))

    # Add muscle to the bounds
    x_bounds[0].concatenate(
        Bounds([activation_min] * biorbd_model.nbMuscles(), [activation_max] * biorbd_model.nbMuscles())
    )

    # Following values are taken from Belaise's matlab code
    x_bounds[0].min[:, 0] = (-0.2, 0.1, -0.25, 0.1, 0, -0, -0.2, 0.05, -0.15, -0.02, 0, 0.28) + (0,) * biorbd_model.nbMuscles()
    x_bounds[0].max[:, 0] = (-0.2, 0.1, -0.25, 0.1, 0, -0, -0.2, 0.05, -0.15, -0.02, 0, 0.28) + (0,) * biorbd_model.nbMuscles()
    x_bounds[0].min[:biorbd_model.nbQ()*2, -1] = (-0.03, 0.1, -0.1, 0.2, -0.76, 1., 2., -1.5, -0.17, -0.62, 1.4, -0.57)
    x_bounds[0].max[:biorbd_model.nbQ()*2, -1] = (-0.03, 0.1, -0.1, 0.2, -0.76, 1., 2., -1.5, -0.17, -0.62, 1.4, -0.57)


    # Initial guess

    x_init = InitialConditionsList()
    x_init.add([1.] * biorbd_model.nbQ() + [0] * biorbd_model.nbQdot() + [0] * biorbd_model.nbMuscles())


    # Define control path constraint
    u_bounds = BoundsList()
    u_init = InitialConditionsList()

    u_bounds.add(
        [
            [tau_min] * biorbd_model.nbGeneralizedTorque() + [excitation_min] * biorbd_model.nbMuscles(),
            [tau_max] * biorbd_model.nbGeneralizedTorque() + [excitation_max] * biorbd_model.nbMuscles(),
        ]
    )
    u_init.add([tau_init] * biorbd_model.nbGeneralizedTorque() + [excitation_init] * biorbd_model.nbMuscles())
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
        use_SX=use_SX,
        nb_threads=nb_threads
    )


if __name__ == "__main__":
    nb_shooting_points = 101
    Tf = 1.0
    generate_with = 'acados'

    if generate_with == 'acados':
        ocp = prepare_ocp(biorbd_model_path="arm_Belaise.bioMod", final_time=Tf,
                             number_shooting_points=nb_shooting_points, use_SX=True)

        # --- Solve the program --- #
        sol = ocp.solve(
            solver=Solver.ACADOS,
            show_online_optim=False,
            solver_options={"nlp_solver_tol_comp": 1e-3, "nlp_solver_tol_eq": 1e-3, "nlp_solver_tol_stat": 1e-3,
                            "sim_method_num_steps": 3, },
        )

    if generate_with == 'ipopt':
        ocp = prepare_ocp(biorbd_model_path="arm_Belaise.bioMod", final_time=Tf,
                             number_shooting_points=nb_shooting_points, use_SX=False, nb_threads=6)
        sol = ocp.solve(
            solver=Solver.IPOPT,
            show_online_optim=False,
            solver_options={
                "tol": 1e-3,
                "dual_inf_tol": 1e-3,
                "constr_viol_tol": 1e-3,
                "compl_inf_tol": 1e-3,
                "linear_solver": "ma57",
                "max_iter": 100,
                "hessian_approximation": "limited-memory",
            },
        )

    print(f"Time to solve: {sol['time_tot']}sec")


    # --- Show results --- #
    ocp.save_get_data(sol, f"arm_flexion_sim_{nb_shooting_points}sn.bob")
    result = ShowResult(ocp, sol)
    result.graphs()
    result.animate()
