import numpy as np
import matplotlib.pyplot as plt
from fourier_model import FourierNonlinearModel
from Core.envar_solver import EnVarSolver

def main():
    # --- 1. Problem Setup (4th Order) ---
    N_modes = 4
    
    # Initialize physical model
    model = FourierNonlinearModel(N_modes=N_modes)
    
    # Set ground truth (Target coefficients)
    a_target = np.array([-30, 15, -30, 8])
    y_target = model.forward(a_target)

    # Set initial guess (Distant from target)
    a_init = np.array([1.5, -1.0, 0.5, -0.5])

    print("--- Fourier EnVar Test (N=4) ---")
    print(f"Target:  {a_target}")
    print(f"Initial: {a_init}")

    # --- 2. Initialize Solver ---
    solver_config = {
        'Nen': 4,           # Ensemble size
        'lam': 0.0,          # No regularization
        'max_iter': 200,
        'alpha': 0.01,        # Step size
        'variance': 0.05     # Variance for gradient estimation
    }
    solver = EnVarSolver(model, a_init, y_target, config=solver_config)

    # --- 3. Visualization Setup ---
    plt.ion()
    # CHANGED: 3 Subplots now (18 inches wide)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # --- Subplot 1: Parameter Space Slice (a1, a2) ---
    _generate_contour_background(ax1, model, y_target, N_modes)
    h_target, = ax1.plot(a_target[0], a_target[1], 'y*', markersize=15, label='Target', zorder=5)
    h_mean, = ax1.plot([a_init[0]], [a_init[1]], 'ro', label='Mean', zorder=4)
    h_ens, = ax1.plot([], [], 'b.', markersize=4, alpha=0.6, label='Ensemble', zorder=3)
    h_traj, = ax1.plot([], [], 'k-', linewidth=1.5, label='Trajectory')
    
    ax1.set_title('Parameter Slice ($a_1, a_2$)')
    ax1.set_xlabel('$a_1$')
    ax1.set_ylabel('$a_2$')
    ax1.legend(loc='upper right', fontsize='small')

    # --- Subplot 2: Observation Space ---
    t = model.get_time_grid()
    ax2.plot(t, y_target, 'k-', linewidth=2, label='Target')
    h_obs_curr, = ax2.plot(t, model.forward(a_init), 'r--', label='Current')
    ax2.set_title('Observation Fit')
    ax2.set_xlabel('Time')
    ax2.legend(loc='upper right')

    # --- Subplot 3: Convergence Plot (NEW) ---
    h_cost, = ax3.semilogy([], [], 'r-o', linewidth=2, markersize=4)
    ax3.set_title('Convergence History')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Cost $J_{true}$ (Log Scale)')
    ax3.grid(True, which="both", linestyle='--', alpha=0.5)

    # --- 4. Run Optimization Loop ---
    traj_a = [a_init.copy()]
    cost_history = []
    iter_history = []

    for state in solver.solve():
        iter_ = state['iter']
        a_curr = state['a_mean']
        a_ens = state['ensemble']
        cost = state['J_true']

        print(f"Iter {iter_:02d} | Cost={cost:.4e}")

        # Update Data Lists
        traj_a.append(a_curr.copy())
        traj_arr = np.array(traj_a)
        cost_history.append(cost)
        iter_history.append(iter_)

        # Update Plots
        # 1. Parameters
        h_mean.set_data([a_curr[0]], [a_curr[1]])
        h_traj.set_data(traj_arr[:, 0], traj_arr[:, 1])
        if a_ens is not None:
            max_viz = 10
            h_ens.set_data(a_ens[0, :max_viz], a_ens[1, :max_viz])

        # 2. Observations
        h_obs_curr.set_ydata(state['y_current'])

        # 3. Convergence (NEW)
        h_cost.set_data(iter_history, cost_history)
        ax3.relim()
        ax3.autoscale_view()

        fig.canvas.draw_idle()
        plt.pause(0.1)

        if state['converged']:
            print(">>> Converged!")
            break

    plt.ioff()
    plt.show()

def _generate_contour_background(ax, model, y_target, N_modes):
    """
    Generates a 2D slice of the cost landscape (varying a1, a2).
    Keeps a3, a4... fixed at 0.0 for the background visualization.
    """
    a1_range = np.arange(-2.0, 3.0, 0.1)
    a2_range = np.arange(-2.0, 3.0, 0.1)
    A1, A2 = np.meshgrid(a1_range, a2_range)
    J_grid = np.zeros_like(A1)

    for i in range(A1.size):
        a_vec = np.zeros(N_modes)
        a_vec[0] = A1.flat[i]
        a_vec[1] = A2.flat[i]
        
        y_vec = model.forward(a_vec)
        misfit = y_vec - y_target
        J_grid.flat[i] = 0.5 * (misfit @ misfit)
        
    cf = ax.contourf(A1, A2, J_grid, 30, cmap='viridis', alpha=0.6)
    plt.colorbar(cf, ax=ax, label='$J_{slice}$')
    ax.plot(0, 0, 'w+', markersize=10, alpha=0.5)

if __name__ == "__main__":
    main()