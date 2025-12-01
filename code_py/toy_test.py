import numpy as np
import matplotlib.pyplot as plt
from fourier_model import FourierNonlinearModel
from envar_solver import EnVarSolver

def main():
    # --- 1. Problem Setup ---
    N_modes = 7
    
    # Initialize physical model
    model = FourierNonlinearModel(N_modes=N_modes)
    
    # Set ground truth (Target)
    a_target = np.zeros(N_modes)
    a_target[0:7] = [-0.4, 1.5, 0.0, 0.5, 11, 5, -2]
    y_target = model.forward(a_target)

    # Set initial guess
    a_init = np.zeros(N_modes)
    a_init[0:3] = [2.0, 4.0, 0.3]

    print("--- Fourier EnVar Toy Problem ---")
    print(f"Target: {a_target[:3]}...")
    print(f"Initial: {a_init[:3]}...")

    # --- 2. Initialize Solver ---
    solver_config = {
        'Nen': 7,
        'lam': 0.0,
        'max_iter': 100,
        'alpha': 1
    }
    solver = EnVarSolver(model, a_init, y_target, config=solver_config)

    # --- 3. Visualization Setup (Matplotlib) ---
    plt.ion() # Enable interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Parameter space trajectory (a1, a2)
    _generate_contour_background(ax1, model, y_target, N_modes)
    
    h_target, = ax1.plot(a_target[0], a_target[1], 'ys', markersize=10, label='Target', zorder=5)
    h_mean, = ax1.plot([a_init[0]], [a_init[1]], 'ro', label='Current Mean', zorder=4)
    h_ens, = ax1.plot([], [], 'b.', markersize=5, label='Ensemble', zorder=3)
    h_traj, = ax1.plot([], [], 'k-', linewidth=1.5, label='Trajectory')
    ax1.set_title('Parameter Space ($a_1, a_2$)')
    ax1.set_xlabel('$a_1$')
    ax1.set_ylabel('$a_2$')
    ax1.legend()

    # Subplot 2: Observation Space (Time domain)
    t = model.get_time_grid()
    ax2.plot(t, y_target, 'k-', linewidth=2, label='Target Obs')
    h_obs_curr, = ax2.plot(t, model.forward(a_init), 'r--', label='Current Obs')
    ax2.set_title('Observation Space')
    ax2.legend()

    # --- 4. Run Optimization Loop ---
    traj_a = [a_init.copy()]

    for state in solver.solve():
        iter_ = state['iter']
        a_curr = state['a_mean']
        a_ens = state['ensemble']
        
        # Print log
        print(f"Iter {iter_:02d} | J_loc={state['J_loc']:.4e} | "
              f"J_true={state['J_true']:.4e} | err={state['error']:.4e}")

        # Update data
        traj_a.append(a_curr)
        traj_arr = np.array(traj_a)

        # Update plots
        h_mean.set_data([a_curr[0]], [a_curr[1]])
        h_ens.set_data(a_ens[0, :], a_ens[1, :]) 
        h_traj.set_data(traj_arr[:, 0], traj_arr[:, 1])
        h_obs_curr.set_ydata(state['y_current'])

        fig.canvas.draw_idle()
        plt.pause(0.1)
        print(f"control vector = {state['a_mean']}")

        if state['converged']:
            print(">>> Converged!")
            break

    plt.ioff()
    plt.show()

def _generate_contour_background(ax, model, y_target, N_modes):
    """
    Helper function: Only for generating background J_true contour (scans a1, a2)
    """
    a1_range = np.arange(0.0, 5, 0.05)
    a2_range = np.arange(-5, 3, 0.05)
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
    plt.colorbar(cf, ax=ax, label='$J_{true}$')

if __name__ == "__main__":
    main()