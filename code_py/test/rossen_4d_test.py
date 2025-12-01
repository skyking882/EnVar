import numpy as np
import matplotlib.pyplot as plt
from rossen_4d_model import Rosenbrock4DModel
from Core.envar_solver import EnVarSolver

def main():
    # --- 1. Problem Setup ---
    N_modes = 4
    model = Rosenbrock4DModel()
    
    # Target: Flat line at 1.0
    a_target = np.ones(N_modes)
    y_target = np.zeros(2 * (N_modes - 1)) # Residuals = 0 at solution

    # Initial Guess: Difficult Zig-Zag
    # x1 is far negative (-1.2), others are scattered
    a_init = np.array([-1.2, 1.0, -0.5, 0.5])

    print(f"--- 4D Rosenbrock Benchmark ---")
    print(f"Target: {a_target}")
    print(f"Initial: {a_init}")

    # --- 2. Solver Config ---
    solver_config = {
        'Nen': 10,       
        'lam': 0.0,
        'max_iter': 100,
        'alpha': 1.0,    
        'tol': 1e-4,
        'variance' : 0.1
    }
    solver = EnVarSolver(model, a_init, y_target, config=solver_config)

    # --- 3. Visualization Setup ---
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Parameter Evolution (Time Series)
    # We create 4 lines, one for each dimension
    colors = ['r', 'g', 'b', 'm']
    labels = [f'$x_{i}$' for i in range(1, 5)]
    lines = []
    
    # Initialize empty lines
    for i in range(N_modes):
        ln, = ax1.plot([], [], color=colors[i], linewidth=2, label=labels[i])
        lines.append(ln)
        
    ax1.axhline(1.0, color='k', linestyle='--', label='Target (1.0)')
    ax1.set_title('Parameter Convergence History')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Parameter Value')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Current "Shape" (Profile)
    # Visualizes the vector [x1, x2, x3, x4] as a connected line/bar
    indices = np.arange(1, 5)
    h_profile, = ax2.plot(indices, a_init, 'o-', linewidth=2, color='k', label='Current State')
    h_target_prof, = ax2.plot(indices, a_target, 's--', color='gray', alpha=0.5, label='Target')
    
    # Ensemble Cloud (Vertical bars at each index)
    h_ens_cloud, = ax2.plot([], [], 'b.', alpha=0.3, label='Ensemble')

    ax2.set_title('Current State Vector')
    ax2.set_xlabel('Dimension Index')
    ax2.set_ylabel('Value')
    ax2.set_ylim(-2.0, 2.0)
    ax2.set_xticks(indices)
    ax2.legend()
    ax2.grid(True)

    # --- 4. Optimization Loop ---
    # History storage: Shape (Iteration, N_modes)
    traj_a = [a_init.copy()]
    iter_history = [0]
    
    for state in solver.solve():
        iter_ = state['iter']
        a_curr = state['a_mean']
        a_ens = state['ensemble']
        cost = state['J_true']

        print(f"Iter {iter_:02d} | Cost={cost:.4e} | State={np.round(a_curr, 2)}")

        # Update History
        traj_a.append(a_curr.copy())
        iter_history.append(iter_)
        history_arr = np.array(traj_a) # Shape: (N_iters, 4)

        # Update Plot 1: Time Series
        # Update each line individually
        for i in range(N_modes):
            lines[i].set_data(iter_history, history_arr[:, i])
            
        ax1.relim()
        ax1.autoscale_view()

        # Update Plot 2: Current Profile
        h_profile.set_data(indices, a_curr)
        
        # Update Ensemble Cloud
        if a_ens is not None:
            # Flatten for plotting: X is repeated indices, Y is values
            # X: [1,1,1..., 2,2,2..., 3,3,3...]
            ens_x = np.repeat(indices, a_ens.shape[1])
            ens_y = a_ens.ravel() # Flatten row-major
            h_ens_cloud.set_data(ens_x, ens_y)

        fig.canvas.draw_idle()
        plt.pause(0.1)

        if state['converged']:
            print(">>> Converged!")
            break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()