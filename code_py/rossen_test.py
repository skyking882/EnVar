import numpy as np
import matplotlib.pyplot as plt
from rossenbrock_model import RosenbrockModel  # Adjusted import
# from envar_solver import EnVarSolver # Assuming this exists in your directory

# Mocking EnVarSolver for demonstration if you don't have the file handy
# (Delete this block if you are running with your actual solver)
# ------------------------------------------------------------------
class EnVarSolver:
    def __init__(self, model, a_init, y_target, config):
        self.model = model
        self.a = a_init.copy()
        self.y_target = y_target
        self.cov = np.eye(len(a_init)) * 0.1
        self.config = config
    
    def solve(self):
        # A simple Gradient Descent with momentum mock to simulate solver behavior
        velocity = np.zeros_like(self.a)
        for i in range(self.config['max_iter']):
            # Finite difference gradient
            eps = 1e-4
            J0 = np.sum((self.model.forward(self.a) - self.y_target)**2)
            grad = np.zeros_like(self.a)
            for k in range(len(self.a)):
                a_pert = self.a.copy(); a_pert[k] += eps
                J_pert = np.sum((self.model.forward(a_pert) - self.y_target)**2)
                grad[k] = (J_pert - J0)/eps
            
            velocity = 0.9 * velocity - 0.001 * grad
            self.a += velocity
            
            # Create Ensemble for visualization (fake cloud)
            ens = self.a[:, None] + np.random.randn(2, 10) * 0.05
            
            yield {
                'iter': i,
                'a_mean': self.a,
                'ensemble': ens,
                'J_loc': J0,
                'J_true': J0,
                'error': J0,
                'converged': J0 < 1e-6,
                'y_current': self.model.forward(self.a)
            }
# ------------------------------------------------------------------


def main():
    # --- 1. Problem Setup ---
    N_modes = 2  # Rosenbrock is 2D
    
    # Initialize physical model
    model = RosenbrockModel(N_modes=N_modes)
    
    # Set ground truth (Target)
    # The Global Minimum of Rosenbrock is at (1, 1). 
    a_target_ref = np.array([1.0, 1.0]) 
    y_target = np.zeros(2) #residual=0

    # Set initial guess
    a_init = np.array([-1.2, -1.0])

    print("--- Rosenbrock Optimization Benchmark ---")
    print(f"Global Min Location: {a_target_ref}")
    print(f"Initial Guess: {a_init}")

    # --- 2. Initialize Solver ---
    solver_config = {
        'Nen': 30,       # Ensemble size
        'lam': 0.0,      # Regularization
        'max_iter': 250,
        'alpha': 1.0
    }
    solver = EnVarSolver(model, a_init, y_target, config=solver_config)

    # --- 3. Visualization Setup ---
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Parameter space (The Banana Valley)
    _generate_contour_background(ax1, model, y_target)
    
    h_target, = ax1.plot(a_target_ref[0], a_target_ref[1], 'y*', markersize=15, label='Global Min (1,1)', zorder=5)
    h_mean, = ax1.plot([a_init[0]], [a_init[1]], 'ro', label='Current Mean', zorder=4)
    h_ens, = ax1.plot([], [], 'b.', markersize=3, label='Ensemble', zorder=3)
    h_traj, = ax1.plot([], [], 'k-', linewidth=1.5, label='Trajectory')
    
    ax1.set_title('Rosenbrock Landscape ($x, y$)')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.legend(loc='upper left')

    # Subplot 2: Convergence History (Instead of Time Series)
    h_cost_line, = ax2.semilogy([], [], 'r-', linewidth=2)
    ax2.set_title('Cost Function Convergence')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost $J$ (Log Scale)')
    ax2.grid(True, which="both", ls="-")

    # --- 4. Run Optimization Loop ---
    traj_a = [a_init.copy()]
    cost_history = []
    iter_history = []

    for state in solver.solve():
        iter_ = state['iter']
        a_curr = state['a_mean']
        a_ens = state['ensemble']
        cost = state['J_true']
        
        # Logging
        if iter_ % 5 == 0:
            print(f"Iter {iter_:03d} | Cost={cost:.4e} | Loc=({a_curr[0]:.2f}, {a_curr[1]:.2f})")

        # Update History
        traj_a.append(a_curr.copy())
        traj_arr = np.array(traj_a)
        cost_history.append(cost)
        iter_history.append(iter_)

        # Update Plots
        # 1. Landscape
        h_mean.set_data([a_curr[0]], [a_curr[1]])
        if a_ens is not None:
            h_ens.set_data(a_ens[0, :], a_ens[1, :]) 
        h_traj.set_data(traj_arr[:, 0], traj_arr[:, 1])
        
        # 2. Convergence
        h_cost_line.set_data(iter_history, cost_history)
        ax2.relim()
        ax2.autoscale_view()

        fig.canvas.draw_idle()
        plt.pause(0.01)

        if state['converged']:
            print(">>> Converged!")
            break

    plt.ioff()
    plt.show()

def _generate_contour_background(ax, model, y_target):
    """
    Generates the classic Rosenbrock 'Banana' contour.
    """
    # Standard Rosenbrock view range
    x_range = np.linspace(-2.0, 2.0, 100)
    y_range = np.linspace(-1.0, 3.0, 100)
    X, Y = np.meshgrid(x_range, y_range)
    J_grid = np.zeros_like(X)

    for i in range(X.size):
        a_vec = np.array([X.flat[i], Y.flat[i]])
        obs_vec = model.forward(a_vec)
        
        # Compute Sum of Squares
        misfit = obs_vec - y_target
        J_grid.flat[i] = np.sum(misfit**2)
        
    # Log scale contour often helps visualize the valley floor better
    # But linear is fine for the general shape. Using LogNorm is advanced, 
    # lets stick to levels that highlight the valley.
    import matplotlib.colors as colors
    
    cf = ax.contourf(X, Y, J_grid, 
                     levels=np.logspace(-1, 3.5, 20), # Log-spaced levels
                     norm=colors.LogNorm(), 
                     cmap='viridis', alpha=0.6)
    plt.colorbar(cf, ax=ax, label='Cost $J$ (Log Scale)')

if __name__ == "__main__":
    main()