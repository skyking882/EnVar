import numpy as np
import matplotlib.pyplot as plt

# --- IMPORT YOUR EXISTING MODULES ---
from rossenbrock_model import RosenbrockModel
from envar_solver import EnVarSolver

def main():
    # ==========================================
    # 1. SETUP PROBLEM (Rosenbrock Benchmark)
    # ==========================================
    N_modes = 2
    model = RosenbrockModel(N_modes=N_modes)

    # Target: The global minimum is at [1.0, 1.0]
    # For Rosenbrock, the residual at the minimum is 0.
    y_target = np.zeros(2) 

    # Initial Guess: Start far away to test convergence
    a_init = np.array([-1.2, -1.0])

    print(f"--- EnVar Execution Skeleton ---")
    print(f"Model: Rosenbrock (N={N_modes})")
    print(f"Target Residual: 0.0")
    print(f"Initial Guess:   {a_init}")

    # ==========================================
    # 2. CONFIGURE SOLVER
    # ==========================================
    # Note: We use the robust settings found in previous steps
    config = {
        'Nen': 20,           # Ensemble Size
        'max_iter': 100,     # Maximum Iterations
        'lam': 0.0,          # Regularization (None for this benchmark)
        'alpha': 1.0,        # Learning Rate/Step Size
        'variance': 0.1,     # Variance (0.1 is robust for Rosenbrock)
        'tol': 1e-4          # Convergence Tolerance
    }

    # Initialize the Solver Class (from envar_solver.py)
    solver = EnVarSolver(model, a_init, y_target, config=config)

    # ==========================================
    # 3. RUN OPTIMIZATION LOOP
    # ==========================================
    cost_history = []
    
    print("\nStarting Optimization...")
    print(f"{'Iter':<5} | {'Cost (J)':<12} | {'Params (Mean)'}")
    print("-" * 45)

    # The solver.solve() is a generator that yields the state dictionary
    for state in solver.solve():
        iter_ = state['iter']
        cost = state['J_true']
        a_curr = state['a_mean']
        
        # Store for plotting
        cost_history.append(cost)

        # Logging
        if iter_ % 5 == 0 or state['converged']:
            # Format array for clean printing
            p_str = np.array2string(a_curr, precision=3, separator=', ')
            print(f"{iter_:<5d} | {cost:.4e}   | {p_str}")

        if state['converged']:
            print(">>> Converged!")
            break

    # ==========================================
    # 4. PLOT CONVERGENCE
    # ==========================================
    plt.figure(figsize=(8, 6))
    
    # Plot Cost vs Iteration (Log Scale)
    plt.semilogy(cost_history, 'r-o', linewidth=2, markersize=5, label='J_true')
    
    plt.title('EnVar Convergence (Rosenbrock)')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function $J$ (Log Scale)')
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()