import numpy as np
import matplotlib
# matplotlib.use('TkAgg') # Uncomment if you have display issues
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Import tools from your module
from Core.ensemble import ensemble, cost

def main():
    # 1. Problem setup
    Nen       = 3       
    N_modes   = 3          
    lam       = 0.0        
    N_t       = 200    
    # Time grid
    t = np.linspace(0.0, 2.0 * np.pi, N_t).reshape(-1, 1)

    # Fourier basis
    Phi = np.zeros((N_t, N_modes))
    for n in range(1, N_modes + 1):
        Phi[:, n - 1] = np.sin(n * t[:, 0])

    # Target
    a_target = np.array([-0.4, 1.5, 0])
    y_target = Phi @ a_target
    
    # Initial guess
    a_e = np.array([2, 4, 0.3], dtype=float)

    # Storage
    history_a = [a_e.copy()]
    history_error = []

    print("Fourier EnVar toy (Python):")
    print(f"Initial a = {a_e}")

    # 2. Visualisation Setup (J_true contour)
    a1_range = np.arange(0.0, 5, 0.03)
    a2_range = np.arange(-5, 3, 0.03)
    A1, A2 = np.meshgrid(a1_range, a2_range)
    J_grid = np.zeros_like(A1)
    
    # Pre-compute grid for visualization
    for i in range(A1.size):
        a_vec = np.array([A1.flat[i], A2.flat[i], 0.0])
        # Direct calculation for visualization speed
        misfit = (Phi @ a_vec) - y_target
        J_val = 0.5 * (misfit @ misfit)
        if lam > 0:
            idx = np.arange(1, N_modes + 1)
            reg = idx * a_vec
            J_val += 0.5 * lam * (reg @ reg)
        J_grid.flat[i] = J_val

    # Plot Setup
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    cf = ax1.contourf(A1, A2, J_grid, 40)
    plt.colorbar(cf, ax=ax1)
    ax1.set_title(r'EnVar Cost: $J_{true}(a_1, a_2; a_3=0)$')
    ax1.set_xlabel(r'$a_1$')
    ax1.set_ylabel(r'$a_2$')
    
    ax1.plot(a_target[0], a_target[1], 'ys', markersize=10, label='Target')
    h_mean, = ax1.plot([a_e[0]], [a_e[1]], 'ro', label='Mean')
    h_ens,  = ax1.plot([], [], 'b.', markersize=8, label='Ensemble')
    h_traj, = ax1.plot([a_e[0]], [a_e[1]], 'k-', linewidth=1.5, label='Trajectory')
    ax1.legend()

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(t, y_target, 'k-', linewidth=1.5, label='Target')
    h_curr, = ax2.plot(t, Phi @ a_e, 'r--', label='Current')
    ax2.legend()
    
    plt.ion()
    plt.show()

    # 3. Optimisation Loop
    maxIter = 20
    alpha = 0.5
    rng = np.random.default_rng(42)
    for iter_ in range(1, maxIter + 1):
        
        # A. Generate Ensemble
        a_r_ensemble = ensemble(Nen, a_e, rng=rng,save_file="a_r_ensemble.txt")        
        # Deviation matrix
        E_prime = a_r_ensemble - a_e[:, None]
        # B. Observations
        y_e = Phi @ a_e
        o_e = y_e - y_target  # Mismatch vector (size N_t)

        # Observation perturbations H
        o_r_mat = np.zeros((N_t, Nen))
        for k in range(Nen):
            y_k = Phi @ a_r_ensemble[:, k]
            o_r_mat[:, k] = y_k - y_target
        
        # H maps weights w -> observation increments
        H = o_r_mat - o_e[:, None]

        # C. Minimise Local Cost J(w) using imported cost function
        def objective_wrapper(w):
            # Call the shared cost function from ensemble.py
            # Note: We ignore dJ and Hess here as you requested
            #print(f"w={w}")
            J_val, dJ, _ = cost(o_e, H, w, a_e, E_prime, 0)
            return J_val
        def gradient_wrapper(w):
            # Call the shared cost function from ensemble.py
            # Note: We ignore dJ and Hess here as you requested
            #print(f"w={w}")
            _, dJ, _ = cost(o_e, H, w, a_e, E_prime, 0)
            return dJ

        w0 = np.zeros(Nen)
        
        # jac=False: scipy will estimate gradients numerically
        res = minimize(objective_wrapper, w0, method='trust-constr', jac='3-point',
                       options={'disp' : True, 'xtol': 1e-12 })
        
        w_opt = res.x
        J_loc = res.fun
        print(res)
        J_val, dJ, _ = cost(o_e, H, w_opt, a_e, E_prime, lam)
        print("dJ=",dJ)
        print("J=",J_val)

        s = np.linalg.svd(H, compute_uv=False)
        print("singular values of H:", s)
        w_ls, *_ = np.linalg.lstsq(H, -o_e, rcond=None)
        r_star = o_e + H @ w_ls
        print("||o_e|| =", np.linalg.norm(o_e))
        print("||residual after best w|| =", np.linalg.norm(r_star))

        # D. Update Mean
        step = E_prime @ w_opt
        a_e = a_e + alpha * step
        y_e = Phi @ a_e
        # Logging & Convergence
        history_a.append(a_e.copy())
        misfit = (y_e - y_target)
        error_obs = np.linalg.norm(misfit)
        J_true = 0.5 * misfit.T @ misfit
        history_error.append(error_obs)
        
        # E. Update Plots
        h_mean.set_data([a_e[0]], [a_e[1]])
        h_ens.set_data(a_r_ensemble[0, :], a_r_ensemble[1, :])
        
        hist_arr = np.array(history_a)
        h_traj.set_data(hist_arr[:, 0], hist_arr[:, 1])
        h_curr.set_ydata(Phi @ a_e)
        fig1.canvas.draw_idle()
        fig2.canvas.draw_idle()
        plt.pause(0.01)
        
        print(f"Iter {iter_} | J_loc={J_loc:.4e} | J_true={J_true:.4e} | err={error_obs:.4e}")
        if error_obs < 1e-6:
            print("Converged.")
            break
       

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()