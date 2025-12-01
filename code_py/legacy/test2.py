import numpy as np
import matplotlib
# matplotlib.use('TkAgg') # Uncomment if you have display issues
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Import tools from your module
from Core.ensemble import ensemble, cost

def nonlinear_obs(Phi, a):
    """
    Nonlinear observation operator:
        y = Phi @ (a + 0.1 * a^2)
    works for arbitrary N_modes = len(a).
    """
    a = np.asarray(a, dtype=float)
    a_eff = a + 0.1 * (a**2)
    return Phi @ a_eff

def main():
    # 1. Problem setup
    Nen       = 7        # number of ensemble members
    N_modes   = 7       # number of DoFs (easily change this)
    lam       = 0.0      # regularisation parameter (global J_true)
    N_t       = 200      # number of time samples

    # Time grid
    t = np.linspace(0.0, 2.0 * np.pi, N_t).reshape(-1, 1)

    # Fourier basis Phi_{i,n} = sin(n t_i)
    Phi = np.zeros((N_t, N_modes))
    for n in range(1, N_modes + 1):
        Phi[:, n - 1] = np.sin(n * t[:, 0])

    # Target coefficients (generalised to N_modes >= 3)
    a_target = np.zeros(N_modes)
    a_target[0:7] = [-0.4, 1.5, 0.0, 0.5, 11, 5, -2]

    # Nonlinear target observation
    y_target = nonlinear_obs(Phi, a_target)

    # Initial guess (also generalised to N_modes >= 3)
    a_e = np.zeros(N_modes, dtype=float)
    a_e[0:3] = [2.0, 4.0, 0.3]

    # Storage
    history_a = [a_e.copy()]
    history_error = []

    print("Fourier EnVar toy (Python) with nonlinear observation:")
    print(f"Initial a = {a_e}")

    # 2. Visualisation Setup (J_true contour in (a1, a2), other modes = 0)
    a1_range = np.arange(0.0, 5, 0.03)
    a2_range = np.arange(-5, 3, 0.03)
    A1, A2 = np.meshgrid(a1_range, a2_range)
    J_grid = np.zeros_like(A1)

    for i in range(A1.size):
        # Full vector of length N_modes, only first two vary, others zero
        a_vec = np.zeros(N_modes)
        a_vec[0] = A1.flat[i]
        a_vec[1] = A2.flat[i]

        y_vec = nonlinear_obs(Phi, a_vec)
        misfit = y_vec - y_target
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
    ax1.set_title(r'EnVar Cost: $J_{true}(a_1, a_2; a_{j>2}=0)$')
    ax1.set_xlabel(r'$a_1$')
    ax1.set_ylabel(r'$a_2$')

    ax1.plot(a_target[0], a_target[1], 'ys', markersize=10, label='Target')
    h_mean, = ax1.plot([a_e[0]], [a_e[1]], 'ro', label='Mean')
    h_ens,  = ax1.plot([], [], 'b.', markersize=8, label='Ensemble')
    h_traj, = ax1.plot([a_e[0]], [a_e[1]], 'k-', linewidth=1.5, label='Trajectory')
    ax1.legend()

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(t, y_target, 'k-', linewidth=1.5, label='Target')
    h_curr, = ax2.plot(t, nonlinear_obs(Phi, a_e), 'r--', label='Current')
    ax2.legend()

    plt.ion()
    plt.show()

    # 3. Optimisation Loop
    maxIter = 20
    alpha = 0.5
    rng = np.random.default_rng(42)

    for iter_ in range(1, maxIter + 1):

        # A. Generate Ensemble (dimension automatically = N_modes)
        a_r_ensemble = ensemble(Nen, a_e, rng=rng, save_file="a_r_ensemble.txt")

        # Deviation matrix in parameter space
        E_prime = a_r_ensemble - a_e[:, None]

        # B. Observations at mean and ensemble (nonlinear)
        y_e = nonlinear_obs(Phi, a_e)
        o_e = y_e - y_target  # Mismatch vector (size N_t)

        o_r_mat = np.zeros((N_t, Nen))
        for k in range(Nen):
            y_k = nonlinear_obs(Phi, a_r_ensemble[:, k])
            o_r_mat[:, k] = y_k - y_target

        # H maps weights w -> observation increments
        H = o_r_mat - o_e[:, None]

        # C. Minimise Local Cost J(w) using imported cost function
        def objective_wrapper(w):
            J_val, dJ, _ = cost(o_e, H, w, a_e, E_prime, 0)
            return J_val

        def gradient_wrapper(w):
            _, dJ, _ = cost(o_e, H, w, a_e, E_prime, 0)
            return dJ

        w0 = np.zeros(Nen)

        # Keep the minimizer setup unchanged
        res = minimize(
            objective_wrapper,
            w0,
            method='trust-constr',
            jac='3-point',
            options={'disp': None, 'xtol': 1e-12}
        )

        w_opt = res.x
        J_loc = res.fun
        J_val, dJ, _ = cost(o_e, H, w_opt, a_e, E_prime, lam)



        # D. Update Mean in parameter space
        step = E_prime @ w_opt
        a_e = a_e + alpha * step
        y_e = nonlinear_obs(Phi, a_e)

        # Logging & Convergence
        history_a.append(a_e.copy())
        misfit = (y_e - y_target)
        error_obs = np.linalg.norm(misfit)
        print(f"misfit = {misfit}")
        J_true = 0.5 * misfit.T @ misfit
        history_error.append(error_obs)

        # E. Update Plots
        h_mean.set_data([a_e[0]], [a_e[1]])
        h_ens.set_data(a_r_ensemble[0, :], a_r_ensemble[1, :])

        hist_arr = np.array(history_a)
        h_traj.set_data(hist_arr[:, 0], hist_arr[:, 1])
        h_curr.set_ydata(nonlinear_obs(Phi, a_e))

        fig1.canvas.draw_idle()
        fig2.canvas.draw_idle()
        plt.pause(0.01)

        print(f"Iter {iter_} | J_loc={J_loc:.4e} | J_true={J_true:.4e} | err={error_obs:.4e}")
        if error_obs < 1e-3:
            print("Converged.")
            break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()