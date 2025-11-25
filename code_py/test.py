import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ----------------------------------------------------------------------
# Ensemble generation (Python version of Ensemble.m)
# ----------------------------------------------------------------------
def ensemble(Nen, a_e, Y=1000, variance=0.005, rng=None):
    """
    Generate an ensemble around mean vector a_e with prescribed covariance
    (port of your Ensemble.m, adapted for coefficient vector a_e).

    Parameters
    ----------
    Nen : int
        Number of ensemble members.
    a_e : array_like, shape (N_dofs,)
        Mean coefficients.
    Y : int
        Oversampling factor for initial random set.
    variance : float
        Target variance (diagonal in "energy" space).
    rng : np.random.Generator or None
        Random generator for reproducibility.

    Returns
    -------
    a_r_ensemble : ndarray, shape (N_dofs, Nen)
        Ensemble of coefficient vectors.
    """
    a_e = np.asarray(a_e, dtype=float).reshape(-1)
    N_dofs = a_e.size

    # Covariance with zero row sum
    A = np.full((N_dofs, N_dofs), -1.0 / (N_dofs - 1.0))
    np.fill_diagonal(A, 1.0)
    covar = A.copy()

    # Augment and SVD to ensure positive-definite
    C = np.zeros((N_dofs + 1, N_dofs + 1))
    C[0, 0] = 1.0
    C[1:, 1:] = covar

    Uc, s_c, Vtc = np.linalg.svd(C, full_matrices=True)
    U1 = Uc[:, :-1]
    s1 = s_c[:-1]
    V1 = Vtc[:-1, :]
    SS = U1 @ (np.diag(s1) @ V1)
    covar = SS[:-1, :-1]

    # Cholesky
    R = np.linalg.cholesky(covar)
    Lc = R.T

    if rng is None:
        rng = np.random.default_rng()

    n_samples = Y * Nen

    # Standardised random vectors with zero mean, unit std
    y_hat = rng.random((N_dofs, n_samples)) * 2.0 - 1.0
    y_hat = y_hat - y_hat.mean(axis=0, keepdims=True)
    std = y_hat.std(axis=0, ddof=1, keepdims=True)
    std[std == 0.0] = 1.0
    y_hat = y_hat / std

    # Impose covariance
    P_hat = Lc @ y_hat

    # Extract Nen directions via SVD
    Uh, s_h, Vth = np.linalg.svd(P_hat, full_matrices=True)
    U = Uh[:, :Nen]
    S = np.diag(s_h[:Nen])
    V = Vth.T[:Nen, :Nen]
    P = U @ ((1.0 / math.sqrt(Y)) * (S @ V.T))

    # Work in "energy" variables e = a^2
    e_e = a_e ** 2
    S_var = np.eye(N_dofs) * variance

    a_r_ensemble = np.zeros((N_dofs, Nen))
    neg_mask = a_e < 0.0

    for i in range(Nen):
        e_r = S_var @ P[:, i] + e_e
        # Reflect if negative
        if np.any(e_r <= 0.0):
            e_r = -S_var @ P[:, i] + e_e
        a_r = np.sqrt(np.abs(e_r))
        a_r[neg_mask] *= -1.0
        a_r_ensemble[:, i] = a_r

    return a_r_ensemble


# ----------------------------------------------------------------------
# Local cost in weight space J(w)
# J(w) = 0.5 || o_e + H w ||^2 + 0.5 * lambda * || L (a_e + E_prime w) ||^2
# ----------------------------------------------------------------------
def local_cost(w, o_e, H, a_e, E_prime, lam, L):
    w = np.asarray(w, dtype=float).reshape(-1)
    o_e = np.asarray(o_e, dtype=float).reshape(-1)   # N_t
    H = np.asarray(H, dtype=float)                   # N_t x Nen
    a_e = np.asarray(a_e, dtype=float).reshape(-1)   # N_modes
    E_prime = np.asarray(E_prime, dtype=float)       # N_modes x Nen
    L = np.asarray(L, dtype=float)                   # N_modes x N_modes

    resid = o_e + H @ w
    J = 0.5 * float(resid @ resid)

    if lam != 0.0:
        a_new = a_e + E_prime @ w
        reg = L @ a_new
        J += 0.5 * lam * float(reg @ reg)

    return J


# ----------------------------------------------------------------------
# Main script: Fourier EnVar toy
# ----------------------------------------------------------------------
def main():
    # 1. Problem setup
    Nen       = 3          # Number of ensemble members
    N_modes   = 3          # Number of Fourier modes (controls)
    lambda_   = 0.0        # Regularisation strength (0 = off)
    N_t       = 200        # Number of time samples in [0, 2*pi]

    # Time grid
    t = np.linspace(0.0, 2.0 * np.pi, N_t).reshape(-1, 1)   # (N_t x 1)

    # Fourier basis: phi_n(t) = sin(n t), n = 1...N_modes
    Phi = np.zeros((N_t, N_modes))
    for n in range(1, N_modes + 1):
        Phi[:, n - 1] = np.sin(n * t[:, 0])

    # Target coefficients
    a_target = np.zeros((N_modes, 1))
    a_target[0, 0] = 2.4
    a_target[1, 0] = -1.1   # (you labelled "pure sin(2t)", but this is generic)

    y_target = Phi @ a_target   # N_t x 1

    # Initial guess (deliberately wrong)
    a_e = np.array([0.8, -0.5, 0.3], dtype=float)

    # Storage
    history_a = [a_e.copy()]
    history_error = []

    print("Fourier EnVar toy:")
    print("y(t; a) = sum_n a_n sin(n t), target = Phi * a_target")
    print("J_true(a) = 0.5 * ||Phi a - Phi a_target||^2 + 0.5*lambda*||L a||^2")
    print(f"Initial a = [{a_e[0]:.2f} {a_e[1]:.2f} {a_e[2]:.2f}]^T\n")

    # 2. Precompute true cost J_true on (a1,a2) grid for visualisation (a3=0)
    a1_range = np.arange(0.0, 4.0 + 1e-12, 0.03)
    a2_range = np.arange(-1.5, 1.5 + 1e-12, 0.03)
    A1, A2 = np.meshgrid(a1_range, a2_range)

    J_grid = np.zeros_like(A1)
    # Diagonal regularisation operator L (same as in J_true)
    L = np.diag(np.arange(1, N_modes + 1))

    for i in range(A1.size):
        a_vec = np.array([A1.flat[i], A2.flat[i], 0.0])
        y = Phi @ a_vec
        misfit = y.reshape(-1, 1) - y_target
        J_true_val = 0.5 * float(misfit.T @ misfit)
        if lambda_ > 0:
            J_true_val += 0.5 * lambda_ * np.linalg.norm(L @ a_vec) ** 2
        J_grid.flat[i] = J_true_val

    # 3. Visualisation: contour of J_true(a1,a2) and trajectory
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    cf = ax1.contourf(A1, A2, J_grid, 40)
    fig1.colorbar(cf, ax=ax1)
    ax1.set_title(r'EnVar on Fourier coefficients: $J_{\mathrm{true}}(a_1,a_2; a_3=0)$')
    ax1.set_xlabel(r'$a_1$ (sin t)')
    ax1.set_ylabel(r'$a_2$ (sin 2t)')
    ax1.axis('equal')
    ax1.axis('tight')
    ax1.grid(True)

    # Target point
    ax1.plot(a_target[0, 0], a_target[1, 0], 'ys',
             markersize=10, markerfacecolor='y', label='Target $a^*$')

    # Handles for mean, ensemble, trajectory
    line_mean, = ax1.plot(a_e[0], a_e[1], 'ro',
                          markerfacecolor='r', markersize=8, label='Mean')
    line_ens, = ax1.plot([], [], 'b.', markersize=10, label='Ensemble')
    # Trajectory line; initial point only
    hist_arr = np.column_stack(history_a)
    line_traj, = ax1.plot(hist_arr[0, :], hist_arr[1, :],
                          'k-', linewidth=1.5, label='Trajectory')
    ax1.legend(loc='best')

    # 4. Second figure: signals (target vs current)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    line_target, = ax2.plot(t[:, 0], y_target[:, 0], 'k-',
                            linewidth=1.5, label=r'Target $y^*(t)$')
    y_init = Phi @ a_e
    line_curr, = ax2.plot(t[:, 0], y_init, 'r--',
                          linewidth=1.5, label=r'Current $y(t;a)$')
    ax2.set_xlabel('t')
    ax2.set_ylabel('y')
    ax2.set_title('Signal reconstruction: target vs current')
    ax2.grid(True)
    ax2.legend(loc='best')

    plt.ion()  # interactive updates

    # 5. EnVar optimisation loop
    maxIter = 100
    alpha = 2.0   # relaxation factor

    rng = np.random.default_rng()

    for iter_ in range(1, maxIter + 1):

        # A. Generate ensemble in coefficient space
        a_r_ensemble = ensemble(Nen, a_e, rng=rng)      # N_modes x Nen
        E_prime = a_r_ensemble - a_e.reshape(-1, 1)     # N_modes x Nen

        # B. Build observations (signal mismatch)
        y_e = Phi @ a_e
        o_e = y_e.reshape(-1, 1) - y_target            # N_t x 1

        o_r_mat = np.zeros((N_t, Nen))
        for k in range(Nen):
            a_k = a_r_ensemble[:, k]
            y_k = Phi @ a_k
            o_r_mat[:, k] = (y_k.reshape(-1, 1) - y_target)[:, 0]

        # Observation perturbation matrix H (N_t x Nen)
        H = o_r_mat - o_e

        # C. Local cost in weight space, solve for w
        def fun(w):
            return local_cost(w, o_e[:, 0], H, a_e, E_prime, lambda_, L)

        w0 = np.zeros(Nen)
        bounds = [(-np.inf, np.inf)] * Nen

        res = minimize(fun, w0, method='SLSQP', bounds=bounds,
                       options={'disp': False, 'maxiter': 100})
        w_opt = res.x
        J_loc = res.fun

        # D. Update mean in coefficient space
        step_vec = E_prime @ w_opt
        a_e = a_e + alpha * step_vec

        # Store trajectory
        history_a.append(a_e.copy())

        # Compute true cost and error in observation space
        y_e = Phi @ a_e
        misfit = y_e.reshape(-1, 1) - y_target
        J_true = 0.5 * float(misfit.T @ misfit)
        if lambda_ > 0:
            J_true += 0.5 * lambda_ * np.linalg.norm(L @ a_e) ** 2
        err_obs = np.linalg.norm(misfit)
        history_error.append(err_obs)

        # E. Update plots
        # Coefficient space
        line_ens.set_data(a_r_ensemble[0, :], a_r_ensemble[1, :])
        line_mean.set_data(a_e[0], a_e[1])

        hist_arr = np.column_stack(history_a)
        line_traj.set_data(hist_arr[0, :], hist_arr[1, :])

        # Signal plot
        line_curr.set_ydata(y_e)

        fig1.canvas.draw_idle()
        fig2.canvas.draw_idle()
        plt.pause(0.01)

        # F. Logging
        print(
            f"Iter {iter_:2d} | J_loc(w)={J_loc:.3e} | "
            f"J_true(a)={J_true:.3e} | ||Phi a - y^*||={err_obs:.3e} | "
            f"a=[{a_e[0]:.3f} {a_e[1]:.3f} {a_e[2]:.3f}]^T"
        )

        if err_obs < 1e-6:
            print('Converged: ||Phi a - y^*|| < 1e-6.')
            break

    # 6. Error convergence figure
    fig3, ax3 = plt.subplots()
    ax3.plot(history_error, '-o', linewidth=1.5)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel(r'$||\Phi a - y^*||_2$')
    ax3.set_title('EnVar convergence (signal misfit)')
    ax3.grid(True)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()