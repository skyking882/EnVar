import math
from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------
# obs_vec.m
# ----------------------------------------------------------------------
def obs_vec(q):
    """
    Python equivalent of MATLAB obs_vec(q).

    Parameters
    ----------
    q : array_like
        Input array (e.g. time history of drag).

    Returns
    -------
    o : ndarray
        Mean of q along axis 0 (column-wise mean in MATLAB).
    """
    q_arr = np.asarray(q)
    return q_arr.mean(axis=0)


# ----------------------------------------------------------------------
# Ensemble.m
# ----------------------------------------------------------------------
def ensemble(Nen, u_e, Y=1000, variance=0.005, rng=None):
    """
    Python equivalent of Ensemble.m.

    Parameters
    ----------
    Nen : int
        Number of ensemble members.
    u_e : array_like, shape (N_dofs,)
        Mean control vector (current design).
    Y : int, optional
        Oversampling factor used to generate Y*Nen provisional members.
    variance : float, optional
        Desired variance for each DOF in energy space (diagonal S matrix).
    rng : np.random.Generator, optional
        RNG to control reproducibility.

    Returns
    -------
    u_r_ensemble : ndarray, shape (N_dofs, Nen)
        Ensemble of control vectors (one column per member).
    """
    u_e = np.asarray(u_e, dtype=float).reshape(-1)
    N_dofs = u_e.size

    # Generate symmetric covariance matrix with zero row-sum
    A = np.full((N_dofs, N_dofs), -1.0 / (N_dofs - 1.0))
    np.fill_diagonal(A, 1.0)
    covar = A.copy()

    # Augmented covariance matrix
    C = np.zeros((N_dofs + 1, N_dofs + 1))
    C[0, 0] = 1.0
    C[1:, 1:] = covar

    # SVD of augmented covariance, enforce positive definiteness
    Uc, s_c, Vtc = np.linalg.svd(C, full_matrices=True)
    U1 = Uc[:, :-1]
    s1 = s_c[:-1]
    V1 = Vtc[:-1, :]
    SS = U1 @ (np.diag(s1) @ V1)
    covar = SS[:-1, :-1]

    # Cholesky factorisation of positive-definite covariance
    R = np.linalg.cholesky(covar)
    L = R.T

    if rng is None:
        rng = np.random.default_rng()

    n_samples = Y * Nen

    # Generate Y*Nen standardised random vectors with zero mean and unit std
    y_hat = rng.random((N_dofs, n_samples)) * 2.0 - 1.0
    y_hat = y_hat - y_hat.mean(axis=0, keepdims=True)
    std = y_hat.std(axis=0, ddof=1, keepdims=True)
    std[std == 0.0] = 1.0
    y_hat = y_hat / std

    # Impose the desired covariance
    P_hat = L @ y_hat  # (N_dofs, Y*Nen)

    # Extract Nen directions using SVD
    Uh, s_h, Vth = np.linalg.svd(P_hat, full_matrices=True)
    U = Uh[:, :Nen]                    # (N_dofs, Nen)
    S = np.diag(s_h[:Nen])             # (Nen, Nen)
    V = Vth.T[:Nen, :Nen]              # (Nen, Nen)
    P = U @ ((1.0 / math.sqrt(Y)) * (S @ V.T))  # (N_dofs, Nen)

    # Work in "energy" variable e = u^2
    e_e = u_e ** 2  # baseline energy

    S_var = np.eye(N_dofs) * variance

    u_r_ensemble = np.zeros((N_dofs, Nen))
    neg_mask = u_e < 0.0

    for i in range(Nen):
        e_r = S_var @ P[:, i] + e_e
        # If any component becomes negative, reflect the perturbation
        if np.any(e_r <= 0.0):
            e_r = -S_var @ P[:, i] + e_e

        u_r = np.sqrt(np.abs(e_r))
        # Restore sign pattern of u_e
        u_r[neg_mask] *= -1.0
        u_r_ensemble[:, i] = u_r

    return u_r_ensemble


# ----------------------------------------------------------------------
# Cost.m
# ----------------------------------------------------------------------
def cost(o_e, H, a, u_e, E_prime, lam):
    """
    Python equivalent of Cost.m.

    Cost:
        J = 1/2 * || o_e + H a ||^2
            + 1/2 * lam * || v ||^2
        with v_i = i * (u_e_i + (E_prime a)_i).

    Parameters
    ----------
    o_e : float or array_like
        Baseline observation (e.g. mean drag).
    H : array_like, shape (1, Nen) or (Nen,)
        Linear operator mapping weights to observation increments.
    a : array_like, shape (Nen,)
        Weight vector (optimisation variable).
    u_e : array_like, shape (N_dofs,)
        Mean control vector.
    E_prime : array_like, shape (N_dofs, Nen)
        Deviation matrix for ensemble controls.
    lam : float
        Regularisation parameter (lambda).

    Returns
    -------
    J : float
        Cost function value.
    dJ : ndarray, shape (Nen,)
        Gradient of J with respect to 'a'.
    Hessian : ndarray, shape (Nen, Nen)
        Hessian of J with respect to 'a'.
    """
    o_e = float(np.asarray(o_e))
    H = np.atleast_2d(np.asarray(H, dtype=float))
    if H.shape[0] != 1 and H.shape[1] == 1:
        H = H.T  # make it a row vector

    a = np.asarray(a, dtype=float).reshape(-1)
    u_e = np.asarray(u_e, dtype=float).reshape(-1)
    E_prime = np.asarray(E_prime, dtype=float)

    Nen = E_prime.shape[1]
    N_dofs = u_e.size
    assert a.size == Nen, "Length of 'a' must equal number of ensemble members"

    # Penalty vector v_i = i * (u_e_i + (E_prime a)_i)
    idx = np.arange(1, N_dofs + 1, dtype=float)
    Ea = E_prime @ a
    v = idx * (u_e + Ea)

    # Main quadratic term 1/2 || o_e + H a ||^2
    resid = o_e + float(H @ a)
    J_main = 0.5 * resid ** 2

    # Smoothness penalty 1/2 * lam * ||v||^2
    J_pen = 0.5 * lam * float(np.vdot(v, v))
    J = J_main + J_pen

    # Gradient
    dJ_main = (H.T * resid).reshape(-1)

    # grad penalty: lam * E_prime^T (idx * v)
    gv = idx * v
    dJ_pen = lam * (E_prime.T @ gv)
    dJ = dJ_main + dJ_pen

    # Hessian
    H_data = H.T @ H  # (Nen, Nen)
    W = (idx ** 2)
    H_pen = lam * (E_prime.T * W) @ E_prime
    Hessian = H_data + H_pen

    return J, dJ, Hessian


# ----------------------------------------------------------------------
# Constraints.m
# ----------------------------------------------------------------------
def constraints(u_e, E_prime, c, a, step_eps=0.075):
    """
    Python equivalent of Constraints.m.

    Inequalities C(a) <= 0:
      1) Length constraint (on even DOFs):
         u_even + E_even a - (L/2 - a0) <= 0
      2) Step-size constraint:
         |E_prime a| - step_eps <= 0

    Equality Ceq(a) = 0:
         ||u_e + E_prime a||^2 - c = 0

    Parameters
    ----------
    u_e : array_like, shape (N_dofs,)
        Mean control vector.
    E_prime : array_like, shape (N_dofs, Nen)
        Deviation matrix.
    c : float
        Target squared norm of the control vector.
    a : array_like, shape (Nen,)
        Weight vector.
    step_eps : float, optional
        Allowed magnitude for each step component |(E_prime a)_i|.

    Returns
    -------
    C : ndarray, shape (n_ineq,)
        Inequality constraints (C <= 0).
    Ceq : float
        Equality constraint (Ceq = 0).
    gC : ndarray, shape (Nen, n_ineq)
        Gradient of C wrt a (columns correspond to constraints).
    gCeq : ndarray, shape (Nen,)
        Gradient of Ceq wrt a.
    """
    u_e = np.asarray(u_e, dtype=float).reshape(-1)
    E_prime = np.asarray(E_prime, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1)

    N_dofs, Nen = E_prime.shape

    # MATLAB indices = 1:length(u_e); even indices -> Python's 1,3,5,...
    idx_all = np.arange(N_dofs)
    idx_even = idx_all[1::2]

    E_even = E_prime[idx_even, :]
    u_even = u_e[idx_even]

    L = 3.0
    a0 = 0.5

    # 1) Length constraint
    C_length = u_even + E_even @ a - (L / 2.0 - a0)

    # 2) Step-size constraint
    step = E_prime @ a
    C_step = np.abs(step) - step_eps

    C = np.concatenate([C_length, C_step])

    # Equality: norm-squared of updated control equals c
    ua = u_e + E_prime @ a
    Ceq = float(np.vdot(ua, ua) - c)

    # Gradients
    # dC_length/da = E_even^T
    gC_length = E_even.T  # (Nen, n_even)

    # dC_step/da = diag(sign(step)) E_prime
    sign = np.sign(step)
    gC_step = (E_prime.T * sign)  # (Nen, N_dofs)

    gC = np.concatenate([gC_length, gC_step], axis=1)

    # dCeq/da = 2 E_prime^T (u_e + E_prime a)
    gCeq = 2.0 * (E_prime.T @ ua)

    return C, Ceq, gC, gCeq


# ----------------------------------------------------------------------
# Hessianfcn.m (Hessian of Lagrangian)
# ----------------------------------------------------------------------
@dataclass
class LagrangeMultipliers:
    """
    Minimal stand-in for fmincon's lambda structure:
    - eqnonlin: multiplier for nonlinear equality constraint(s)
    - ineqnonlin: multiplier for nonlinear inequality constraint(s)
    """
    eqnonlin: float = 0.0
    ineqnonlin: float = 0.0


def hessian_lagrangian(a, lagrange, H, E_prime, lam):
    """
    Python replacement for Hessianfcn.m: Hessian of the Lagrangian.

    L(a) = J(a) + lambda_eq * Ceq(a) + lambda_ineq * C_ineq(a)

    Here we include:
      - Hessian of J from `cost`
      - Hessian of Ceq (norm constraint) = 2 E_prime^T E_prime
      - Step-size inequality Hessian set to zero (as in original code).

    Parameters
    ----------
    a : array_like, shape (Nen,)
        Weight/control vector.
    lagrange : LagrangeMultipliers
        Multipliers for equality / inequality constraints.
    H : array_like, shape (1, Nen) or (Nen,)
        Linear observation operator (same as in `cost`).
    E_prime : array_like, shape (N_dofs, Nen)
        Deviation matrix.
    lam : float
        Regularisation parameter used in `cost`.

    Returns
    -------
    Hout : ndarray, shape (Nen, Nen)
        Hessian of the Lagrangian with respect to 'a'.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    E_prime = np.asarray(E_prime, dtype=float)

    # Hessian of the objective J(a)
    _, _, H_obj = cost(0.0, H, a, np.zeros(E_prime.shape[0]), E_prime, lam)

    # Hessian of equality constraint: 2 E_prime^T E_prime
    Hh = 2.0 * (E_prime.T @ E_prime)

    # For the inequality (step size) we keep Hessian zero
    Hg = np.zeros_like(Hh)

    Hout = H_obj + lagrange.eqnonlin * Hh + lagrange.ineqnonlin * Hg
    return Houts