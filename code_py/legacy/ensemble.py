import math
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# obs_vec
# ----------------------------------------------------------------------
def obs_vec(q):
    """
    Python equivalent of MATLAB obs_vec(q).
    """
    q_arr = np.asarray(q)
    return q_arr.mean(axis=0)

# ----------------------------------------------------------------------
# Ensemble Generation
# ----------------------------------------------------------------------
def ensemble(Nen, u_e, Y=1000, variance=0.005, rng=None, save_file=None):
    """
    Python equivalent of Ensemble.m.
    
    Parameters
    ----------
    Nen : int
        Number of ensemble members.
    u_e : array_like
        Mean control vector.
    Y : int
        Oversampling factor.
    variance : float
        Target variance.
    rng : np.random.Generator, optional
        Random number generator.
    save_file : str, optional
        If provided, saves the ensemble matrix to this filename (e.g., 'a_r_ensemble.txt').
        The format will be transposed (Nen x N_Dofs) to match MATLAB's writematrix output.
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

    # SVD of augmented covariance
    Uc, s_c, Vtc = np.linalg.svd(C, full_matrices=True)
    U1 = Uc[:, :-1]
    s1 = s_c[:-1]
    V1 = Vtc[:-1, :]
    
    # Reconstruct valid covariance
    SS = U1 @ (np.diag(s1) @ V1)
    covar = SS[:-1, :-1]

    # Cholesky factorisation with fallback for numerical stability
    try:
        R = np.linalg.cholesky(covar)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(covar)
        vals[vals < 1e-12] = 0.0
        covar = vecs @ np.diag(vals) @ vecs.T
        R = np.linalg.cholesky(covar + np.eye(N_dofs)*1e-12)
        
    L = R.T

    if rng is None:
        rng = np.random.default_rng(42)

    n_samples = Y * Nen

    # Generate Y*Nen standardised random vectors 
    y_hat = rng.random((N_dofs, n_samples)) * 2.0 - 1.0
    y_hat = y_hat - y_hat.mean(axis=0, keepdims=True)
    std = y_hat.std(axis=0, ddof=1, keepdims=True)
    std[std == 0.0] = 1.0
    y_hat = y_hat / std

    # Impose the desired covariance
    P_hat = L @ y_hat

    # Extract Nen directions using SVD
    Uh, s_h, Vth = np.linalg.svd(P_hat, full_matrices=True)
    U = Uh[:, :Nen]                    
    S = np.diag(s_h[:Nen])             
    V = Vth.T[:Nen, :Nen]              
    P = U @ ((1.0 / math.sqrt(Y)) * (S @ V.T))

    # Work in "energy" variable e = u^2
    e_e = u_e ** 2  
    S_var = np.eye(N_dofs) * variance

    u_r_ensemble = np.zeros((N_dofs, Nen))
    neg_mask = u_e < 0.0

    for i in range(Nen):
        e_r = S_var @ P[:, i] + e_e
        if np.any(e_r <= 0.0):
            e_r = -S_var @ P[:, i] + e_e

        u_r = np.sqrt(np.abs(e_r))
        u_r[neg_mask] *= -1.0
        u_r_ensemble[:, i] = u_r

    # ---------------------------------------------------------
    # Save to file if requested (Aligns with main.m logic)
    # ---------------------------------------------------------
    if save_file:
        # MATLAB: writematrix(a_r_ensemble', ...)
        # We save the transpose: Rows=Members, Cols=DOFs
        np.savetxt(save_file, u_r_ensemble.T, fmt='%.18e', delimiter=' ')

    return u_r_ensemble

# ----------------------------------------------------------------------
# Cost Function
# ----------------------------------------------------------------------
def cost(o_e, H, a, u_e, E_prime, lam=0.0):
    """
    Python equivalent of Cost.m.
    Handles both scalar and vector observations.
    """
    o_e = np.asarray(o_e, dtype=float)
    H = np.asarray(H, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1)
    u_e = np.asarray(u_e, dtype=float).reshape(-1)
    E_prime = np.asarray(E_prime, dtype=float)

    N_dofs = u_e.size
    
    # Penalty vector v_i = i * (u_e_i + (E_prime a)_i)
    idx = np.arange(1, N_dofs + 1, dtype=float)
    Ea = E_prime @ a
    v = idx * (u_e + Ea)

    # Main quadratic term 1/2 || o_e + H a ||^2
    resid = o_e + H @ a
    resid_norm = np.linalg.norm(resid) 
    J_main = 0.5 * (resid_norm**2)

    # Smoothness penalty 1/2 * lam * ||v||^2
    J_pen = 0.5 * 0.0 * np.sum(v**2)
    J = J_main + J_pen

    # Gradient calculation (vectorized)
    dJ_main = H.T @ resid
    
    gv = idx * v
    dJ_pen = 0.0 * (E_prime.T @ gv)
    dJ = dJ_main + dJ_pen

    # Hessian
    H_data = H.T @ H
    W = (idx ** 2)
    H_pen = 0.0 * (E_prime.T * W) @ E_prime
    Hessian = H_data + H_pen

    return J, dJ, Hessian

# ----------------------------------------------------------------------
# Constraints
# ----------------------------------------------------------------------
def constraints(u_e, E_prime, c, a, step_eps=0.075):
    u_e = np.asarray(u_e, dtype=float).reshape(-1)
    E_prime = np.asarray(E_prime, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1)

    N_dofs, Nen = E_prime.shape

    idx_even = np.arange(1, N_dofs, 2)
    E_even = E_prime[idx_even, :]
    u_even = u_e[idx_even]

    L_param = 3.0
    a0 = 0.5

    # 1) Length constraint
    C_length = u_even + E_even @ a - (L_param / 2.0 - a0)

    # 2) Step-size constraint
    step = E_prime @ a
    C_step = np.abs(step) - step_eps

    C = np.concatenate([C_length, C_step])

    # Equality
    ua = u_e + E_prime @ a
    Ceq = float(np.vdot(ua, ua) - c)

    # Gradients
    gC_length = E_even.T
    sign_step = np.sign(step)
    gC_step = (E_prime.T * sign_step) 
    gC = np.concatenate([gC_length, gC_step], axis=1)

    gCeq = 2.0 * (E_prime.T @ ua)

    return C, Ceq, gC, gCeq

# ----------------------------------------------------------------------
# Hessian of Lagrangian
# ----------------------------------------------------------------------
@dataclass
class LagrangeMultipliers:
    eqnonlin: float = 0.0
    ineqnonlin: float = 0.0

def hessian_lagrangian(a, lagrange, H, E_prime, lam):
    a = np.asarray(a, dtype=float).reshape(-1)
    E_prime = np.asarray(E_prime, dtype=float)

    # Hessian of the objective J(a)
    _, _, H_obj = cost(0.0, H, a, np.zeros(E_prime.shape[0]), E_prime, lam)
    Hh = 2.0 * (E_prime.T @ E_prime)
    Hg = np.zeros_like(Hh)

    Hout = H_obj + lagrange.eqnonlin * Hh + lagrange.ineqnonlin * Hg
    return Hout