import numpy as np

def envar_cost_function(w, o_e, H, a_e, E_prime, lam):
    """
    Calculate EnVar local cost function J(w), gradient dJ, and Hessian.
    w: Weight vector to be optimized
    """
    # Ensure input is numpy array
    w = np.asarray(w, dtype=float)
    a = a_e + E_prime @ w # Current estimate in parameter space
    
    N_dofs = a_e.size
    idx = np.arange(1, N_dofs + 1, dtype=float)
    
    # Penalty vector v_i = i * a_i
    v = idx * a

    # Main term: 1/2 || o_e + H w ||^2 
    # H = (Observation of Ensemble - Observation of Mean)
    # Note: H acts on w, producing increments in observation space
    resid = o_e + H @ w
    J_main = 0.5 * np.sum(resid**2)

    # Regularization/smoothing term: 1/2 * lam * ||v||^2
    J_pen = 0.5 * lam * np.sum(v**2)
    J = J_main + J_pen

    # Gradient calculation
    # dJ_main / dw = H^T @ resid
    dJ_main = H.T @ resid
    
    # dJ_pen / dw:  v = D * (a_e + E w) -> dv/dw = D E
    # d(0.5 v.T v)/dw = v.T (dv/dw) = v.T D E = (E.T D.T v)
    # D is diag(idx)
    gv = idx * v
    dJ_pen = lam * (E_prime.T @ gv)
    dJ = dJ_main + dJ_pen

    # Hessian calculation
    # H_main = H^T H
    H_data = H.T @ H
    
    # H_pen = lam * E^T D^T D E
    W_diag = (idx ** 2)
    # Use broadcasting to simulate diagonal matrix multiplication
    H_pen = lam * (E_prime.T * W_diag) @ E_prime
    
    Hessian = H_data + H_pen

    return J, dJ, Hessian