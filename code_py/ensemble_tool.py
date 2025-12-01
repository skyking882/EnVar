import numpy as np
import math

def ensemble_generator(Nen, u_e, Y=1000, variance=0.005, rng=None):
    """
    Generate perturbation ensemble (Ensemble Generation).
    """
    u_e = np.asarray(u_e, dtype=float).reshape(-1)
    N_dofs = u_e.size

    # Generate symmetric covariance matrix structure with zero row-sum
    A = np.full((N_dofs, N_dofs), -1.0 / (N_dofs - 1.0))
    np.fill_diagonal(A, 1.0)
    
    # Augmented covariance
    C = np.zeros((N_dofs + 1, N_dofs + 1))
    C[0, 0] = 1.0
    C[1:, 1:] = A

    # SVD decomposition
    Uc, s_c, Vtc = np.linalg.svd(C, full_matrices=True)
    U1 = Uc[:, :-1]
    s1 = s_c[:-1]
    V1 = Vtc[:-1, :]
    
    # Reconstruct valid covariance
    SS = U1 @ (np.diag(s1) @ V1)
    covar = SS[:-1, :-1]

    # Cholesky decomposition (with numerical stability handling)
    try:
        R = np.linalg.cholesky(covar)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(covar)
        vals[vals < 1e-12] = 0.0
        covar = vecs @ np.diag(vals) @ vecs.T
        R = np.linalg.cholesky(covar + np.eye(N_dofs)*1e-12)
    L = R.T

    if rng is None:
        rng = np.random.default_rng()

    n_samples = Y * Nen
    # Generate standardized random vectors
    y_hat = rng.random((N_dofs, n_samples)) * 2.0 - 1.0
    y_hat = y_hat - y_hat.mean(axis=0, keepdims=True)
    std = y_hat.std(axis=0, ddof=1, keepdims=True)
    std[std == 0.0] = 1.0
    y_hat = y_hat / std

    # Impose covariance
    P_hat = L @ y_hat

    # Extract Nen directions using SVD
    Uh, s_h, Vth = np.linalg.svd(P_hat, full_matrices=True)
    U = Uh[:, :Nen]                    
    S = np.diag(s_h[:Nen])             
    V = Vth.T[:Nen, :Nen]              
    P = U @ ((1.0 / math.sqrt(Y)) * (S @ V.T))

    # Energy variable transformation e = u^2
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

    return u_r_ensemble