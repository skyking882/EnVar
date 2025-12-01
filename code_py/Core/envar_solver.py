import numpy as np
from scipy.optimize import minimize
from Core.ensemble_tool import ensemble_generator
from Core.cost_functions import envar_cost_function

class EnVarSolver:
    def __init__(self, model, a_init, y_target, config=None):
        self.model = model
        self.a_current = np.asarray(a_init, dtype=float).copy()
        self.y_target = y_target
        
        # Default configuration
        self.cfg = {
            'Nen': 7,
            'lam': 0.0,
            'alpha': 0.5,     # Update step size
            'max_iter': 20,
            'tol': 1e-2,
            'rng_seed': 42
        }
        if config:
            self.cfg.update(config)
            
        self.rng = np.random.default_rng(self.cfg['rng_seed'])
        self.history = []

    def solve(self):
        """
        Generator function, executes optimization iteration step-by-step.
        Yields current state dictionary for external plotting or monitoring.
        """
        Nen = self.cfg['Nen']
        lam = self.cfg['lam']
        alpha = self.cfg['alpha']

        for iter_ in range(1, self.cfg['max_iter'] + 1):
            # A. Generate Ensemble (Parameter space)
            a_ensemble = ensemble_generator(Nen, self.a_current, rng=self.rng)
            
            # Deviation matrix E'
            E_prime = a_ensemble - self.a_current[:, None]

            # B. Calculate observation space and increments
            y_e = self.model.forward(self.a_current)
            o_e = y_e - self.y_target 

            o_r_mat = np.zeros((len(y_e), Nen))
            for k in range(Nen):
                y_k = self.model.forward(a_ensemble[:, k])
                o_r_mat[:, k] = y_k - self.y_target
            
            H = o_r_mat - o_e[:, None]

            # C. Minimize Local Cost J(w)
            def objective_wrapper(w):
                val, dJ, _ = envar_cost_function(w, o_e, H, self.a_current, E_prime, lam)
                return val

            def gradient_wrapper(w):
                _, dJ, _ = envar_cost_function(w, o_e, H, self.a_current, E_prime, lam)
                return dJ

            w0 = np.zeros(Nen)
            res = minimize(
                objective_wrapper, w0, method='trust-constr', jac=gradient_wrapper,
                options={'disp': False, 'xtol': 1e-12}
            )
            w_opt = res.x
            J_loc = res.fun

            # D. Update parameter space mean
            step = E_prime @ w_opt
            self.a_current = self.a_current + alpha * step
            
            # Log and calculate error
            y_new = self.model.forward(self.a_current)
            misfit = y_new - self.y_target
            error_obs = np.linalg.norm(misfit)
            print(f"error_obs = {error_obs}")
            J_true = 0.5 * (misfit.T @ misfit)

            # Construct state packet
            state = {
                'iter': iter_,
                'a_mean': self.a_current.copy(),
                'ensemble': a_ensemble.copy(),
                'y_current': y_new.copy(),
                'J_loc': J_loc,
                'J_true': J_true,
                'error': error_obs,
                'converged': False
            }
            
            self.history.append(state)
            
            if error_obs < self.cfg['tol']:
                state['converged'] = True
                yield state
                break
            
            yield state