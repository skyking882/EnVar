import numpy as np

class Rosenbrock4DModel:
    """
    Generalized Rosenbrock Benchmark (N=4).
    Intermediate dimension: Too complex for 2D contours, but small enough
    to track every single variable individually.
    
    Global Minimum: x = [1, 1, 1, 1]
    """
    def __init__(self, N_modes=4):
        self.N_modes = 4 # Hardcoded for this specific test
        self.x_indices = np.arange(1, self.N_modes + 1)

    def forward(self, a):
        """
        Standard Generalized Rosenbrock Residuals.
        Sum(Residuals^2) = Rosenbrock Function.
        """
        a = np.asarray(a, dtype=float)
        
        # Term 1: Bias (1 - x_i) for i=0..N-2
        r1 = 1.0 - a[:-1]
        
        # Term 2: Linking (x_{i+1} - x_i^2)
        r2 = 10.0 * (a[1:] - a[:-1]**2)
        
        return np.concatenate([r1, r2])

    def get_feature_names(self):
        return [f"$x_{i}$" for i in range(1, 5)]