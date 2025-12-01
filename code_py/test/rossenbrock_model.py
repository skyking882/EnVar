import numpy as np

class RosenbrockModel:
    """
    Rosenbrock Benchmark Model.
    Mapped to an observation operator form for Least-Squares solvers.
    
    Standard Form: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    
    Vector Residual Form (Output):
    y[0] = 1 - x
    y[1] = 10 * (y - x^2)
    """
    def __init__(self, N_modes=2):
        # Rosenbrock is typically 2D, but can be generalized. 
        # Here we strictly enforce N=2 for the standard 'banana' visualization.
        self.N_modes = 2 
        
    def forward(self, a):
        """
        Input: State vector 'a' (size 2: [x, y])
        Output: Residual vector 'y' (size 2)
        """
        x = a[0]
        y = a[1]
        
        # Term 1: Bias (pulls x to 1)
        r1 = 1.0 - x
        
        # Term 2: The Banana Valley (pulls y to x^2)
        # We use sqrt(100) = 10 as the coefficient so that squaring it gives 100
        r2 = 10.0 * (y - x**2)
        
        return np.array([r1, r2])

    def get_feature_names(self):
        return ["Bias Term (1-x)", "Valley Term 10(y-x^2)"]