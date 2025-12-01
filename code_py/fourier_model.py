import numpy as np

class FourierNonlinearModel:
    """
    Toy Problem: Nonlinear observation operator on Fourier basis
    y = Phi @ (a + 0.1 * a^2)
    """
    def __init__(self, N_modes, N_t=200):
        self.N_modes = N_modes
        self.N_t = N_t
        self.t = np.linspace(0.0, 2.0 * np.pi, N_t).reshape(-1, 1)
        self.Phi = self._build_basis()

    def _build_basis(self):
        Phi = np.zeros((self.N_t, self.N_modes))
        for n in range(1, self.N_modes + 1):
            Phi[:, n - 1] = np.sin(n * self.t[:, 0])
        return Phi

    def forward(self, a):
        """
        Observation operator H(a). 
        Input: State vector a (N_modes,)
        Output: Observation vector y (N_t,)
        """
        a = np.asarray(a, dtype=float)
        # Physics/Observation model: y = Phi * (a + 0.1*a^2)
        a_eff = a + 0.1 * (a**2)
        return self.Phi @ a_eff

    def get_time_grid(self):
        return self.t