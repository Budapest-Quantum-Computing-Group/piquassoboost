


import piquasso as pq

from piquasso.api.errors import StatePreparationError
from piquasso._math.linalg import is_symmetric, is_selfadjoint

from .state_wrapper import GaussianState_Wrapper


class GaussianState(GaussianState_Wrapper, pq.GaussianState):
    def __init__(self, C, G, m):
        if not is_selfadjoint(C):
            raise StatePreparationError("C should be self-adjoint matrix.")
        if not is_symmetric(G):
            raise StatePreparationError("G should be symmetric matrix.")

        super().__init__(C, G, m)

    def apply_passive(self, T, modes):
        self.m[modes, ] = T @ self.m[modes, ]
        super().apply_to_C_and_G(T, modes)
