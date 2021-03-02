


import piquasso as pq

from .state_wrapper import GaussianState_Wrapper


class GaussianState(GaussianState_Wrapper, pq.GaussianState):
    def apply_passive(self, T, modes):
        self.m[modes, ] = T @ self.m[modes, ]
        super().apply_to_C_and_G(T, modes)
