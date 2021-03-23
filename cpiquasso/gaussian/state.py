

import numpy as np

import piquasso as pq

from piquasso._math.linalg import is_symmetric, is_selfadjoint

from .state_wrapper import GaussianState_Wrapper



class GaussianState(GaussianState_Wrapper, pq.GaussianState):
    def __init__(self, *, d):
        self.d = d

        vector_shape = (self.d, )
        matrix_shape = vector_shape * 2

        super().__init__(
            m=np.zeros(vector_shape, dtype=complex),
            G=np.zeros(matrix_shape, dtype=complex),
            C=np.zeros(matrix_shape, dtype=complex),
        )

    def apply_passive(self, T, modes):
        self._m[modes, ] = T @ self._m[modes, ]
        super().apply_to_C_and_G(T, modes)

    def __deepcopy__(self, memo):
        """
        NOTE: This method is needed to deepcopy the GaussianState by `copy.deepcopy`
        at state registration, see :class:`pq.Program`.
        """
        obj = GaussianState(d=self.d)

        obj._m = self._m
        obj._G = self._G
        obj._C = self._C

        return obj
