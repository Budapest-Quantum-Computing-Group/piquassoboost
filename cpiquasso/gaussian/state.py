#
# Copyright 2021 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

        obj._m = np.copy(self._m)
        obj._G = np.copy(self._G)
        obj._C = np.copy(self._C)

        return obj
