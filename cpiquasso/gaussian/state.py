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

from .state_wrapper import GaussianState_Wrapper

from piquasso._math.linalg import block_reduce

from cpiquasso.sampling.simulation_strategies.GaussianSimulationStrategy import (
    GaussianSimulationStrategyFast
)
from cpiquasso.sampling.Boson_Sampling_Utilities_wrapper import Torontonian_wrapper

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

    def _apply_threshold_measurement(
        self,
        *,
        shots,
        modes,
    ):
        """
        NOTE: The same logic is used here, as for the particle number measurement.
        However, at threshold measurement there is no sense of cutoff, therefore it is
        set to 2 to make the logic sensible in this case as well.

        Also note, that one could speed up this calculation by not calculating the
        probability of clicks (i.e. 1 as sample), and argue that the probability of a
        click is equal to one minus the probability of no click.
        """
        if not np.allclose(self.mean, np.zeros_like(self.mean)):
            raise NotImplementedError(
                "Threshold measurement for displaced states are not supported: "
                f"mean={self.mean}"
            )

        return self._apply_general_particle_number_measurement(
            cutoff=2,
            modes=modes,
            shots=shots,
            calculation=calculate_threshold_detection_probability,
        )

    def _apply_particle_number_measurement(self, *, cutoff, modes, shots):
        reduced_state = self.reduced(modes)

        return GaussianSimulationStrategyFast(
            covariance_matrix=reduced_state.cov / (2 * pq.api.constants.HBAR),
            m=reduced_state.mean / np.sqrt(pq.api.constants.HBAR),
            fock_cutoff=cutoff,
        ).simulate(shots)

def calculate_threshold_detection_probability(
    state,
    subspace_modes,
    occupation_numbers,
):
    d = len(subspace_modes)

    Q = (state.complex_covariance + np.identity(2 * d)) / 2

    OS = (np.identity(2 * d, dtype=complex) - np.linalg.inv(Q)).conj()

    OS_reduced = block_reduce(OS, reduce_on=occupation_numbers)

    return (
        Torontonian_wrapper(OS_reduced.astype(complex)).calculate()
        # torontonian(OS_reduced.astype(complex))
    ).real / np.sqrt(np.linalg.det(Q).real)
