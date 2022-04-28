#
# Copyright 2021-2022 Budapest Quantum Computing Group
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

from .GaussianSimulationStrategy_wrapper import GaussianSimulationStrategy_wrapper
from .GaussianSimulationStrategyFast_wrapper import GaussianSimulationStrategyFast_wrapper


class GaussianSimulationStrategy(
    GaussianSimulationStrategy_wrapper
):
    def __init__(self, covariance_matrix, seed, m=None, fock_cutoff=5):
        super().__init__(
            covariance_matrix=covariance_matrix,
            m=m,
            fock_cutoff=fock_cutoff,
            seed=seed
        )


    def simulate(self, samples_number: int = 1):
        """
            Returns sample from linear optics experiments given output state.
            :param input_state: Input state in particle basis.
            :return: A resultant state after traversing through interferometer.
        """

        return super().simulate(samples_number)



class GaussianSimulationStrategyFast(
    GaussianSimulationStrategyFast_wrapper
):
    def __init__(self, covariance_matrix, seed, m=None, fock_cutoff=5):
        super().__init__(
            covariance_matrix=covariance_matrix,
            m=m,
            fock_cutoff=fock_cutoff,
            seed=seed
        )

    def simulate(self, samples_number: int = 1):
        """
            Returns sample from linear optics experiments given output state.
            :param input_state: Input state in particle basis.
            :return: A resultant state after traversing through interferometer.
        """

        return super().simulate(samples_number)
