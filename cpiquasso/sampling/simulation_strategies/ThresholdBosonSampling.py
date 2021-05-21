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

from .ThresholdBosonSampling_wrapper import ThresholdBosonSampling_wrapper


class ThresholdBosonSampling(
    ThresholdBosonSampling_wrapper
):
    def __init__(self, covariance_matrix, m=None, fock_cutoff=2, max_photons=20):

        super().__init__(covariance_matrix=covariance_matrix, m=m, fock_cutoff=fock_cutoff, max_photons=max_photons)


    def simulate(self, samples_number: int = 1):
        """
            Returns sample from linear optics experiments given output state.
            :param input_state: Input state in particle basis.
            :return: A resultant state after traversing through interferometer.
        """

        return super().simulate(samples_number)


