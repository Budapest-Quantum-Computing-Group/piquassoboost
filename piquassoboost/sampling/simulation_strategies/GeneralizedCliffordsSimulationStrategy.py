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

from .GeneralizedCliffordsSimulationStrategy_wrapper import (
    GeneralizedCliffordsSimulationStrategy_wrapper
)


class GeneralizedCliffordsSimulationStrategy(
    GeneralizedCliffordsSimulationStrategy_wrapper
):
    def __init__(self, interferometer_matrix, seed):

        super().__init__(interferometer_matrix, seed)


    def simulate(self, input_state, samples_number: int = 1):
        """
            Returns sample from linear optics experiments given output state.
            :param input_state: Input state in particle basis.
            :return: A resultant state after traversing through interferometer.
        """

        return super().simulate(input_state, samples_number)
