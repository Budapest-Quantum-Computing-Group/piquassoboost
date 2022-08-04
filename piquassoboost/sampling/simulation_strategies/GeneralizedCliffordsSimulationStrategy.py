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

from .GeneralizedCliffordsBSimulationStrategy_wrapper import (
    GeneralizedCliffordsBSimulationStrategy_wrapper
)

from .GeneralizedCliffordsBUniformLossesSimulationStrategy_wrapper import (
    GeneralizedCliffordsBUniformLossesSimulationStrategy_wrapper
)


class GeneralizedCliffordsBSimulationStrategy(
    GeneralizedCliffordsBSimulationStrategy_wrapper
):
    def __init__(self, interferometer_matrix, seed):

        super().__init__(interferometer_matrix, seed=seed, lib=0)


    def simulate(self, input_state, samples_number: int = 1):
        """
            Returns sample from linear optics experiments given output state.
            :param input_state: Input state in particle basis.
            :return: A resultant state after traversing through interferometer.
        """

        return super().simulate(input_state, samples_number)


class GeneralizedCliffordsBUniformLossesSimulationStrategy(
    GeneralizedCliffordsBUniformLossesSimulationStrategy_wrapper
):
    def __init__(self, interferometer_matrix, transmissivity, seed):

        super().__init__(interferometer_matrix, transmissivity=transmissivity, seed=seed, lib=0)


    def simulate(self, input_state, samples_number: int = 1):
        """
            Returns sample from linear optics experiments given output state.
            :param input_state: Input state in particle basis.
            :return: A resultant state after traversing through interferometer.
        """

        return super().simulate(input_state, samples_number)





class GeneralizedCliffordsSimulationStrategy(
    GeneralizedCliffordsSimulationStrategy_wrapper
):
    def __init__(self, interferometer_matrix, seed):

        super().__init__(interferometer_matrix, seed=seed, lib=0)


    def simulate(self, input_state, samples_number: int = 1):
        """
            Returns sample from linear optics experiments given output state.
            :param input_state: Input state in particle basis.
            :return: A resultant state after traversing through interferometer.
        """

        return super().simulate(input_state, samples_number)





class GeneralizedCliffordsSimulationStrategyChinHuh(
    GeneralizedCliffordsSimulationStrategy_wrapper
):
    def __init__(self, interferometer_matrix, seed):

        super().__init__(interferometer_matrix, seed=seed, lib=1)


    def simulate(self, input_state, samples_number: int = 1):
        """
            Returns sample from linear optics experiments given output state.
            :param input_state: Input state in particle basis.
            :return: A resultant state after traversing through interferometer.
        """

        return super().simulate(input_state, samples_number)






class GeneralizedCliffordsSimulationStrategySingleDFE(
    GeneralizedCliffordsSimulationStrategy_wrapper
):
    def __init__(self, interferometer_matrix, seed):

        super().__init__(interferometer_matrix, seed=seed, lib=2)


    def simulate(self, input_state, samples_number: int = 1):
        """
            Returns sample from linear optics experiments given output state.
            :param input_state: Input state in particle basis.
            :return: A resultant state after traversing through interferometer.
        """

        return super().simulate(input_state, samples_number)






class GeneralizedCliffordsSimulationStrategyDualDFE(
    GeneralizedCliffordsSimulationStrategy_wrapper
):
    def __init__(self, interferometer_matrix, seed):

        super().__init__(interferometer_matrix, seed=seed, lib=3)


    def simulate(self, input_state, samples_number: int = 1):
        """
            Returns sample from linear optics experiments given output state.
            :param input_state: Input state in particle basis.
            :return: A resultant state after traversing through interferometer.
        """

        return super().simulate(input_state, samples_number)





class GeneralizedCliffordsSimulationStrategyMultiSingleDFE(
    GeneralizedCliffordsSimulationStrategy_wrapper
):
    def __init__(self, interferometer_matrix, seed):

        super().__init__(interferometer_matrix, seed=seed, lib=4)


    def simulate(self, input_state, samples_number: int = 1):
        """
            Returns sample from linear optics experiments given output state.
            :param input_state: Input state in particle basis.
            :return: A resultant state after traversing through interferometer.
        """

        return super().simulate(input_state, samples_number)







class GeneralizedCliffordsSimulationStrategyMultiDualDFE(
    GeneralizedCliffordsSimulationStrategy_wrapper
):
    def __init__(self, interferometer_matrix, seed):

        super().__init__(interferometer_matrix, seed=seed, lib=5)


    def simulate(self, input_state, samples_number: int = 1):
        """
            Returns sample from linear optics experiments given output state.
            :param input_state: Input state in particle basis.
            :return: A resultant state after traversing through interferometer.
        """

        return super().simulate(input_state, samples_number)

