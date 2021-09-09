#
#  Copyright 2021 Budapest Quantum Computing Group
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from .ThresholdBosonSampling_wrapper import ThresholdBosonSampling_wrapper


class ThresholdBosonSampling(
    ThresholdBosonSampling_wrapper
):
    def __init__(self, covariance_matrix):
        super().__init__(covariance_matrix=covariance_matrix)

    def simulate(self, samples_number: int = 1):
        """
            Returns samples from piquasso circuit. It applies threshold measurement.
            :param sample_number: number of samples to be calculated.
            :return: Simulation results.
        """

        return super().simulate(samples_number)


