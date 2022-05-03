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

__author__ = 'Tomasz Rybotycki'
"""
This class would be turned into a C++ extension.
"""

from typing import List

from numpy import ndarray


class BosonSamplingSimulator:

    def __init__(self, simulation_strategy) -> None:
        self.simulation_strategy = simulation_strategy

    def get_classical_simulation_results(self, input_state: ndarray, samples_number: int = 1) -> List[ndarray]:
        return self.simulation_strategy.simulate(input_state, samples_number)
