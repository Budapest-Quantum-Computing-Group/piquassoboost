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

from .BosonSamplingSimulator import BosonSamplingSimulator
from .simulation_strategies.GeneralizedCliffordsSimulationStrategy import (
    GeneralizedCliffordsSimulationStrategy,
)

import piquasso as pq


class SamplingCircuit(pq.SamplingState.circuit_class):
    def _sampling(self, instruction):
        """Simulates a boson sampling using generalized Clifford&Clifford algorithm
        from [Brod, Oszmaniec 2020].

        This method assumes that initial_state is given in the second quantization
        description (mode occupation). BoSS requires input states as numpy arrays,
        therefore the state is prepared as such structure.

        Generalized Cliffords simulation strategy form [Brod, Oszmaniec 2020] was used
        as it allows effective simulation of broader range of input states than original
        algorithm.
        """

        simulation_strategy = GeneralizedCliffordsSimulationStrategy(
            self.state.interferometer
        )
        sampling_simulator = BosonSamplingSimulator(simulation_strategy)

        initial_state = np.array(self.state.initial_state)

        samples = sampling_simulator.get_classical_simulation_results(
            initial_state,
            samples_number=instruction.params["shots"],
        )

        self.results.append(
            pq.api.result.Result(instruction=instruction, samples=samples)
        )
