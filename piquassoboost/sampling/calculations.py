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

import numpy as np

from theboss.boson_sampling_utilities.boson_sampling_utilities import (
    prepare_interferometer_matrix_in_expanded_space
)
from .BosonSamplingSimulator import BosonSamplingSimulator
from .simulation_strategies.GeneralizedCliffordsSimulationStrategy import (
    GeneralizedCliffordsSimulationStrategy,
    GeneralizedCliffordsSimulationStrategyChinHuh,
    GeneralizedCliffordsSimulationStrategyChinHuh,
    GeneralizedCliffordsSimulationStrategySingleDFE,
    GeneralizedCliffordsSimulationStrategyDualDFE,
    GeneralizedCliffordsSimulationStrategyMultiSingleDFE,
    GeneralizedCliffordsSimulationStrategyMultiDualDFE,
)

from piquasso.api.result import Result



def sampling(state, instruction, shots) -> Result:
    """Simulates a boson sampling using generalized Clifford&Clifford algorithm
    from [Brod, Oszmaniec 2020].

    This method assumes that initial_state is given in the second quantization
    description (mode occupation). BoSS requires input states as numpy arrays,
    therefore the state is prepared as such structure.

    Generalized Cliffords simulation strategy form [Brod, Oszmaniec 2020] was used
    as it allows effective simulation of broader range of input states than original
    algorithm.
    """

    interferometer = state.interferometer
    initial_state = np.array(state.initial_state)

    if state.is_lossy:  # Prepare inputs for lossy regime.
        # In case of losses we want specially prepared 2m x 2m interferometer matrix
        interferometer = prepare_interferometer_matrix_in_expanded_space(
            state.interferometer
        )

        # In case of losses we want 2m-modes input state
        # (initialized with 0 on new modes)
        for _ in initial_state:
            initial_state = np.append(initial_state, 0)

    simulation_strategy = GeneralizedCliffordsSimulationStrategy(
        interferometer, state._config.seed_sequence
    )
    sampling_simulator = BosonSamplingSimulator(simulation_strategy)

    samples = sampling_simulator.get_classical_simulation_results(
        initial_state,
        samples_number=shots
    )

    if state.is_lossy:  # Trim lossy state to initial size.
        trimmed_samples = []
        for sample in samples:
            trimmed_samples.append(sample[:len(state.initial_state)])
        samples = trimmed_samples  # We want to return trimmed samples.

    return Result(state=state, samples=samples)
