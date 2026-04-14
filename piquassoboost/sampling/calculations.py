#
# Copyright 2021-2026 Budapest Quantum Computing Group
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

from functools import partial
from fractions import Fraction
from collections import Counter

from piquasso.api.branch import Branch
from piquasso.api.exceptions import NotImplementedCalculation
from piquasso._simulators.sampling.utils import (
    _prepare_interferometer_matrix_in_expanded_space,
)
from piquasso._simulators.sampling.simulation_steps import (
    particle_number_measurement as _pq_particle_number_measurement,
)

from .BosonSamplingSimulator import BosonSamplingSimulator
from .simulation_strategies.GeneralizedCliffordsSimulationStrategy import (
    GeneralizedCliffordsBSimulationStrategy,
    GeneralizedCliffordsSimulationStrategy,
    GeneralizedCliffordsSimulationStrategyChinHuh,
    GeneralizedCliffordsSimulationStrategyChinHuh,
    GeneralizedCliffordsSimulationStrategySingleDFE,
    GeneralizedCliffordsSimulationStrategyDualDFE,
    GeneralizedCliffordsSimulationStrategyMultiSingleDFE,
    GeneralizedCliffordsSimulationStrategyMultiDualDFE,
    GeneralizedCliffordsBUniformLossesSimulationStrategy,
    GeneralizedCliffordsBLossySimulationStrategy,
)


def _is_uniform(array):
    return np.isclose(np.min(array), np.max(array))


def _particle_number_measurement(
    state, instruction, shots, strategy_class, speedup_uniform=False
):
    """Simulates a boson sampling using generalized Clifford&Clifford algorithm
    from [Brod, Oszmaniec 2020].

    This method assumes that initial_state is given in the second quantization
    description (mode occupation). BoSS requires input states as numpy arrays,
    therefore the state is prepared as such structure.

    Generalized Cliffords simulation strategy form [Brod, Oszmaniec 2020] was used
    as it allows effective simulation of broader range of input states than original
    algorithm.
    """

    # PostSelectPhotons mid-circuit and max_sample_generation_trials are not
    # implemented in the boosted path — delegate to piquasso for those cases.
    if state._is_postselected():
        return _pq_particle_number_measurement(state, instruction, shots)

    if (
        state._can_validate_variable(state._coefficients[0])
        and len(state._occupation_numbers) != 1
        and not np.isclose(state._coefficients[0], 1.0)
    ):
        raise NotImplementedCalculation(
            f"The instruction {instruction} is not supported for states defined using "
            "multiple 'StateVector' instructions.\n"
            "If you need this feature to be implemented, please create an issue at "
            "https://github.com/Budapest-Quantum-Computing-Group/piquasso/issues"
        )

    initial_state = state._occupation_numbers[0]

    if state._config.number_of_approximated_modes is not None:
        interferometer = state.interferometer

        simulation_strategy = GeneralizedCliffordsBLossySimulationStrategy(
            interferometer,
            state._config.number_of_approximated_modes,
            state._config.seed_sequence,
        )

        sampling_simulator = BosonSamplingSimulator(simulation_strategy)

        samples = np.array(
            sampling_simulator.get_classical_simulation_results(
                initial_state, samples_number=shots
            ),
            dtype=int,
        )

        counts = Counter(map(tuple, samples.tolist()))
        return [
            Branch(state=None, outcome=outcome, frequency=Fraction(count, shots))
            for outcome, count in counts.items()
        ]

    interferometer_svd = np.linalg.svd(state.interferometer)

    singular_values = interferometer_svd[1]

    is_uniform = _is_uniform(singular_values)

    if state.is_lossy:
        if speedup_uniform and is_uniform:
            # revert interferometer to unitary
            interferometer = state.interferometer / singular_values[0]

            simulation_strategy = GeneralizedCliffordsBUniformLossesSimulationStrategy(
                interferometer, singular_values[0], state._config.seed_sequence
            )

        else:
            # In case of losses we want specially prepared 2m x 2m interferometer matrix
            interferometer = _prepare_interferometer_matrix_in_expanded_space(
                interferometer_svd
            )

            simulation_strategy = strategy_class(
                interferometer, state._config.seed_sequence
            )

            # In case of losses we want 2m-modes input state
            # (initialized with 0 on new modes)
            initial_state = np.append(initial_state, np.zeros_like(initial_state))

    else:
        interferometer = state.interferometer
        simulation_strategy = strategy_class(
            interferometer, state._config.seed_sequence
        )

    sampling_simulator = BosonSamplingSimulator(simulation_strategy)

    samples = sampling_simulator.get_classical_simulation_results(
        initial_state, samples_number=shots
    )

    samples = np.array(samples, dtype=int)

    if state.is_lossy:  # Trim lossy state to initial size.
        samples = samples[:, : len(instruction.modes)]

    counts = Counter(map(tuple, samples.tolist()))
    return [
        Branch(state=None, outcome=outcome, frequency=Fraction(count, shots))
        for outcome, count in counts.items()
    ]


particle_number_measurement = partial(
    _particle_number_measurement,
    strategy_class=GeneralizedCliffordsBSimulationStrategy,
    speedup_uniform=True,
)

sampling_GeneralizedCliffords = partial(
    _particle_number_measurement, strategy_class=GeneralizedCliffordsSimulationStrategy
)
sampling_GeneralizedCliffords_chinhuh = partial(
    _particle_number_measurement,
    strategy_class=GeneralizedCliffordsSimulationStrategyChinHuh,
)
sampling_GeneralizedCliffords_single_dfe = partial(
    _particle_number_measurement,
    strategy_class=GeneralizedCliffordsSimulationStrategySingleDFE,
)
sampling_GeneralizedCliffords_dual_dfe = partial(
    _particle_number_measurement,
    strategy_class=GeneralizedCliffordsSimulationStrategyDualDFE,
)
sampling_GeneralizedCliffords_multi_single_dfe = partial(
    _particle_number_measurement,
    strategy_class=GeneralizedCliffordsSimulationStrategyMultiSingleDFE,
)
sampling_GeneralizedCliffords_multi_dual_dfe = partial(
    _particle_number_measurement,
    strategy_class=GeneralizedCliffordsSimulationStrategyMultiDualDFE,
)
