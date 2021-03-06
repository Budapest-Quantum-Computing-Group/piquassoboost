import numpy as np

from .BosonSamplingSimulator import BosonSamplingSimulator
from .simulation_strategies.GeneralizedCliffordsSimulationStrategy import (
    GeneralizedCliffordsSimulationStrategy,
)

import piquasso as pq


class SamplingCircuit(pq.sampling.circuit.SamplingCircuit):
    def sampling(self, operation):
        """Simulates a boson sampling using generalized Clifford&Clifford algorithm
        from [Brod, Oszmaniec 2020].

        This method assumes that initial_state is given in the second quantization
        description (mode occupation). BoSS requires input states as numpy arrays,
        therefore the state is prepared as such structure.

        Generalized Cliffords simulation strategy form [Brod, Oszmaniec 2020] was used
        as it allows effective simulation of broader range of input states than original
        algorithm.
        """

        params = operation.params
        simulation_strategy = GeneralizedCliffordsSimulationStrategy(
            self.state.interferometer
        )
        sampling_simulator = BosonSamplingSimulator(simulation_strategy)

        initial_state = np.array(self.state.initial_state)
        shots = params[0]
        self.state.results = sampling_simulator.get_classical_simulation_results(
            initial_state,
            samples_number=shots,
        )
