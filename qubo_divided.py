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
import piquasso as pq

import piquassoboost as pqb

import functools

from typing import List, Tuple


def _map_to_bitstring(sample: np.ndarray, parity: int) -> np.ndarray:
    return list(map(lambda x: (x + parity) % 2, sample))


def _get_samples(
    thetas: List[float],
    config,
):
    iterator = iter(thetas)
    samples = [tuple()] * config["shots"]

    d = config["d_per_computer"]

    for _ in range(config["number_of_computers"]):

        simulator = pqb.BoostedSamplingSimulator(d=d)

        with pq.Program() as program:
            pq.Q(all) | pq.StateVector((1,) * d)

            for column in range(d):
                start_index = column % 2
                for element in range(start_index, d - 1, 2):
                    pq.Q(element, element + 1) | pq.Beamsplitter(next(iterator), phi=0)

            pq.Q() | pq.Sampling()

        samples_in_round = simulator.execute(program, config["shots"]).samples

        samples = list(map(lambda x, y: tuple([*x, *y]), samples, samples_in_round))

    return samples


def _calculate_energy_from_bitstrings(matrix, bitstrings):
    return [bitstring @ matrix @ bitstring for bitstring in bitstrings]


def _get_energy(
    thetas: List[float],
    matrix,
    config,
    mapper,
) -> float:
    samples = _get_samples(thetas, config)
    bitstrings = list(map(mapper, samples))
    energies = _calculate_energy_from_bitstrings(matrix, bitstrings)

    return sum(energies) / config["shots"]


def _update_thetas(thetas: List[float], matrix, config, mapper):
    new_thetas = np.copy(thetas)

    for j in range(len(thetas)):
        upshifted_thetas = np.copy(thetas)
        upshifted_thetas[j] += np.pi / 2
        upshifted_energy = _get_energy(upshifted_thetas, matrix, config, mapper=mapper)

        downshifted_thetas = np.copy(thetas)
        downshifted_thetas[j] -= np.pi / 2
        downshifted_energy = _get_energy(
            upshifted_thetas, matrix, config, mapper=mapper
        )

        derivative = (upshifted_energy - downshifted_energy) / 2

        new_thetas[j] -= config["learning_rate"] * derivative

    return new_thetas


def solve_qubo_by_dividing(
    matrix: np.ndarray,
    *,
    learning_rate: float = 0.1,
    shots: int = 5,
    updates: int = 10,
    d_per_computer: int = 2
) -> Tuple[float, Tuple[int, ...]]:
    d = len(matrix)

    assert d % d_per_computer == 0

    number_of_computers = d // d_per_computer

    simulator = pqb.BoostedSamplingSimulator(d=d_per_computer)

    min_energy = None

    config = {
        "simulator": simulator,
        "d": d,
        "d_per_computer": d_per_computer,
        "number_of_computers": number_of_computers,
        "shots": shots,
        "learning_rate": learning_rate,
    }

    thetas = np.random.uniform(
        0.0,
        2 * np.pi,
        number_of_computers * d_per_computer * (d_per_computer - 1) // 2
    )

    mean_energies = {
        0: [],
        1: [],
    }

    for parity in (0, 1):
        print("parity:", parity)
        map_to_bistring_with_parity = functools.partial(
            _map_to_bitstring, parity=parity
        )
        for _ in range(updates):
            samples = _get_samples(
                thetas=thetas,
                config=config,
            )

            bitstrings = list(map(map_to_bistring_with_parity, samples))
            energies = _calculate_energy_from_bitstrings(matrix, bitstrings)

            mean_energy = sum(energies)/len(energies)

            mean_energies[parity].append(mean_energy)

            print("Mean energies", mean_energy)

            if min_energy is None or min(energies) < min_energy:
                min_energy = min(energies)
                min_bistring = bitstrings[np.argmin(energies)]

            thetas = _update_thetas(thetas, matrix, config, map_to_bistring_with_parity)

    from matplotlib import pyplot as plt

    plt.plot(mean_energies[0])
    plt.plot(mean_energies[1])
    plt.show()

    return min_energy, min_bistring
