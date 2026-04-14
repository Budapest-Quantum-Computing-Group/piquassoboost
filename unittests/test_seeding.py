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

"""Seeding tests for the boosted path.

The expected sample values below are specific to the piquassoboost C++ RNG
(std::mt19937 seeded with the integer seed_sequence). They differ from piquasso's
native expected values, which use numpy's SeedSequence-based RNG, but are fully
reproducible across runs.
"""

import numpy as np
import piquasso as pq
import piquassoboost as pqb

pqb.patch()

_U = np.array(
    [
        [-0.17959207 - 0.29175972j, -0.64550941 - 0.02897055j,
         -0.023922 - 0.38854671j, -0.07908932 + 0.35617293j,
         -0.39779191 + 0.14902272j],
        [-0.36896208 + 0.19468375j, -0.11545557 + 0.20434514j,
          0.25548079 + 0.05220164j, -0.51002161 + 0.38442256j,
          0.48106678 - 0.25210091j],
        [ 0.25912844 + 0.16131742j,  0.11886251 + 0.12632645j,
          0.69028213 - 0.25734432j,  0.01276639 + 0.05841739j,
          0.03713264 + 0.57364845j],
        [-0.20314019 - 0.18973473j,  0.59146854 + 0.28605532j,
         -0.11096495 - 0.26870144j, -0.47290354 - 0.0489408j,
         -0.42459838 + 0.01554643j],
        [-0.5021973 - 0.53474291j,   0.24524545 - 0.0741398j,
          0.37786104 + 0.10225255j,  0.46696955 + 0.10636677j,
          0.07171789 - 0.09194236j],
    ]
)


def test_boson_sampling_seeded():
    seed_sequence = 123

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([2, 1, 1, 0, 1])
        pq.Q(all) | pq.Interferometer(_U)
        pq.Q(all) | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5, config=pq.Config(seed_sequence=seed_sequence))
    samples = simulator.execute(program, shots=20).samples

    expected_samples = [
        (4, 0, 0, 0, 1),
        (0, 0, 0, 5, 0),
        (0, 1, 0, 3, 1),
        (0, 1, 1, 2, 1),
        (1, 0, 2, 0, 2),
        (1, 0, 2, 2, 0),
        (4, 0, 0, 1, 0),
        (0, 0, 2, 2, 1),
        (0, 4, 0, 0, 1),
        (0, 0, 3, 0, 2),
        (2, 0, 3, 0, 0),
        (1, 2, 1, 0, 1),
        (0, 0, 2, 1, 2),
        (0, 2, 1, 1, 1),
        (0, 1, 1, 1, 2),
        (2, 1, 1, 0, 1),
        (0, 3, 1, 1, 0),
        (0, 0, 3, 1, 1),
        (4, 0, 0, 0, 1),
        (0, 0, 3, 1, 1),
    ]

    assert samples == expected_samples


def test_LossyInterferometer_boson_sampling_seeded():
    seed_sequence = 123

    singular_values = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    lossy_interferometer_matrix = _U @ np.diag(singular_values) @ _U @ _U.T

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([2, 1, 1, 0, 1])
        pq.Q(all) | pq.LossyInterferometer(lossy_interferometer_matrix)
        pq.Q(all) | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5, config=pq.Config(seed_sequence=seed_sequence))
    samples = simulator.execute(program, shots=20).samples

    expected_samples = [
        (0, 3, 1, 0, 0),
        (0, 0, 0, 0, 4),
        (0, 1, 0, 0, 2),
        (0, 0, 2, 0, 1),
        (0, 2, 1, 0, 0),
        (0, 0, 1, 0, 1),
        (1, 0, 0, 1, 2),
        (0, 0, 0, 1, 2),
        (0, 0, 0, 1, 3),
        (2, 0, 0, 1, 1),
        (0, 1, 0, 0, 0),
        (0, 1, 0, 1, 0),
        (0, 0, 0, 1, 1),
        (0, 0, 1, 2, 0),
        (0, 0, 2, 0, 0),
        (3, 0, 0, 0, 0),
        (0, 1, 1, 0, 2),
        (0, 0, 1, 1, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 1, 1, 0),
    ]

    assert samples == expected_samples


def test_LossyInterferometer_boson_sampling_uniform_losses():
    seed_sequence = 123

    singular_values = np.array([0.9] * 5)
    lossy_interferometer_matrix = _U @ np.diag(singular_values) @ _U @ _U.T

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([2, 1, 1, 0, 1])
        pq.Q(all) | pq.LossyInterferometer(lossy_interferometer_matrix)
        pq.Q(all) | pq.ParticleNumberMeasurement()

    simulator = pq.SamplingSimulator(d=5, config=pq.Config(seed_sequence=seed_sequence))
    samples = simulator.execute(program, shots=20).samples

    expected_samples = [
        (0, 1, 1, 2, 0),
        (0, 0, 1, 1, 1),
        (1, 0, 4, 0, 0),
        (1, 0, 2, 1, 0),
        (0, 1, 2, 0, 0),
        (0, 0, 5, 0, 0),
        (1, 0, 1, 0, 2),
        (0, 1, 1, 1, 0),
        (0, 0, 2, 3, 0),
        (0, 0, 1, 2, 0),
        (0, 0, 0, 1, 2),
        (0, 0, 1, 2, 1),
        (2, 1, 1, 0, 1),
        (2, 0, 2, 0, 0),
        (0, 2, 1, 0, 0),
        (1, 0, 0, 2, 1),
        (3, 0, 0, 0, 1),
        (2, 1, 1, 0, 1),
        (0, 0, 1, 2, 2),
        (2, 1, 1, 0, 1),
    ]

    assert samples == expected_samples


def test_ThresholdMeasurement_use_torontonian_seeding():
    d = 5
    shots = 10
    seed_sequence = 123

    A = np.array(
        [
            [0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0],
        ]
    )

    with pq.Program() as program:
        pq.Q(all) | pq.Graph(A)
        pq.Q(all) | pq.ThresholdMeasurement()

    simulator = pq.GaussianSimulator(
        d=d,
        connector=pq.NumpyConnector(),
        config=pq.Config(seed_sequence=seed_sequence, use_torontonian=True),
    )
    result = simulator.execute(program, shots=shots)

    assert result.samples == [
        (1, 0, 1, 1, 1),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (1, 1, 0, 1, 1),
        (0, 0, 1, 1, 0),
        (1, 1, 0, 1, 1),
        (1, 1, 0, 1, 1),
        (1, 1, 0, 1, 1),
        (0, 0, 0, 0, 0),
        (1, 1, 0, 1, 1),
    ]


def test_ThresholdMeasurement_use_torontonian_seeding_float32():
    d = 5
    shots = 10
    seed_sequence = 123

    A = np.array(
        [
            [0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0],
        ]
    )

    with pq.Program() as program:
        pq.Q(all) | pq.Graph(A)
        pq.Q(all) | pq.ThresholdMeasurement()

    simulator = pq.GaussianSimulator(
        d=d,
        connector=pq.NumpyConnector(),
        config=pq.Config(
            seed_sequence=seed_sequence, use_torontonian=True, dtype=np.float32
        ),
    )
    result = simulator.execute(program, shots=shots)

    assert result.samples == [
        (1, 0, 1, 1, 1),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (1, 1, 0, 1, 1),
        (0, 0, 1, 1, 0),
        (1, 1, 0, 1, 1),
        (1, 1, 0, 1, 1),
        (1, 1, 0, 1, 1),
        (0, 0, 0, 0, 0),
        (1, 1, 0, 1, 1),
    ]
