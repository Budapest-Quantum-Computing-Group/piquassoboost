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

"""
Run me with
.. code-block:: bash
    python boostbenchmarks/graph_gbs_comparison.py
"""

import os
import json
import networkx as nx
import time
import itertools
import subprocess
from networkx.linalg.graphmatrix import adjacency_matrix
from networkx.readwrite.json_graph.adjacency import adjacency_graph
import numpy as np

import matplotlib.pyplot as plt

import piquasso as pq
import strawberryfields as sf

from cpiquasso import patch


patch()


SHOTS = 1
ITERATIONS = 1

NUMBER_OF_MODES = range(2, 6)  # NOTE: The range must start with 2 at least.
MEAN_PHOTON_NUMBER = 0.5
EDGE_CREATION_PROBABILITY = 0.5

LOGARITHMIZE = True


def _run_pq_simulation(params, d) -> float:
    with pq.Program() as pq_program:
        pq.Q() | pq.Graph(
            params["adjacency_matrix"],
            mean_photon_number=MEAN_PHOTON_NUMBER,
        )

        # NOTE: In SF the cutoff is 5, and couldn't be changed
        pq.Q(all) | pq.ParticleNumberMeasurement(cutoff=5, shots=SHOTS)

    state = pq.GaussianState(d=d)

    start_time = time.time()
    state.apply(program=pq_program)
    return time.time() - start_time

def _run_sf_simulation(params, d) -> float:
    sf_program = sf.Program(d)
    sf_engine = sf.Engine(backend="gaussian")

    with sf_program.context as q:
        sf.ops.GraphEmbed(
            params["adjacency_matrix"],
            MEAN_PHOTON_NUMBER
        ) | tuple([q[i] for i in range(d)])

        sf.ops.MeasureFock() | tuple([q[i] for i in range(d)])

    start_time = time.time()
    sf_engine.run(sf_program, shots=SHOTS)
    return time.time() - start_time

def _is_invertible(matrix):
    return np.linalg.matrix_rank(matrix) == matrix.shape[0]

def _generate_adjacency_matrix(d):
    graph = nx.erdos_renyi_graph(n=d, p=EDGE_CREATION_PROBABILITY)
    return nx.adjacency_matrix(graph).toarray()

def _generate_parameters(d):
    """
    NOTE: We need to make sure that no singular matrices enter the simulation,
    which is currently done by re-generating the adjacency matrix if needed.
    There might be better solution for doing this, however...
    """
    adjacency_matrix = _generate_adjacency_matrix(d)
    while not _is_invertible(adjacency_matrix):
        adjacency_matrix = _generate_adjacency_matrix(d)

    return {
        "adjacency_matrix": adjacency_matrix
    }

def _benchmark():
    results = {
        "shots": SHOTS,
        "iterations": ITERATIONS,
        "mean_photon_number": MEAN_PHOTON_NUMBER,
        "number_of_modes": list(NUMBER_OF_MODES),
        "edge_creation_probability": EDGE_CREATION_PROBABILITY,
        "pq_averages": [],
        "sf_averages": [],
    }

    for d in NUMBER_OF_MODES:
        pq_times = []
        sf_times = []
        for _ in itertools.repeat(None, ITERATIONS):
            params = _generate_parameters(d)

            pq_times.append(_run_pq_simulation(params, d))
            sf_times.append(_run_sf_simulation(params, d))

        pq_average = sum(pq_times) / len(pq_times)
        sf_average = sum(sf_times) / len(sf_times)

        results["pq_averages"].append(pq_average)
        results["sf_averages"].append(sf_average)

    print("\nSIMULATION RESULTS:\n")

    print("Mean photon number:", MEAN_PHOTON_NUMBER)
    print("d:", list(NUMBER_OF_MODES))
    print("piquasso   (s):", results["pq_averages"])
    print("strawberry (s):", results["sf_averages"])

    # PLOTTING

    x_axis = list(NUMBER_OF_MODES)

    plt.rcParams.update({'font.size': 12})
    plt.rcParams['lines.markersize'] *= 2

    plt.plot(
        x_axis,
        results["pq_averages"],
        'bx',
        x_axis,
        results["sf_averages"],
        'rx',
    )

    plt.xlim([min(x_axis) - 0.5, max(x_axis) + 0.5])

    plt.xlabel("Number of modes (d)")
    plt.ylabel(
        "Logarithmic computation time (s)"
        if LOGARITHMIZE
        else "Computation time (s)"
    )

    plt.xticks(x_axis)

    if not LOGARITHMIZE:
        plt.gca().set_ylim(bottom=0)
    else:
        plt.yscale("log")

    file_prefix = f"{os.path.basename(__file__)}_{int(time.time())}"

    svg_filename = f"{file_prefix}.png"
    json_filename = f"{file_prefix}.json"

    with open(json_filename, "w") as f:
        json.dump(results, f)

    plt.savefig(svg_filename)

    subprocess.call(('xdg-open', svg_filename))

if __name__ == "__main__":
    _benchmark()
