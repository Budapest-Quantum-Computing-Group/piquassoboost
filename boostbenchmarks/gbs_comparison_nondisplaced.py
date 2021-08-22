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
    python boostbenchmarks/gbs_comparison_nondisplaced.py
"""

import os
import json
import time
import scipy
import itertools
import subprocess
import numpy as np

from scipy.optimize import root_scalar

import matplotlib.pyplot as plt

import piquasso as pq
import strawberryfields as sf

from cpiquasso import patch


patch()


SHOTS = 1
ITERATIONS = 1

NUMBER_OF_MODES = range(2, 6)  # NOTE: The range must start with 2 at least.
MEAN_PHOTON_NUMBER = 0.5

LOGARITHMIZE = True


def _run_pq_simulation(params, d) -> float:
    with pq.Program() as pq_program:
        pq.Q(all) | pq.Squeezing(r=params["squeezing"])

        pq.Q(all) | pq.Interferometer(params["interferometer"])

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
        for mode, parameter in enumerate(params["squeezing"]):
            sf.ops.Sgate(parameter) | q[mode]

        sf.ops.Interferometer(params["interferometer"]) | tuple([q[i] for i in range(d)])

        sf.ops.MeasureFock() | tuple([q[i] for i in range(d)])

    start_time = time.time()
    sf_engine.run(sf_program, shots=SHOTS)
    return time.time() - start_time

def _get_scaling(singular_values, mean_photon_number):
    def mean_photon_number_equation(scaling):
        return sum(
            (scaling * singular_value) ** 2 / (1 - (scaling * singular_value) ** 2)
            for singular_value
            in singular_values
        ) / len(singular_values) - mean_photon_number

    def mean_photon_number_gradient(scaling):
        return (
            (2.0 / scaling)
            * np.sum(
                (
                    singular_values * scaling
                    / (1 - (singular_values * scaling) ** 2)
                ) ** 2
            )
        )

    lower_bound = 0.0

    tolerance = 1e-10  # Needed to avoid zero division.

    upper_bound = 1.0 / (max(singular_values) + tolerance)

    result = root_scalar(
        mean_photon_number_equation,
        fprime=mean_photon_number_gradient,
        x0=(lower_bound - upper_bound) / 2.0,
        bracket=(lower_bound, upper_bound),
    )

    if not result.converged:
        raise RuntimeError("No scaling found.")

    return result.root


def _scale_squeezing(squeezing, mean_photon_number):
    scaling = _get_scaling(
        singular_values=squeezing,
        mean_photon_number=mean_photon_number
    )

    return scaling * squeezing

def _generate_parameters(d, mean_photon_number):
    squeezing = np.random.uniform(low=0.0, high=1.0, size=d)

    scaled_squeezing = _scale_squeezing(squeezing, mean_photon_number)

    return {
        "squeezing": scaled_squeezing,
        "interferometer": scipy.stats.unitary_group.rvs(d)
    }


def _benchmark():
    results = {
        "shots": SHOTS,
        "iterations": ITERATIONS,
        "mean_photon_number": MEAN_PHOTON_NUMBER,
        "number_of_modes": list(NUMBER_OF_MODES),
        "pq_averages": [],
        "sf_averages": [],
    }

    for d in NUMBER_OF_MODES:
        pq_times = []
        sf_times = []
        for _ in itertools.repeat(None, ITERATIONS):
            params = _generate_parameters(d, MEAN_PHOTON_NUMBER)

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
