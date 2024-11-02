import time

import random

import json

import matplotlib.pyplot as plt

import numpy as np

import perceval as pcvl
from perceval.algorithm import Sampler

import piquasso as pq
import piquassoboost as pqb


N = 100  # number of samplings


def Generating_Input(n, m):
    "This function randomly chooses an input with n photons in m modes."
    modes = sorted(random.sample(range(m), n))

    state = "|"
    for i in range(m):
        state = (
            state + "0" * (1 - (i in modes)) + "1" * (i in modes) + "," * (i < m - 1)
        )
    return pcvl.BasicState(state + ">")


mzi = (
    pcvl.BS()
    // (0, pcvl.PS(phi=pcvl.Parameter("φ_a")))
    // pcvl.BS()
    // (1, pcvl.PS(phi=pcvl.Parameter("φ_b")))
)


def get_perceval_samples(unitary):
    m = len(unitary)

    n = m // 2

    Linear_Circuit = pcvl.Circuit.decomposition(
        unitary,
        mzi,
        phase_shifter_fn=pcvl.PS,
        shape=pcvl.InterferometerShape.TRIANGLE,
    )

    QPU = pcvl.Processor("CliffordClifford2017", Linear_Circuit)

    input_state = Generating_Input(n, m)
    QPU.with_input(input_state)

    # Keep all outputs
    QPU.min_detected_photons_filter(0)

    sampler = Sampler(QPU)

    return sampler.samples(N)["results"]

def generate_random_fock_state(m, n):
    modes = sorted(random.sample(range(m), n))

    state_vector = []
    for i in range(m):
        if i in modes:
            state_vector.append(1)
        else:
            state_vector.append(0)

    return state_vector


def get_piquassoboost_samples(unitary):
    m = len(unitary)
    n = m // 2

    input_state = generate_random_fock_state(m=m, n=n)
    program = pq.Program(
        instructions=[
            pq.StateVector(input_state),
            pq.Interferometer(unitary),
            pq.ParticleNumberMeasurement(),
        ]
    )

    simulator = pqb.BoostedSamplingSimulator(d=m)

    return simulator.execute(program, shots=N).samples


def get_piquasso_samples(unitary):
    m = len(unitary)
    n = m // 2

    input_state = generate_random_fock_state(m=m, n=n)
    program = pq.Program(
        instructions=[
            pq.StateVector(input_state),
            pq.Interferometer(unitary),
            pq.ParticleNumberMeasurement(),
        ]
    )

    simulator = pq.SamplingSimulator(d=m)

    return simulator.execute(program, shots=N).samples


if __name__ == "__main__":

    # Warmup
    m = 2
    unitary = pcvl.Matrix.random_unitary(m)
    get_perceval_samples(unitary)
    get_piquasso_samples(np.array(unitary))
    ####

    FILENAME = f"boson_sampling_{int(time.time())}.json"

    x = []
    pv_times = []
    pq_times = []
    pqb_times = []

    for m in range(2, 30):
        print("m:", m)
        x.append(m)
        unitary = pcvl.Matrix.random_unitary(m)

        start_time = time.time()
        get_perceval_samples(unitary)
        runtime = time.time() - start_time
        print("PV:", runtime)
        pv_times.append(runtime)

        start_time = time.time()
        get_piquassoboost_samples(np.array(unitary))
        runtime = time.time() - start_time
        print("PQB:", runtime)
        pqb_times.append(runtime)

        start_time = time.time()
        get_piquasso_samples(np.array(unitary))
        runtime = time.time() - start_time
        print("PQ:", runtime)
        pq_times.append(runtime)

        with open(FILENAME, "w") as f:
            json.dump(dict(x=x, pqb_times=pqb_times, pv_times=pv_times, pq_times=pq_times), f)

    plt.scatter(x, pv_times, label="Perceval")
    plt.scatter(x, pqb_times, label="PiquassoBoost")
    plt.scatter(x, pq_times, label="Piquasso")

    plt.yscale("log")

    plt.legend()
    plt.show()
