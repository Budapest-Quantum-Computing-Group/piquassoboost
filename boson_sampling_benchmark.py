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

import time
import pytest
import itertools

import numpy as np

import piquasso as pq
import piquassoboost as pqb

try:
    from mpi4py import MPI
    MPI_imported = True
except ModuleNotFoundError:
    MPI_imported = False


from scipy.stats import unitary_group
import random

if MPI_imported:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

def print_histogram(samples):
    hist = dict()

    for sample in samples:
        key = tuple(sample)
        if key in hist.keys():
            hist[key] += 1
        else:
            hist[key] = 1

    for key in hist.keys():
        print(f"{key}: {hist[key]}")

    return


dim = 72
photon_number = 36

# generate random matrix
U = unitary_group.rvs(dim)#generate_random_unitary(dim)

# generate random input state
input_state = np.zeros(dim)
list_of_indices = [index for index in range(dim)]
for photon in range(photon_number):
    rand_int = random.randint(0, len(list_of_indices)-1)
    input_state[list_of_indices.pop(rand_int)] = 1

shots = 100

if MPI_imported:
    [input_state, U, shots] = comm.bcast([input_state, U, shots], root=0)


print('input state:')
print(input_state)

print( sum(sum(U)))



with pq.Program() as program:
    pq.Q() | pq.StateVector(input_state)
    pq.Q() | pq.Interferometer(U)

    pq.Q() | pq.ParticleNumberMeasurement()

simulator = pqb.BoostedSamplingSimulator(d=dim)

t0 = time.time()

result = simulator.execute(program=program, shots=shots)
print("C++ time elapsed:", time.time() - t0, "s")

#print_histogram(result.samples)


