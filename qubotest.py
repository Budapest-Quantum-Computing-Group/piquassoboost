
from qubo_divided import solve_qubo_by_dividing


import numpy as np

import itertools


d = 5

Q = 2 * np.random.rand(d, d) - 1


min_energy, min_bistring = solve_qubo_by_dividing(
    Q,
    learning_rate=0.1,
    shots=100,
    updates=1000,
    d_per_computer=d,
)

print("Algo:", min_energy, min_bistring)


def calculate_QUBO_explicitely(Q):
    d = len(Q)

    bitstrings = list(map(np.array, list(itertools.product([0, 1], repeat=d))))

    values = []

    for bitstring in bitstrings:
        values.append(bitstring @ Q @ bitstring)

    return min(values), bitstrings[np.argmin(values)]


print("Exact solution:", calculate_QUBO_explicitely(Q))
