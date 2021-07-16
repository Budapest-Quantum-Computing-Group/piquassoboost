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
import sys
import json
import time
import random
import itertools
import numpy as np
import networkx as nx

import piquasso as pq
from piquassoboost import patch

import strawberryfields as sf
from strawberryfields.apps import sample, subgraph


patch()


SHOTS = 100
ITERATIONS = 1

TIME_LIMIT = 10

NUMBER_OF_VERTICES = range(6, 30)  # NOTE: The range must start with 6 at least.

MEAN_PHOTON_NUMBER = 0.7
EDGE_CREATION_PROBABILITY = 0.8
EDGE_CREATION_PROBABILITY_DENSE = 0.9
EDGE_CREATION_PROBABILITY_SPARSE = 0.5

LOGARITHMIZE = True
# Comparing the results of SF and PQ: the dense subgraph density shoud be equal. This can be done at many shots.
COMPARING_RESULTS = True and SHOTS >= 5000
# Postprocessing all of the possible subgraph sizes or just the relevant part of them (relevant: d/3 - 2*d/3).
WIDE_RANGE = True


def range_for_postprocessing(d):
    if WIDE_RANGE:
        return 1, d-1

    sparse_d = int(2 * d / 3)
    dense_d = d - sparse_d
    return dense_d, sparse_d


def _is_invertible(matrix):
    return np.linalg.matrix_rank(matrix) == matrix.shape[0]


def _generate_adjacency_matrix_third_dense(d):
    sparse_d = int(2 * d / 3)
    dense_d = d - sparse_d
    dense_graph = nx.erdos_renyi_graph(n=dense_d, p=EDGE_CREATION_PROBABILITY_DENSE)
    sparse_graph = nx.erdos_renyi_graph(n=sparse_d, p=EDGE_CREATION_PROBABILITY_SPARSE)

    graph = nx.disjoint_union(sparse_graph, dense_graph)

    for i in range(int(dense_d / 2) + 1):
        u = random.randint(0,sparse_d-1)
        v = random.randint(sparse_d,sparse_d + dense_d-1)
        graph.add_edge(u,v)

    adjacency_matrix = nx.adjacency_matrix(graph).toarray()
    if not _is_invertible(adjacency_matrix):
        adjacency_matrix = _generate_adjacency_matrix_third_dense(d)

    return adjacency_matrix


def postprocess(samples, adj, minimal_subgraph_size, maximal_subgraph_size):
    postselected = sample.postselect(
        samples,
        minimal_subgraph_size,
        maximal_subgraph_size
    )
    pl_graph = nx.to_networkx_graph(adj)
    samples = sample.to_subgraphs(postselected, pl_graph)

    dense = subgraph.search(
        samples,
        pl_graph,
        minimal_subgraph_size,
        maximal_subgraph_size,
        max_count=3,
    )

    return dense


def pq_graph_simulation(adj, mean_photon_number=MEAN_PHOTON_NUMBER, shots=SHOTS ):
    with pq.Program() as pq_program:
        pq.Q() | pq.GaussianState(d=len(adj))

        pq.Q() | pq.Graph(
            adj,
            mean_photon_number=mean_photon_number,
        )
        pq.Q(all) | pq.ThresholdMeasurement(shots=shots)


    start_time = time.time()
    results= pq_program.execute()
    end_time = time.time()

    duration = end_time - start_time

    return results, duration



def pq_subgraph(d = 10, shots = 1, adj = None):
    if adj is None:
        adj = _generate_adjacency_matrix_third_dense(d)
    else:
        d = len(adj)

    results, duration = pq_graph_simulation(adj, MEAN_PHOTON_NUMBER, shots)

    minimal_subgraph_size, maximal_subgraph_size = range_for_postprocessing(d)

    pq_samples = results[0].samples
    dense = postprocess(pq_samples, adj, minimal_subgraph_size, maximal_subgraph_size)

    return dense, duration



def sf_subgraph(d = 10, shots = 1, adj = None):
    if adj is None:
        adj = _generate_adjacency_matrix_third_dense(d)
    else:
        d = len(adj)

    sf_program = sf.Program(d)
    sf_engine = sf.Engine(backend="gaussian")

    with sf_program.context as q:
        sf.ops.GraphEmbed(
            adj,
            MEAN_PHOTON_NUMBER
        ) | tuple([q[i] for i in range(d)])

        sf.ops.MeasureThreshold() | tuple([q[i] for i in range(d)])

    start_time = time.time()
    results = sf_engine.run(sf_program, shots=shots)
    duration = time.time() - start_time

    sf_samples = results.samples
    minimal_subgraph_size, maximal_subgraph_size = range_for_postprocessing(d)
    dense = postprocess(sf_samples, adj, minimal_subgraph_size, maximal_subgraph_size)

    return dense, duration


def benchmark():
    params = {
        "shots": SHOTS,
        "iterations": ITERATIONS,
        "mean_photon_number": MEAN_PHOTON_NUMBER,
        "number_of_vertices": list(NUMBER_OF_VERTICES),
        "limit_vertices_pq": max(list(NUMBER_OF_VERTICES))+1,
        "limit_vertices_sf": max(list(NUMBER_OF_VERTICES))+1,
        "edge_creation_probability_dense": EDGE_CREATION_PROBABILITY_DENSE,
        "edge_creation_probability_sparse": EDGE_CREATION_PROBABILITY_SPARSE,
        "pq_averages": [],
        "sf_averages": [],
        "time_limit": TIME_LIMIT,
    }

    results_pq = None
    results_sf = None
    for d in params["number_of_vertices"]:
        for _ in itertools.repeat(None, ITERATIONS):

            pq_times = []
            sf_times = []
            if d < params['limit_vertices_pq'] or d < params['limit_vertices_sf']:
                adj = _generate_adjacency_matrix_third_dense(d)

            if d < params['limit_vertices_pq']:
                results_pq, duration = pq_subgraph(shots=params['shots'], adj=adj)
                pq_times.append(duration)

            if d < params['limit_vertices_sf']:
                results_sf, duration = sf_subgraph(shots=params['shots'], adj=adj)
                sf_times.append(duration)

            if (
                COMPARING_RESULTS
                and d < params['limit_vertices_pq']
                and d < params['limit_vertices_sf']
            ):
                min_size = min(results_sf)
                max_size = max(results_sf)
                for key in range(min_size, max_size+1):
                    diff = abs(results_pq[key][0][0] - results_sf[key][0][0])
                    if (diff > 0.00000001):
                        print("ERROR")
                        sys.exit(1)

        if d < params['limit_vertices_pq']:
            pq_average_time = sum(pq_times) / len(pq_times)
            params["pq_averages"].append(pq_average_time)

            if pq_average_time > params["time_limit"]:
                params["limit_vertices_pq"] = d

        if d < params['limit_vertices_sf']:
            sf_average_time = sum(sf_times) / len(sf_times)
            params["sf_averages"].append(sf_average_time)

            if sf_average_time > params["time_limit"]:
                params["limit_vertices_sf"] = d

    file_prefix = f"{os.path.basename(__file__)}_{int(time.time())}"

    json_filename = f"{file_prefix}.json"

    with open(json_filename, "w") as f:
        json.dump(params, f)

if __name__ == "__main__":
    benchmark()


