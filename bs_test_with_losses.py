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

try:
    from mpi4py import MPI
    MPI_imported = True
except ModuleNotFoundError:
    MPI_imported = False


if __name__ == '__main__':    


    import numpy as np

    import piquasso as pq
    import piquassoboost as pqb

    from piquassoboost.config import BoostConfig

    from scipy.special import binom


    from scipy.stats import unitary_group
    import random

    from math import sqrt

    run_uniform_loss = True
    run_approximate = True
    run_original_bs = True

    dim = 10
    photon_number = dim

    shots = 100
    _loss_probabilities = [0.8] * dim #[0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

    _loss_probabilities_len = len(_loss_probabilities)
    _loss_probabilities = _loss_probabilities + [1.0] * (dim - _loss_probabilities_len)

    _transmittance_values = [sqrt(loss_probability) for loss_probability in _loss_probabilities]

    # generate random matrix
    U = unitary_group.rvs(dim)

    """
    U = np.array([[-0.03363698-0.03579313j, -0.26051209-0.26545676j,
         0.47036895+0.06177485j, -0.38481035+0.07869226j,
         0.66095287-0.20750976j],
       [-0.38603005-0.54239288j, -0.16588722+0.15671419j,
         0.39594098+0.36173318j, -0.14248361+0.02684226j,
        -0.44112776+0.03843068j],
       [-0.10511587-0.14308881j,  0.32863915-0.80969798j,
         0.16833111-0.2320513j ,  0.18125927+0.0211107j ,
        -0.25357129-0.15836539j],
       [ 0.46180097-0.12768244j, -0.17856577-0.04076248j,
         0.27839407+0.06418562j,  0.44054666+0.56488985j,
         0.03284234+0.37550789j],
       [ 0.53821774+0.06010947j, -0.01486077+0.11047893j,
         0.46599365-0.32612953j, -0.24475134-0.46724804j,
        -0.29084286+0.08920772j]])
    """
    input_state = np.ones(dim)
    # generate random input state
    #input_state = np.zeros(dim)
    #list_of_indices = list(range(dim))
    #for _ in range(photon_number):
    #    rand_int = random.randint(0, len(list_of_indices)-1)
    #    input_state[list_of_indices.pop(rand_int)] = 1


    with pq.Program() as program:
        pq.Q() | pq.StateVector(input_state)

        pq.Q() | pq.Interferometer(U)
        pq.Q() | pq.Loss(transmissivity=_transmittance_values)

        pq.Q() | pq.ParticleNumberMeasurement()


    with pq.Program() as program2:
        pq.Q() | pq.StateVector(input_state)

        pq.Q() | pq.Interferometer(U)

        pq.Q() | pq.ParticleNumberMeasurement()


    boost_config = BoostConfig()



    # create samples
    if run_approximate:
        config1 = BoostConfig()
        config1.number_of_approximated_modes = 7

        simulator1 = pqb.BoostedSamplingSimulator(d=dim, config=config1)

        result = simulator1.execute(program=program, shots=shots)
        print(result.samples)
        print("sum of result1:", sum([sum(sample) for sample in result.samples]))
        
    if run_uniform_loss:
        simulator2 = pqb.BoostedSamplingSimulator(d=dim, config=boost_config)

        result2 = simulator2.execute(program=program, shots=shots)
        print(result2.samples)
        print("sum of result2:", sum([sum(sample) for sample in result2.samples]))
    
    if run_original_bs:
        simulator3 = pqb.BoostedSamplingSimulator(d=dim, config=boost_config)

        result3 = simulator3.execute(program=program2, shots=shots)
        print(result3.samples)
        print("sum of result3:", sum([sum(sample) for sample in result3.samples]))




    """
    def tomek_function(photon_number, _transmissivity, i):
        n = photon_number
        eta = _transmissivity
        l = i

        return binom(n, l) * pow(eta, l) * pow(1 - eta, n - l)

    for i,n in enumerate(numbers):
        print(
            "{:2d}".format(i),
            "{:1.4f}".format(tomek_function(photon_number, _transmissivity, i)),
            "{:1.4f}".format(float(n)/shots),
        sep="  ")
    """
