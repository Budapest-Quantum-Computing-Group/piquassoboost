#
# Copyright 2021-2022 Budapest Quantum Computing Group
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

import time
import pytest


from thewalrus import perm as sf_perm

from scipy.stats import unitary_group

from piquassoboost.sampling.Boson_Sampling_Utilities import ChinHuhPermanentCalculator
from piquassoboost.sampling.Boson_Sampling_Utilities import GlynnPermanent

class TestPermanentCalculators:
    """bechmark tests for permanent calculator algorithms
    """

    def test_value_two_dimensional(self):
        """Check permanent value calculated by C++ Glynn permanent calculator"""

        d = 10
        n = 10

        matrices = []
        for _ in range(n):
            matrices.append(unitary_group.rvs(d))

        sf_permanents = []
        sf_time = 0
        for i in range(n):
            matrix = matrices[i]
            start = time.time()
            sf_permanent = sf_perm(matrix)
            end = time.time()
            sf_time += end - start
            sf_permanents.append(sf_permanent)

        print("permanents by SF      :", sf_permanents)
        

        pq_permanents_glynn = []
        pq_glynn_time = 0
        for i in range(n):
            matrix = matrices[i]
            permanent_calculator_pq_glynn = GlynnPermanent(matrix)
            start = time.time()
            pq_permanent_glynn = permanent_calculator_pq_glynn.calculate()
            end = time.time()
            pq_glynn_time += end - start
            pq_permanents_glynn.append(pq_permanent_glynn)


        print("permanent(matrix) by PQ Glynn:", pq_permanents_glynn)


        pq_permanents_chinhuh = []
        pq_chinhuh_time = 0
        for i in range(n):
            matrix = matrices[i]
            input_state = [1] * d
            output_state = input_state

            permanent_calculator_pq_chinhuh = ChinHuhPermanentCalculator(matrix, input_state, output_state)

            start = time.time()
            pq_permanent_chinhuh = permanent_calculator_pq_chinhuh.calculate()
            end = time.time()
            pq_chinhuh_time += end - start
            pq_permanents_chinhuh.append(pq_permanent_chinhuh)

        print("permanent(matrix) by PQ ChinH:", pq_permanents_chinhuh)


        for i in range(n):
            sf_value = sf_permanents[i]
            pq_gl_value = pq_permanents_glynn[i]
            pq_ch_value = pq_permanents_chinhuh[i]

            if ((abs(sf_value - pq_gl_value) > 1.e-9) and (abs(sf_value - pq_ch_value) > 1.e-9)):
                print("permanent values:", sf_value, pq_gl_value, pq_ch_value)

            
        
        print("time elapsed in SF          :", sf_time)
        print("time elapsed in PQ (Glynn)  :", pq_glynn_time)
        print("time elapsed in PQ (Chinhuh):", pq_chinhuh_time)
        
