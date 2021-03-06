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
import numpy as np

from thewalrus import perm as sf_perm

from scipy.stats import unitary_group

from piquassoboost.sampling.Boson_Sampling_Utilities import (
    ChinHuhPermanentCalculator,
    GlynnPermanent,
    GlynnRepeatedPermanentCalculator,
    GlynnRepeatedPermanentCalculatorDouble,
    BBFGPermanentDouble,
    GlynnPermanentSimpleDouble,
    GlynnPermanentSimpleLongDouble,
)

from piquassoboost.sampling.permanent_calculators import (
    permanent_CPU_repeated_double,
    permanent_CPU_repeated_long_double
)

class TestPermanentCalculators:
    """bechmark tests for permanent calculator algorithms
    """

    def test_repeated_input(self):
        """Check input data handling in repeated Glynn calculator"""

        dim = 20
        A = unitary_group.rvs(dim)

       
        # multiplicities of input/output states
        input_state = np.ones(dim-1, np.int64)
        output_state = np.ones(dim, np.int64)

        try:
            # Glynn repeated permanent calculator
            permanent_Glynn_calculator = GlynnRepeatedPermanentCalculator( A, input_state=input_state, output_state=output_state )
            permanent_Glynn_calculator.calculate()
            assert(0)
        except: 
            assert(1)



        # multiplicities of input/output states
        input_state = np.ones(dim, np.int64)
        output_state = np.ones(dim, np.int64)

        try:
            # Glynn repeated permanent calculator
            permanent_Glynn_calculator = GlynnRepeatedPermanentCalculator( A, input_state=input_state, output_state=output_state )
            permanent_Glynn_Cpp_repeated = permanent_Glynn_calculator.calculate()
        except: 
            assert(0)




    def test_value_repeated(self):
        """Check permanent value calculated by C++ Glynn permanent calculator"""

        d = 15
        n = 10

        matrices = []
        for _ in range(n):
            matrices.append(unitary_group.rvs(d))
        

        pq_permanents_glynn = []
        pq_glynn_time = 0
        for i in range(n):
            matrix = matrices[i]
            input_state = [1] * d
            input_state[0] = 3
            input_state[4] = 2
            input_state[8] = 3
            input_state = np.array(input_state, dtype=np.int64)
            output_state = input_state

            permanent_calculator_pq_glynn = GlynnRepeatedPermanentCalculator(matrix, input_state=input_state, output_state=output_state)
            start = time.time()
            pq_permanent_glynn = permanent_calculator_pq_glynn.calculate()
            end = time.time()
            pq_glynn_time += end - start
            pq_permanents_glynn.append(pq_permanent_glynn)


        #print("permanent(matrix) by PQ Glynn:", pq_permanents_glynn)


        pq_permanents_chinhuh = []
        pq_chinhuh_time = 0
        for i in range(n):
            matrix = matrices[i]
            input_state = [1] * d
            input_state[0] = 3
            input_state[4] = 2
            input_state[8] = 3
            input_state = np.array(input_state, dtype=np.int64)
            output_state = input_state

            permanent_calculator_pq_chinhuh = ChinHuhPermanentCalculator(matrix, input_state, output_state)

            start = time.time()
            pq_permanent_chinhuh = permanent_calculator_pq_chinhuh.calculate()
            end = time.time()
            pq_chinhuh_time += end - start
            pq_permanents_chinhuh.append(pq_permanent_chinhuh)

        #print("permanent(matrix) by PQ ChinH:", pq_permanents_chinhuh)


        for i in range(n):
            pq_gl_value = pq_permanents_glynn[i]
            pq_ch_value = pq_permanents_chinhuh[i]

            assert( abs(pq_ch_value - pq_ch_value) < 1.e-9 )



    def test_value_two_dimensional(self):
        """Check permanent value calculated by C++ Glynn permanent calculator"""

        d = 15
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

        #print("permanents by SF      :", sf_permanents)
        
        
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


        #print("permanent(matrix) by PQ Glynn:", pq_permanents_glynn)

        
        pq_permanents_chinhuh = []
        pq_chinhuh_time = 0
        for i in range(n):
            matrix = matrices[i]
            input_state = [1] * d
            input_state = np.array(input_state, dtype=np.int64)
            output_state = input_state

            permanent_calculator_pq_chinhuh = ChinHuhPermanentCalculator(matrix, input_state, output_state)

            start = time.time()
            pq_permanent_chinhuh = permanent_calculator_pq_chinhuh.calculate()
            end = time.time()
            pq_chinhuh_time += end - start
            pq_permanents_chinhuh.append(pq_permanent_chinhuh)

        #print("permanent(matrix) by PQ ChinH:", pq_permanents_chinhuh)
        

        for i in range(n):
            sf_value = sf_permanents[i]
            pq_gl_value = pq_permanents_glynn[i]
            pq_ch_value = pq_permanents_chinhuh[i]

            assert( abs(sf_value - pq_gl_value) < 1.e-9 )
            assert( abs(sf_value - pq_ch_value) < 1.e-9 )

            
        print("time elapsed in SF          :", sf_time)
        print("time elapsed in PQ (Glynn)  :", pq_glynn_time)
        print("time elapsed in PQ (Chinhuh):", pq_chinhuh_time)
        



    """
        performance test for measuring the calculation precision
    """

    def test_calculator_value(self):
        """Check input data handling in Glynn calculators"""

        dim = 4
        A = unitary_group.rvs(dim)

        print("Matrix input:")
        print(A)
        

        value_from_double = 1.0 - 1.0j
        
        # BB/FG permanent calculator with double precision
        permanent_calculator_BBFG_double_precision = BBFGPermanentDouble( A )
        value_from_double = permanent_calculator_BBFG_double_precision.calculate()



        value_from_long_double = 1.0 - 1.0j

        # Glynn permanent calculator with long double precision
        permanent_calculator_Glynn_long_double_precision = GlynnPermanent( A )
        value_from_long_double = permanent_calculator_Glynn_long_double_precision.calculate()

        # Glynn simple permanent calculator with double precision
        permanent_calculator_Glynn_simple_double_precision = GlynnPermanentSimpleDouble( A )
        value_simple_double = permanent_calculator_Glynn_simple_double_precision.calculate()

        # Glynn simple permanent calculator with long double precision
        permanent_calculator_Glynn_simple_long_double_precision = GlynnPermanentSimpleLongDouble( A )
        value_simple_long_double = permanent_calculator_Glynn_simple_long_double_precision.calculate()



        # multiplicities of input/output states
        input_state = np.ones(dim, np.int64)
        output_state = np.ones(dim, np.int64)
        # repeated Glynn permanent calculator with double precision
        RepeatedGlynnDouble = GlynnRepeatedPermanentCalculatorDouble( A, input_state=input_state, output_state=output_state )
        rep_value_from_double = RepeatedGlynnDouble.calculate()
        rep_value_from_double2 = permanent_CPU_repeated_double( A, input_state, output_state )

        # repeated Glynn permanent calculator with long double precision
        RepeatedGlynnLongDouble = GlynnRepeatedPermanentCalculator( A, input_state=input_state, output_state=output_state )
        rep_value_from_long_double = RepeatedGlynnLongDouble.calculate()
        rep_value_from_long_double2 = permanent_CPU_repeated_long_double( A, input_state, output_state )


        print("value_from_double          :", value_from_double)
        print("value_from_long_double     :", value_from_long_double)
        print("value_simple_double        :", value_simple_double)
        print("value_simple_long_double   :", value_simple_long_double)
        print("rep_value_from_double      :", rep_value_from_double)
        print("rep_value_from_double2     :", rep_value_from_double2)
        print("rep_value_from_long_double :", rep_value_from_long_double)
        print("rep_value_from_long_double2:", rep_value_from_long_double2)
