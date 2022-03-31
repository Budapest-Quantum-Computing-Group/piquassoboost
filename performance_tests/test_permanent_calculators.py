#
# Copyright (C) 2021 by TODO - All rights reserved.
#

import time
import pytest
import numpy as np

from thewalrus import perm as sf_perm

from scipy.stats import unitary_group

from piquassoboost.sampling.Boson_Sampling_Utilities import ChinHuhPermanentCalculator
from piquassoboost.sampling.Boson_Sampling_Utilities import GlynnPermanent

class TestPermanentCalculators:
    """bechmark tests for permanent calculator algorithms
    """

    def test_repeated_input(self):
        """Check input data handling in repeated Glynn calculator"""

        dim = 20
        A = unitary_group.rvs(dim)

        # multiplicities of input/output states
        input_state = np.ones(dim, np.int64)
        output_state = None

        try:
            # Glynn repeated permanent calculator
            permanent_Glynn_calculator = GlynnPermanent( A, input_state=input_state, output_state=output_state )
            permanent_Glynn_calculator.calculate()
            assert(0)
        except: 
            assert(1)


        # multiplicities of input/output states
        input_state = np.ones(dim, np.int64)
        output_state = None

        try:
            # Glynn repeated permanent calculator
            permanent_Glynn_calculator = GlynnPermanent( A, input_state=input_state, output_state=output_state )
            permanent_Glynn_calculator.output_state = np.ones(dim, np.int64)
            permanent_Glynn_calculator.calculate()
        except: 
            assert(0)


        # multiplicities of input/output states
        input_state = None
        output_state = np.ones(dim, np.int64)

        try:
            # Glynn repeated permanent calculator
            permanent_Glynn_calculator = GlynnPermanent( A, input_state=input_state, output_state=output_state )
            permanent_Glynn_calculator.calculate()
            assert(0)
        except: 
            assert(1)



        # multiplicities of input/output states
        input_state = None
        output_state = np.ones(dim, np.int64)

        try:
            # Glynn repeated permanent calculator
            permanent_Glynn_calculator = GlynnPermanent( A, input_state=input_state, output_state=output_state )
            permanent_Glynn_calculator.input_state = np.ones(dim, np.int64)
            permanent_Glynn_calculator.calculate()
        except: 
            assert(0)



        # multiplicities of input/output states
        input_state = np.ones(dim-1, np.int64)
        output_state = np.ones(dim, np.int64)

        try:
            # Glynn repeated permanent calculator
            permanent_Glynn_calculator = GlynnPermanent( A, input_state=input_state, output_state=output_state )
            permanent_Glynn_calculator.calculate()
            assert(0)
        except: 
            assert(1)



        # multiplicities of input/output states
        input_state = np.ones(dim, np.int64)
        output_state = np.ones(dim, np.int64)

        try:
            # Glynn repeated permanent calculator
            permanent_Glynn_calculator = GlynnPermanent( A, input_state=input_state, output_state=output_state )
            permanent_Glynn_Cpp_repeated = permanent_Glynn_calculator.calculate()
        except: 
            assert(0)


        # multiplicities of input/output states
        input_state = None
        output_state = None

        try:
            # Glynn repeated permanent calculator
            permanent_Glynn_calculator = GlynnPermanent( A, input_state=input_state, output_state=output_state )
            permanent_Glynn_Cpp = permanent_Glynn_calculator.calculate()
        except: 
            assert(0)


        assert( abs(permanent_Glynn_Cpp_repeated - permanent_Glynn_Cpp) < 1e-6 )


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
        
