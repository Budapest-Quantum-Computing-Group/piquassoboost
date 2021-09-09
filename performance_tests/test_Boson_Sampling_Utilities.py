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

import numpy as np

import time
from piquassoboost.sampling.Boson_Sampling_Utilities import ChinHuhPermanentCalculator

from typing import List, Optional

from numpy import complex128, ndarray
from numpy.linalg import svd
from scipy.special import binom


def generate_random_matrix( dim ):
    """ creading positive definite selfadjoint matrix A of dimension dim and with eigenvalues between 0 and 1 """

    A = np.complex128(np.random.random([dim, dim]))
    A_adjoint = A.conj().T

    P = A @ A_adjoint
    P += np.identity(len(P))

    P_inverse = np.linalg.inv(P)
    
    return P_inverse


# the python implementation class to calculate the permanent 
class ChinHuhPermanentCalculator_python:
    """
        This class is designed to calculate permanent of effective scattering matrix of a boson sampling instance.
        Note, that it can be used to calculate permanent of given matrix. All that is required that input and output
        states are set to [1, 1, ..., 1] with proper dimensions.
    """
    

    def __init__(self, matrix: ndarray, input_state: Optional[List[int]] = None,
                 output_state: Optional[List[int]] = None):
        if output_state is None:
            output_state = []
        if input_state is None:
            input_state = []

        self.__matrix = matrix
        self.__input_state = input_state
        self.__output_state = output_state

    @property
    def matrix(self) -> ndarray:
        return self.__matrix

    @matrix.setter
    def matrix(self, matrix) -> None:
        self.__matrix = matrix

    @property
    def input_state(self) -> List[int]:
        return self.__input_state

    @input_state.setter
    def input_state(self, input_state) -> None:
        self.__input_state = input_state

    @property
    def output_state(self) -> List[int]:
        return self.__output_state

    @output_state.setter
    def output_state(self, output_state) -> None:
        self.__output_state = output_state

    def calculate(self) -> complex128:
        """
            This is the main method of the calculator. Assuming that input state, output state and the matrix are
            defined correctly (that is we've got m x m matrix, and vectors of with length m) this calculates the
            permanent of an effective scattering matrix related to probability of obtaining output state from given
            input state.
            :return: Permanent of effective scattering matrix.
        """

        if not self.__can_calculation_be_performed():
            raise AttributeError


        v_vectors = self.__calculate_v_vectors()


        permanent = 0
        for v_vector in v_vectors:
            v_sum = sum(v_vector)
            addend = pow(-1, v_sum)
            # Binomials calculation
            for i in range(len(v_vector)):
                addend *= binom(self.__input_state[i], v_vector[i])

            # Product calculation
            product = 1
            for j in range(len(self.__input_state)):
                if self.__output_state[j] == 0:  # There's no reason to calculate the sum if t_j = 0
                    continue
                # Otherwise we calculate the sum
                product_part = 0
                for i in range(len(self.__input_state)):
                    product_part += (self.__input_state[i] - 2 * v_vector[i]) * self.__matrix[j][i]
                product_part = pow(product_part, self.__output_state[j])
                product *= product_part
            addend *= product
            permanent += addend

        permanent /= pow(2, sum(self.__input_state))       

        return permanent

    def __can_calculation_be_performed(self) -> bool:
        """
            Checks if calculation can be performed. For this to happen sizes of given matrix and states have
            to match.
            :return: Information if the calculation can be performed.
        """
        return self.__matrix.shape[0] == self.__matrix.shape[1] \
               and len(self.__output_state) == len(self.__input_state) \
               and len(self.__output_state) == self.__matrix.shape[0]

    def __calculate_v_vectors(self, input_vector: Optional[list] = None) -> List[List[int]]:
        if input_vector is None:
            input_vector = []
        v_vectors = []
        for i in range(self.__input_state[len(input_vector)] + 1):
            input_state = input_vector.copy()
            input_state.append(i)

            if len(input_state) == len(self.__input_state):
                v_vectors.append(input_state)
            else:
                v_vectors.extend(self.__calculate_v_vectors(input_state))

        return v_vectors


class TestBoson_Sampling_Utilities:


    def test_ChinHuhPermanentCalculator(self):


        # create inputs for the permanent calculations
        initial_state = [1, 2, 1, 0, 0]
        output_state = [1, 0, 1, 1, 1]

        permutation_matrix = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, complex(0.5,0.7), 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ], dtype=complex)

        #create an instance of python implementation to calculate the permanent
        permanent_calculator_python = ChinHuhPermanentCalculator_python( permutation_matrix, initial_state, output_state)

        # do iterations to calculate the permanent and choose the fastest one
        iter_loops = 100
        time_python = 10
        for idx in range(iter_loops):
            start = time.time()   
            permanent_python = permanent_calculator_python.calculate()
            time_loc = time.time() - start
            start = time.time()   
       
            if time_python > time_loc:
                time_python = time_loc

        # converting the states into np arrays for C++ compatibility
        initial_state = np.array(initial_state, dtype=np.int64)
        output_state = np.array(output_state, dtype=np.int64)


        #create an instance of C++ implementation to calculate the permanent
        permanent_calculator_Cpp = ChinHuhPermanentCalculator( permutation_matrix, initial_state, output_state)

        # do iterations to calculate the permanent and choose the fastest one
        time_Cpp = 10
        for idx in range(iter_loops):
            start = time.time()   
            permanent_Cpp = permanent_calculator_Cpp.calculate()

            time_loc = time.time() - start
            start = time.time()   
       
            if time_Cpp > time_loc:
                time_Cpp = time_loc

        print(' ')
        print('*******************************************')
        print('Time elapsed with python: ' + str(time_python))
        print('Time elapsed with C++: ' + str(time_Cpp))
        print( "speedup: " + str(time_python/time_Cpp) )
        print(' ')
        print(' ')

        print( 'Difference between python and C++ result: ' + str(abs(permanent_Cpp-permanent_python)))

        assert abs(permanent_Cpp-permanent_python) < 1e-13
