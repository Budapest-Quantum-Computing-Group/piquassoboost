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

__author__ = 'Tomasz Rybotycki', 'Peter Rakyta'

from typing import List, Optional

import numpy as np

from .Boson_Sampling_Utilities_wrapper import ChinHuhPermanentCalculator_wrapper
from .Boson_Sampling_Utilities_wrapper import PowerTraceHafnian_wrapper
from .Boson_Sampling_Utilities_wrapper import PowerTraceHafnianRecursive_wrapper
from .Boson_Sampling_Utilities_wrapper import PowerTraceLoopHafnian_wrapper
from .Boson_Sampling_Utilities_wrapper import PowerTraceLoopHafnianRecursive_wrapper
from .Boson_Sampling_Utilities_wrapper import Torontonian_wrapper
from .Boson_Sampling_Utilities_wrapper import TorontonianRecursive_wrapper


class ChinHuhPermanentCalculator(ChinHuhPermanentCalculator_wrapper):
    """
        This class is designed to calculate permanent of effective scattering matrix of a boson sampling instance.
        Note, that it can be used to calculate permanent of given matrix. All that is required that input and output
        states are set to [1, 1, ..., 1] with proper dimensions.
    """
    

    def __init__(self, matrix, input_state, output_state):

        # call the constructor of the wrapper class
        super(ChinHuhPermanentCalculator, self).__init__(matrix=matrix, input_state=input_state, output_state=output_state)
        pass

       
    def calculate(self):
        """
            This is the main method of the calculator. Assuming that input state, output state and the matrix are
            defined correctly (that is we've got m x m matrix, and vectors of with length m) this calculates the
            permanent of an effective scattering matrix related to probability of obtaining output state from given
            input state.
            :return: Permanent of effective scattering matrix.
        """

        if not self.__can_calculation_be_performed():
            raise AttributeError


        # call the permanent calculator of the parent class
        return super(ChinHuhPermanentCalculator, self).calculate()


    def __can_calculation_be_performed(self) -> bool:
        """
            Checks if calculation can be performed. For this to happen sizes of given matrix and states have
            to match.
            :return: Information if the calculation can be performed.
        """
        return self.matrix.shape[0] == self.matrix.shape[1] \
               and len(self.output_state) == len(self.input_state) \
               and len(self.output_state) == self.matrix.shape[0]


class PowerTraceHafnian(PowerTraceHafnian_wrapper):
    """
        This class is designed to calculate the hafnian of a symetrix matrix using the power trace method.
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        super(PowerTraceHafnian, self).__init__(matrix=matrix)
        pass

       
    def calculate(self):
        """
            ?????????????????.
            :return: The hafnian of the matrix.
        """



        # call the permanent calculator of the parent class
        return super(PowerTraceHafnian, self).calculate()





class PowerTraceHafnianRecursive(PowerTraceHafnianRecursive_wrapper):
    """
        This class is designed to calculate the hafnian of a symetrix matrix using the power trace method.
    """
    

    def __init__(self, matrix, occupancy):

        # call the constructor of the wrapper class
        super(PowerTraceHafnianRecursive, self).__init__(matrix=matrix, occupancy=occupancy)
        pass

       
    def calculate(self):
        """
            ?????????????????.
            :return: The hafnian of the matrix.
        """



        # call the permanent calculator of the parent class
        return super(PowerTraceHafnianRecursive, self).calculate()



class PowerTraceLoopHafnian(PowerTraceLoopHafnian_wrapper):
    """
        This class is designed to calculate the loop hafnian of a symetrix matrix using the power trace method.
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        super(PowerTraceLoopHafnian, self).__init__(matrix=matrix)
        pass

       
    def calculate(self):
        """
            ?????????????????.
            :return: The hafnian of the matrix.
        """



        # call the permanent calculator of the parent class
        return super(PowerTraceLoopHafnian, self).calculate()



class PowerTraceLoopHafnianRecursive(PowerTraceLoopHafnianRecursive_wrapper):
    """
        This class is designed to calculate the hafnian of a symetrix matrix using the power trace method.
    """
    

    def __init__(self, matrix, occupancy):

        # call the constructor of the wrapper class
        super(PowerTraceLoopHafnianRecursive, self).__init__(matrix=matrix, occupancy=occupancy)
        pass

       
    def calculate(self):
        """
            ?????????????????.
            :return: The hafnian of the matrix.
        """



        # call the permanent calculator of the parent class
        return super(PowerTraceLoopHafnianRecursive, self).calculate()


class Torontonian(Torontonian_wrapper):
    """
        This class is designed to calculate the torontonian of a selfadjoint positive definite matrix with eigenvalues between 0 and 1.
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        super(Torontonian, self).__init__(matrix=matrix)
        pass

       
    def calculate(self):
        """
            :return: The Torontonian of the matrix.
        """

        # call the torontonian calculator of the parent class
        return super(Torontonian, self).calculate()





class TorontonianRecursive(TorontonianRecursive_wrapper):
    """
        This class is designed to calculate the torontonian using the recursive algorithm of a selfadjoint positive definite matrix with eigenvalues between 0 and 1.
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        super(TorontonianRecursive, self).__init__(matrix=matrix)
        pass

       
    def calculate(self, use_extended=False):
        """
            :return: The Torontonian of the matrix.
        """

        # call the torontonian calculator of the parent class
        return super(TorontonianRecursive, self).calculate(use_extended=use_extended)
