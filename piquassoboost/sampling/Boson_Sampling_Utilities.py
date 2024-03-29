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
from .Boson_Sampling_Utilities_wrapper import GlynnPermanentCalculator_wrapper
from .Boson_Sampling_Utilities_wrapper import PowerTraceHafnian_wrapper
from .Boson_Sampling_Utilities_wrapper import PowerTraceHafnianRecursive_wrapper
from .Boson_Sampling_Utilities_wrapper import PowerTraceLoopHafnian_wrapper
from .Boson_Sampling_Utilities_wrapper import PowerTraceLoopHafnianRecursive_wrapper


class RepeatedPermanentCalculator(ChinHuhPermanentCalculator_wrapper):
    """
        This class is designed to calculate permanent of effective scattering matrix of a boson sampling instance.
        Note, that it can be used to calculate permanent of given matrix. All that is required that input and output
        states are set to [1, 1, ..., 1] with proper dimensions.
    """
    

    def __init__(self, lib, matrix, input_state, output_state):

        if (not (type(input_state) is np.ndarray)) :
            if (not isinstance(input_state, list)):
                input_state = np.array(input_state, dtype=np.int64)
            else:
                for item in input_state:
                    if (not (type(item) is np.ndarray)):
                        raise Exception('The input state should be a numpy array, or a list of numpy arrays')
                    
                
        if (not (type(output_state) is np.ndarray)) :
            if (not isinstance(output_state, list)):
                output_state = np.array(output_state, dtype=np.int64)
            else:
                for item in output_state:
                    if (not (type(item) is np.ndarray)):
                        raise Exception('The output state should be a numpy array, or a list of numpy arrays')



        # call the constructor of the wrapper class
        super(RepeatedPermanentCalculator, self).__init__(lib=lib, matrix=matrix, input_state=input_state, output_state=output_state)
        pass

       
    def calculate(self):
        """
            This is the main method of the calculator. Assuming that input state, output state and the matrix are
            defined correctly (that is we've got m x m matrix, and vectors of with length m) this calculates the
            permanent of an effective scattering matrix related to probability of obtaining output state from given
            input state.
            :return: Permanent of effective scattering matrix.
        """

#        if not self.__can_calculation_be_performed():
#            raise AttributeError


        # call the permanent calculator of the parent class
        return super(RepeatedPermanentCalculator, self).calculate()


    def __can_calculation_be_performed(self) -> bool:
        """
            Checks if calculation can be performed. For this to happen sizes of given matrix and states have
            to match.
            :return: Information if the calculation can be performed.
        """
        return self.matrix.shape[0] == self.matrix.shape[1] \
               and len(self.output_state) == len(self.input_state) \
               and len(self.output_state) == self.matrix.shape[0]






class ChinHuhPermanentCalculator(RepeatedPermanentCalculator):
    def __init__(self, matrix, input_state, output_state):
        super(ChinHuhPermanentCalculator, self).__init__(0, matrix, input_state, output_state)
        pass





class GlynnRepeatedPermanentCalculator(RepeatedPermanentCalculator):
    def __init__(self, matrix, input_state, output_state):
        """
            This class is designed to calculate the permanent of
            matrix using Glynn's algorithm
            (Balasubramanian-Bax-Franklin-Glynn (BBFG) formula)
            with long double precision and multipled rows or columns

            1 shall be equal to GlynnRep
        """
        super(GlynnRepeatedPermanentCalculator, self).__init__(1, matrix, input_state, output_state)
        pass



class GlynnRepeatedInfPermanentCalculator(RepeatedPermanentCalculator):
    def __init__(self, matrix, input_state, output_state):
        super(GlynnRepeatedInfPermanentCalculator, self).__init__(9, matrix, input_state, output_state)
        pass



class GlynnRepeatedSingleDFEPermanentCalculator(RepeatedPermanentCalculator):
    def __init__(self, matrix, input_state, output_state):
        super(GlynnRepeatedSingleDFEPermanentCalculator, self).__init__(2, matrix, input_state, output_state)
        pass







class GlynnRepeatedDualDFEPermanentCalculator(RepeatedPermanentCalculator):
    def __init__(self, matrix, input_state, output_state):
        super(GlynnRepeatedDualDFEPermanentCalculator, self).__init__(3, matrix, input_state, output_state)
        pass






class GlynnRepeatedMultiSingleDFEPermanentCalculator(RepeatedPermanentCalculator):
    def __init__(self, matrix, input_state, output_state):
        super(GlynnRepeatedMultiSingleDFEPermanentCalculator, self).__init__(4, matrix, input_state, output_state)
        pass







class GlynnRepeatedMultiDualDFEPermanentCalculator(RepeatedPermanentCalculator):
    def __init__(self, matrix, input_state, output_state):
        super(GlynnRepeatedMultiDualDFEPermanentCalculator, self).__init__(5, matrix, input_state, output_state)
        pass




class GlynnRepeatedPermanentCalculatorDouble(RepeatedPermanentCalculator):
    def __init__(self, matrix, input_state, output_state):
        """
            This class is designed to calculate the permanent of
            matrix using Glynn's algorithm
            (Balasubramanian-Bax-Franklin-Glynn (BBFG) formula)
            with double precision and multipled rows or columns
        """
        super(GlynnRepeatedPermanentCalculatorDouble, self).__init__(6, matrix, input_state, output_state)
        pass



class BBFGRepeatedPermanentCalculatorDouble(RepeatedPermanentCalculator):
    def __init__(self, matrix, input_state, output_state):
        """
            This class is designed to calculate the permanent of
            matrix using Glynn's algorithm
            (Balasubramanian-Bax-Franklin-Glynn (BBFG) formula)
            with double precision and multipled rows or columns
        """
        super(BBFGRepeatedPermanentCalculatorDouble, self).__init__(7, matrix, input_state, output_state)
        pass



class BBFGRepeatedPermanentCalculatorLongDouble(RepeatedPermanentCalculator):
    def __init__(self, matrix, input_state, output_state):
        """
            This class is designed to calculate the permanent of
            matrix using Glynn's algorithm
            (Balasubramanian-Bax-Franklin-Glynn (BBFG) formula)
            with double precision and multipled rows or columns
        """
        super(BBFGRepeatedPermanentCalculatorLongDouble, self).__init__(8, matrix, input_state, output_state)
        pass



class GlynnPermanent(GlynnPermanentCalculator_wrapper):
    """
        This class is designed to calculate the permanent of matrix using Glynn's algorithm (Balasubramanian-Bax-Franklin-Glynn (BBFG) formula)
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        super(GlynnPermanent, self).__init__(matrix, 0)
        pass

class GlynnPermanentInf(GlynnPermanentCalculator_wrapper):
    """
        This class is designed to calculate the permanent of matrix using Glynn's algorithm (Balasubramanian-Bax-Franklin-Glynn (BBFG) formula)
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        super(GlynnPermanentInf, self).__init__(matrix, 1)
        pass






class GlynnPermanentSingleDFE(GlynnPermanentCalculator_wrapper):
    """
        This class is designed to calculate the permanent of matrix using Glynn's algorithm (Balasubramanian-Bax-Franklin-Glynn (BBFG) formula)
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        super(GlynnPermanentSingleDFE, self).__init__(matrix, 2)
        pass






class GlynnPermanentDualDFE(GlynnPermanentCalculator_wrapper):
    """
        This class is designed to calculate the permanent of matrix using Glynn's algorithm (Balasubramanian-Bax-Franklin-Glynn (BBFG) formula)
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        super(GlynnPermanentDualDFE, self).__init__(matrix, 3)
        pass

       
class GlynnPermanentDouble(GlynnPermanentCalculator_wrapper):
    """
        This class is designed to calculate the permanent of matrix using Glynn's algorithm (Balasubramanian-Bax-Franklin-Glynn (BBFG) formula) with double precision
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        # 6 shall mean macro GlynnDoubleCPU
        super(GlynnPermanentDouble, self).__init__(matrix, 6)
        pass





class BBFGPermanentDouble(GlynnPermanentCalculator_wrapper):
    """
        This class is designed to calculate the permanent of matrix using Glynn's algorithm (Balasubramanian-Bax-Franklin-Glynn (BBFG) formula) with double precision
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        # 6 shall mean macro GlynnDoubleCPU
        super(BBFGPermanentDouble, self).__init__(matrix, 7)
        pass


class BBFGPermanentLongDouble(GlynnPermanentCalculator_wrapper):
    """
        This class is designed to calculate the permanent of matrix using Glynn's algorithm (Balasubramanian-Bax-Franklin-Glynn (BBFG) formula) with long double precision
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        # 6 shall mean macro GlynnDoubleCPU
        super(BBFGPermanentLongDouble, self).__init__(matrix, 8)
        pass


class GlynnPermanentSimpleDouble(GlynnPermanentCalculator_wrapper):
    """
        This class is designed to calculate the permanent of matrix using Glynn's algorithm (Balasubramanian-Bax-Franklin-Glynn (BBFG) formula) with double precision by simple algorithm
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        # 10 shall mean macro GlynnSimpleDouble
        super(GlynnPermanentSimpleDouble, self).__init__(matrix, 10)
        pass


class GlynnPermanentSimpleLongDouble(GlynnPermanentCalculator_wrapper):
    """
        This class is designed to calculate the permanent of matrix using Glynn's algorithm (Balasubramanian-Bax-Franklin-Glynn (BBFG) formula) with long double precision by simple algorithm
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        # 11 shall mean macro GlynnSimpleLongDouble
        super(GlynnPermanentSimpleLongDouble, self).__init__(matrix, 11)
        pass


class PowerTraceHafnian(PowerTraceHafnian_wrapper):
    """
        This class is designed to calculate the hafnian of a symetrix matrix using the power trace method.
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        super(PowerTraceHafnian, self).__init__(lib=1, matrix=matrix)
        pass

       
    def calculate(self):
        """
            ?????????????????.
            :return: The hafnian of the matrix.
        """



        # call the permanent calculator of the parent class
        return super(PowerTraceHafnian, self).calculate()



class PowerTraceHafnianLongDouble(PowerTraceHafnian_wrapper):
    """
        This class is designed to calculate the hafnian of a symetrix matrix using the power trace method.
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        super(PowerTraceHafnianLongDouble, self).__init__(lib=2, matrix=matrix)
        pass

       
    def calculate(self):
        """
            ?????????????????.
            :return: The hafnian of the matrix.
        """



        # call the permanent calculator of the parent class
        return super(PowerTraceHafnianLongDouble, self).calculate()




class PowerTraceHafnianInf(PowerTraceHafnian_wrapper):
    """
        This class is designed to calculate the hafnian of a symetrix matrix using the power trace method.
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        super(PowerTraceHafnianInf, self).__init__(lib=3, matrix=matrix)
        pass

       
    def calculate(self):
        """
            ?????????????????.
            :return: The hafnian of the matrix.
        """



        # call the permanent calculator of the parent class
        return super(PowerTraceHafnianInf, self).calculate()





class PowerTraceHafnianRecursive(PowerTraceHafnianRecursive_wrapper):
    """
        This class is designed to calculate the hafnian of a symetrix matrix using the power trace method.
    """
    

    def __init__(self, matrix, occupancy):

        # call the constructor of the wrapper class
        super(PowerTraceHafnianRecursive, self).__init__(lib=1, matrix=matrix, occupancy=occupancy)
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
        This class is designed to calculate the loop hafnian of a symetrix matrix using the power trace method calculated with double precision.
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        super(PowerTraceLoopHafnian, self).__init__(matrix=matrix, lib=1)
        pass

       
    def calculate(self):
        """
            ?????????????????.
            :return: The loop hafnian of the matrix.
        """



        # call the permanent calculator of the parent class
        return super(PowerTraceLoopHafnian, self).calculate()



class PowerTraceLoopHafnianLongDouble(PowerTraceLoopHafnian_wrapper):
    """
        This class is designed to calculate the loop hafnian of a symetrix matrix using the power trace method calculated with extended precision.
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        super(PowerTraceLoopHafnianLongDouble, self).__init__(lib=2, matrix=matrix )
        pass

       
    def calculate(self):
        """
            ?????????????????.
            :return: The loop hafnian of the matrix.
        """



        # call the permanent calculator of the parent class
        return super(PowerTraceLoopHafnianLongDouble, self).calculate()
        
        
        
class PowerTraceLoopHafnianInf(PowerTraceLoopHafnian_wrapper):
    """
        This class is designed to calculate the loop hafnian of a symetrix matrix using the power trace method calculated with infinite precision (using the MPFR library).
    """
    

    def __init__(self, matrix):

        # call the constructor of the wrapper class
        super(PowerTraceLoopHafnianInf, self).__init__(lib=3, matrix=matrix )
        pass

       
    def calculate(self):
        """
            ?????????????????.
            :return: The loop hafnian of the matrix.
        """



        # call the permanent calculator of the parent class
        return super(PowerTraceLoopHafnianInf, self).calculate()        



class PowerTraceLoopHafnianRecursive(PowerTraceLoopHafnianRecursive_wrapper):
    """
        This class is designed to calculate the loop hafnian of a symetrix matrix using the power trace method.
    """
    

    def __init__(self, matrix, occupancy):

        # call the constructor of the wrapper class
        super(PowerTraceLoopHafnianRecursive, self).__init__(matrix=matrix, occupancy=occupancy)
        pass

       
    def calculate(self):
        """
            ?????????????????.
            :return: The loop hafnian of the matrix.
        """



        # call the permanent calculator of the parent class
        return super(PowerTraceLoopHafnianRecursive, self).calculate()

