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

import numpy as np

from .Boson_Sampling_Utilities_wrapper import ChinHuhPermanentCalculator_wrapper
from .Boson_Sampling_Utilities_wrapper import GlynnPermanentCalculator_wrapper
from .Boson_Sampling_Utilities_wrapper import PowerTraceHafnian_wrapper
from .Boson_Sampling_Utilities_wrapper import PowerTraceHafnianRecursive_wrapper
from .Boson_Sampling_Utilities_wrapper import PowerTraceLoopHafnian_wrapper
from .Boson_Sampling_Utilities_wrapper import PowerTraceLoopHafnianRecursive_wrapper


class ChinHuhPermanentCalculator(ChinHuhPermanentCalculator_wrapper):
    """
    This class is designed to calculate permanent of effective scattering matrix of a
    boson sampling instance. Note, that it can be used to calculate permanent of given
    matrix. All that is required that input and output states are set to [1, 1, ..., 1]
    with proper dimensions.

    Assuming that input state, output state and the matrix are defined correctly (that
    is we've got m x m matrix, and vectors of with length m) this calculates the
    permanent of an effective scattering matrix related to probability of obtaining
    output state from given input state.
    """

    # NOTE: This is needed because we use different names in `piquassoboost` and
    # in `theboss`.
    compute_permanent = ChinHuhPermanentCalculator_wrapper.calculate

    def __init__(self, matrix, input_state, output_state):

        # input/output states should be numpy arrays
        input_state = np.asarray(input_state, dtype=np.int64)
        output_state = np.asarray(output_state, dtype=np.int64)

        # call the constructor of the wrapper class
        super().__init__(
            matrix=matrix,
            input_state=input_state,
            output_state=output_state
        )

    def __can_calculation_be_performed(self) -> bool:
        """
            Checks if calculation can be performed. For this to happen sizes of given matrix and states have
            to match.
            :return: Information if the calculation can be performed.
        """
        return self.matrix.shape[0] == self.matrix.shape[1] \
               and len(self.output_state) == len(self.input_state) \
               and len(self.output_state) == self.matrix.shape[0]


class GlynnPermanent(GlynnPermanentCalculator_wrapper):
    """
        This class is designed to calculate the permanent of matrix using Glynn's algorithm (Balasubramanian-Bax-Franklin
Glynn (BBFG) formula)
        :DFE: set 0 to use CPU implementation, set 1 to use single DFE implementation, set 2 to use dual DFE implementation
        :precision: set 1 (default) to use long double precision, set 2 to use infinite computational precision using the GNU MPFR library. Has no effect if DFE>0 is set.
    """
    def __init__(self, matrix, input_state=None, output_state=None, DFE=0, precision=1):
        # call the constructor of the wrapper class

        if (input_state is None)  and (output_state is None):
            super().__init__(matrix=matrix, DFE=DFE, precision=precision)

        elif (input_state is None):

            # input/output states should be numpy arrays
            output_state = np.asarray(output_state, dtype=np.int64)
            super().__init__(matrix=matrix, output_state=output_state, DFE=DFE, precision=precision)

        elif (output_state is None):

            # input/output states should be numpy arrays
            input_state = np.asarray(input_state, dtype=np.int64)
            super().__init__(matrix=matrix, input_state=input_state, DFE=DFE, precision=precision)

        else:

            # input/output states should be numpy arrays
            input_state = np.asarray(input_state, dtype=np.int64)
            output_state = np.asarray(output_state, dtype=np.int64)
            super().__init__(matrix=matrix, input_state=input_state, output_state=output_state, DFE=DFE, precision=precision)



    def calculate(self):
        """
            Run the permanent calculation process according to the initialized values in the constructor
            :return: The permanent of the matrix.
        """
        # call the permanent calculator of the parent class
        return super().calculate()




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
            Run the hafnian calculation process according to the initialized values in the constructor
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
