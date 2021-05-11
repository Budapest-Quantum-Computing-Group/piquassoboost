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


