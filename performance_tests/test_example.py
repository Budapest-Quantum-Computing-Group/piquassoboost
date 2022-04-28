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

#from performance_tests.test_wrappers.test_example_wrapper import wrapper_for_test_example


class Test_Example:
    """This is an example class to demonstrate how to interface with a C++ part of the piquasso project."""

    def qtest_example(self):
        r"""
        This method is called by pytest. 
        The caal is forwarded to a cython class to test some functionalities of a C++ class. 

        """
        
        # creating an instance of wrapping class to call the test function by pytest
        wrapper_for_test_example_instance = wrapper_for_test_example()

        # call the test function in the wrapper class
        wrapper_for_test_example_instance.some_wrapped_test_function()

