#from performance_tests.test_wrappers.test_matrix_wrapper import wrapper_for_test_matrix


class Test_Example:
    """This is an example class to demonstrate how to interface with a C++ part of the piquasso project."""

    def xtest_transpose(self):
        r"""
        This method is called by pytest. 
        The caal is forwarded to a cython class to test some functionalities of a C++ class. 

        """
        
        # creating an instance of wrapping class to call the test function by pytest
        wrapper_for_matrix_instance = wrapper_for_test_matrix()

        # call the test function in the wrapper class
        wrapper_for_matrix_instance.transpose_test_function()


    def xtest_dote(self):
        r"""
        This method test the dot product of two matrices. 
        The call is forwarded to a cython class to test some functionalities of a C++ class. 

        """
        
        # creating an instance of wrapping class to call the test function by pytest
        wrapper_for_matrix_instance = wrapper_for_test_matrix( matrix_size = 16 )

        # call the test function A*B
        wrapper_for_matrix_instance.dot_test_function()

        # call the test function A * B^*
        wrapper_for_matrix_instance.dot_test_function5()

        # call the test function A^T * B
        wrapper_for_matrix_instance.dot_test_function2()

        # call the test function A^+ * B
        wrapper_for_matrix_instance.dot_test_function3()

        # call the test function A^* * B
        wrapper_for_matrix_instance.dot_test_function4()

        # call the test function A * B^T
        wrapper_for_matrix_instance.dot_test_function6()



