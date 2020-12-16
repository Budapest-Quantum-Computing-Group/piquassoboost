from libcpp.vector cimport vector
from piquasso.common.matrix cimport matrix 
cimport piquasso.common.tbb as tbb
from piquasso.gaussian.source.tasks_apply_to_C_and_G cimport Extract_Rows

cdef extern from "tasks_apply_to_C_and_G.h" namespace "pic":

    cdef cppclass Extract_Rows:
        Extract_Rows() except +
        Extract_Rows( matrix &mtx_in, matrix &rows_out, vector[size_t] &modes_in ) except +
        const tbb.continue_msg& operator()(tbb.continue_msg &msg)

    

