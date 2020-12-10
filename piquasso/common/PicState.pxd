from libcpp cimport bool #import bool type 
from numpy cimport int64_t

cdef extern from "source/PicState.h" namespace "pic":


    cdef cppclass PicState_int64:
        size_t rows
        size_t cols
        int64_t* data
        bool conjugated
        bool transposed
        bool owner

        PicState_int64() except +
        PicState_int64( int64_t* data_in, size_t cols_in) except +
        bool is_conjugated()
        void conjugate()
        bool is_transposed()
        void transpose()
        int64_t* get_data();
