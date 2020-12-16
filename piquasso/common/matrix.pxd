from libcpp cimport bool #import bool type 
from numpy cimport int64_t

cdef extern from "source/matrix.h" namespace "pic":

    cdef cppclass matrix:
        size_t rows
        size_t cols
        double complex* data
        bool conjugated
        bool transposed
        bool owner

        matrix() except +
        matrix( double complex* data_in, size_t rows_in, size_t cols_in) except +
        matrix( size_t rows_in, size_t cols_in) except +
        bool is_conjugated()
        void conjugate()
        bool is_transposed()
        void transpose()
        double complex* get_data()
        void set_owner( bool owner_in)


    cdef cppclass matrix_int64:
        size_t rows
        size_t cols
        int64_t* data
        bool conjugated
        bool transposed
        bool owner

        matrix_int64() except +
        matrix_int64( int64_t* data_in, size_t rows_in, size_t cols_in) except +
        matrix_int64( size_t rows_in, size_t cols_in) except +
        bool is_conjugated()
        void conjugate()
        bool is_transposed()
        void transpose()
        int64_t* get_data()
        void set_owner( bool owner_in)
