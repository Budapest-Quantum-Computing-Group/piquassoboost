from libcpp cimport bool #import bool type 
from numpy cimport int64_t
from piquasso.common.matrix cimport matrix # import the C++ version of class matrix

# matrix dot product definition
cdef extern from "source/dot.h" namespace "pic":
    cdef matrix dot( matrix& A, matrix& B )


