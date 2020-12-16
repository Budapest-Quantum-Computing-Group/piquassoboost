
# external declaration of several TBB headers used in cython wrappers
cdef extern from "tbb/flow_graph.h" namespace "tbb::flow":
    cdef cppclass continue_msg:
        continue_msg()
