import numpy as np
cimport numpy as np
np.import_array()

from piquasso.common.matrix cimport matrix
from piquasso.common.PicState cimport PicState_int64


cdef class numpy_interface:
    cdef np.ndarray PicState_int64_to_numpy(self, PicState_int64 &cstate )
    cdef np.ndarray matrix_to_numpy(self, matrix &cmtx )
    cdef np.ndarray array_from_ptr(self, void * ptr, int dim, np.npy_intp* shape, int np_type)
