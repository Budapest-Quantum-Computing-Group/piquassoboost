#ifndef PicTypes_H
#define PicTypes_H

#include <assert.h>
#include <cstddef>
#include <complex>

// platform independent types
#include <stdint.h>

#ifndef CACHELINE
#define CACHELINE 64
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#if BLAS==0 // undefined blas
    /// Set the number of threads on runtime in MKL
    void omp_set_num_threads(int num_threads);
    /// get the number of threads in MKL
    int omp_get_max_threads();
#elif BLAS==1 // MKL
    /// Set the number of threads on runtime in MKL
    void MKL_Set_Num_Threads(int num_threads);
    /// get the number of threads in MKL
    int mkl_get_max_threads();
#elif BLAS==2 // OpenBLAS
    /// Set the number of threads on runtime in OpenBlas
    void openblas_set_num_threads(int num_threads);
    /// get the number of threads in OpenBlas
    int openblas_get_num_threads();
#endif

#ifdef __cplusplus
}
#endif


/// The namespace of the Picasso project
namespace pic {


/// @brief Structure type representing complex numbers in the Picasso package
using Complex16 = std::complex<double>;


}  //PIC

#endif
