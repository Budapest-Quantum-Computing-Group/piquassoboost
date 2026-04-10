#
#  Copyright 2021 Budapest Quantum Computing Group
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

cmake_minimum_required(VERSION 3.10.2)


# set the project name and version
project(PiuqassoBoost VERSION 0.1)

# reuse compilation time linking for use runtime linking 
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Export all symbols from the shared library on Windows (no __declspec needed)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Windows-specific early settings
if(WIN32)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # /bigobj: allow object files with more than 65535 sections
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj" CACHE STRING "Flags used by the compiler during all build types" FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /bigobj" CACHE STRING "Flags used by the compiler during release builds" FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj" CACHE STRING "Flags used by the compiler during debug builds" FORCE)
    message(STATUS "MSVC: Added /bigobj flag")
  endif()
endif()

# include CMAKE modules
include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckFunctionExists)
include(cmake/check_AVX.cmake)

# Setup Windows/conda search paths BEFORE BLAS/CBLAS discovery
if(WIN32 AND DEFINED ENV{CONDA_PREFIX})
  set(CONDA_PREFIX "$ENV{CONDA_PREFIX}")
  list(APPEND CMAKE_PREFIX_PATH "${CONDA_PREFIX}/Library")
  list(APPEND CMAKE_LIBRARY_PATH "${CONDA_PREFIX}/Library/lib")
  list(APPEND CMAKE_INCLUDE_PATH "${CONDA_PREFIX}/Library/include")
  set(ENV{PATH} "${CONDA_PREFIX}/Library/bin;$ENV{PATH}")
  message(STATUS "Windows conda environment detected: ${CONDA_PREFIX}")
endif()

include(cmake/FindBLASEXT.cmake)
include(cmake/FindCBLAS.cmake)

# variables for compile and link options
set(CXX_FLAGS)
set(CXX_FLAGS_DEBUG)
set(CXX_FLAGS_RELEASE)
set(EXTRA_INCLUDES)
set(BLAS_DIR)
set(PYTHON_PLAT_LIB)


#enable test target
enable_testing()

#################################################################
# checking python developer packages

# Use modern CMake Python discovery (CMake >= 3.14).
# skbuild sets Python3_EXECUTABLE / PYTHON_EXECUTABLE before invoking cmake;
# on a bare cmake invocation pass -DPython3_EXECUTABLE=/path/to/python.
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Provide legacy variable names used by the rest of this file.
set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
set(PYTHON_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
set(PYTHON_LIBRARIES "${Python3_LIBRARIES}")
set(NumPy_INCLUDE_DIR "${Python3_NumPy_INCLUDE_DIRS}")
set(NumPy_FOUND TRUE)

# On Windows the Python import library directory must be explicitly added to
# the linker search path so that targets linking python3XX.lib can find it.
if(WIN32 AND Python3_LIBRARIES)
    get_filename_component(_python_lib_dir "${Python3_LIBRARIES}" DIRECTORY)
    link_directories("${_python_lib_dir}")
    message(STATUS "Added Python lib dir to linker path: ${_python_lib_dir}")
endif()


exec_program(${PYTHON_EXECUTABLE}
             ARGS "-c \"from sysconfig import get_paths; info = get_paths(); platlib=info.get('platlib',' '); print(platlib)\""
             OUTPUT_VARIABLE PYTHON_PLAT_LIB
             RETURN_VALUE PLATLIB_NOT_FOUND
            )
if(PLATLIB_NOT_FOUND)
    message(FATAL_ERROR "Python platlib not found")
endif()

set(ENV{PYTHONPATH} "$ENV{PYTHONPATH};${PYTHON_PLAT_LIB}")
message("PYTHONPATH is set to "$ENV{PYTHONPATH})

# Detect Python version for sysconfig vs site compatibility
exec_program(${PYTHON_EXECUTABLE}
             ARGS "-c \"import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')\""
             OUTPUT_VARIABLE PYTHON_VERSION_STRING
             RETURN_VALUE PYTHON_VERSION_NOT_FOUND
            )
if(PYTHON_VERSION_NOT_FOUND)
    message(FATAL_ERROR "Python version detection failed")
endif()
message("Detected Python version: " ${PYTHON_VERSION_STRING})

# construct library directories (sysconfig for Python >= 3.10, distutils removed in 3.12)
if(${PYTHON_VERSION_STRING} VERSION_GREATER_EQUAL "3.10")
    exec_program(${PYTHON_EXECUTABLE}
                 ARGS "-c \"import sysconfig; lib = sysconfig.get_path('purelib'); tmp = [lib + '/../..']; ret=';'.join(tmp); print(ret)\""
                 OUTPUT_VARIABLE PYTHON_SYS_PATH
                 RETURN_VALUE SYSPATH_NOT_FOUND
                )
else()
    exec_program(${PYTHON_EXECUTABLE}
                 ARGS "-c \"import site; tmp = [s + '/../..' for s in site.getsitepackages()]; ret=';'.join(tmp); print(ret)\""
                 OUTPUT_VARIABLE PYTHON_SYS_PATH
                 RETURN_VALUE SYSPATH_NOT_FOUND
                )
endif()
if(SYSPATH_NOT_FOUND)
    message(FATAL_ERROR "Python syspath not found")
endif()

message("Python syspaths: " ${PYTHON_SYS_PATH})



# construct include directories
if(${PYTHON_VERSION_STRING} VERSION_GREATER_EQUAL "3.10")
    exec_program(${PYTHON_EXECUTABLE}
                 ARGS "-c \"import sysconfig; lib = sysconfig.get_path('purelib'); tmp = [lib + '/../../../include']; ret=';'.join(tmp); print(ret)\""
                 OUTPUT_VARIABLE CMAKE_REQUIRED_INCLUDES
                 RETURN_VALUE SYSPATH_NOT_FOUND
                )
else()
    exec_program(${PYTHON_EXECUTABLE}
                 ARGS "-c \"import site; tmp = [s + '/../../../include' for s in site.getsitepackages()]; ret=';'.join(tmp); print(ret)\""
                 OUTPUT_VARIABLE CMAKE_REQUIRED_INCLUDES
                 RETURN_VALUE SYSPATH_NOT_FOUND
                )
endif()
if(SYSPATH_NOT_FOUND)
    message(FATAL_ERROR "Python include path not found")
endif()

message("Include paths: " ${CMAKE_REQUIRED_INCLUDES})


set(NUMPY_INC_DIR ${NumPy_INCLUDE_DIR})


if(NOT NumPy_FOUND)
    message(FATAL_ERROR "NumPy headers not found")
endif()


# adding BLAS library dir if given by environment variable
if(DEFINED ENV{BLAS_LIB_DIR})

  set(BLAS_DIR "$ENV{BLAS_LIB_DIR}")

else()

  # Determine BLAS library directory using a cross-platform helper script
  set(FIND_NUMPY_BLAS_SCRIPT "${PROJECT_SOURCE_DIR}/cmake/find_numpy_blas_dir.py")
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${FIND_NUMPY_BLAS_SCRIPT}
    OUTPUT_VARIABLE BLAS_DIR
    RESULT_VARIABLE NUMPY_BLAS_NOT_FOUND
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  # Fallback: simple numpy-relative path
  if(NUMPY_BLAS_NOT_FOUND OR "${BLAS_DIR}" STREQUAL "")
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "import numpy, os; print(os.path.abspath(os.path.join(os.path.dirname(numpy.__file__), '..', 'lib')))"
      OUTPUT_VARIABLE BLAS_DIR
      RESULT_VARIABLE NUMPY_BLAS_NOT_FOUND2
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endif()

endif()

message(STATUS "BLAS_DIR: ${BLAS_DIR}")



find_package(PythonExtensions REQUIRED)


set(CMAKE_VERBOSE_MAKEFILE ON)


#################################################################
# find MPI libraries if MPI is enables by environment variable PIQUASSOBOOST_MPI

if(DEFINED ENV{PIQUASSOBOOST_MPI})
  find_package(MPI REQUIRED)

  # openmpi which has a different c++ bindings
  #add_definitions(-DOMPI_SKIP_MPICXX)

  # setting basic compile flags
  list(APPEND CXX_FLAGS_DEBUG "-D__MPI__")
  list(APPEND CXX_FLAGS_RELEASE "-D__MPI__")

  list(APPEND EXTRA_INCLUDES "${MPI_C_INCLUDE_PATH}") 

endif()


#################################################################
# find CBLAS libraries. Hints are given by numpy library directory via variable BLAS_DIR

# Set library search paths from BLAS_DIR
if(BLAS_DIR AND NOT "${BLAS_DIR}" STREQUAL "")
  list(APPEND CMAKE_PREFIX_PATH "${BLAS_DIR}")
  list(APPEND CMAKE_LIBRARY_PATH "${BLAS_DIR}")
endif()

# Update runtime library paths
if(WIN32)
  if(BLAS_DIR AND NOT "${BLAS_DIR}" STREQUAL "")
    set(ENV{PATH} "${BLAS_DIR};$ENV{PATH}")
  endif()
else()
  set(ENV{LD_LIBRARY_PATH} "${BLAS_DIR}:$ENV{LD_LIBRARY_PATH}")
endif()

# BLAS may already be found by FindBLASEXT.cmake; if not, search with Windows-friendly approach
if(NOT BLAS_FOUND)
  if(WIN32)
    find_library(_BLAS_WIN
      NAMES openblas blas mkl_rt
      PATHS ${CMAKE_LIBRARY_PATH}
      NO_DEFAULT_PATH
    )
    if(_BLAS_WIN)
      set(BLAS_LIBRARIES "${_BLAS_WIN}")
      set(BLAS_FOUND TRUE)
      message(STATUS "Found BLAS library directly on Windows: ${BLAS_LIBRARIES}")
    endif()
  endif()
  if(NOT BLAS_FOUND)
    find_package(BLAS MODULE REQUIRED)
  endif()
endif()

find_package(LAPACK REQUIRED)

# Detect BLAS vendor by probing for vendor-specific functions
set(CMAKE_REQUIRED_LIBRARIES "${BLAS_LIBRARIES}" "${LAPACK_LIBRARIES}")
if(NOT WIN32)
  list(APPEND CMAKE_REQUIRED_LIBRARIES "-lm")
endif()
check_function_exists(MKL_Set_Num_Threads BLAS_IS_MKL)
check_function_exists(openblas_set_num_threads BLAS_IS_OPENBLAS)
message(STATUS "BLAS_IS_MKL: ${BLAS_IS_MKL}")
message(STATUS "BLAS_IS_OPENBLAS: ${BLAS_IS_OPENBLAS}")

# Find LAPACKE library (platform-specific)
set(LAPACKE_LIBRARIES)
if(APPLE)
  execute_process(
    COMMAND brew --prefix lapack
    OUTPUT_VARIABLE HOMEBREW_LAPACK_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(HOMEBREW_LAPACK_PREFIX)
    find_library(LAPACKE_LIBRARIES
      NAMES lapacke
      PATHS ${HOMEBREW_LAPACK_PREFIX}/lib /opt/homebrew/opt/lapack/lib /usr/local/opt/lapack/lib
      NO_DEFAULT_PATH
    )
  endif()
  if(NOT LAPACKE_LIBRARIES)
    find_library(LAPACKE_LIBRARIES NAMES lapacke)
  endif()
elseif(WIN32)
  # On Windows, OpenBLAS and MKL include LAPACKE
  if(BLAS_IS_MKL OR BLAS_IS_OPENBLAS)
    set(LAPACKE_LIBRARIES "${BLAS_LIBRARIES}")
    message(STATUS "Using BLAS libraries for LAPACKE on Windows")
  endif()
  if(NOT LAPACKE_LIBRARIES)
    if(DEFINED ENV{CONDA_PREFIX})
      find_library(LAPACKE_LIBRARIES
        NAMES lapacke
        PATHS "${CONDA_PREFIX}/Library/lib" ${CMAKE_LIBRARY_PATH}
        NO_DEFAULT_PATH
      )
    endif()
    if(NOT LAPACKE_LIBRARIES)
      find_library(LAPACKE_LIBRARIES NAMES lapacke PATHS ${CMAKE_LIBRARY_PATH})
    endif()
  endif()
else()
  set(CMAKE_REQUIRED_LIBRARIES "${BLAS_LIBRARIES}" "${LAPACK_LIBRARIES}" "-lm")
  check_function_exists(LAPACKE_zgetri LAPACKE_FOUND_IN_BLAS)
  if(LAPACKE_FOUND_IN_BLAS)
    set(LAPACKE_LIBRARIES)
  else()
    find_library(LAPACKE_LIBRARIES lapacke REQUIRED)
  endif()
endif()
set(CMAKE_REQUIRED_LIBRARIES "")

# If MKL is enabled
if(BLAS_IS_MKL)
  message(STATUS "MKL is enabled")
  list(APPEND CXX_FLAGS_DEBUG "-DBLAS=1")
  list(APPEND CXX_FLAGS_RELEASE "-DBLAS=1")

  # If OpenBlas is enabled
elseif(BLAS_IS_OPENBLAS)
  message(STATUS "OpenBLAS is enabled")
  list(APPEND CXX_FLAGS_DEBUG "-DBLAS=2")
  list(APPEND CXX_FLAGS_RELEASE "-DBLAS=2")
else()

  list(APPEND CXX_FLAGS_DEBUG "-DBLAS=0")
  list(APPEND CXX_FLAGS_RELEASE "-DBLAS=0")
endif()


if (CBLAS_CONJ_NO_TRANS_PRESENT AND NOT BLAS_IS_MKL)
  list(APPEND CXX_FLAGS_DEBUG "-DCBLAS_CONJ_NO_TRANS_PRESENT=1")
  list(APPEND CXX_FLAGS_RELEASE "-DCBLAS_CONJ_NO_TRANS_PRESENT=1")
else()
  list(APPEND CXX_FLAGS_DEBUG "-DCBLAS_CONJ_NO_TRANS_PRESENT=0")
  list(APPEND CXX_FLAGS_RELEASE "-DCBLAS_CONJ_NO_TRANS_PRESENT=0")
endif()

# checking for AVX/AVX2 support
CHECK_FOR_AVX()



# macOS: detect Homebrew libomp before find_package(OpenMP)
if(APPLE)
  execute_process(
    COMMAND brew --prefix libomp
    OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(HOMEBREW_LIBOMP_PREFIX)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include" CACHE STRING "C compiler flags for OpenMP" FORCE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include" CACHE STRING "C++ compiler flags for OpenMP" FORCE)
    set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "C library names for OpenMP" FORCE)
    set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "C++ library names for OpenMP" FORCE)
    find_library(OpenMP_omp_LIBRARY NAMES omp PATHS ${HOMEBREW_LIBOMP_PREFIX}/lib NO_DEFAULT_PATH)
    if(OpenMP_omp_LIBRARY)
      set(OpenMP_omp_LIBRARY ${OpenMP_omp_LIBRARY} CACHE PATH "Path to OpenMP omp library" FORCE)
    endif()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${HOMEBREW_LIBOMP_PREFIX}/lib -lomp")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${HOMEBREW_LIBOMP_PREFIX}/lib -lomp")
    message(STATUS "Found Homebrew libomp at: ${HOMEBREW_LIBOMP_PREFIX}")
  endif()
endif()

find_package(OpenMP REQUIRED)

# Convert OpenMP flags from string to list for proper target_compile_options usage on macOS
if(APPLE AND OpenMP_CXX_FLAGS)
  separate_arguments(OpenMP_CXX_FLAGS_LIST UNIX_COMMAND "${OpenMP_CXX_FLAGS}")
else()
  set(OpenMP_CXX_FLAGS_LIST ${OpenMP_CXX_FLAGS})
endif()

list(APPEND EXTRA_INCLUDES "${OpenMP_CXX_INCLUDE_DIRS}")

# Identify the compiler type and set compiler specific options
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # using Clang
  message("-- Using Clang compiler")
  list(APPEND CXX_FLAGS_DEBUG "-Wall" "-Wpedantic" "-Wextra" "-fexceptions" "-DDEBUG" "-fno-builtin-malloc" "-fno-builtin-calloc" "-fno-builtin-realloc" "-fno-builtin-free" "-fpermissive")
  list(APPEND CXX_FLAGS_RELEASE "-Wall" "-O3" "-m64" "-DNDEBUG" "-fno-builtin-malloc" "-fno-builtin-calloc" "-fno-builtin-realloc" "-fno-builtin-free" "-fpermissive")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  message("-- Using GNU compiler")
  list(APPEND CXX_FLAGS_DEBUG "-Wall" "-Wpedantic" "-Wextra" "-fexceptions" "-DDEBUG" "-fno-builtin-malloc" "-fno-builtin-calloc" "-fno-builtin-realloc" "-fno-builtin-free" "-fpermissive" "-g3" "-ggdb")
  list(APPEND CXX_FLAGS_RELEASE "-Wall" "-O3" "-m64" "-DNDEBUG" "-fno-builtin-malloc" "-fno-builtin-calloc" "-fno-builtin-realloc" "-fno-builtin-free" "-fpermissive" "-ftree-vectorize")

  if (${HAVE_AVX512F_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "-mavx512f" "-DUSE_AVX")
  elseif (${HAVE_AVX2_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "-mavx2" "-DUSE_AVX")
  elseif (${HAVE_AVX_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "-mavx" "-DUSE_AVX")
  endif()


elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # using Intel C++
  message("-- Using Intel compiler")
  list(APPEND CXX_FLAGS_DEBUG "-Wall" "-DDEBUG")
  list(APPEND CXX_FLAGS_RELEASE "-O3" "-DNDEBUG")
  if (BLAS_IS_MKL)
    list(APPEND CXX_FLAGS_DEBUG "-mkl" "-tbb")
    list(APPEND CXX_FLAGS_RELEASE "-mkl" "-tbb")
  endif()

  if (${HAVE_AVX512F_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "-mavx512f" "-DUSE_AVX512F")
  elseif (${HAVE_AVX2_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "-mavx2" "-DUSE_AVX")
  elseif (${HAVE_AVX_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "-mavx" "-DUSE_AVX")
  endif()

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # using Visual Studio C++
  message("-- Using Visual Studio C++ compiler")
  list(APPEND CXX_FLAGS "/bigobj")
  list(APPEND CXX_FLAGS_DEBUG "/W3" "/DDEBUG"
    "/wd4018"   # signed/unsigned mismatch
    "/wd4244"   # narrowing conversion (e.g. int64_t to int)
    "/wd4267"   # size_t to smaller integral type
    "/wd4146"   # unary minus on unsigned (in gmp.h)
    "/D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING"  # std::complex<RationalInf>
  )
  list(APPEND CXX_FLAGS_RELEASE "/W3" "/O2" "/DNDEBUG"
    "/wd4018"   # signed/unsigned mismatch
    "/wd4244"   # narrowing conversion (e.g. int64_t to int)
    "/wd4267"   # size_t to smaller integral type
    "/wd4146"   # unary minus on unsigned (in gmp.h)
    "/D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING"  # std::complex<RationalInf>
  )

  if (${HAVE_AVX512F_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "/arch:AVX512" "/DUSE_AVX512F")
  elseif (${HAVE_AVX2_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "/arch:AVX2" "/DUSE_AVX")
  elseif (${HAVE_AVX_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "/arch:AVX" "/DUSE_AVX")
  endif()

endif()

############################################################xx
# checking TBB libraries and headers

# Collect TBB search paths
set(TBB_SEARCH_PATHS)

if(WIN32 AND DEFINED ENV{CONDA_PREFIX})
  set(CONDA_TBB_LIB_DIR "${CONDA_PREFIX}/Library/lib")
  set(CONDA_TBB_INC_DIR "${CONDA_PREFIX}/Library/include")
  message(STATUS "Windows conda TBB lib dir: ${CONDA_TBB_LIB_DIR}")
endif()

# adding TBB library dir if given by environment variable
if(DEFINED ENV{TBB_LIB_DIR})
  set(TBB_LIB_DIR "$ENV{TBB_LIB_DIR}")
  message(STATUS "Using TBB_LIB_DIR from environment: ${TBB_LIB_DIR}")
elseif(WIN32 AND DEFINED CONDA_TBB_LIB_DIR)
  set(TBB_LIB_DIR "${CONDA_TBB_LIB_DIR}")
  message(STATUS "Using conda TBB library directory: ${TBB_LIB_DIR}")
else()
  set(TBB_LIB_DIR "${PYTHON_SYS_PATH}")
endif()

if(TBB_LIB_DIR)
  list(APPEND TBB_SEARCH_PATHS "${TBB_LIB_DIR}")
endif()
if(WIN32 AND DEFINED CONDA_TBB_LIB_DIR)
  list(APPEND TBB_SEARCH_PATHS "${CONDA_TBB_LIB_DIR}" "${CONDA_PREFIX}/Library/bin")
endif()
list(APPEND TBB_SEARCH_PATHS ${PYTHON_SYS_PATH})

# Find TBB library
set(TBB_LIB)
if(WIN32)
  find_library(TBB_LIB
    NAMES tbb12 tbb tbb12.lib tbb.lib
    PATHS ${TBB_SEARCH_PATHS}
    PATH_SUFFIXES lib Library/lib
    NO_DEFAULT_PATH
  )
  if(NOT TBB_LIB)
    find_library(TBB_LIB
      NAMES tbb12 tbb tbb12.lib tbb.lib
      PATHS ${TBB_SEARCH_PATHS}
      PATH_SUFFIXES lib Library/lib
    )
  endif()
else()
  find_library(TBB_LIB
    NAMES tbb
    PATHS ${TBB_SEARCH_PATHS}
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
  )
  if(NOT TBB_LIB)
    find_library(TBB_LIB NAMES tbb PATHS ${TBB_SEARCH_PATHS} PATH_SUFFIXES lib)
  endif()
endif()

if(NOT TBB_LIB)
  message(FATAL_ERROR "Could not find TBB library. Searched in: ${TBB_SEARCH_PATHS}")
else()
  message(STATUS "Found TBB library: ${TBB_LIB}")
  # On Windows add TBB lib dir to the linker search path so extension DLLs
  # that transitively depend on tbb12.lib can find it.
  if(WIN32)
    get_filename_component(_tbb_lib_dir "${TBB_LIB}" DIRECTORY)
    link_directories("${_tbb_lib_dir}")
    message(STATUS "Added TBB lib dir to linker path: ${_tbb_lib_dir}")
  endif()
endif()

# Find TBBMALLOC library
find_library(TBBMALLOC_LIB
  NAMES tbbmalloc tbbmalloc.lib
  PATHS ${TBB_SEARCH_PATHS}
  PATH_SUFFIXES lib Library/lib
  NO_DEFAULT_PATH
)
if(NOT TBBMALLOC_LIB)
  find_library(TBBMALLOC_LIB NAMES tbbmalloc tbbmalloc.lib PATHS ${TBB_SEARCH_PATHS} PATH_SUFFIXES lib Library/lib)
endif()
if(NOT TBBMALLOC_LIB)
  message(WARNING "Could not find TBBMALLOC library. Continuing without it.")
else()
  message(STATUS "Found TBBMALLOC library: ${TBBMALLOC_LIB}")
endif()

# Find TBBMALLOC_PROXY library
find_library(TBBMALLOC_PROXY_LIB
  NAMES tbbmalloc_proxy tbbmalloc_proxy.lib
  PATHS ${TBB_SEARCH_PATHS}
  PATH_SUFFIXES lib Library/lib
  NO_DEFAULT_PATH
)
if(NOT TBBMALLOC_PROXY_LIB)
  find_library(TBBMALLOC_PROXY_LIB NAMES tbbmalloc_proxy tbbmalloc_proxy.lib PATHS ${TBB_SEARCH_PATHS} PATH_SUFFIXES lib Library/lib)
endif()
if(NOT TBBMALLOC_PROXY_LIB)
  message(WARNING "Could not find TBBMALLOC_PROXY library. Continuing without it.")
else()
  message(STATUS "Found TBBMALLOC_PROXY library: ${TBBMALLOC_PROXY_LIB}")
endif()

# adding TBB include dir
set(TBB_INC_DIR)
set(TBB_HEADER_FOUND FALSE)

if(DEFINED ENV{TBB_INC_DIR})
  set(TBB_INC_DIR "$ENV{TBB_INC_DIR}")
  message(STATUS "TBB_INC_DIR from environment: ${TBB_INC_DIR}")
elseif(WIN32 AND DEFINED CONDA_TBB_INC_DIR)
  set(TBB_INC_DIR "${CONDA_TBB_INC_DIR}")
  message(STATUS "Using conda TBB include directory: ${TBB_INC_DIR}")
endif()

set(TBB_INC_SEARCH_PATHS)
if(TBB_INC_DIR)
  list(APPEND TBB_INC_SEARCH_PATHS "${TBB_INC_DIR}")
endif()
if(WIN32 AND DEFINED CONDA_TBB_INC_DIR)
  list(APPEND TBB_INC_SEARCH_PATHS "${CONDA_TBB_INC_DIR}" "${CONDA_PREFIX}/include")
  list(APPEND TBB_INC_SEARCH_PATHS "C:/Program Files/Intel/oneAPI/tbb/latest/include")
endif()
list(APPEND TBB_INC_SEARCH_PATHS "${CMAKE_REQUIRED_INCLUDES}")

foreach(SEARCH_PATH ${TBB_INC_SEARCH_PATHS})
  if(EXISTS "${SEARCH_PATH}/tbb/tbb.h")
    set(TBB_INC_DIR "${SEARCH_PATH}")
    set(TBB_HEADER_FOUND TRUE)
    message(STATUS "Found TBB header at: ${TBB_INC_DIR}/tbb/tbb.h")
    break()
  endif()
endforeach()

if(NOT TBB_HEADER_FOUND)
  set(CMAKE_REQUIRED_FLAGS "-c")
  check_include_file_cxx(tbb/tbb.h TBB_HEADER)
  if(TBB_HEADER)
    set(TBB_HEADER_FOUND TRUE)
    list(APPEND EXTRA_INCLUDES "${CMAKE_REQUIRED_INCLUDES}")
    message(STATUS "Found TBB header in system paths")
  endif()
endif()

if(NOT TBB_HEADER_FOUND)
  message(FATAL_ERROR "TBB header tbb.h not found. Please set TBB_INC_DIR environment variable.")
endif()

if(TBB_INC_DIR)
  list(APPEND EXTRA_INCLUDES "${TBB_INC_DIR}")
endif()



############################################################xx
# checking MPFR libraries and headers

if(DEFINED ENV{PIQUASSOBOOST_MPFR})
  # adding MPFR library dir if given by environment variable
  if(DEFINED ENV{MPFR_LIB_DIR})

    find_library(MPFR_LIB mpfr
               PATHS $ENV{MPFR_LIB_DIR}
               NO_DEFAULT_PATH
               REQUIRED)
    find_library(GMP_LIB gmp
               PATHS $ENV{MPFR_LIB_DIR}
               NO_DEFAULT_PATH)

  else()

    find_library(MPFR_LIB mpfr
               PATHS ${PYTHON_SYS_PATH}
               NO_DEFAULT_PATH
               REQUIRED)
    find_library(GMP_LIB gmp
               PATHS ${PYTHON_SYS_PATH}
               NO_DEFAULT_PATH)
  
  endif()


  # adding MPFR include dir
  if(DEFINED ENV{MPFR_INC_DIR})

    set(CMAKE_REQUIRED_FLAGS "-c")
    check_include_file_cxx(mpfr.h MPFR_HEADER "-I$ENV{MPFR_INC_DIR}")

    if(NOT MPFR_HEADER)
      message(FATAL_ERROR "MPFR header mpfr.h not found")
    endif()

    message("-- Adding include directory $ENV{MPFR_INC_DIR}")
    list(APPEND EXTRA_INCLUDES "$ENV{MPFR_INC_DIR}")

  else()
 
    set(CMAKE_REQUIRED_FLAGS "-c")
    check_include_file_cxx(mpfr.h MPFR_HEADER )
    list(APPEND EXTRA_INCLUDES "${CMAKE_REQUIRED_INCLUDES}") 

    if(NOT MPFR_HEADER)
      message(FATAL_ERROR "MPFR header mpfr.h not found")
    endif()

  endif()

else()

  set(MPFR_LIB)
  set(GMP_LIB)

endif()


list(APPEND piquassoboost_files 
    ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/PicTypes.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/dependency_graph.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/matrix.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/matrix32.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/PicState.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/dot.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/n_aryGrayCodeCounter.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/common_functionalities.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian/source/CalculationHelpers.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian/source/GaussianState_Cov.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian/source/tasks_apply_to_C_and_G/extract_rows.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian/source/tasks_apply_to_C_and_G/transform_rows.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian/source/tasks_apply_to_C_and_G/extract_corner.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian/source/tasks_apply_to_C_and_G/transform_cols.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian/source/tasks_apply_to_C_and_G/insert_transformed_cols.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian/source/tasks_apply_to_C_and_G/insert_transformed_rows.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/CChinHuhPermanentCalculator.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/BatchedPermanentCalculator.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/TorontonianUtilities.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/Torontonian.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/TorontonianRecursive.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/PowerTraceHafnianUtilities.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/PowerTraceHafnian.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/PowerTraceHafnianRecursive.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/PowerTraceLoopHafnian.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/PowerTraceLoopHafnianRecursive.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/BruteForceHafnian.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/BruteForceLoopHafnian.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/GlynnPermanentCalculators_implementation.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/GlynnPermanentCalculatorSimple.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/BBFGPermanentCalculator.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/BBFGPermanentCalculatorRepeated.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/simulation_strategies/source/CGeneralizedCliffordsSimulationStrategy.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/simulation_strategies/source/CGeneralizedCliffordsBSimulationStrategy.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/simulation_strategies/source/CGeneralizedCliffordsBUniformLossesSimulationStrategy.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/simulation_strategies/source/ThresholdBosonSampling.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/simulation_strategies/source/GaussianSimulationStrategy.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/simulation_strategies/source/GaussianSimulationStrategyFast.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/simulation_strategies/source/RepeatedColumnPairs.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/simulation_strategies/source/CGeneralizedCliffordsBLossySimulationStrategy.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/simulation_strategies/source/samplingHelperFunctions.cpp

)

# add infinite precision implementations if required
if (DEFINED ENV{PIQUASSOBOOST_MPFR})
  list(APPEND CXX_FLAGS_DEBUG "-D__MPFR__=1")
  list(APPEND CXX_FLAGS_RELEASE "-D__MPFR__=1")

endif()

if (DEFINED ENV{PIQUASSOBOOST_DFE})
  list(APPEND piquassoboost_files 
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/GlynnPermanentCalculatorDFE.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/GlynnPermanentCalculatorRepeatedDFE.cpp
  )

  list(APPEND CXX_FLAGS_DEBUG "-D__DFE__=1")
  list(APPEND CXX_FLAGS_RELEASE "-D__DFE__=1")

endif()

if (DEFINED ENV{PIQUASSOBOOST_RYSER_HAFNIAN})

  # Glynn formula giver better numerical accuracy (it has been made default)
  list(APPEND CXX_FLAGS_DEBUG "-D__RYSER_HAFNIAN__=1")
  list(APPEND CXX_FLAGS_RELEASE "-D__RYSER_HAFNIAN__=1")

else()
  # Glynn formula giver better numerical accuracy (it has been made default)
  list(APPEND CXX_FLAGS_DEBUG "-D__GLYNN_HAFNIAN__=1")
  list(APPEND CXX_FLAGS_RELEASE "-D__GLYNN_HAFNIAN__=1")

endif()



# add architecture specific kernels
if (${HAVE_AVX512F_EXTENSIONS})
  list(APPEND piquassoboost_files 
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/kernels/get_reflection_vector_AVX512F.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/kernels/calc_vH_times_A_AVX512F.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/kernels/calc_vov_times_A_AVX512F.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/kernels/apply_householder_cols_AVX512F.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/kernels/loop_correction_AVX512F.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/kernels/calc_cholesky_decomposition_AVX.cpp
  )
elseif (${HAVE_AVX2_EXTENSIONS})
  list(APPEND piquassoboost_files 
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/kernels/get_reflection_vector_AVX.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/kernels/calc_vH_times_A_AVX.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/kernels/calc_vov_times_A_AVX.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/kernels/apply_householder_cols_AVX.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/kernels/loop_correction_AVX.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/kernels/calc_cholesky_decomposition_AVX.cpp
  )
elseif (${HAVE_AVX_EXTENSIONS})
  list(APPEND piquassoboost_files 
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/kernels/get_reflection_vector_AVX.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/kernels/calc_vH_times_A_AVX.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/kernels/calc_vov_times_A_AVX.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/kernels/apply_householder_cols_AVX.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/kernels/loop_correction_AVX.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/kernels/calc_cholesky_decomposition_AVX.cpp
  )
endif()

add_library(piquassoboost SHARED
    ${piquassoboost_files})

# adding compile options
target_compile_options(piquassoboost PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    ${OpenMP_CXX_FLAGS_LIST}
)

# adding linking options
target_link_libraries(piquassoboost PUBLIC
    ${TBBMALLOC_LIB}
    ${TBBMALLOC_PROXY_LIB}
    ${TBB_LIB}
)
target_link_libraries(piquassoboost PRIVATE
    ${BLAS_LIBRARIES}
    ${CBLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    ${MPI_C_LIBRARIES}    
    ${MPFR_LIB}
    ${GMP_LIB}
    ${OpenMP_CXX_LIBRARIES}
)

target_include_directories(piquassoboost PRIVATE
                            .
                            ./piquassoboost/common/source
                            ./piquassoboost/gaussian/source
                            ./piquassoboost/sampling/source
                            ./piquassoboost/sampling/simulation_strategies/source
                            ${TBB_HEADER}
                            ./piquassoboost/sampling/source/kernels
                            ${EXTRA_INCLUDES})



set_target_properties(piquassoboost PROPERTIES
    VERSION ${PROJECT_VERSION}
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/PicTypes.hpp
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/dependency_graph.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/matrix.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/matrix32.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/matrix_real.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/matrix_real16.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/PicState.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/PicStateHash.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/dot.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/n_aryGrayCodeCounter.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/common_functionalities.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian/source/CalculationHelpers.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian/source/GaussianState_Cov.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian/source/tasks_apply_to_C_and_G/extract_rows.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian/source/tasks_apply_to_C_and_G/transform_rows.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian/source/tasks_apply_to_C_and_G/extract_corner.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian/source/tasks_apply_to_C_and_G/transform_cols.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian/source/tasks_apply_to_C_and_G/insert_transformed_cols.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian/source/tasks_apply_to_C_and_G/insert_transformed_rows.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/CChinHuhPermanentCalculator.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/GlynnPermanentCalculator.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/BBFGPermanentCalculator.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/BBFGPermanentCalculatorRepeated.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/GlynnPermanentCalculatorRepeated.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/GlynnPermanentCalculatorSimple.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/Torontonian.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/TorontonianUtilities.hpp
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/PowerTraceHafnianUtilities.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/PowerTraceHafnian.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/PowerTraceHafnianRecursive.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/PowerTraceLoopHafnian.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/PowerTraceLoopHafnianRecursive.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/BruteForceHafnian.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/BruteForceLoopHafnian.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/simulation_strategies/source/CGeneralizedCliffordsSimulationStrategy.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/simulation_strategies/source/CGeneralizedCliffordsBSimulationStrategy.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/simulation_strategies/source/CGeneralizedCliffordsBUniformLossesSimulationStrategy.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/simulation_strategies/source/ThresholdBosonSampling.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/simulation_strategies/source/GaussianSimulationStrategy.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/simulation_strategies/source/GaussianSimulationStrategyFast.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/simulation_strategies/source/RepeatedColumnPairs.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/simulation_strategies/source/CGeneralizedCliffordsBLossySimulationStrategy.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/simulation_strategies/source/samplingHelperFunctions.h
    
)

configure_file(${PROJECT_SOURCE_DIR}/piquassoboost/common/source/Config.h.in
               ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/Config.h)

set_target_properties( piquassoboost PROPERTIES
                        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/piquassoboost
)

# On Windows, the DLL goes to RUNTIME_OUTPUT_DIRECTORY;
# put it alongside the Python extensions so ctypes/import can find it
if(WIN32)
  set_target_properties( piquassoboost PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/piquassoboost
  )
endif()



###########################################################################




# adding subdirectories for python extensions
add_subdirectory (piquassoboost)

if(DEFINED ENV{PIQUASSOBOOST_CTEST})
  # adding CMAKE files for test executables
  add_subdirectory (ctests)
endif()

install(TARGETS piquassoboost
        LIBRARY DESTINATION piquassoboost
        RUNTIME DESTINATION piquassoboost  # Windows DLL
)
