cmake_minimum_required(VERSION 3.10.2)


# set the project name and version
project(CPiquasso VERSION 0.1)

# reuse compilation time linking for use runtime linking 
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# include CMAKE modules
include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckFunctionExists)
include(check_AVX.cmake)

# variables for compile and link options
set(CXX_FLAGS_DEBUG)
set(CXX_FLAGS_RELEASE)
set(EXTRA_INCLUDES)
set(BLAS_DIR)
set(PYTHON_PLAT_LIB)


#enable test target
enable_testing()


#################################################################
# checking python developer packages


# find out python packages
find_package(PythonInterp)
find_package(PythonLibs)


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

# contruct library directories
exec_program(${PYTHON_EXECUTABLE}
             ARGS "-c \"import sys; ret=';'.join(sys.path); print(ret)\""
             OUTPUT_VARIABLE PYTHON_SYS_PATH
             RETURN_VALUE SYSPATH_NOT_FOUND
            )
if(SYSPATH_NOT_FOUND)
    message(FATAL_ERROR "Python syspath not found")
endif()

message("Python syspaths: " ${PYTHON_SYS_PATH})




# contruct include directories
exec_program(${PYTHON_EXECUTABLE}
             ARGS "-c \"import sys; tmp = [s + '/../../../include' for s in sys.path]; ret=';'.join(tmp); print(ret)\""
             OUTPUT_VARIABLE CMAKE_REQUIRED_INCLUDES
             RETURN_VALUE SYSPATH_NOT_FOUND
            )
if(SYSPATH_NOT_FOUND)
    message(FATAL_ERROR "Python include path not found")
endif()

message("Include paths: " ${CMAKE_REQUIRED_INCLUDES})



# Determine Numpy include directory
exec_program(${PYTHON_EXECUTABLE}
             ARGS "-c \"import numpy; print(numpy.get_include())\""
             OUTPUT_VARIABLE NUMPY_INC_DIR
             RETURN_VALUE NUMPY_NOT_FOUND
            )

if(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "NumPy headers not found")
endif()


# adding BLAS library dir if given by environment variable
if(DEFINED ENV{BLAS_LIB_DIR})

  set(BLAS_DIR "$ENV{BLAS_LIB_DIR}")

else()

# Determine CBLAS library directory behind Numpy
exec_program(
  ${PYTHON_EXECUTABLE}
  ARGS "-c \"import numpy; blas_info=numpy.__config__.get_info('blas_opt_info'); libs = blas_info.get('library_dirs'); print(libs[0])\""
  OUTPUT_VARIABLE BLAS_DIR
  RETURN_VALUE NUMPY_BLAS_NOT_FOUND
)

endif()



find_package(PythonExtensions REQUIRED)


set(CMAKE_VERBOSE_MAKEFILE ON)


#################################################################
# find MPI libraries if MPI is enables by environment variable CPIQUASSO_MPI

if(DEFINED ENV{CPIQUASSO_MPI})
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

find_package(BLAS REQUIRED)

# make difference between MKL and OPENBLAS by checking specific functions
set(CMAKE_REQUIRED_LIBRARIES "${BLAS_LIBRARIES}")
check_function_exists(MKL_Set_Num_Threads BLAS_IS_MKL)
check_function_exists(openblas_set_num_threads BLAS_IS_OPENBLAS)
set(CMAKE_REQUIRED_LIBRARIES "")

# If MKL is enabled
if(${BLAS_IS_MKL})
  list(APPEND CXX_FLAGS_DEBUG "-DBLAS=1")
  list(APPEND CXX_FLAGS_RELEASE "-DBLAS=1")

  # If OpenBlas is enabled
elseif(${BLAS_IS_OPENBLAS})
  list(APPEND CXX_FLAGS_DEBUG "-DBLAS=2")
  list(APPEND CXX_FLAGS_RELEASE "-DBLAS=2")
else()

  list(APPEND CXX_FLAGS_DEBUG "-DBLAS=0")
  list(APPEND CXX_FLAGS_RELEASE "-DBLAS=0")
endif()


# setting basic compile flags
list(APPEND CXX_FLAGS_DEBUG "-Wall" "-Wpedantic" "-Wextra" "-fexceptions" "-DDEBUG" "-fno-builtin-malloc" "-fno-builtin-calloc" "-fno-builtin-realloc" "-fno-builtin-free" "-fpermissive")
list(APPEND CXX_FLAGS_RELEASE "-Wall" "-O3" "-m64" "-DNDEBUG" "-fno-builtin-malloc" "-fno-builtin-calloc" "-fno-builtin-realloc" "-fno-builtin-free" "-fpermissive")

# checking for AVX/AVX2 support
CHECK_FOR_AVX()



# Identify the compiler type and set compiler specific options
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # using Clang
  message("-- Using Clang compiler")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  message("-- Using GNU compiler")
  list(APPEND CXX_FLAGS_DEBUG "-g3" "-ggdb")
  list(APPEND CXX_FLAGS_RELEASE "-ftree-vectorize")

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

  if (${HAVE_AVX512F_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "/arch:AVX512F" "-DUSE_AVX512F")
  elseif (${HAVE_AVX2_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "/arch:AVX2" "-DUSE_AVX")
  elseif (${HAVE_AVX_EXTENSIONS})
    list(APPEND CXX_FLAGS_RELEASE "/arch:AVX" "-DUSE_AVX")
  endif()

endif()

############################################################xx
# checking TBB libraries and headers

# adding TBB library dir if given by environment variable
if(DEFINED ENV{TBB_LIB_DIR})

  find_library(TBB_LIB tbb
               PATHS $ENV{TBB_LIB_DIR}
               NO_DEFAULT_PATH
               REQUIRED)

  find_library(TBBMALLOC_LIB tbbmalloc
               PATHS $ENV{TBB_LIB_DIR}
               NO_DEFAULT_PATH
               REQUIRED)

  find_library(TBBMALLOC_PROXY_LIB tbbmalloc_proxy
               PATHS $ENV{TBB_LIB_DIR}
               NO_DEFAULT_PATH
               REQUIRED)

else()

  find_library(TBB_LIB tbb
               PATHS ${PYTHON_SYS_PATH}
               PATH_SUFFIXES "/../../../lib"
               NO_DEFAULT_PATH
               REQUIRED)

  find_library(TBBMALLOC_LIB tbbmalloc
               PATHS ${PYTHON_SYS_PATH}
               PATH_SUFFIXES "/../../../lib"
               NO_DEFAULT_PATH
               REQUIRED)

  find_library(TBBMALLOC_PROXY_LIB tbbmalloc_proxy
               PATHS ${PYTHON_SYS_PATH}
               PATH_SUFFIXES "/../../../lib"
               NO_DEFAULT_PATH
               REQUIRED)
  
endif()


# adding TBB include dir
if(DEFINED ENV{TBB_INC_DIR})

  set(CMAKE_REQUIRED_FLAGS "-c")
  check_include_file_cxx(tbb/tbb.h TBB_HEADER "-I$ENV{TBB_INC_DIR}")

  if(NOT TBB_HEADER)
    message(FATAL_ERROR "TBB header tbb.h not found")
  endif()

  message("-- Adding include directory $ENV{TBB_INC_DIR}")
  list(APPEND EXTRA_INCLUDES "$ENV{TBB_INC_DIR}")

else()
 
  set(CMAKE_REQUIRED_FLAGS "-c")
  check_include_file_cxx(tbb/tbb.h TBB_HEADER )
  list(APPEND EXTRA_INCLUDES "${CMAKE_REQUIRED_INCLUDES}") 

  if(NOT TBB_HEADER)
    message(FATAL_ERROR "TBB header tbb.h not found")
  endif()


endif()


list(APPEND cpiquasso_files 
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/PicTypes.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/dependency_graph.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/matrix.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/matrix32.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/PicState.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/PicStateHash.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/dot.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/common_functionalities.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/CGaussianState.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/GaussianState_Cov.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/tasks_apply_to_C_and_G/extract_rows.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/tasks_apply_to_C_and_G/transform_rows.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/tasks_apply_to_C_and_G/extract_corner.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/tasks_apply_to_C_and_G/transform_cols.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/tasks_apply_to_C_and_G/insert_transformed_cols.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/tasks_apply_to_C_and_G/insert_transformed_rows.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/CChinHuhPermanentCalculator.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/TorontonianUtilities.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/Torontonian.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/TorontonianRecursive.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/PowerTraceHafnianUtilities.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/PowerTraceHafnian.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/PowerTraceHafnianRecursive.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/PowerTraceLoopHafnian.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/PowerTraceLoopHafnianRecursive.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/BruteForceHafnian.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/BruteForceLoopHafnian.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/simulation_strategies/source/CGeneralizedCliffordsSimulationStrategy.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/simulation_strategies/source/GaussianSimulationStrategy.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/simulation_strategies/source/GaussianSimulationStrategyFast.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/simulation_strategies/source/RepeatedColumnPairs.cpp

)

# add architecture specific kernels
if (${HAVE_AVX512F_EXTENSIONS})
  list(APPEND cpiquasso_files 
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/kernels/get_reflection_vector_AVX512F.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/kernels/calc_vH_times_A_AVX512F.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/kernels/calc_vov_times_A_AVX512F.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/kernels/apply_householder_cols_AVX512F.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/kernels/loop_correction_AVX512F.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/kernels/calc_cholesky_decomposition_AVX.cpp
  )
elseif (${HAVE_AVX2_EXTENSIONS})
  list(APPEND cpiquasso_files 
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/kernels/get_reflection_vector_AVX.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/kernels/calc_vH_times_A_AVX.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/kernels/calc_vov_times_A_AVX.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/kernels/apply_householder_cols_AVX.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/kernels/loop_correction_AVX.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/kernels/calc_cholesky_decomposition_AVX.cpp
  )
elseif (${HAVE_AVX_EXTENSIONS})
  list(APPEND cpiquasso_files 
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/kernels/get_reflection_vector_AVX.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/kernels/calc_vH_times_A_AVX.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/kernels/calc_vov_times_A_AVX.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/kernels/apply_householder_cols_AVX.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/kernels/loop_correction_AVX.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/kernels/calc_cholesky_decomposition_AVX.cpp
  )
endif()

add_library(cpiquasso SHARED
    ${cpiquasso_files}
)

# adding compile options
target_compile_options(cpiquasso PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)


# adding linking options
target_link_libraries(cpiquasso PRIVATE
    "$<$<CONFIG:Debug>:${BLAS_LIBRARIES}>"
    "$<$<CONFIG:Release>:${BLAS_LIBRARIES}>"
    ${TBBMALLOC_LIB}
    ${TBBMALLOC_PROXY_LIB}
    ${TBB_LIB}
    ${MPI_C_LIBRARIES}    
)

target_include_directories(cpiquasso PRIVATE
                            .
                            ./cpiquasso/common/source
                            ./cpiquasso/gaussian/source
                            ./cpiquasso/sampling/source
                            ${TBB_HEADER}
                            ./cpiquasso/sampling/source/kernels
                            ${EXTRA_INCLUDES})



set_target_properties(cpiquasso PROPERTIES
    VERSION ${PROJECT_VERSION}
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/PicTypes.hpp
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/dependency_graph.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/matrix.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/matrix32.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/PicState.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/PicStateHash.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/dot.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/common_functionalities.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/CGaussianState.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/GaussianState_Cov.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/tasks_apply_to_C_and_G/extract_rows.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/tasks_apply_to_C_and_G/transform_rows.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/tasks_apply_to_C_and_G/extract_corner.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/tasks_apply_to_C_and_G/transform_cols.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/tasks_apply_to_C_and_G/insert_transformed_cols.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/tasks_apply_to_C_and_G/insert_transformed_rows.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/CChinHuhPermanentCalculator.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/Torontonian.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/TorontonianUtilities.hpp
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/PowerTraceHafnianUtilities.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/PowerTraceHafnian.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/PowerTraceHafnianRecursive.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/PowerTraceLoopHafnian.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/PowerTraceLoopHafnianRecursive.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/BruteForceHafnian.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/BruteForceLoopHafnian.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/simulation_strategies/source/CGeneralizedCliffordsSimulationStrategy.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/simulation_strategies/source/GaussianSimulationStrategy.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/simulation_strategies/source/GaussianSimulationStrategyFast.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/simulation_strategies/source/RepeatedColumnPairs.h
)

configure_file(${PROJECT_SOURCE_DIR}/cpiquasso/common/source/Config.h.in
               ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/Config.h)

set_target_properties( cpiquasso PROPERTIES
                        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/cpiquasso
)



###########################################################################




# adding subdirectories for python extensions
add_subdirectory (cpiquasso)

# adding CMAKE files for test executables
add_subdirectory (ctests)

install(TARGETS cpiquasso LIBRARY DESTINATION cpiquasso)


