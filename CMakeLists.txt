cmake_minimum_required(VERSION 3.10.2)


# set the project name and version
project(CPiquasso VERSION 0.1)



# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# include CMAKE modules
include(CheckIncludeFile)
include(CheckIncludeFileCXX)

# variables for compile and link options
set(CXX_FLAGS_DEBUG)
set(CXX_FLAGS_RELEASE)
set(TBB_LIBS)
set(PYTHON_LIBS)
set(EXTRA_INCLUDES)
set(BLAS_LIBRARIES)
set(NUMPY_BLAS_LIB_DIR)
set(NUMPY_BLAS_LIB)
set(PYTHON_DATA_PATH)
set(TBB_INC_DIR)


#enable test target
enable_testing()


#################################################################
# checking python developer packages


# find out python packages
find_package(PythonInterp)
find_package(PythonLibs)


# Determine Numpy input directory
exec_program(${PYTHON_EXECUTABLE}
             ARGS "-c \"import numpy; print(numpy.get_include())\""
             OUTPUT_VARIABLE NUMPY_INC_DIR
             RETURN_VALUE NUMPY_NOT_FOUND
            )
if(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "NumPy headers not found")
endif()

# Determine CBLAS library behind Numpy
exec_program(
  ${PYTHON_EXECUTABLE}
  ARGS "-c \"import numpy; blas_info=numpy.__config__.get_info('blas_opt_info'); libs = blas_info.get('libraries'); print(libs[0])\""
  OUTPUT_VARIABLE NUMPY_BLAS_LIB
  RETURN_VALUE NUMPY_BLAS_NOT_FOUND
)

# Determine CBLAS library directorybehind Numpy
exec_program(
  ${PYTHON_EXECUTABLE}
  ARGS "-c \"import numpy; blas_info=numpy.__config__.get_info('blas_opt_info'); libs = blas_info.get('library_dirs'); print(libs[0])\""
  OUTPUT_VARIABLE NUMPY_BLAS_LIB_DIR
  RETURN_VALUE NUMPY_BLAS_NOT_FOUND
)


# Determine include and library path for additional dependencies
exec_program(${PYTHON_EXECUTABLE}
             ARGS "-c \"from sysconfig import get_paths; info = get_paths(); data_path=info.get('data',' '); print(data_path)\""
             OUTPUT_VARIABLE PYTHON_DATA_PATH
             RETURN_VALUE DATA_PATH_NOT_FOUND
            )
if(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "Python DATA not found")
endif()

find_package(PythonExtensions REQUIRED)


# setting paths for library dependencies
set(PYTHON_LIBS "${PYTHON_DATA_PATH}/lib" )
set(TBB_INC_DIR "${PYTHON_DATA_PATH}/include/")

set(CMAKE_VERBOSE_MAKEFILE ON)

#################################################################


# If MKL is enabled, ObenBLAS should be disabled
if(NUMPY_BLAS_LIB MATCHES "mkl")
  set(USE_OPENBLAS OFF)
  list(APPEND CXX_FLAGS_DEBUG "-DBLAS=1")
  list(APPEND CXX_FLAGS_RELEASE "-DBLAS=1")
  list(APPEND BLAS_LIBRARIES "-Wl,-rpath=${NUMPY_BLAS_LIB_DIR}" "-L${NUMPY_BLAS_LIB_DIR}" "-lmkl_rt" "-lpthread")

  # If OpenBlas is enabled, MKL should be disabled
elseif(NUMPY_BLAS_LIB MATCHES "openblas")
  set(USE_MKL OFF)
  list(APPEND CXX_FLAGS_DEBUG "-DBLAS=2")
  list(APPEND CXX_FLAGS_RELEASE "-DBLAS=2")
  list(APPEND BLAS_LIBRARIES "-Wl,-rpath=${NUMPY_BLAS_LIB_DIR}" "-L${NUMPY_BLAS_LIB_DIR}" "-lopenblas")
else()
  find_package(BLAS)

  if(NOT BLAS_FOUND)
    message(FATAL_ERROR "BLAS library not found")
  endif()

  list(APPEND BLAS_LIBRARIES ${BLAS_LIBRARIES})
  list(APPEND CXX_FLAGS_DEBUG "-DBLAS=0")
  list(APPEND CXX_FLAGS_RELEASE "-DBLAS=0")
endif()


# setting basic compile flags
list(APPEND CXX_FLAGS_DEBUG "-Wall" "-Wpedantic" "-Wextra" "-fexceptions" "-DDEBUG" "-fno-builtin-malloc" "-fno-builtin-calloc" "-fno-builtin-realloc" "-fno-builtin-free" "-fpermissive")
list(APPEND CXX_FLAGS_RELEASE "-Wall" "-O3" "-m64" "-msse3" "-mssse3" "-msse4.1" "-mavx2" "-DNDEBUG" "-fno-builtin-malloc" "-fno-builtin-calloc" "-fno-builtin-realloc" "-fno-builtin-free" "-fpermissive")


# Identify the compiler type and set compiler specific options
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # using Clang
  message("-- Using Clang compiler")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  message("-- Using GNU compiler")
  list(APPEND CXX_FLAGS_DEBUG "-g3" "-ggdb")
  list(APPEND CXX_FLAGS_RELEASE "-ftree-vectorize")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # using Intel C++
  message("-- Using Intel compiler")
  if (USE_MKL)
    list(APPEND CXX_FLAGS_DEBUG "-mkl" "-tbb")
    list(APPEND CXX_FLAGS_RELEASE "-mkl" "-tbb")
  endif()

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # using Visual Studio C++
  message("-- Using Visual Studio C++ compiler")
endif()

############################################################xx
# checking TBB libraries and headers

# adding TBB library dir if given
if(DEFINED ENV{TBB_LIB_DIR})
  find_library(TBB_LIB tbb PATH $ENV{TBB_LIB_DIR} )

  if(NOT TBB_LIB)
    message(FATAL_ERROR "TBB library not found")
  endif()

  find_library(TBBMALLOC_PROXY_LIB tbbmalloc_proxy PATH $ENV{TBB_LIB_DIR} )

  if(NOT TBBMALLOC_PROXY_LIB)
    message(FATAL_ERROR "TBBMALLOC_PROXY library not found")
  endif()

  message("-- Adding library directory $ENV{TBB_LIB_DIR}")
  list(APPEND TBB_LIBS "-Wl,-rpath=$ENV{TBB_LIB_DIR}" "-L$ENV{TBB_LIB_DIR}" "-ltbb" "-ltbbmalloc_proxy" "-ltbbmalloc")

else()

  find_library(TBB_LIB tbb PATH ${PYTHON_LIBS})

  if(NOT TBB_LIB)
    message(FATAL_ERROR "TBB library not found")
  endif()

  find_library(TBBMALLOC_PROXY_LIB tbbmalloc_proxy PATH ${PYTHON_LIBS} )

  if(NOT TBBMALLOC_PROXY_LIB)
    message(FATAL_ERROR "TBBMALLOC_PROXY library not found")
  endif()

  list(APPEND TBB_LIBS "-Wl,-rpath=${PYTHON_LIBS}" "-L${PYTHON_LIBS}" "-ltbb" "-ltbbmalloc_proxy" "-ltbbmalloc")

endif()

#set(CMAKE_REQUIRED_LIBRARIES "tbb")


# adding TBB include dir
if(DEFINED ENV{TBB_INC_DIR})

  check_include_file_cxx(tbb/tbb.h TBB_HEADER "-I$ENV{TBB_INC_DIR}")

  if(NOT TBB_HEADER)
    #message(FATAL_ERROR "TBB header tbb.h not found")
  endif()

  message("-- Adding include directory $ENV{TBB_INC_DIR}")
  list(APPEND EXTRA_INCLUDES "$ENV{TBB_INC_DIR}")

else()

  check_include_file_cxx(tbb/tbb.h TBB_HEADER "-I${TBB_INC_DIR}")
  list(APPEND EXTRA_INCLUDES "${TBB_INC_DIR}") 

  if(NOT TBB_HEADER)
    #message(FATAL_ERROR "TBB header tbb.h not found")
  endif()


endif()


add_library(cpiquasso SHARED
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/PicTypes.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/dependency_graph.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/matrix.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/PicState.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/PicStateHash.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/dot.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/CGaussianState.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/tasks_apply_to_C_and_G.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/CChinHuhPermanentCalculator.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/PowerTraceHafnian.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/simulation_strategies/source/CGeneralizedCliffordsSimulationStrategy.cpp
)

# adding compile options
target_compile_options(cpiquasso PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)

# adding linking options
target_link_libraries(cpiquasso PRIVATE
    "$<$<CONFIG:Debug>:${TBB_LIBS} ${BLAS_LIBRARIES}>"
    "$<$<CONFIG:Release>:${TBB_LIBS} ${BLAS_LIBRARIES}>"
)

target_include_directories(cpiquasso PRIVATE
                            .
                            ./cpiquasso/common/source
                            ./cpiquasso/sampling/source
                            ${EXTRA_INCLUDES})

set_target_properties(cpiquasso PROPERTIES
    VERSION ${PROJECT_VERSION}
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/PicTypes.hpp
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/dependency_graph.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/matrix.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/PicState.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/PicStateHash.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/dot.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/CGaussianState.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/tasks_apply_to_C_and_G.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/CChinHuhPermanentCalculator.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/PowerTraceHafnian.h
    PUBLIC_HEADER ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/simulation_strategies/source/CGeneralizedCliffordsSimulationStrategy.h
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


