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

set(EXT_DIR ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian)

set(CMAKE_VERBOSE_MAKEFILE ON)

add_library(calculation_extension MODULE
    ${EXT_DIR}/calculation_extension.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/common/numpy_interface.cpp
)

ADD_DEPENDENCIES (calculation_extension piquassoboost)

target_link_libraries(calculation_extension piquassoboost ${BLAS_LIBRARIES})

python_extension_module(calculation_extension)

# adding compile options
target_compile_options(calculation_extension PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)


target_include_directories(calculation_extension PUBLIC
                            ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian/source/
                            ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/
                            ${PROJECT_SOURCE_DIR}/piquassoboost/common/
                            ${EXTRA_INCLUDES}
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR})

set_target_properties( calculation_extension PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)

install(TARGETS calculation_extension LIBRARY 
         DESTINATION piquassoboost/gaussian)
