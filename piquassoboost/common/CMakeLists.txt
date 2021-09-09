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

set(EXT_DIR ${PROJECT_SOURCE_DIR}/piquassoboost/common)

add_library(dot_wrapper MODULE
    ${EXT_DIR}/dot_wrapper.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/common/numpy_interface.cpp
)

ADD_DEPENDENCIES(dot_wrapper piquassoboost)

target_link_libraries(dot_wrapper piquassoboost ${BLAS_LIBRARIES})

python_extension_module(dot_wrapper)

# adding compile options
target_compile_options(dot_wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(dot_wrapper PUBLIC
                            ${PROJECT_SOURCE_DIR}/piquassoboost/gaussian/source/
                            ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/
                            ${PROJECT_SOURCE_DIR}/piquassoboost/common/
                            ${EXTRA_INCLUDES}
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR})

set_target_properties( dot_wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)

install(TARGETS dot_wrapper LIBRARY DESTINATION piquassoboost/common)
