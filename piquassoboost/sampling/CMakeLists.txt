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

set(EXT_DIR ${PROJECT_SOURCE_DIR}/piquassoboost/sampling)


add_library( Boson_Sampling_Utilities_wrapper MODULE
    ${EXT_DIR}/Boson_Sampling_Utilities_wrapper.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/common/numpy_interface.cpp
)
add_library(permanent_calculators MODULE
    ${EXT_DIR}/permanent_calculators_wrapper.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/common/numpy_interface.cpp
)

target_link_libraries(Boson_Sampling_Utilities_wrapper 
                          piquassoboost 
                          ${BLAS_LIBRARIES})
target_link_libraries(permanent_calculators piquassoboost ${BLAS_LIBRARIES})

python_extension_module(Boson_Sampling_Utilities_wrapper)
python_extension_module(permanent_calculators)

ADD_DEPENDENCIES (Boson_Sampling_Utilities_wrapper piquassoboost)
ADD_DEPENDENCIES (permanent_calculators piquassoboost)


# adding compile options
target_compile_options(Boson_Sampling_Utilities_wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)
target_compile_options(permanent_calculators PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)




target_include_directories(Boson_Sampling_Utilities_wrapper PUBLIC
                            ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/
                            ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/
                            ${PROJECT_SOURCE_DIR}/piquassoboost/common/
                            ${EXTRA_INCLUDES}
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR})
target_include_directories(permanent_calculators PUBLIC
                            ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/
                            ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/
                            ${PROJECT_SOURCE_DIR}/piquassoboost/common/
                            ${EXTRA_INCLUDES}
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR})



set_target_properties( Boson_Sampling_Utilities_wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)
set_target_properties( permanent_calculators PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)


install(TARGETS Boson_Sampling_Utilities_wrapper LIBRARY DESTINATION piquassoboost/sampling)
install(TARGETS permanent_calculators LIBRARY 
         DESTINATION piquassoboost/sampling)
