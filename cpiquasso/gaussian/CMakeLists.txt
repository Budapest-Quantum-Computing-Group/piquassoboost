set(EXT_DIR ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian)

set(CMAKE_VERBOSE_MAKEFILE ON)

add_library(state_wrapper MODULE
    ${EXT_DIR}/state_wrapper.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/numpy_interface.cpp
)

ADD_DEPENDENCIES (state_wrapper cpiquasso)

target_link_libraries(state_wrapper cpiquasso ${BLAS_LIBRARIES})

python_extension_module(state_wrapper)

# adding compile options
target_compile_options(state_wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)


target_include_directories(state_wrapper PUBLIC
                            ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/
                            ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/
                            ${PROJECT_SOURCE_DIR}/cpiquasso/common/
                            ${EXTRA_INCLUDES}
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR})

set_target_properties( state_wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)

install(TARGETS state_wrapper LIBRARY 
         DESTINATION cpiquasso/gaussian)
