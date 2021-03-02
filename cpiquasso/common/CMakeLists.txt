set(EXT_DIR ${PROJECT_SOURCE_DIR}/cpiquasso/common)

add_library(dot_wrapper MODULE
    ${EXT_DIR}/dot_wrapper.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/numpy_interface.cpp
)

ADD_DEPENDENCIES(dot_wrapper cpiquasso)

target_link_libraries(dot_wrapper cpiquasso ${BLAS_LIBRARIES})

python_extension_module(dot_wrapper)

# adding compile options
target_compile_options(dot_wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)


target_include_directories(dot_wrapper PUBLIC
                            ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/
                            ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/
                            ${PROJECT_SOURCE_DIR}/cpiquasso/common/
                            ${EXTRA_INCLUDES}
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR})

set_target_properties( dot_wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)

install(TARGETS dot_wrapper LIBRARY DESTINATION cpiquasso/common)
