set(EXT_DIR ${PROJECT_SOURCE_DIR}/cpiquasso/sampling)


add_library( Boson_Sampling_Utilities_wrapper MODULE
    ${EXT_DIR}/Boson_Sampling_Utilities_wrapper.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/numpy_interface.cpp
)

add_library( Torontonian_wrapper MODULE
    ${EXT_DIR}/Torontonian_wrapper.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/numpy_interface.cpp
)

target_link_libraries(Boson_Sampling_Utilities_wrapper 
                          cpiquasso 
                          ${BLAS_LIBRARIES})

target_link_libraries(Torontonian_wrapper 
                          cpiquasso 
                          ${BLAS_LIBRARIES})

python_extension_module(Boson_Sampling_Utilities_wrapper)

python_extension_module(Torontonian_wrapper)

ADD_DEPENDENCIES (Boson_Sampling_Utilities_wrapper cpiquasso)

ADD_DEPENDENCIES (Torontonian_wrapper cpiquasso)


# adding compile options
target_compile_options(Boson_Sampling_Utilities_wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)

# adding compile options
target_compile_options(Torontonian_wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)


target_include_directories(Boson_Sampling_Utilities_wrapper PUBLIC
                            ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/
                            ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/
                            ${PROJECT_SOURCE_DIR}/cpiquasso/common/
                            ${EXTRA_INCLUDES}
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR})


target_include_directories(Torontonian_wrapper PUBLIC
                            ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/
                            ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/
                            ${PROJECT_SOURCE_DIR}/cpiquasso/common/
                            ${EXTRA_INCLUDES}
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR})

set_target_properties( Boson_Sampling_Utilities_wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)

set_target_properties( Torontonian_wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)



install(TARGETS Boson_Sampling_Utilities_wrapper LIBRARY DESTINATION cpiquasso/sampling)

install(TARGETS Torontonian_wrapper LIBRARY DESTINATION cpiquasso/sampling)
