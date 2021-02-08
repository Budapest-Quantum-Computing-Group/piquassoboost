set(EXT_DIR ${PROJECT_SOURCE_DIR}/piquasso/sampling)


add_library( Boson_Sampling_Utilities_wrapper SHARED
    ${EXT_DIR}/Boson_Sampling_Utilities_wrapper.cpp  
)


ADD_DEPENDENCIES (Boson_Sampling_Utilities_wrapper piquasso)

set_target_properties(
    Boson_Sampling_Utilities_wrapper
    PROPERTIES
        PREFIX ""
        SUFFIX ""
        OUTPUT_NAME "Boson_Sampling_Utilities_wrapper${EXT_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
        LINKER_LANGUAGE C
    )



# adding compile options
target_compile_options(Boson_Sampling_Utilities_wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)

# adding linking options
target_link_libraries( Boson_Sampling_Utilities_wrapper PRIVATE
    "$<$<CONFIG:Debug>:${BLAS_LIBRARIES} ${PYTHON_LIBRARIES}>"
    "$<$<CONFIG:Release>:${BLAS_LIBRARIES} ${PYTHON_LIBRARIES}>"
)

target_include_directories(Boson_Sampling_Utilities_wrapper PUBLIC
                            ${PROJECT_SOURCE_DIR}/piquasso/sampling/source/
                            ${PROJECT_SOURCE_DIR}/piquasso/common/source/
                            ${EXTRA_INCLUDES}
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR})

target_link_libraries (Boson_Sampling_Utilities_wrapper PRIVATE
                        ${PIQUASSO_LIBRARIES})






