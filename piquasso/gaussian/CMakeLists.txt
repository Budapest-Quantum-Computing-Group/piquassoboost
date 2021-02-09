set(EXT_DIR ${PROJECT_SOURCE_DIR}/piquasso/gaussian)

add_library( state_wrapper SHARED
    ${EXT_DIR}/state_wrapper.cpp  
)


ADD_DEPENDENCIES (state_wrapper piquasso)

set_target_properties(
    state_wrapper
    PROPERTIES
        PREFIX ""
        SUFFIX ""
        OUTPUT_NAME "state_wrapper${EXT_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
        LINKER_LANGUAGE C
    )




# adding compile options
target_compile_options(state_wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)


target_include_directories(state_wrapper PUBLIC
                            ${PROJECT_SOURCE_DIR}/piquasso/gaussian/source/
                            ${PROJECT_SOURCE_DIR}/piquasso/common/source/
                            ${EXTRA_INCLUDES}
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR})

#set_property(state_wrapper python PROPERTY IMPORTED_LOCATION ${PYTHON_LIBRARIES})

target_link_libraries (state_wrapper PRIVATE
                        piquasso
                        ${PYTHON_LIBRARIES}
                        ${BLAS_LIBRARIES})




