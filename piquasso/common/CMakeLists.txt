set(EXT_DIR ${PROJECT_SOURCE_DIR}/piquasso/common)

add_library( dot_wrapper SHARED
    ${EXT_DIR}/dot_wrapper.cpp  
)


ADD_DEPENDENCIES (dot_wrapper piquasso)

set_target_properties(
    dot_wrapper
    PROPERTIES
        PREFIX ""
        SUFFIX ""
        OUTPUT_NAME "dot_wrapper${EXT_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
        LINKER_LANGUAGE C
    )



# adding compile options
target_compile_options(dot_wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)


target_include_directories(dot_wrapper PUBLIC
                            ${PROJECT_SOURCE_DIR}/piquasso/gaussian/source/
                            ${PROJECT_SOURCE_DIR}/piquasso/common/source/
                            ${EXTRA_INCLUDES}
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR})

target_link_libraries (dot_wrapper PRIVATE
                        piquasso
                        ${PYTHON_LIBRARIES}
                        ${BLAS_LIBRARIES})




