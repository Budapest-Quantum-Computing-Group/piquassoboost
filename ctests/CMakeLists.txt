# CMAKE to create standalone executables


# add tests to the build
add_test(hafnian_complex_matrix hafnian_complex_matrix ...)
add_test(determinant_testing determinant_testing ...)
add_test(hafnian_recursive_complex_matrix hafnian_recursive_complex_matrix ...)
add_test(loop_hafnian_recursive_complex_matrix loop_hafnian_recursive_complex_matrix ...)
add_test(brute_force_hafnian_complex_matrix brute_force_hafnian_complex_matrix ...)
add_test(brute_force_loop_hafnian_complex_matrix brute_force_loop_hafnian_complex_matrix ...)
add_test(dot_stride dot_stride ...)
add_test(extract_rows extract_rows ...)
add_test(extract_corner extract_corner ...)
add_test(insert_transformed_cols insert_transformed_cols ...)
add_test(insert_transformed_rows insert_transformed_rows ...)
add_test(common_functionalities common_functionalities ...)

add_executable (hafnian_complex_matrix hafnian_complex_matrix.cpp)
add_executable (determinant_testing determinant_testing.cpp)
add_executable (hafnian_recursive_complex_matrix hafnian_recursive_complex_matrix.cpp)
add_executable (loop_hafnian_recursive_complex_matrix loop_hafnian_recursive_complex_matrix.cpp)
add_executable (brute_force_hafnian_complex_matrix brute_force_hafnian_complex_matrix.cpp)
add_executable (brute_force_loop_hafnian_complex_matrix brute_force_loop_hafnian_complex_matrix.cpp)
add_executable (dot_stride dot_stride.cpp)
add_executable (extract_rows extract_rows.cpp)
add_executable (extract_corner extract_corner.cpp)
add_executable (insert_transformed_cols insert_transformed_cols.cpp)
add_executable (insert_transformed_rows insert_transformed_rows.cpp)
add_executable (common_functionalities common_functionalities.cpp)


target_include_directories(hafnian_complex_matrix PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/sampling/source
                            ${TBB_HEADER}
                            ${EXTRA_INCLUDES})
                            
target_include_directories(determinant_testing PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/sampling/source
                            ../cpiquasso/sampling/simulation_strategies/source
                            ../cpiquasso/gaussian/source
                            ${EXTRA_INCLUDES})

target_include_directories(hafnian_recursive_complex_matrix PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/sampling/source
                            ../cpiquasso/sampling/simulation_strategies/source
                            ${TBB_HEADER}
                            ${EXTRA_INCLUDES})

target_include_directories(loop_hafnian_recursive_complex_matrix PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/sampling/source
                            ${TBB_HEADER}
                            ${EXTRA_INCLUDES})

target_include_directories(brute_force_hafnian_complex_matrix PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/sampling/source
                            ${TBB_HEADER}
                            ${EXTRA_INCLUDES})


target_include_directories(brute_force_loop_hafnian_complex_matrix PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/sampling/source
                            ${TBB_HEADER}
                            ${EXTRA_INCLUDES})

target_include_directories(dot_stride PRIVATE
                            ../cpiquasso/common/source
                            ${EXTRA_INCLUDES})

target_include_directories(extract_rows PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/gaussian/source
                            ${TBB_HEADER}
                            ${EXTRA_INCLUDES})
target_include_directories(extract_corner PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/gaussian/source
                            ${EXTRA_INCLUDES})
                            
target_include_directories(insert_transformed_cols PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/gaussian/source
                            ${TBB_HEADER}
                            ${EXTRA_INCLUDES})
                            
target_include_directories(insert_transformed_rows PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/gaussian/source
                            ${TBB_HEADER}
                            ${EXTRA_INCLUDES})

target_include_directories(common_functionalities PRIVATE
                            ../cpiquasso/common/source
                            ${TBB_HEADER}
                            ${EXTRA_INCLUDES})
                            


# Link the executable to the qgd library. Since the qgd library has
# public include directories we will use those link directories when building
# decomposition_test
target_link_libraries (hafnian_complex_matrix 
                           cpiquasso
                           ${TBBMALLOC_LIB}
                           ${TBBMALLOC_PROXY_LIB}
                           ${TBB_LIB}
                           ${BLAS_LIBRARIES}
                           ${MPI_C_LIBRARIES})


target_link_libraries (determinant_testing 
                           cpiquasso
                           ${TBB_LIBS}
                           ${BLAS_LIBRARIES})


target_link_libraries (hafnian_recursive_complex_matrix 
                           cpiquasso
                           ${TBBMALLOC_LIB}
                           ${TBBMALLOC_PROXY_LIB}
                           ${TBB_LIB}
                           ${BLAS_LIBRARIES}
                           ${MPI_C_LIBRARIES})

target_link_libraries (loop_hafnian_recursive_complex_matrix 
                           cpiquasso
                           ${TBBMALLOC_LIB}
                           ${TBBMALLOC_PROXY_LIB}
                           ${TBB_LIB}
                           ${BLAS_LIBRARIES}
                           ${MPI_C_LIBRARIES})


target_link_libraries (brute_force_hafnian_complex_matrix
                           cpiquasso
                           ${TBBMALLOC_LIB}
                           ${TBBMALLOC_PROXY_LIB}
                           ${TBB_LIB}
                           ${BLAS_LIBRARIES}
                           ${MPI_C_LIBRARIES})

target_link_libraries (brute_force_loop_hafnian_complex_matrix
                           cpiquasso
                           ${TBBMALLOC_LIB}
                           ${TBBMALLOC_PROXY_LIB}
                           ${TBB_LIB}
                           ${BLAS_LIBRARIES}
                           ${MPI_C_LIBRARIES})

target_link_libraries (dot_stride
                           cpiquasso
                           ${TBBMALLOC_LIB}
                           ${TBBMALLOC_PROXY_LIB}
                           ${TBB_LIB}
                           ${BLAS_LIBRARIES}
                           ${MPI_C_LIBRARIES})
                           
target_link_libraries (extract_rows
                           cpiquasso
                           ${TBBMALLOC_LIB}
                           ${TBBMALLOC_PROXY_LIB}
                           ${TBB_LIB}
                           ${BLAS_LIBRARIES}
                           ${MPI_C_LIBRARIES})

target_link_libraries (extract_corner
                           cpiquasso
                           ${TBBMALLOC_LIB}
                           ${TBBMALLOC_PROXY_LIB}
                           ${TBB_LIB}
                           ${BLAS_LIBRARIES}
                           ${MPI_C_LIBRARIES})
                           
target_link_libraries (insert_transformed_cols
                           cpiquasso
                           ${TBBMALLOC_LIB}
                           ${TBBMALLOC_PROXY_LIB}
                           ${TBB_LIB}
                           ${BLAS_LIBRARIES}
                           ${MPI_C_LIBRARIES})
                           
                           
target_link_libraries (insert_transformed_rows
                           cpiquasso
                           ${TBBMALLOC_LIB}
                           ${TBBMALLOC_PROXY_LIB}
                           ${TBB_LIB}
                           ${BLAS_LIBRARIES}
                           ${MPI_C_LIBRARIES})
                           
                           
target_link_libraries (common_functionalities
                           cpiquasso
                           ${TBBMALLOC_LIB}
                           ${TBBMALLOC_PROXY_LIB}
                           ${TBB_LIB}
                           ${BLAS_LIBRARIES}
                           ${MPI_C_LIBRARIES})


# adding compile options
target_compile_options(hafnian_complex_matrix PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)


# adding compile options
target_compile_options(hafnian_recursive_complex_matrix PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)


# adding compile options
target_compile_options(loop_hafnian_recursive_complex_matrix PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)


# adding compile options
target_compile_options(brute_force_hafnian_complex_matrix PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)

# adding compile options
target_compile_options(brute_force_loop_hafnian_complex_matrix PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)

# adding compile options
target_compile_options(dot_stride PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)

# adding compile options
target_compile_options(extract_rows PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)

# adding compile options
target_compile_options(extract_corner PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)

# adding compile options
target_compile_options(insert_transformed_cols PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)

# adding compile options
target_compile_options(insert_transformed_rows PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)


