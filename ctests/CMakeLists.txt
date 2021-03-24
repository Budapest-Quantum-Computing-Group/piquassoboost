# CMAKE to create standalone executables


# add tests to the build
add_test(hafnian_complex_matrix hafnian_complex_matrix ...)
add_test(hafnian_recursive_complex_matrix hafnian_recursive_complex_matrix ...)
add_test(brute_force_hafnian_complex_matrix brute_force_hafnian_complex_matrix ...)
add_test(brute_force_loop_hafnian_complex_matrix brute_force_loop_hafnian_complex_matrix ...)
add_test(dot_stride dot_stride ...)
add_test(extract_rows extract_rows ...)
add_test(extract_corner extract_corner ...)
add_test(insert_transformed_cols insert_transformed_cols ...)
add_test(insert_transformed_rows insert_transformed_rows ...)


# Add executable called "decomposition_test" that is built from the source files
# "decomposition_test.cpp". The extensions are automatically found.
add_executable (hafnian_complex_matrix hafnian_complex_matrix.cpp)
add_executable (hafnian_recursive_complex_matrix hafnian_recursive_complex_matrix.cpp)
add_executable (brute_force_hafnian_complex_matrix brute_force_hafnian_complex_matrix.cpp)
add_executable (brute_force_loop_hafnian_complex_matrix brute_force_loop_hafnian_complex_matrix.cpp)
add_executable (dot_stride dot_stride.cpp)
add_executable (extract_rows extract_rows.cpp)
add_executable (extract_corner extract_corner.cpp)
add_executable (insert_transformed_cols insert_transformed_cols.cpp)
add_executable (insert_transformed_rows insert_transformed_rows.cpp)


target_include_directories(hafnian_complex_matrix PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/sampling/source
                            ${EXTRA_INCLUDES})

target_include_directories(hafnian_recursive_complex_matrix PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/sampling/source
                            ${EXTRA_INCLUDES})

target_include_directories(brute_force_hafnian_complex_matrix PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/sampling/source
                            ${EXTRA_INCLUDES})


target_include_directories(brute_force_loop_hafnian_complex_matrix PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/sampling/source
                            ${EXTRA_INCLUDES})

target_include_directories(dot_stride PRIVATE
                            ../cpiquasso/common/source
                            ${EXTRA_INCLUDES})

target_include_directories(extract_rows PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/gaussian/source
                            ${EXTRA_INCLUDES})
target_include_directories(extract_corner PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/gaussian/source
                            ${EXTRA_INCLUDES})
                            
target_include_directories(insert_transformed_cols PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/gaussian/source
                            ${EXTRA_INCLUDES})
                            
target_include_directories(insert_transformed_rows PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/gaussian/source
                            ${EXTRA_INCLUDES})



# Link the executable to the qgd library. Since the qgd library has
# public include directories we will use those link directories when building
# decomposition_test
target_link_libraries (hafnian_complex_matrix 
                           cpiquasso
                           ${TBB_LIBS}
                           ${BLAS_LIBRARIES})


target_link_libraries (hafnian_recursive_complex_matrix 
                           cpiquasso
                           ${TBB_LIBS}
                           ${BLAS_LIBRARIES})


target_link_libraries (brute_force_hafnian_complex_matrix
                           cpiquasso
                           ${TBB_LIBS}
                           ${BLAS_LIBRARIES})

target_link_libraries (brute_force_loop_hafnian_complex_matrix
                           cpiquasso
                           ${TBB_LIBS}
                           ${BLAS_LIBRARIES})

target_link_libraries (dot_stride
                           cpiquasso
                           ${TBB_LIBS}
                           ${BLAS_LIBRARIES})
                           
target_link_libraries (extract_rows
                           cpiquasso
                           ${TBB_LIBS}
                           ${BLAS_LIBRARIES})

target_link_libraries (extract_corner
                           cpiquasso
                           ${TBB_LIBS}
                           ${BLAS_LIBRARIES})
                           
target_link_libraries (insert_transformed_cols
                           cpiquasso
                           ${TBB_LIBS}
                           ${BLAS_LIBRARIES})
                           
                           
target_link_libraries (insert_transformed_rows
                           cpiquasso
                           ${TBB_LIBS}
                           ${BLAS_LIBRARIES})


