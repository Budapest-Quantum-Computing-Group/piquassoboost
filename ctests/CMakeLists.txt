# CMAKE to create standalone executables


# add tests to the build
add_test(hafnian_complex_matrix hafnian_complex_matrix ...)
add_test(brute_force_hafnian_complex_matrix brute_force_hafnian_complex_matrix ...)
add_test(dot_stride dot_stride ...)


# Add executable called "decomposition_test" that is built from the source files
# "decomposition_test.cpp". The extensions are automatically found.
add_executable (hafnian_complex_matrix hafnian_complex_matrix.cpp)
add_executable (brute_force_hafnian_complex_matrix brute_force_hafnian_complex_matrix.cpp)
add_executable (dot_stride dot_stride.cpp)


target_include_directories(hafnian_complex_matrix PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/sampling/source
                            ${EXTRA_INCLUDES})

target_include_directories(brute_force_hafnian_complex_matrix PRIVATE
                            ../cpiquasso/common/source
                            ../cpiquasso/sampling/source
                            ${EXTRA_INCLUDES})

target_include_directories(dot_stride PRIVATE
                            ../cpiquasso/common/source
                            ${EXTRA_INCLUDES})



# Link the executable to the qgd library. Since the qgd library has
# public include directories we will use those link directories when building
# decomposition_test
target_link_libraries (hafnian_complex_matrix 
                           cpiquasso
                           ${TBB_LIBS}
                           ${BLAS_LIBRARIES})


target_link_libraries (brute_force_hafnian_complex_matrix
                           cpiquasso
                           ${TBB_LIBS}
                           ${BLAS_LIBRARIES})


target_link_libraries (dot_stride
                           cpiquasso
                           ${TBB_LIBS}
                           ${BLAS_LIBRARIES})

