# CMAKE to create standalone executables


# add tests to the build
add_test(hafnian_complex_matrix hafnian_complex_matrix ...)


# Add executable called "decomposition_test" that is built from the source files
# "decomposition_test.cpp". The extensions are automatically found.
add_executable (hafnian_complex_matrix hafnian_complex_matrix.cpp)


target_include_directories(hafnian_complex_matrix PRIVATE
                            ../piquasso/common/source
                            ../piquasso/sampling/source
                            ${EXTRA_INCLUDES})



# Link the executable to the qgd library. Since the qgd library has
# public include directories we will use those link directories when building
# decomposition_test
target_link_libraries (hafnian_complex_matrix 
                           piquasso
                           ${TBB_LIBS}
                           ${BLAS_LIBRARIES})





