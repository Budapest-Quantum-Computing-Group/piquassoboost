add_test(torontonian_flop_benchmark torontonian_flop_benchmark ...)

add_executable (torontonian_flop_benchmark torontonian_flop_benchmark.cpp)

target_include_directories(torontonian_flop_benchmark PRIVATE
                            ../piquassoboost/common/source
                            ../piquassoboost/sampling/source
                            ${TBB_HEADER}
                            ${EXTRA_INCLUDES})

target_link_libraries (torontonian_flop_benchmark
                           piquassoboost
                           ${TBBMALLOC_LIB}
                           ${TBBMALLOC_PROXY_LIB}
                           ${TBB_LIB}
                           ${BLAS_LIBRARIES}
                           ${MPI_C_LIBRARIES})

target_compile_options(torontonian_flop_benchmark PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)
