# CMAKE to create standalone executables


# add tests to the build



add_executable (TorontonianDFE 
               TorontonianDFE.cpp)



target_include_directories(TorontonianDFE PRIVATE
                            ../piquassoboost/common/source
                            ../piquassoboost/sampling/source
                            ../ctests
                            ${TBB_HEADER}
                            ${EXTRA_INCLUDES})
                            



target_link_libraries (TorontonianDFE 
                           piquassoboost
                           ${TBBMALLOC_LIB}
                           ${TBBMALLOC_PROXY_LIB}
                           ${TBB_LIB}
                           ${MPI_C_LIBRARIES}
                           "-L/opt/maxcompiler-2021.1/lib"
                           "-L/opt/maxeler/maxeleros/lib"
                           "-lslic" "-lmaxeleros" "-lcurl"
                           "/home/rakytap/workspace/TorontonianCPU/dist/release/lib/libTorontonianDFE.so")



# adding compile options
target_compile_options(TorontonianDFE PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)














add_executable (PermanentGlynnDFE 
               Random_Unitary.cpp
               PermanentGlynnDFE.cpp)



target_include_directories(PermanentGlynnDFE PRIVATE
                            ../piquassoboost/common/source
                            ../piquassoboost/sampling/source
                            ../ctests
                            ${TBB_HEADER}
                            ${EXTRA_INCLUDES})


target_link_libraries (PermanentGlynnDFE 
                           piquassoboost
                           ${TBBMALLOC_LIB}
                           ${TBBMALLOC_PROXY_LIB}
                           ${TBB_LIB}
                           ${MPI_C_LIBRARIES}
                           "-L/opt/maxcompiler-2021.1/lib"
                           "-L/opt/maxeler/maxeleros/lib"
                           "-lslic" "-lmaxeleros" "-lcurl"
                           "/home/rakytap/workspace/PermanentGlynnCPU/dist/release/lib/libPermanentGlynnDFE.so")



# adding compile options
target_compile_options(PermanentGlynnDFE PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)





