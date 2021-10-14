set(EXT_DIR ${PROJECT_SOURCE_DIR}/piquassoboost/sampling)


add_library( Boson_Sampling_Utilities_wrapper MODULE
    ${EXT_DIR}/Boson_Sampling_Utilities_wrapper.cpp
    ${PROJECT_SOURCE_DIR}/piquassoboost/common/numpy_interface.cpp
)

target_link_libraries(Boson_Sampling_Utilities_wrapper 
                          piquassoboost 
                          ${BLAS_LIBRARIES}
			   "-L/opt/maxcompiler-2021.1/lib"
                           "-L/opt/maxeler/maxeleros/lib"
                           "/home/rakytap/workspace/PermanentGlynnCPU/dist/release/lib/libPermanentGlynnDFE.so")

python_extension_module(Boson_Sampling_Utilities_wrapper)


ADD_DEPENDENCIES (Boson_Sampling_Utilities_wrapper piquassoboost)



# adding compile options
target_compile_options(Boson_Sampling_Utilities_wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
    "-DCPYTHON"
)



target_include_directories(Boson_Sampling_Utilities_wrapper PUBLIC
                            ${PROJECT_SOURCE_DIR}/piquassoboost/sampling/source/
                            ${PROJECT_SOURCE_DIR}/piquassoboost/common/source/
                            ${PROJECT_SOURCE_DIR}/piquassoboost/common/
                            ${EXTRA_INCLUDES}
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR}
"/opt/maxcompiler-2021.1/lib/simulator/include/")



set_target_properties( Boson_Sampling_Utilities_wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)



install(TARGETS Boson_Sampling_Utilities_wrapper LIBRARY DESTINATION piquassoboost/sampling)

