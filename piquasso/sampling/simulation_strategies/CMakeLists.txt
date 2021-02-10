
set(EXT_DIR ${PROJECT_SOURCE_DIR}/piquasso/sampling/simulation_strategies)


add_library( GeneralizedCliffordsSimulationStrategy_wrapper SHARED
    ${EXT_DIR}/GeneralizedCliffordsSimulationStrategy_wrapper.cpp  
    ${PROJECT_SOURCE_DIR}/piquasso/common/numpy_interface.cpp
)


ADD_DEPENDENCIES (GeneralizedCliffordsSimulationStrategy_wrapper piquasso)

set_target_properties(
    GeneralizedCliffordsSimulationStrategy_wrapper
    PROPERTIES
        PREFIX ""
        SUFFIX ""
        OUTPUT_NAME "GeneralizedCliffordsSimulationStrategy_wrapper${EXT_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
        LINKER_LANGUAGE C
    )



# adding compile options
target_compile_options(GeneralizedCliffordsSimulationStrategy_wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)


target_include_directories(GeneralizedCliffordsSimulationStrategy_wrapper PUBLIC
                            ${PROJECT_SOURCE_DIR}/piquasso/sampling/simulation_strategies/source/
                            ${PROJECT_SOURCE_DIR}/piquasso/sampling/source/
                            ${PROJECT_SOURCE_DIR}/piquasso/common/source/
                            ${PROJECT_SOURCE_DIR}/piquasso/common/
                            ${EXTRA_INCLUDES}
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR})

target_link_libraries (GeneralizedCliffordsSimulationStrategy_wrapper PRIVATE
                        piquasso
                        ${PYTHON_LIBRARIES}
                        ${BLAS_LIBRARIES})




