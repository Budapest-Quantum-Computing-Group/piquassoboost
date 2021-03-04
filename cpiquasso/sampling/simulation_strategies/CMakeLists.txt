
set(EXT_DIR ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/simulation_strategies)


add_library( GeneralizedCliffordsSimulationStrategy_wrapper MODULE
    ${EXT_DIR}/GeneralizedCliffordsSimulationStrategy_wrapper.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/numpy_interface.cpp
)


target_link_libraries(GeneralizedCliffordsSimulationStrategy_wrapper cpiquasso ${BLAS_LIBRARIES})

python_extension_module(GeneralizedCliffordsSimulationStrategy_wrapper)

ADD_DEPENDENCIES (GeneralizedCliffordsSimulationStrategy_wrapper cpiquasso)


# adding compile options
target_compile_options(GeneralizedCliffordsSimulationStrategy_wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)


target_include_directories(GeneralizedCliffordsSimulationStrategy_wrapper PUBLIC
                            ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/simulation_strategies/source/
                            ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/
                            ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/
                            ${PROJECT_SOURCE_DIR}/cpiquasso/common/
                            ${EXTRA_INCLUDES}
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR})


set_target_properties( GeneralizedCliffordsSimulationStrategy_wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/../.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)

install(TARGETS GeneralizedCliffordsSimulationStrategy_wrapper LIBRARY DESTINATION cpiquasso/sampling/simulation_strategies)



##########################################################



add_library( GaussianSimulationStrategy_wrapper MODULE
    ${EXT_DIR}/GaussianSimulationStrategy_wrapper.cpp
    ${PROJECT_SOURCE_DIR}/cpiquasso/common/numpy_interface.cpp
)


target_link_libraries(GaussianSimulationStrategy_wrapper cpiquasso ${BLAS_LIBRARIES})

python_extension_module(GaussianSimulationStrategy_wrapper)

ADD_DEPENDENCIES (GaussianSimulationStrategy_wrapper cpiquasso)


# adding compile options
target_compile_options(GaussianSimulationStrategy_wrapper PRIVATE
    ${CXX_FLAGS}
    "$<$<CONFIG:Debug>:${CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CXX_FLAGS_RELEASE}>"
)


target_include_directories(GaussianSimulationStrategy_wrapper PUBLIC
                            ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/simulation_strategies/source/
                            ${PROJECT_SOURCE_DIR}/cpiquasso/gaussian/source/
                            ${PROJECT_SOURCE_DIR}/cpiquasso/sampling/source/
                            ${PROJECT_SOURCE_DIR}/cpiquasso/common/source/
                            ${PROJECT_SOURCE_DIR}/cpiquasso/common/
                            ${EXTRA_INCLUDES}
                            ${PYTHON_INCLUDE_DIR}
                            ${NUMPY_INC_DIR})


set_target_properties( GaussianSimulationStrategy_wrapper PROPERTIES
                        INSTALL_RPATH "$ORIGIN/../.."
                        LIBRARY_OUTPUT_DIRECTORY ${EXT_DIR}
)

install(TARGETS GaussianSimulationStrategy_wrapper LIBRARY DESTINATION cpiquasso/sampling/simulation_strategies)
