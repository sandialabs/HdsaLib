# Add subdirectories for the tests
add_subdirectory(PDE_Test_Problem)
add_subdirectory(PDE_Test_Problem_Continuation)
add_subdirectory(synthetic_test)
add_subdirectory(synthetic_test_elliptic_prior)
add_subdirectory(synthetic_test_hessian_gevp)
add_subdirectory(synthetic_test_continuation)
add_subdirectory(synthetic_test_hyperparam_1D)
add_subdirectory(synthetic_test_transient)
add_subdirectory(synthetic_test_transient_control)
add_subdirectory(synthetic_test_multi_state)
add_subdirectory(synthetic_test_transient_multi_state)
add_subdirectory(synthetic_test_OED)
add_subdirectory(synthetic_test_OUU)
add_subdirectory(synthetic_test_lumped_mass)
add_subdirectory(lumped_mass_unit_test)
add_subdirectory(synthetic_test_bilaplacian)

# Define the source directory and the target directory
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TARGET_DIR ${CMAKE_BINARY_DIR}/tests/model_discrepancy/)

# Copy the run_tests.py file to the build directory above the test directories
add_custom_command(OUTPUT ${TARGET_DIR}/run_tests.py
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/run_tests.py
        ${TARGET_DIR}/run_tests.py
    COMMENT "Copying run_tests.py to the build directory"
)

# Copy the tester.py file to the build directory above the test directories
add_custom_command(OUTPUT ${TARGET_DIR}/tester.py
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/tester.py
        ${TARGET_DIR}/tester.py
    COMMENT "Copying tester.py to the build directory"
)

# Ensure the custom command runs after the subdirectories are processed
add_custom_target(copy_run_tests_model_discrepancy ALL DEPENDS ${TARGET_DIR}/run_tests.py)
add_custom_target(copy_tester_model_discrepancy ALL DEPENDS ${TARGET_DIR}/tester.py)