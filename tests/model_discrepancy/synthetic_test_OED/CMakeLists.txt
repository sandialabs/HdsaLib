# Include directories for the test
include_directories(${CMAKE_CURRENT_SOURCE_DIR})  # Include the current directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../..)  # Include the src directory

# Add the test executable
add_executable(synthetic_test_OED_example_sensitivity.exe example_sensitivity.cpp)

# Link the test executable against the HdsaLib interface library and any necessary libraries
target_link_libraries(synthetic_test_OED_example_sensitivity.exe HdsaLib ${Trilinos_LIBRARIES} ${MPI_LIBRARIES})

# Set the output directory for the executable
set_target_properties(synthetic_test_OED_example_sensitivity.exe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/model_discrepancy/synthetic_test_OED
)

# Define the source directory and the target directory
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TARGET_DIR ${CMAKE_BINARY_DIR}/tests/model_discrepancy/synthetic_test_OED)

# Copy the required files to the output directory
add_custom_command(TARGET synthetic_test_OED_example_sensitivity.exe POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/random_numbers.txt
        ${TARGET_DIR}/random_numbers.txt
    # Copy the specified files from the reference_output directory
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/reference_output/sequential_oed_design_3.txt
        ${TARGET_DIR}/reference_output/sequential_oed_design_3.txt
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/reference_output/sequential_oed_design_5.txt
        ${TARGET_DIR}/reference_output/sequential_oed_design_5.txt
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/reference_output/sequential_oed_posterior_z_bar_2.txt
        ${TARGET_DIR}/reference_output/sequential_oed_posterior_z_bar_2.txt
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/reference_output/sequential_oed_posterior_z_bar_4.txt
        ${TARGET_DIR}/reference_output/sequential_oed_posterior_z_bar_4.txt
    COMMENT "Copying files to the output directory"
)
