# Include directories for the test
include_directories(${CMAKE_CURRENT_SOURCE_DIR})  # Include the current directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../..)  # Include the src directory

# Add the test executable
add_executable(bf_synthetic_test_OUU_example_sensitivity.exe example_sensitivity.cpp)

# Link the test executable against the HdsaLib interface library and any necessary libraries
target_link_libraries(bf_synthetic_test_OUU_example_sensitivity.exe HdsaLib ${Trilinos_LIBRARIES} ${MPI_LIBRARIES})

# Set the output directory for the executable
set_target_properties(bf_synthetic_test_OUU_example_sensitivity.exe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/bifidelity_correction/synthetic_test_OUU
)

# Define the source directory and the target directory
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TARGET_DIR ${CMAKE_BINARY_DIR}/tests/bifidelity_correction/synthetic_test_OUU)

# Copy the required files to the output directory
add_custom_command(TARGET bf_synthetic_test_OUU_example_sensitivity.exe POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/Plotting_Script.m
        ${TARGET_DIR}/Plotting_Script.m
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/random_numbers.txt
        ${TARGET_DIR}/random_numbers.txt
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/z_opt.txt
        ${TARGET_DIR}/z_opt.txt
    # Copy the specified files from the reference_output directory
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/reference_output/z_update.txt
        ${TARGET_DIR}/reference_output/z_update.txt
    COMMENT "Copying files to the output directory"
)
