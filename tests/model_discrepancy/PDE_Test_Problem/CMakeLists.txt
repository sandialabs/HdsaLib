# Include directories for the test
include_directories(${CMAKE_CURRENT_SOURCE_DIR})  # Include the current directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../..)  # Include the src directory

# Add the test executable
add_executable(PDE_Test_Problem_example_sensitivity.exe example_sensitivity.cpp)

# Link the test executable against the HdsaLib interface library and any necessary libraries
target_link_libraries(PDE_Test_Problem_example_sensitivity.exe HdsaLib ${Trilinos_LIBRARIES} ${MPI_LIBRARIES})

# Set the output directory for the executable
set_target_properties(PDE_Test_Problem_example_sensitivity.exe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/model_discrepancy/PDE_Test_Problem
)

# Define the source directory and the target directory
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TARGET_DIR ${CMAKE_BINARY_DIR}/tests/model_discrepancy/PDE_Test_Problem)

# Copy the required files to the output directory
add_custom_command(TARGET PDE_Test_Problem_example_sensitivity.exe POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/Plotting_Script.m
        ${TARGET_DIR}/Plotting_Script.m
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/random_numbers.txt
        ${TARGET_DIR}/random_numbers.txt
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/Sabl_Output.mat
        ${TARGET_DIR}/Sabl_Output.mat
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/u_opt.txt
        ${TARGET_DIR}/u_opt.txt
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/z_opt.txt
        ${TARGET_DIR}/z_opt.txt
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/D.txt
        ${TARGET_DIR}/D.txt
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/Z.txt
        ${TARGET_DIR}/Z.txt
    # Copy the specified files from the reference_output directory
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/reference_output/prior_discrepancy_sample_1/Vector_3.txt
        ${TARGET_DIR}/reference_output/prior_discrepancy_sample_1/Vector_3.txt
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/reference_output/prior_discrepancy_evaluated_at_z_opt/Vector_10.txt
        ${TARGET_DIR}/reference_output/prior_discrepancy_evaluated_at_z_opt/Vector_10.txt
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/reference_output/posterior_discrepancy_mean_1.txt
        ${TARGET_DIR}/reference_output/posterior_discrepancy_mean_1.txt
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/reference_output/posterior_discrepancy_mean_3.txt
        ${TARGET_DIR}/reference_output/posterior_discrepancy_mean_3.txt
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/reference_output/posterior_discrepancy_samples_3/Vector_20.txt
        ${TARGET_DIR}/reference_output/posterior_discrepancy_samples_3/Vector_20.txt
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/reference_output/posterior_update_mean.txt 
        ${TARGET_DIR}/reference_output/posterior_update_mean.txt 
    COMMAND ${CMAKE_COMMAND} -E copy
        ${SOURCE_DIR}/reference_output/posterior_update_samples/Vector_40.txt
        ${TARGET_DIR}/reference_output/posterior_update_samples/Vector_40.txt
    COMMENT "Copying files to the output directory"
)
