import os
import subprocess

# ANSI escape codes for colored output
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

# Set the file path to the "mrhyde" executable
mrhyde_path = os.path.expanduser('~/software/MrHyDE/build_opt/src/mrhyde')  

# Set the base directory to the current working directory
base_directory = os.getcwd()  

# Function to execute commands and handle errors
def execute_command(command, cwd=None):
    try:
        subprocess.run(command, check=True, shell=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}\n{e}")

# Walk through all subdirectories of the base directory
for subdir, _, _ in os.walk(base_directory):
    # Skip the base directory itself
    if subdir == base_directory:
        continue

    # Print only the folder name
    folder_name = os.path.basename(subdir)
    print(f"Processing: {folder_name}")

    # Create a symbolic link to "mrhyde" in the current subdirectory
    link_command = f"ln -s {mrhyde_path} ."
    execute_command(link_command, cwd=subdir)

    test_cases = [("input.yaml", "output.log", "output.gold")]
    if os.path.exists(os.path.join(subdir, "input_jacobian_check.yaml")):
        test_cases.append(
            (
                "input_jacobian_check.yaml",
                "output_jacobian_check.log",
                "output_jacobian_check.gold",
            )
        )

    for input_file, output_log, gold_file in test_cases:
        # Execute the mpi command from the current subdirectory
        mpi_command = f"mpiexec -n 2 ./mrhyde {input_file} > {output_log}"
        execute_command(mpi_command, cwd=subdir)

        # Compare the log files
        compare_command = f"diff {output_log} {gold_file}"
        comparison_result = subprocess.run(compare_command, shell=True, cwd=subdir)

        if comparison_result.returncode != 0:
            print(f"{RED}Test Failed: {output_log} differs from {gold_file}{RESET}")
            input(f"Press Enter to continue...")
        else:
            print(f"{GREEN}Test Passed: {output_log} matches {gold_file}{RESET}")

    # Clean up the output files
    cleanup_command = f"rm -rf hdsa_output output.log output_jacobian_check.log mrhyde"
    execute_command(cleanup_command, cwd=subdir)
