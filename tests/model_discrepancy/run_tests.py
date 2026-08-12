import os
import subprocess

def execute_script_in_subdirectories(base_directory, subdirectories):
    """Execute 'python3 run_test.py' in each specified subdirectory."""
    RESET = "\033[0m"
    for subdirectory in subdirectories:
        # Construct the full path to the subdirectory
        full_path = os.path.join(base_directory, subdirectory)
        
        # Check if the subdirectory exists
        if os.path.isdir(full_path):
            print(RESET + f"Executing in: {full_path}")
            try:
                # Execute the command in the subdirectory
                subprocess.run(['python3', '../tester.py'], cwd=full_path, check=True)
                #print(f"Successfully executed 'python3 run_test.py' in {full_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error executing 'python3 tester.py' in {full_path}: {e}")
        else:
            print(f"Subdirectory {full_path} does not exist.")

def main():
    # Define the base directory where the subdirectories are located
    base_directory = '.'  # Change this to the appropriate base directory if needed
    subdirectories = [
        'PDE_Test_Problem',
        'PDE_Test_Problem_Continuation',
        'synthetic_test',
        'synthetic_test_continuation',
        'synthetic_test_bilaplacian',
        'synthetic_test_elliptic_prior',
        'synthetic_test_hessian_gevp',
        'synthetic_test_hyperparam_1D',
        'synthetic_test_lumped_mass',
        'synthetic_test_transient',
        'synthetic_test_transient_control',
        'synthetic_test_multi_state',
        'synthetic_test_transient_multi_state',
        'synthetic_test_OED',
        'synthetic_test_OUU',
        'lumped_mass_unit_test'
    ]

    # Execute the script in each subdirectory
    execute_script_in_subdirectories(base_directory, subdirectories)

if __name__ == "__main__":
    main()
