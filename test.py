import os
import subprocess

# Path to the project2.py script
project_script = 'p2cpy.py'

# Directory containing the test files
test_dir = './tests/'

# Directory to save the output files
output_dir = './outputs/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through the test files and run project2.py with each file as an input
for i in range(1, 15):
    test_file = os.path.join(test_dir, f'{i}.png')
    output_file = os.path.join(output_dir, f'output{i}.txt')
    with open(output_file, 'w') as f:
        subprocess.run(['python', project_script], input=test_file.encode(), stdout=f)