import os
import subprocess

# Set the path to the colmap executable
colmap_path = 'colmap'

# Set the path to the COLMAP project file
project_file = 'colmap_localization/project.colmap'

# Ensure the project file exists
if not os.path.exists(project_file):
    print(f"Error: The project file '{project_file}' does not exist.")
    exit(1)

# Function to launch COLMAP GUI with the project file
def launch_colmap_gui(project_path):
    command = f'{colmap_path} gui --project_path {project_path}'
    subprocess.run(command, shell=True)

# Launch COLMAP GUI
launch_colmap_gui(project_file)
