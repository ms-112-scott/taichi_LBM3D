import subprocess
import yaml
import os

def main():
    config_file = 'config.yml'
    os.environ['CONFIG_FILE'] = config_file

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if config.get('run_mesh', True):  # Default to True if not specified
        print("Running voxelization...")
        subprocess.run(['python', 'voxelize_geometry.py'], check=True)
        print("Voxelization complete.")
    else:
        print("Skipping voxelization as per config.yml.")

    if config.get('run_simulation', True):  # Default to True if not specified
        print("Running LBM simulation...")
        subprocess.run(['python', 'lbm_solver_3d.py'], check=True)
        print("LBM simulation complete.")
    else:
        print("Skipping LBM simulation as per config.yml.")

if __name__ == '__main__':
    main()
