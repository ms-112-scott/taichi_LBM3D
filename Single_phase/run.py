import subprocess
import yaml
import os
import sys


def main():
    config_file = 'config.yml'

    with open(config_file, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    case_name = config.get("case_name", "default_case")
    case_dir = os.path.join(
        "..", "cases", case_name
    )  # Assuming run.py is in Single_phase

    os.makedirs(case_dir, exist_ok=True)
    print(f"Created case directory: {case_dir}")

    # Set environment variables for child processes
    os.environ["CONFIG_FILE"] = config_file
    os.environ["CASE_DIR"] = case_dir

    if config.get("run_mesh", True):  # Default to True if not specified
        print("Running voxelization...")
        subprocess.run([sys.executable, "voxelize_geometry.py"], check=True)
        print("Voxelization complete.")
    else:
        print("Skipping voxelization as per config.yml.")

    if config.get("run_simulation", True):  # Default to True if not specified
        print("Running LBM simulation...")
        subprocess.run([sys.executable, "lbm_solver_3d.py"], check=True)
        print("LBM simulation complete.")
    else:
        print("Skipping LBM simulation as per config.yml.")


if __name__ == "__main__":
    main()
