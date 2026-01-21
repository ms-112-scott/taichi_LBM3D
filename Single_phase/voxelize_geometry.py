import trimesh
import numpy as np
import yaml

def voxelize_stl(stl_file_path, output_file_path, pitch):
    """
    Voxelizes an STL file and saves the result to a text file.

    Args:
        stl_file_path (str): Path to the input STL file.
        output_file_path (str): Path to the output text file.
        pitch (float): The size of a single voxel.
    """
    # Load the STL file
    mesh = trimesh.load_mesh(stl_file_path)

    # Voxelize the mesh
    voxelized_mesh = mesh.voxelized(pitch=pitch)

    # Get the binary matrix representation of the voxelized mesh
    binary_matrix = voxelized_mesh.matrix

    # Convert boolean to integer (0 for fluid, 1 for solid)
    int_matrix = binary_matrix.astype(int)

    # Save the dimensions to the config file
    with open('config.yml', 'r+') as f:
        config = yaml.safe_load(f)
        config['simulation']['nx'] = int_matrix.shape[0]
        config['simulation']['ny'] = int_matrix.shape[1]
        config['simulation']['nz'] = int_matrix.shape[2]
        f.seek(0)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        f.truncate()


    # Reshape and save to the file
    with open(output_file_path, 'w') as f:
        for z in range(int_matrix.shape[2]):
            for y in range(int_matrix.shape[1]):
                for x in range(int_matrix.shape[0]):
                    f.write(str(int_matrix[x, y, z]) + ' ')
                f.write('\n')
            f.write('\n')
    
    print(f"Voxelized geometry saved to {output_file_path}")
    print(f"Dimensions: {int_matrix.shape}")


if __name__ == '__main__':
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    stl_file = config['geometry']['stl_file']
    output_file = config['geometry']['output_file']
    voxel_size = config['geometry']['voxel_size']
    
    voxelize_stl(stl_file, output_file, voxel_size)
