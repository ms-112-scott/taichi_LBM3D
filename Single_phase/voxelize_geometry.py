import trimesh
import numpy as np
import yaml
import os
from pyevtk.hl import gridToVTK  # 新增引用


def voxelize_stl(stl_file_path, output_file_path, pitch):
    """
    Voxelizes an STL file and saves the result to a text file AND a VTK file.
    """
    # Load the STL file
    mesh = trimesh.load_mesh(stl_file_path)

    # Voxelize the mesh
    voxelized_mesh = mesh.voxelized(pitch=pitch)

    # Get the binary matrix representation (Boolean: True=Solid, False=Fluid)
    binary_matrix = voxelized_mesh.matrix

    # Convert boolean to integer (0 for fluid, 1 for solid)
    # 注意：通常模擬軟體需要特定的排列順序 (Fortran vs C order)，請確認你的需求
    int_matrix = binary_matrix.astype(int)

    # 取得維度
    nx, ny, nz = int_matrix.shape

    # --- 既有邏輯：更新 Config ---
    config_file = os.environ.get("CONFIG_FILE", "config.yml")
    with open(config_file, "r+", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        config["simulation"]["nx"] = nx
        config["simulation"]["ny"] = ny
        config["simulation"]["nz"] = nz
        f.seek(0)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        f.truncate()

    case_dir = os.environ.get("CASE_DIR", ".")
    output_full_path = os.path.join(case_dir, output_file_path)

    # --- 既有邏輯：寫入自定義文字檔 ---
    print(f"Saving text output to {output_full_path}...")
    with open(output_full_path, "w") as f:
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    f.write(str(int_matrix[x, y, z]) + " ")
                f.write("\n")
            f.write("\n")

    # --- 新增邏輯：生成 VTK 檔案 ---
    # 建立檔案名稱 (去掉副檔名，因為 gridToVTK 會自動加上 .vtr 或 .vts)
    vtk_filename = os.path.join(case_dir, "voxelized_geometry")

    # 定義座標軸 (Coordinates)
    # 這裡假設起點是 (0,0,0)，每個體素大小為 pitch
    x = np.arange(0, nx + 1) * pitch
    y = np.arange(0, ny + 1) * pitch
    z = np.arange(0, nz + 1) * pitch

    # gridToVTK 需要的是連續的數據，並且通常需要將數據展平或保持正確形狀
    # cellData 是一個字典，key 是變數名稱，value 是 numpy array
    # 注意：gridToVTK 預期數據形狀為 (nx, ny, nz)
    print(f"Saving VTK output to {vtk_filename}.vtr ...")
    gridToVTK(vtk_filename, x, y, z, cellData={"Geometry": int_matrix})

    print(f"Done. Dimensions: {int_matrix.shape}")


if __name__ == "__main__":
    config_file = os.environ.get("CONFIG_FILE", "config.yml")
    # 檢查設定檔是否存在
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found.")
    else:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        stl_file = config["geometry"]["stl_file"]
        output_file = config["geometry"]["output_file"]
        voxel_size = config["geometry"]["voxel_size"]

        voxelize_stl(stl_file, output_file, voxel_size)
