# Gemini Code Companion: taichi_LBM3D 客製化

## 目標

使用 `taichi_LBM3D` 專案進行建築或都市尺度的風場模擬，並為未來加入熱模擬做準備。

## 待辦事項

1.  **幾何準備：**
    *   [已完成] 了解如何將建築或都市的幾何模型轉換為 LBM 求解器所需的格式。
    *   [已完成] 建立一個 Python 腳本 (`voxelize_geometry.py`)，使用 `trimesh` 函式庫將 `.stl` 檔案轉換為求解器所需的格式。

2.  **參數設定：**
    *   [進行中] 根據風場模擬的物理特性，調整 `Single_phase/lbm_solver_3d.py` 中的參數。
    *   [進行中] 學習如何設定合適的邊界條件來模擬風的入口、出口及建築物表面。

3.  **執行與視覺化：**
    *   運行一個簡單的範例來驗證設定。
    *   使用 ParaView 將結果視覺化，以分析風速、風壓等資訊。

4.  **未來工作 (熱模擬)：**
    *   研究如何在 LBM 框架中加入熱傳遞模型。
    *   規劃修改求解器以支援溫度場的計算。

## 操作指南

### 1. 準備幾何模型

我們已經建立了一個 Python 腳本 `Single_phase/voxelize_geometry.py` 來將您的 3D 模型轉換為 LBM 模擬所需的格式。

**如何使用:**

1.  **放置您的 STL 檔案:** 將您的 `.stl` 格式 3D 模型檔案（例如 `my_building.stl`）複製到 `Single_phase` 目錄下。

2.  **編輯腳本:** 打開 `Single_phase/voxelize_geometry.py` 檔案，並修改以下變數：
    ```python
    # ...
    if __name__ == '__main__':
        # 將 'path_to_your_stl_file.stl' 替換為您的 STL 檔案名稱
        stl_file = 'my_building.stl'  
        output_file = 'geometry.txt'
        # 設定您希望的體素大小 (解析度)
        voxel_size = 0.5
        voxelize_stl(stl_file, output_file, voxel_size)
    ```

3.  **執行腳本:** 在您的終端機中，執行以下指令：
    ```bash
    python Single_phase/voxelize_geometry.py
    ```

4.  **檢查輸出:** 腳本會產生一個名為 `geometry.txt` 的檔案。這個檔案就包含了您的 3D 模型的體素化表示。同時，腳本也會在終端機中印出模型的維度 (dimensions)，請將這個維度記錄下來。

### 2. 設定 LBM 求解器

接下來，我們需要設定 `Single_phase/lbm_solver_3d.py` 來進行風場模擬。

1.  **修改輸入檔案:**
    打開 `Single_phase/lbm_solver_3d.py`，找到以下這行：
    ```python
    # solid_np = init_geo('./img_ftb131.txt')
    ```
    將它修改為您剛剛產生的檔案：
    ```python
    solid_np = init_geo('./geometry.txt')
    ```

2.  **更新網格尺寸:**
    根據您在**步驟 1.4** 中記錄下來的維度，更新 `nx, ny, nz` 的值。例如，如果您的維度是 `(100, 80, 50)`，那麼就修改為：
    ```python
    nx,ny,nz = 100, 80, 50
    ```

3.  **設定物理參數:**
    *   **黏滯係數 (`niu`):** 對於風場模擬，您可能需要一個較小的值來模擬較高的雷諾數，例如 `niu = 0.01`。
    *   **外部力 (`fx`, `fy`, `fz`):** 這是驅動風場的主要力量。您可以設定一個 x 方向的力來模擬從左到右的風。例如：`fx,fy,fz = 1.0e-5, 0.0, 0.0`。

4.  **設定邊界條件:**
    這是非常關鍵的一步。假設風是從 x 軸的左邊吹向右邊：
    *   **入口 (x-left):** 設定為 "fix velocity" 或 "fix pressure" 邊界。例如，設定一個固定的速度入口：
        ```python
        # bc_x_left: 2 代表 fix velocity
        # vx_bcxl: 設定入口的 x 方向速度
        bc_x_left, rho_bcxl, vx_bcxl, vy_bcxl, vz_bcxl = 2, 1.0, 1.0e-2, 0.0, 0.0
        ```
    *   **出口 (x-right):** 設定為 "fix pressure" 邊界，讓空氣可以自由流出。
        ```python
        # bc_x_right: 1 代表 fix pressure
        # rho_bcxr: 設定出口的壓力
        bc_x_right, rho_bcxr, vx_bcxr, vy_bcxr, vz_bcxr = 1, 1.0, 0.0, 0.0, 0.0
        ```
    *   **側面 (y-left, y-right, z-left, z-right):** 建議設定為 "periodic" 邊界，來模擬一個無限寬廣的空間。
        ```python
        # bc_y_left: 0 代表 periodic
        bc_y_left, rho_bcyl, vx_bcyl, vy_bcyl, vz_bcyl = 0, 1.0, 0.0, 0.0, 0.0
        bc_y_right, rho_bcyr, vx_bcyr, vy_bcyr, vz_bcyr = 0, 1.0, 0.0, 0.0, 0.0
        # ... 對 z 軸也做同樣的設定
        ```

### 3. 執行與視覺化

完成以上設定後，您就可以執行模擬了：
```bash
python Single_phase/lbm_solver_3d.py
```
模擬會產生 `.vtk` 檔案，您可以使用 [ParaView](https://www.paraview.org/) 來開啟這些檔案，並將風速、壓力等物理量視覺化。

## 筆記

*   `Single_phase/lbm_solver_3d.py` 是我們主要的修改對象。
*   `Single_phase/Convert_stl_to_binary.cpp` 可能是將 STL 檔案轉換為求解器所需格式的關鍵。
*   幾何檔案格式為三維陣列，其中 `0` 代表流體 (空氣)，`1` 代表固體 (建築物)。
