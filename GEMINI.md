# Gemini Code Companion: taichi_LBM3D

## 專案概述 (Project Overview)

`taichi_LBM3D` 專案是使用 Taichi 程式語言實作的 3D 晶格波茲曼方法 (LBM) 求解器集合。它旨在模擬流體流動，特別是多孔介質中的流動。Taichi 的使用使得這些求解器能夠在多核 CPU 和大規模並行 GPU 上執行，從而實現顯著的性能提升。

該專案包含三種主要類型的求解器：

- **單相流 (Single-phase):** 模擬單一流體的流動。
- **兩相流 (Two-phase):** 模擬兩種不混溶流體的流動，包括界面張力和接觸角效應。
- **灰度流 (Grey-Scale):** 一種單相求解器，使用孔隙空間的「灰度」表示來模擬多孔介質中的流動，允許比簡單的固體/流體區分更複雜的幾何形狀。

這些求解器使用 Python 編寫，核心計算透過 Taichi 加速。結果以 VTK 格式儲存，可用於 ParaView 等軟體進行視覺化。

## 專案結構 (Project Structure)

儲存庫分為幾個目錄，每個目錄包含一個特定的求解器或相關文件：

- `Single_phase/`: 包含單相 LBM 求解器 (`lbm_solver_3d.py`) 和相關範例。
- `2phase/`: 包含兩相 LBM 求解器 (`lbm_solver_3d_2phase.py`)。
- `Grey_Scale/`: 包含灰度 LBM 求解器 (`lbm_solver_3d_Macro_Sukop.py`)。
- `img/`: 包含模擬結果的圖像和動畫。
- `docs/`: 包含專案文件。
- `requirements.txt`: 列出 Python 的依賴項。

## 安裝 (Installation)

1.  **Taichi:** 核心依賴項是 Taichi 程式語言。

    ```bash
    python3 -m pip install taichi
    ```

    **注意:** 兩相流求解器 (`2phase/lbm_solver_3d_2phase.py`) 特別要求 `taichi<=0.8.5` 和 `taichi_glsl` 庫。

2.  **PyEVTK:** 用於將模擬結果匯出為 VTK 格式。
    ```bash
    pip install pyevtk
    ```

## 如何運行 (How to Run)

所有求解器都透過編輯其各自 Python 文件頂部的參數進行配置。要運行模擬，您通常需要：

1.  **選擇求解器:** 選擇與您要運行模擬類型相對應的 Python 文件（例如，`Single_phase/lbm_solver_3d.py`）。
2.  **設定計算後端:** 在 Python 文件中，將後端設置為 CPU 或 GPU：

    ```python
    # 適用於 CPU
    ti.init(arch=ti.cpu)

    # 適用於 GPU
    ti.init(arch=ti.gpu)
    ```

3.  **配置模擬參數:** 編輯文件以設定參數，例如：
    - 網格尺寸 (`nx`, `ny`, `nz`)
    - 流體屬性 (例如，粘度 `niu`)
    - 外部力 (`fx`, `fy`, `fz`)
    - 邊界條件
4.  **指定輸入文件:** 求解器需要輸入文件來定義幾何形狀，對於兩相流求解器，還需要初始相分佈。
5.  **運行腳本:** 從終端執行 Python 腳本：
    ```bash
    python Single_phase/lbm_solver_3d.py
    ```

### 求解器特定說明 (Solver-Specific Instructions)

- **單相流 (`Single_phase/lbm_solver_3d.py`):**
  - **輸入:** 需要一個幾何文件（例如，`img_ftb131.txt`），其中 `0` 表示流體，`1` 表示固體。
- **兩相流 (`2phase/lbm_solver_3d_2phase.py`):**
  - **輸入:** 需要一個幾何文件和一個相分佈文件（例如，`phase_ftb131.dat`），其中 `-1` 和 `1` 分別代表兩個相。
  - **依賴項:** 需要 `taichi<=0.8.5` 和 `taichi_glsl`。
- **灰度流 (`Grey_Scale/lbm_solver_3d_Macro_Sukop.py`):**
  - **輸入:** 需要一個「灰度」幾何文件（例如，`BC.dat`），其中值表示多孔介質的屬性。

## 開發慣例 (Development Conventions)

- **配置:** 所有模擬參數都硬編碼在求解器文件的頂部。沒有單獨的配置文件。
- **可擴展性:** 程式碼設計為可擴展的，但需要直接修改 Python 腳本。
- **Taichi 用法:** 核心計算迴圈使用 `@ti.kernel` 裝飾，由 Taichi 運行時執行。`@ti.func` 用於從 Taichi 核心內部呼叫的函數。

## 檔案歸類

- gemini 的說明md文件都放置於my_docs/gemini/底下 可以自行在創建子資料夾用於歸類

## 語言習慣

- 解釋性文字(程式註解 或是 說明文件) 都使用繁體中文
- code使用python 並且符合pythonic原則 使用清楚簡潔註解與標示，加上type hint
