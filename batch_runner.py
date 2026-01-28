import os
import sys
import yaml
import subprocess
import glob
from datetime import datetime

import os
import sys
import yaml
import subprocess
import glob
import argparse
from datetime import datetime

# --- 全域變數 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BATCH_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "cases", "batch_runs")

def main(args):
    """
    主要函數，用於協調 LBM 模擬的批量處理。
    """
    print("--- 啟動 LBM 批量模擬執行器 ---")

    # --- 1. 處理路徑 ---
    # 將使用者傳入的相對路徑轉為絕對路徑
    solver_script_path = os.path.abspath(args.solver)
    base_config_path = os.path.abspath(args.config)
    geometry_input_dir = os.path.abspath(args.geodir)

    # 2. 創建主要的批量輸出目錄 (如果不存在)
    os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)
    print(f"批量輸出目錄: {BATCH_OUTPUT_DIR}")

    # 3. 載入基礎配置檔
    try:
        with open(base_config_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
        print(f"已載入基礎配置: {base_config_path}")
    except FileNotFoundError:
        print(f"錯誤: 基礎配置檔未在 {base_config_path} 找到")
        sys.exit(1)

    # 4. 尋找所有待處理的幾何文件
    geometry_files = glob.glob(os.path.join(geometry_input_dir, "*.txt"))
    if not geometry_files:
        print(f"錯誤: 未在 {geometry_input_dir} 找到任何幾何文件 (*.txt)")
        sys.exit(1)
    
    print(f"在 {geometry_input_dir} 中找到 {len(geometry_files)} 個幾何文件待處理。")

    # 5. 遍歷每個幾何文件並執行模擬
    for geo_file_path in geometry_files:
        try:
            # --- a. 準備本次運行的特定配置 ---
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            geo_filename = os.path.basename(geo_file_path)
            case_name = os.path.splitext(geo_filename)[0]
            run_name = f"{case_name}_{timestamp}"
            
            run_dir = os.path.join(BATCH_OUTPUT_DIR, run_name)
            os.makedirs(run_dir, exist_ok=True)
            
            print(f"\n--- 處理中: {geo_filename} ---")
            print(f"創建運行目錄: {run_dir}")

            # --- b. 在記憶體中修改配置 ---
            run_config = base_config.copy()
            run_config["case_name"] = run_name
            run_config["simulation"]["input_file"] = os.path.abspath(geo_file_path)

            # --- c. 將新的配置檔寫入運行目錄 ---
            new_config_path = os.path.join(run_dir, "config.yml")
            with open(new_config_path, "w", encoding="utf-8") as f:
                yaml.dump(run_config, f, allow_unicode=True)
            print(f"已將本次運行配置寫入: {new_config_path}")

            # --- d. 建構並執行命令 ---
            command = [
                sys.executable,
                solver_script_path,
                "--config",
                new_config_path,
                "--case-dir",
                run_dir,
            ]
            
            print(f"執行命令: {' '.join(command)}")
            
            # 將工作目錄設定為求解器所在的目錄，以處理潛在的相對路徑依賴
            working_dir = os.path.dirname(solver_script_path)

            # 準備子進程的環境變數，確保能找到虛擬環境中的套件
            sub_env = os.environ.copy()
            site_packages_path = os.path.join(PROJECT_ROOT, ".venv", "Lib", "site-packages")
            sub_env['PYTHONPATH'] = f"{site_packages_path}{os.pathsep}{sub_env.get('PYTHONPATH', '')}"
            # 強制子進程使用 UTF-8 編碼
            sub_env['PYTHONIOENCODING'] = 'utf-8'

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                cwd=working_dir,
                env=sub_env,
                encoding='utf-8',
                errors='replace' # 如果仍有解碼錯誤，則替換無效字符
            )
            
            # --- e. 記錄結果 ---
            print("--- 標準輸出 (STDOUT) ---")
            print(result.stdout)
            
            if result.returncode == 0:
                print(f"--- 成功: 完成處理 {geo_filename} ---")
            else:
                print(f"--- 錯誤: 處理 {geo_filename} 失敗 ---")
                print(f"返回碼: {result.returncode}")
                if result.stderr:
                    print("--- 標準錯誤 (STDERR) ---")
                    print(result.stderr)
                
        except Exception as e:
            print(f"處理 {geo_file_path} 時發生意外錯誤: {e}")

    print("\n--- 批量處理完成。 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通用 LBM 批量模擬執行器")
    parser.add_argument(
        "-s", "--solver", 
        type=str, 
        required=True,
        help="要運行的求解器腳本路徑 (例如: Single_phase/lbm_solver_2d_user_refactored.py)"
    )
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        required=True,
        help="用於模擬的基礎 YAML 配置檔路徑 (例如: Single_phase/config_2d_user.yml)"
    )
    parser.add_argument(
        "-g", "--geodir", 
        type=str, 
        required=True,
        help="包含幾何檔案 (*.txt) 的目錄路徑 (例如: cases/geometries_for_batch_2d/)"
    )
    
    args = parser.parse_args()
    main(args)

