# Fluid solver based on lattice boltzmann method using taichi language
# Original Author : Wang (hietwll@gmail.com)
# Refactored for batch processing and integration into the Taichi-LBM-Dataset project.

import sys
import yaml
import argparse
import os
import shutil
import math
import numpy as np
import taichi as ti
import taichi.math as tm
import zarr
import time

ti.init(arch=ti.gpu)

@ti.data_oriented
class LBM_Solver_User_Refactored:
    def __init__(self, config_file, case_dir):
        self.config_file = config_file
        self.case_dir = case_dir
        self.load_config()
        self.allocate_fields()
        self.init_constants()

    def load_config(self):
        """從YAML檔案載入模擬參數"""
        with open(self.config_file, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        sim_config = self.config["simulation"]
        
        self.nx = sim_config["nx"]
        self.ny = sim_config["ny"]
        self.niu = sim_config["niu"]
        self.max_timestep = sim_config["max_timestep"]
        
        # 邊界條件 (可選)
        self.bc_type_np = np.array(sim_config.get("bc_type", [1, 1, 1, 1]), dtype=np.int32)
        self.bc_value_np = np.array(sim_config.get("bc_value", [[0,0],[0,0],[0,0],[0,0]]), dtype=np.float32)

        # 輸出設定
        self.output_frequency = sim_config.get("output_frequency", 1000)
        self.zarr_output_frequency = sim_config.get("zarr_output_frequency", 100)
        self.zarr_data_path = sim_config.get("zarr_data_path", "zarr_data_2d_user")

    def allocate_fields(self):
        """根據參數配置Taichi Fields"""
        self.rho = ti.field(float, shape=(self.nx, self.ny))
        self.vel = ti.Vector.field(2, float, shape=(self.nx, self.ny))
        self.mask = ti.field(float, shape=(self.nx, self.ny))
        self.f_old = ti.Vector.field(9, float, shape=(self.nx, self.ny))
        self.f_new = ti.Vector.field(9, float, shape=(self.nx, self.ny))

    def init_constants(self):
        """初始化LBM方法所用的常數"""
        self.tau = 3.0 * self.niu + 0.5
        self.inv_tau = 1.0 / self.tau
        
        self.w = ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0
        self.e = ti.types.matrix(9, 2, int)([0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1])

        self.bc_type = ti.field(int, 4)
        self.bc_type.from_numpy(self.bc_type_np)
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(self.bc_value_np)

    @ti.func
    def f_eq(self, i, j):
        """計算平衡態分佈函數"""
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * self.rho[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.kernel
    def init_fields(self):
        """初始化流場與分佈函數"""
        self.vel.fill(0)
        self.rho.fill(1)
        for i, j in self.rho:
            # mask field 從 load_geometry 載入，這裡不再處理
            self.f_old[i, j] = self.f_new[i, j] = self.f_eq(i, j)

    def load_geometry(self):
        """從檔案載入幾何遮罩"""
        geo_file = self.config["simulation"]["input_file"]
        if not os.path.isabs(geo_file):
             geo_file = os.path.join(self.case_dir, geo_file)
        
        try:
            in_dat = np.loadtxt(geo_file)
            in_dat = np.reshape(in_dat, (self.nx, self.ny), order="F")
            self.mask.from_numpy(in_dat)
            print(f"成功從 {geo_file} 載入幾何。")
        except Exception as e:
            print(f"錯誤: 無法載入幾何檔案 {geo_file}。錯誤訊息: {e}")
            print("將使用無障礙物的空白流場。")
            self.mask.fill(0)


    @ti.kernel
    def collide_and_stream(self):
        """LBM 核心: 碰撞與遷移"""
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                feq = self.f_eq(ip, jp)
                # BGK 碰撞
                self.f_new[i, j][k] = (1 - self.inv_tau) * self.f_old[ip, jp][k] + feq[k] * self.inv_tau

    @ti.kernel
    def update_macro_var(self):
        """計算宏觀物理量 (密度、速度)"""
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.rho[i, j] = 0.0
            self.vel[i, j] = 0.0, 0.0
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                self.rho[i, j] += self.f_new[i, j][k]
                self.vel[i, j] += tm.vec2(self.e[k, 0], self.e[k, 1]) * self.f_new[i, j][k]

            self.vel[i, j] /= self.rho[i, j]
            
            # 將遮罩區域速度設為0
            if self.mask[i,j] == 1.0:
                self.vel[i,j] = 0.0, 0.0


    @ti.kernel
    def apply_bc(self):
        """施加邊界條件"""
        # 左邊界和右邊界
        for j in range(1, self.ny - 1):
            self.apply_bc_core(1, 0, 0, j, 1, j) # 左
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j) # 右

        # 上邊界和下邊界
        for i in range(self.nx):
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2) # 上
            self.apply_bc_core(1, 3, i, 0, i, 1) # 下

        # 圓柱障礙物 (在此版本中由 mask 取代)
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.mask[i, j] == 1.0:
                self.apply_bc_core(0, 0, i, j, i + 1, j) # 簡易處理，可再改進


    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
        """邊界條件核心函數"""
        if outer == 1:  # 處理外邊界
            if self.bc_type[dr] == 0: # Dirichlet
                self.vel[ibc, jbc] = self.bc_value[dr]
            elif self.bc_type[dr] == 1: # Neumann
                self.vel[ibc, jbc] = self.vel[inb, jnb]

        self.rho[ibc, jbc] = self.rho[inb, jnb]
        self.f_old[ibc, jbc] = self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]

    def setup_zarr(self):
        """初始化 Zarr 檔案儲存"""
        self.zarr_write_index = 0
        if self.zarr_output_frequency > 0:
            zarr_path = os.path.join(self.case_dir, self.zarr_data_path)
            if os.path.exists(zarr_path):
                shutil.rmtree(zarr_path)
            
            num_frames = math.ceil(self.max_timestep / self.zarr_output_frequency)
            root = zarr.open(zarr_path, mode='w')

            self.zarr_v = root.create_dataset(
                'velocity', shape=(num_frames, self.nx, self.ny, 2), chunks=(1, self.nx, self.ny, 2), dtype='f4')
            self.zarr_rho = root.create_dataset(
                'density', shape=(num_frames, self.nx, self.ny), chunks=(1, self.nx, self.ny), dtype='f4')
            zarr_solid = root.create_dataset(
                'geometry', shape=(self.nx, self.ny), chunks=(self.nx, self.ny), dtype='f4')
            zarr_solid[...] = self.mask.to_numpy()
            print(f"Zarr store initialized at: {zarr_path}")

    def write_zarr(self, iter):
        """將當前幀寫入 Zarr 檔案"""
        if self.zarr_output_frequency > 0 and iter > 0 and iter % self.zarr_output_frequency == 0:
            if self.zarr_write_index < self.zarr_v.shape[0]:
                self.zarr_v[self.zarr_write_index] = self.vel.to_numpy()
                self.zarr_rho[self.zarr_write_index] = self.rho.to_numpy()
                print(f"--- Saved frame {self.zarr_write_index} to Zarr store at iteration {iter}")
                self.zarr_write_index += 1

    def run(self):
        """執行模擬的主循環"""
        self.load_geometry()
        self.init_fields()
        self.setup_zarr()
        
        time_init = time.time()
        for iter in range(self.max_timestep + 1):
            self.collide_and_stream()
            self.update_macro_var()
            self.apply_bc()
            
            if iter % self.output_frequency == 0:
                print(f"Iter {iter}/{self.max_timestep}, Time: {time.time() - time_init:.2f}s")

            self.write_zarr(iter)
        
        print(f"--- Simulation complete. Total time: {time.time() - time_init:.2f}s ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refactored Taichi 2D LBM Solver")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    parser.add_argument("--case-dir", type=str, required=True, help="Path to the case directory for inputs and outputs.")
    args = parser.parse_args()

    solver = LBM_Solver_User_Refactored(config_file=args.config, case_dir=args.case_dir)
    solver.run()
