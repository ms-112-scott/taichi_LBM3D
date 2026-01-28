# 基於 Taichi 語言的格子玻爾茲曼方法 (Lattice Boltzmann Method, LBM) 流體求解器
# 原作者 : Wang (hietwll@gmail.com)
# 修改與繁體中文註解 : Gemini (針對碩士論文資料生成優化)

import sys
import os  # 新增：用於建立資料夾
import matplotlib
import numpy as np
from matplotlib import cm

import taichi as ti
import taichi.math as tm

# 初始化 Taichi，使用 GPU 加速
ti.init(arch=ti.gpu)


@ti.data_oriented
class lbm_solver:
    def __init__(
        self,
        name,  # 案例名稱
        nx,  # 網格寬度
        ny,  # 網格高度
        niu,  # 黏滯係數
        bc_type,  # 邊界類型
        bc_value,  # 邊界數值
        cy=0,  # 障礙物開關
        cy_para=[0.0, 0.0, 0.0],  # 障礙物參數
    ):
        self.name = name
        self.nx = nx
        self.ny = ny
        self.niu = niu
        self.tau = 3.0 * niu + 0.5
        self.inv_tau = 1.0 / self.tau

        # --- [新增功能 1] 自動計算特徵時間 (Tc) ---
        # Tc = L / U_in (特徵長度 / 入口速度)
        # 假設左邊界 (Index 0) 是入口
        input_vel = bc_value[0][0]
        if input_vel == 0:
            input_vel = 0.1  # 防止除以 0，給個默認值

        self.Tc = int(nx / input_vel)
        print(f"[{self.name}] 特徵時間 Tc 計算結果: 每 {self.Tc} 步 為一個週期。")

        # --- [新增功能 2] 建立輸出資料夾 ---
        self.output_dir = os.path.join("output", self.name)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[{self.name}] 數據將儲存於: {self.output_dir}")

        # 定義場變量
        self.rho = ti.field(float, shape=(nx, ny))
        self.vel = ti.Vector.field(2, float, shape=(nx, ny))
        self.mask = ti.field(float, shape=(nx, ny))
        self.f_old = ti.Vector.field(9, float, shape=(nx, ny))
        self.f_new = ti.Vector.field(9, float, shape=(nx, ny))

        self.w = (
            ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0
        )
        self.e = ti.types.matrix(9, 2, int)(
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]
        )

        self.bc_type = ti.field(int, 4)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))

        self.cy = cy
        self.cy_para = tm.vec3(cy_para)

    @ti.func
    def f_eq(self, i, j):
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * self.rho[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.kernel
    def init(self):
        self.vel.fill(0)
        self.rho.fill(1)
        self.mask.fill(0)
        for i, j in self.rho:
            self.f_old[i, j] = self.f_new[i, j] = self.f_eq(i, j)
            if self.cy == 1:
                if (i - self.cy_para[0]) ** 2 + (
                    j - self.cy_para[1]
                ) ** 2 <= self.cy_para[2] ** 2:
                    self.mask[i, j] = 1.0

    @ti.kernel
    def collide_and_stream(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                feq = self.f_eq(ip, jp)
                self.f_new[i, j][k] = (1 - self.inv_tau) * self.f_old[ip, jp][k] + feq[
                    k
                ] * self.inv_tau

    @ti.kernel
    def update_macro_var(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.rho[i, j] = 0
            self.vel[i, j] = 0, 0
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                self.rho[i, j] += self.f_new[i, j][k]
                self.vel[i, j] += (
                    tm.vec2(self.e[k, 0], self.e[k, 1]) * self.f_new[i, j][k]
                )
            self.vel[i, j] /= self.rho[i, j]

    @ti.kernel
    def apply_bc(self):
        for j in range(1, self.ny - 1):
            self.apply_bc_core(1, 0, 0, j, 1, j)
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j)
        for i in range(self.nx):
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2)
            self.apply_bc_core(1, 3, i, 0, i, 1)
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.cy == 1 and self.mask[i, j] == 1:
                self.vel[i, j] = 0, 0
                inb = 0
                jnb = 0
                if i >= self.cy_para[0]:
                    inb = i + 1
                else:
                    inb = i - 1
                if j >= self.cy_para[1]:
                    jnb = j + 1
                else:
                    jnb = j - 1
                self.apply_bc_core(0, 0, i, j, inb, jnb)

    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
        if outer == 1:
            if self.bc_type[dr] == 0:
                self.vel[ibc, jbc] = self.bc_value[dr]
            elif self.bc_type[dr] == 1:
                self.vel[ibc, jbc] = self.vel[inb, jnb]
        self.rho[ibc, jbc] = self.rho[inb, jnb]
        self.f_old[ibc, jbc] = (
            self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]
        )

    # --- [新增功能 3] 儲存 Snapshot 的函數 ---
    def save_snapshot(self, step, gui_img=None):
        filename_base = os.path.join(self.output_dir, f"step_{step:06d}")

        # 1. 儲存 NPY (原始數據，訓練 NCA 用)
        vel_data = self.vel.to_numpy()
        np.save(f"{filename_base}.npy", vel_data)

        # 2. 儲存 PNG (視覺圖片，論文展示用)
        if gui_img is not None:
            ti.tools.imwrite(gui_img, f"{filename_base}.png")

        print(f"Saved snapshot at step {step}")

    def solve(self):
        gui = ti.GUI(self.name, (self.nx, 2 * self.ny))
        self.init()

        # 準備計數器
        total_steps = 0
        is_running = True

        # 設定存檔間隔：這裡設為 Tc (滿足你的要求)
        # *注意*：如果要訓練 NCA，建議未來把這裡改成 50 或 100
        snapshot_interval = self.Tc

        # --- 特殊處理：儲存初始狀態 (Step 0) ---
        # 為了要存 Step 0 的圖，我們需要先「假跑」一次視覺化計算 (但不推進物理時間)
        # 這裡我們手動算一次顏色，或者簡單地存 NPY 即可
        self.save_snapshot(0)

        while is_running and gui.running:
            # 為了加速，每個 Loop 跑 10 個物理步
            steps_per_batch = 10
            for _ in range(steps_per_batch):
                self.collide_and_stream()
                self.update_macro_var()
                self.apply_bc()

            # 更新總步數
            total_steps += steps_per_batch

            # --- 視覺化代碼 ---
            vel = self.vel.to_numpy()
            ugrad = np.gradient(vel[:, :, 0])
            vgrad = np.gradient(vel[:, :, 1])
            vor = ugrad[1] - vgrad[0]
            vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5
            colors = [
                (1, 1, 0),
                (0.953, 0.490, 0.016),
                (0, 0, 0),
                (0.176, 0.976, 0.529),
                (0, 1, 1),
            ]
            my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "my_cmap", colors
            )
            vor_img = cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=-0.02, vmax=0.02), cmap=my_cmap
            ).to_rgba(vor)
            vel_img = cm.plasma(vel_mag / 0.15)
            img = np.concatenate((vor_img, vel_img), axis=1)

            gui.set_image(img)
            gui.show()

            # --- 檢查是否需要存檔 (每一個 Tc) ---
            # 我們使用取餘數的方式。由於 total_steps 是一批批增加的，
            # 我們檢查是否剛好跨過 Tc 的倍數，或者簡單地用接近判斷
            if total_steps % snapshot_interval < steps_per_batch:
                # 為了避免重複存檔，我們可以校正步數顯示
                save_step_label = (total_steps // snapshot_interval) * snapshot_interval
                if save_step_label > 0:  # 0 已經存過了
                    self.save_snapshot(save_step_label, img)


if __name__ == "__main__":
    flow_case = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    # 案例設定保持不變...
    if flow_case == 0:
        lbm = lbm_solver(
            "Karman_Vortex",
            801,
            201,
            0.01,
            [0, 0, 1, 0],
            [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            1,
            [160.0, 100.0, 20.0],
        )
        lbm.solve()
    elif flow_case == 1:
        lbm = lbm_solver(
            "Cavity_Flow",
            256,
            256,
            0.0255,
            [0, 0, 0, 0],
            [[0.0, 0.0], [0.1, 0.0], [0.0, 0.0], [0.0, 0.0]],
        )
        lbm.solve()
