
import taichi as ti
import numpy as np
import yaml
import os
import argparse
import zarr
import math
import shutil

@ti.data_oriented
class LBM_D2Q9_BGK:
    def __init__(self, config_file, case_dir):
        self.config_file = config_file
        self.case_dir = case_dir
        self.load_config()
        self.init_taichi()
        self.allocate_fields()
        self.init_constants()

    def load_config(self):
        with open(self.config_file, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        sim_config = self.config["simulation"]
        
        self.nx, self.ny = sim_config["nx"], sim_config["ny"]
        self.fx, self.fy = sim_config.get("fx", 0.0), sim_config.get("fy", 0.0)
        self.niu = sim_config["niu"]
        self.max_timestep = sim_config["max_timestep"]
        self.output_frequency = sim_config.get("output_frequency", 1000)
        self.vtk_frequency = sim_config.get("vtk_frequency", 0) # VTK is optional
        self.zarr_output_frequency = sim_config.get("zarr_output_frequency", 100)
        self.zarr_data_path = sim_config.get("zarr_data_path", "zarr_data")

    def init_taichi(self):
        ti.init(arch=ti.gpu, kernel_profiler=False, print_ir=False)

    def allocate_fields(self):
        self.f = ti.field(ti.f32, shape=(self.nx, self.ny, 9))
        self.F = ti.field(ti.f32, shape=(self.nx, self.ny, 9))
        self.rho = ti.field(ti.f32, shape=(self.nx, self.ny))
        self.v = ti.Vector.field(2, ti.f32, shape=(self.nx, self.ny))
        self.solid = ti.field(ti.i32, shape=(self.nx, self.ny))

    def init_constants(self):
        self.tau = 3.0 * self.niu + 0.5
        self.inv_tau = 1.0 / self.tau

        # D2Q9 lattice vectors and weights
        self.e = ti.Vector.field(2, dtype=ti.i32, shape=9)
        self.w = ti.field(ti.f32, shape=9)
        self.LR = ti.field(ti.i32, shape=9) # Opposite directions

        e_np = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], 
                         [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)
        self.e.from_numpy(e_np)
        
        w_np = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32)
        self.w.from_numpy(w_np)

        self.LR.from_numpy(np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32))

    def init_simulation_state(self):
        self.init_fields()
        self.setup_zarr()

    @ti.func
    def feq(self, i, j, k):
        eu = self.e[k].dot(self.v[i, j])
        uv = self.v[i, j].dot(self.v[i, j])
        return self.w[k] * self.rho[i, j] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.kernel
    def init_fields(self):
        for i, j in self.rho:
            self.rho[i, j] = 1.0
            self.v[i, j] = ti.Vector([0.0, 0.0])
            for k in ti.static(range(9)):
                self.f[i, j, k] = self.feq(i, j, k)
                self.F[i, j, k] = self.feq(i, j, k)

    def load_geometry(self):
        geo_file = self.config["simulation"]["input_file"]
        if not os.path.isabs(geo_file):
             geo_file = os.path.join(self.case_dir, geo_file)

        in_dat = np.loadtxt(geo_file)
        in_dat = np.reshape(in_dat, (self.nx, self.ny), order="F")
        self.solid.from_numpy(in_dat)

    @ti.kernel
    def collision_and_streaming(self):
        for i, j in self.rho:
            if self.solid[i, j] == 0:
                # 1. Collision
                rho_local = 0.0
                v_local = ti.Vector([0.0, 0.0])
                for k in ti.static(range(9)):
                    rho_local += self.f[i, j, k]
                    v_local += self.e[k] * self.f[i, j, k]
                
                self.rho[i, j] = rho_local
                v_local /= rho_local
                
                # Add force
                v_local += self.tau * ti.Vector([self.fx, self.fy])
                self.v[i, j] = v_local
                
                for k in ti.static(range(9)):
                    feq_k = self.feq(i, j, k)
                    self.f[i, j, k] += -self.inv_tau * (self.f[i, j, k] - feq_k)

                # 2. Streaming
                for k in ti.static(range(9)):
                    ip, jp = i - self.e[k][0], j - self.e[k][1]
                    
                    # Periodic boundary
                    if ip < 0: ip = self.nx - 1
                    if ip >= self.nx: ip = 0
                    if jp < 0: jp = self.ny - 1
                    if jp >= self.ny: jp = 0

                    if self.solid[ip, jp] == 0:
                        self.F[i, j, k] = self.f[ip, jp, k]
                    else: # Bounce back
                        self.F[i, j, k] = self.f[i, j, self.LR[k]]
    
    @ti.kernel
    def update_f(self):
         for i, j, k in self.f:
             self.f[i, j, k] = self.F[i, j, k]

    def setup_zarr(self):
        self.zarr_write_index = 0
        if self.zarr_output_frequency > 0:
            zarr_path = os.path.join(self.case_dir, self.zarr_data_path)
            if os.path.exists(zarr_path):
                shutil.rmtree(zarr_path)
            
            num_frames = math.ceil(self.max_timestep / self.zarr_output_frequency)
            root = zarr.open(zarr_path, mode='w')

            self.zarr_v = root.create_dataset(
                'velocity',
                shape=(num_frames, self.nx, self.ny, 2),
                chunks=(1, self.nx, self.ny, 2),
                dtype='f4')
            self.zarr_rho = root.create_dataset(
                'density',
                shape=(num_frames, self.nx, self.ny),
                chunks=(1, self.nx, self.ny),
                dtype='f4')
            zarr_solid = root.create_dataset(
                'geometry',
                shape=(self.nx, self.ny),
                chunks=(self.nx, self.ny),
                dtype='i4')
            zarr_solid[...] = self.solid.to_numpy()

    def write_zarr(self, iter):
        if self.zarr_output_frequency > 0 and iter > 0 and iter % self.zarr_output_frequency == 0:
            self.zarr_v[self.zarr_write_index] = self.v.to_numpy()
            self.zarr_rho[self.zarr_write_index] = self.rho.to_numpy()
            self.zarr_write_index += 1
            print(f"--- Saved frame {self.zarr_write_index} to Zarr store at iteration {iter}")

    def run(self):
        self.load_geometry()
        self.init_simulation_state()
        
        time_init = time.time()
        for iter in range(self.max_timestep + 1):
            self.collision_and_streaming()
            self.update_f()
            
            if iter % self.output_frequency == 0:
                print(f"Iter {iter}, Time: {time.time() - time_init:.2f}s")

            self.write_zarr(iter)
        
        print("--- Simulation complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Taichi 2D LBM Solver (D2Q9-BGK)")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    parser.add_argument("--case-dir", type=str, required=True, help="Path to the case directory for inputs and outputs.")
    args = parser.parse_args()

    solver = LBM_D2Q9_BGK(config_file=args.config, case_dir=args.case_dir)
    solver.run()
