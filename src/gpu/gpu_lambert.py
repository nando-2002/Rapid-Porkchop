import numpy as np
from utils.lambert import lamsolve
from gpu.gpu_lambert import device_lamsolve_total_dv

R1 = np.array([1.0e8, 0.0, 0.0])    # example state in km
R2 = np.array([8.0e7, 2.0e7, 1.0e7])
tof = 200 * 24 * 3600.0             # 200 days
mu = astroConstants(4)

# CPU Lambert
v1_cpu, v2_cpu = lamsolve(R1, R2, tof, mu)
dv_cpu = np.linalg.norm(v1_cpu) + np.linalg.norm(v2_cpu)

# GPU Lambert: wrap in a tiny kernel to call the device function
from numba import cuda

@cuda.jit
def test_kernel(out, r1, r2, tof, mu):
    dv = device_lamsolve_total_dv(r1[0], r1[1], r1[2],
                                  r2[0], r2[1], r2[2],
                                  tof, mu)
    out[0] = dv

out = cuda.to_device(np.zeros(1, np.float64))
d_R1 = cuda.to_device(R1.astype(np.float64))
d_R2 = cuda.to_device(R2.astype(np.float64))
test_kernel[1,1](out, d_R1, d_R2, tof, mu)
dv_gpu = out.copy_to_host()[0]

print("CPU ΔV:", dv_cpu, "GPU ΔV:", dv_gpu)
