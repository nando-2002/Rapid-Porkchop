# gpu_porkchop.py

import time
import numpy as np
from numba import cuda

from utils.uplanet import uplanet
from utils.utils import date2mjd2000, kep2car
from utils.astroConstants import astroConstants
from gpu.gpu_lambert import porkchop_kernel

# -------------------------
# User settings
# -------------------------

npoints = 100  # adjust as desired (e.g. 100, 200, 500, 1000)
ibody_dep = 3  # Earth
ibody_arr = 4  # Mars

# Example date ranges (same style as your original porkchop_test.py)
departure_start = np.array([2003, 4, 2, 0, 0, 0])
departure_end   = np.array([2003, 8, 1, 0, 0, 0])

arrival_start   = np.array([2003, 9, 1, 0, 0, 0])
arrival_end     = np.array([2004, 3, 1, 0, 0, 0])

mu = astroConstants(4)  # km^3/s^2

# -------------------------
# Build time grids
# -------------------------

dep_start_mjd  = date2mjd2000(departure_start)
dep_end_mjd    = date2mjd2000(departure_end)
arr_start_mjd  = date2mjd2000(arrival_start)
arr_end_mjd    = date2mjd2000(arrival_end)

deprange = np.linspace(dep_start_mjd, dep_end_mjd, npoints)
arrrange = np.linspace(arr_start_mjd, arr_end_mjd, npoints)

# -------------------------
# Precompute states on CPU
# -------------------------

print(f"Precomputing planet states for npoints={npoints}...")
t0 = time.time()

re_earth = np.empty((npoints, 3), dtype=np.float64)
ve_earth = np.empty((npoints, 3), dtype=np.float64)
rm_mars  = np.empty((npoints, 3), dtype=np.float64)
vm_mars  = np.empty((npoints, 3), dtype=np.float64)

for i in range(npoints):
    # Earth at departure
    kep_e = uplanet(deprange[i], ibody_dep)
    h, e, i_deg, Om, om, theta = kep_e
    r_e, v_e = kep2car(h, e, i_deg, Om, om, theta, mu)
    re_earth[i, :] = np.asarray(r_e).flatten()
    ve_earth[i, :] = np.asarray(v_e).flatten()

for j in range(npoints):
    # Mars at arrival
    kep_m = uplanet(arrrange[j], ibody_arr)
    h, e, i_deg, Om, om, theta = kep_m
    r_m, v_m = kep2car(h, e, i_deg, Om, om, theta, mu)
    rm_mars[j, :] = np.asarray(r_m).flatten()
    vm_mars[j, :] = np.asarray(v_m).flatten()

t1 = time.time()
print(f"State precomputation time: {t1 - t0:.3f} s")

# -------------------------
# Copy to GPU
# -------------------------

print("Copying data to GPU...")
d_deprange = cuda.to_device(deprange)
d_arrrange = cuda.to_device(arrrange)
d_re_earth = cuda.to_device(re_earth)
d_ve_earth = cuda.to_device(ve_earth)
d_rm_mars  = cuda.to_device(rm_mars)
d_vm_mars  = cuda.to_device(vm_mars)

d_mu = cuda.to_device(np.array([mu], dtype=np.float64))

d_deltav = cuda.to_device(np.full((npoints, npoints),
                                  np.inf,
                                  dtype=np.float64))

# -------------------------
# Kernel launch config
# -------------------------

threadsperblock = (16, 16)
blockspergrid_x = (npoints + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (npoints + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

print(f"Launching kernel with grid={blockspergrid}, block={threadsperblock}...")
t2 = time.time()
porkchop_kernel[blockspergrid, threadsperblock](
    d_deprange, d_arrrange,
    d_re_earth, d_ve_earth,
    d_rm_mars, d_vm_mars,
    d_mu,
    d_deltav
)
cuda.synchronize()
t3 = time.time()

print(f"GPU kernel time: {t3 - t2:.3f} s")

# -------------------------
# Fetch results
# -------------------------

deltavsolutions = d_deltav.copy_to_host()

# Example: print minimum DV
finite_mask = np.isfinite(deltavsolutions)
if np.any(finite_mask):
    min_dv = np.min(deltavsolutions[finite_mask])
    print(f"Lowest Delta-V in grid: {min_dv:.4f} km/s")
else:
    print("No valid Lambert solutions found in this grid.")

# Optional: plotting (if you want to visualize)
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    cs = plt.contourf(deprange, arrrange, deltavsolutions.T, levels=50)
    plt.colorbar(cs, label="Delta-V [km/s]")
    plt.xlabel("Departure MJD2000")
    plt.ylabel("Arrival MJD2000")
    plt.title("GPU Porkchop Plot (Lambert ΔV)")
    plt.tight_layout()
    plt.show()
