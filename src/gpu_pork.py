import time
import numpy as np
from numba import cuda, types
import math
import matplotlib.pyplot as plt
import os

from utils.uplanet import uplanet
from utils.utils import date2mjd2000, kep2car, jd2date
from utils.astroConstants import astroConstants
from utils.lambert import lam_solve             # CPU reference
from gpu.gpu_lambert import lam_solve_dev       # GPU device Lambert

def save_porkchop_plot(dep_range, arr_range, dv_matrix, filename=None, dpi=150, n_ticks=6, cmap='rainbow'):
    # dv_matrix expected shape (n_dep, n_arr)
    Z = np.array(dv_matrix, dtype=float).copy()
    Z[Z == 0] = np.nan  # mask invalid entries
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), "gpu_porkchop.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    # ensure grid orientation matches (x: dep, y: arr)
    X, Y = np.meshgrid(dep_range, arr_range)
    mesh = ax.pcolormesh(X, Y, Z.T, cmap=cmap, shading='auto')
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label('Delta-V (km/s)')

    def mjd2000_to_str(mjd):
        jd = mjd + 2400000.5 + 51544.5
        Y_, M_, D_, hrs, mins, secs = jd2date(jd)
        return f"{int(Y_)}-{int(M_):02d}-{int(D_):02d}"

    # set reasonable ticks
    xticks = np.linspace(dep_range[0], dep_range[-1], n_ticks)
    yticks = np.linspace(arr_range[0], arr_range[-1], n_ticks)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([mjd2000_to_str(x) for x in xticks], rotation=30)
    ax.set_yticklabels([mjd2000_to_str(y) for y in yticks])
    ax.set_xlabel('Departure date (YYYY-MM-DD)')
    ax.set_ylabel('Arrival date (YYYY-MM-DD)')

    # annotate minimum
    if np.any(~np.isnan(Z)):
        min_val = np.nanmin(Z)
        min_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
        min_dep_mjd = dep_range[min_idx[0]]
        min_arr_mjd = arr_range[min_idx[1]]
        ax.plot(min_dep_mjd, min_arr_mjd, 'ro', markersize=5)
        ax.annotate(f"Min {min_val:.2f} km/s", (min_dep_mjd, min_arr_mjd),
                    textcoords="offset points", xytext=(10, -10), color='white')

    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return filename

def gpu_porkchop_test():
    # heliocentric
    mu = astroConstants(4)

    # departure date range : 2 April 2003 to 1 August 2003
    departure_start = np.array([2023, 11, 1, 0, 0, 0])
    departure_end   = np.array([2025, 1, 1, 0, 0, 0])

    # arrival date range : 1 Sept 2003 to 1 March 2004
    arrival_start = np.array([2024, 4, 1, 0, 0, 0])
    arrival_end   = np.array([2025, 3, 1, 0, 0, 0])

    # convert to mjd2000
    departure_start_mjd = date2mjd2000(departure_start)
    departure_end_mjd   = date2mjd2000(departure_end)
    arrival_start_mjd   = date2mjd2000(arrival_start)
    arrival_end_mjd     = date2mjd2000(arrival_end)

    # grid
    npoints = 1000  # same as original test
    dep_range = np.linspace(departure_start_mjd, departure_end_mjd, npoints)
    arr_range = np.linspace(arrival_start_mjd, arrival_end_mjd, npoints)

    # precompute states on CPU
    re_earth = np.empty((npoints, 3))
    ve_earth = np.empty((npoints, 3))
    rm_mars  = np.empty((npoints, 3))
    vm_mars  = np.empty((npoints, 3))

    print("Precomputing states...")
    for i in range(npoints):
        # Earth at departure
        a, e, inc, Om, om, theta = uplanet(dep_range[i], 3)
        h = np.sqrt(mu * a * (1 - e**2))
        r, v = kep2car(h, e, inc, Om, om, theta, mu)
        re_earth[i] = np.asarray(r).flatten()
        ve_earth[i] = np.asarray(v).flatten()

        # Mars at arrival
        aa, ea, ia, Oma, oma, thetaa = uplanet(arr_range[i], 1)
        ha = np.sqrt(mu * aa * (1 - ea**2))
        r2, v2 = kep2car(ha, ea, ia, Oma, oma, thetaa, mu)
        rm_mars[i] = np.asarray(r2).flatten()
        vm_mars[i] = np.asarray(v2).flatten()

    # allocate result
    delta_v_solutions_gpu = np.zeros((npoints, npoints), dtype=np.float64)

    # CUDA kernel
    @cuda.jit
    def porkchop_kernel(dep_mjd, arr_mjd,
                        re_e, ve_e,
                        rm_m, vm_m,
                        mu_val,
                        dv_out):
        dep_idx, arr_idx = cuda.grid(2)
        n_dep = dep_mjd.shape[0]
        n_arr = arr_mjd.shape[0]
        if dep_idx >= n_dep or arr_idx >= n_arr:
            return

        d_mjd = dep_mjd[dep_idx]
        a_mjd = arr_mjd[arr_idx]
        if a_mjd <= d_mjd:
            return

        # states
        r1x = re_e[dep_idx, 0]
        r1y = re_e[dep_idx, 1]
        r1z = re_e[dep_idx, 2]
        v1x = ve_e[dep_idx, 0]
        v1y = ve_e[dep_idx, 1]
        v1z = ve_e[dep_idx, 2]

        r2x = rm_m[arr_idx, 0]
        r2y = rm_m[arr_idx, 1]
        r2z = rm_m[arr_idx, 2]
        v2x = vm_m[arr_idx, 0]
        v2y = vm_m[arr_idx, 1]
        v2z = vm_m[arr_idx, 2]

        tof = (a_mjd - d_mjd) * 86400.0

        # call device Lambert
        vt = cuda.local.array(3, dtype=types.float64)
        vt2 = cuda.local.array(3, dtype=types.float64)
        lam_solve_dev(r1x, r1y, r1z,
                      r2x, r2y, r2z,
                      tof, mu_val,
                      vt, vt2,
                      MAX_ITER=100,
                      traj_pro=True)

        # delta-v
        dv1x = vt[0] - v1x
        dv1y = vt[1] - v1y
        dv1z = vt[2] - v1z
        dv1 = math.sqrt(dv1x*dv1x + dv1y*dv1y + dv1z*dv1z)

        dv2x = v2x - vt2[0]
        dv2y = v2y - vt2[1]
        dv2z = v2z - vt2[2]
        dv2 = math.sqrt(dv2x*dv2x + dv2y*dv2y + dv2z*dv2z)

        dv_out[dep_idx, arr_idx] = dv1 + dv2

    # copy to GPU and run
    d_dep = cuda.to_device(dep_range.astype(np.float64))
    d_arr = cuda.to_device(arr_range.astype(np.float64))
    d_re  = cuda.to_device(re_earth.astype(np.float64))
    d_ve  = cuda.to_device(ve_earth.astype(np.float64))
    d_rm  = cuda.to_device(rm_mars.astype(np.float64))
    d_vm  = cuda.to_device(vm_mars.astype(np.float64))
    d_dv  = cuda.to_device(delta_v_solutions_gpu)

    threadsperblock = (16, 16)
    blockspergrid_x = (npoints + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (npoints + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    print("Running GPU porkchop...")
    t0 = time.time()
    porkchop_kernel[blockspergrid, threadsperblock](
        d_dep, d_arr,
        d_re, d_ve,
        d_rm, d_vm,
        mu,
        d_dv
    )
    cuda.synchronize()
    t1 = time.time()
    delta_v_solutions_gpu = d_dv.copy_to_host()

    finite = delta_v_solutions_gpu[np.nonzero(delta_v_solutions_gpu)]
    dv_min_gpu = np.min(finite)
    print(f"GPU Lowest Delta-V Possible: {dv_min_gpu:.4f} km/s")
    print(f"GPU kernel time: {t1 - t0:.4f} s")

    # save porkchop figure to same directory
    plot_fn = os.path.join(os.path.dirname(__file__), "gpu_porkchop.png")
    save_porkchop_plot(dep_range, arr_range, delta_v_solutions_gpu, filename=plot_fn)

if __name__ == "__main__":
    gpu_porkchop_test()
