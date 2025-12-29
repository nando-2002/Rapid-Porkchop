import sys
sys.path.append("..")

import time
import numpy as np
from numba import cuda, types
import os
import math

from utils.console import printLine
from utils.uplanet import uplanet
from utils.utils import date2mjd2000, kep2car
from utils.astroConstants import astroConstants
from gpu.gpu_lambert import lam_solve_cuda
from gpu.gpu_flyby import powered_flyby_dv

def call_flyby(nPoints, depPlanet, flyPlanet, arrPlanet):
    """Compute 3D ΔV grid for dep->flyby->arr mission on GPU."""
    
    mu_sun = astroConstants(4)
    mu_flyby = astroConstants(flyPlanet)
    
    printLine()
    print(f"Computing {nPoints}^3 = {nPoints**3:,} flyby trajectories (2030-2061)")
    printLine()
    
    # Time grids: 1 Jan 2030 to 1 Jan 2061
    dep_range = np.linspace(date2mjd2000([2030,1,1,0,0,0]), 
                           date2mjd2000([2061,1,1,0,0,0]), nPoints)
    fly_range = dep_range.copy()
    arr_range = dep_range.copy()
    
    # Precompute planet states [3 planets × nPoints × 3]
    print("Precomputing ephemerides...")
    r_planets = np.empty((3, nPoints, 3))
    v_planets = np.empty((3, nPoints, 3))
    
    planet_ids = [depPlanet, flyPlanet, arrPlanet]
    for i, pid in enumerate(planet_ids):
        for j in range(nPoints):
            a, e, inc, Om, om, theta = uplanet(dep_range[j], pid)
            h = np.sqrt(mu_sun * a * (1 - e**2))
            r, v = kep2car(h, e, inc, Om, om, theta, mu_sun)
            r_planets[i,j] = np.asarray(r).flatten()
            v_planets[i,j] = np.asarray(v).flatten()
    
    # Allocate 3D result arrays
    dv_total = np.full((nPoints, nPoints, nPoints), 99999.0, dtype=np.float32)
    
    # GPU memory
    dep_mjd_gpu = cuda.to_device(dep_range.astype(np.float32))
    fly_mjd_gpu = cuda.to_device(fly_range.astype(np.float32))
    arr_mjd_gpu = cuda.to_device(arr_range.astype(np.float32))
    
    r_dep_gpu = cuda.to_device(r_planets[0].astype(np.float32))
    v_dep_gpu = cuda.to_device(v_planets[0].astype(np.float32))
    r_fly_gpu = cuda.to_device(r_planets[1].astype(np.float32))
    v_fly_gpu = cuda.to_device(v_planets[1].astype(np.float32))
    r_arr_gpu = cuda.to_device(r_planets[2].astype(np.float32))
    v_arr_gpu = cuda.to_device(v_planets[2].astype(np.float32))
    
    dv_total_gpu = cuda.to_device(dv_total)
    
    # 3D Grid launch config
    threadsperblock = (8, 8, 8)
    blockspergrid_x = (nPoints + 7) // 8
    blockspergrid = (blockspergrid_x, blockspergrid_x, blockspergrid_x)
    
    print(f"Launching {blockspergrid[0]}x{blockspergrid[1]}x{blockspergrid[2]} blocks")
    
    t_start = time.time()
    
    @cuda.jit
    def flyby_kernel(dep_mjd, fly_mjd, arr_mjd, r_dep, v_dep, r_fly, v_fly, r_arr, v_arr,
                     mu_sun, mu_flyby, dv_out):
        dep_idx, fly_idx, arr_idx = cuda.grid(3)
        n = dep_mjd.shape[0]
        
        if dep_idx >= n or fly_idx >= n or arr_idx >= n:
            return
            
        d_mjd = dep_mjd[dep_idx]
        f_mjd = fly_mjd[fly_idx]
        a_mjd = arr_mjd[arr_idx]
        
        if f_mjd <= d_mjd or a_mjd <= f_mjd:
            dv_out[dep_idx, fly_idx, arr_idx] = 99999.0
            return
        
        # Extract states
        r1x, r1y, r1z = r_dep[dep_idx, 0], r_dep[dep_idx, 1], r_dep[dep_idx, 2]
        v1x, v1y, v1z = v_dep[dep_idx, 0], v_dep[dep_idx, 1], v_dep[dep_idx, 2]
        
        r2x, r2y, r2z = r_fly[fly_idx, 0], r_fly[fly_idx, 1], r_fly[fly_idx, 2]
        v2x, v2y, v2z = v_fly[fly_idx, 0], v_fly[fly_idx, 1], v_fly[fly_idx, 2]
        vfx, vfy, vfz = v2x, v2y, v2z  # Flyby planet velocity
        
        r3x, r3y, r3z = r_arr[arr_idx, 0], r_arr[arr_idx, 1], r_arr[arr_idx, 2]
        v3x, v3y, v3z = v_arr[arr_idx, 0], v_arr[arr_idx, 1], v_arr[arr_idx, 2]
        
        # Leg 1: dep -> flyby planet (Lambert arrival velocity at flyby)
        tof1 = (f_mjd - d_mjd) * 86400.0
        vt1_dep = cuda.local.array(3, dtype=types.float64)
        vt1_fly = cuda.local.array(3, dtype=types.float64)  # v∞⁻
        lam_solve_cuda(r1x, r1y, r1z, r2x, r2y, r2z, tof1, mu_sun, vt1_dep, vt1_fly)
        
        dv_dep = math.sqrt((vt1_dep[0]-v1x)**2 + (vt1_dep[1]-v1y)**2 + (vt1_dep[2]-v1z)**2)
        
        # Leg 2: flyby planet -> arr (Lambert departure velocity from flyby)
        tof2 = (a_mjd - f_mjd) * 86400.0
        vt2_fly = cuda.local.array(3, dtype=types.float64)  # v∞⁺
        vt2_arr = cuda.local.array(3, dtype=types.float64)
        lam_solve_cuda(r2x, r2y, r2z, r3x, r3y, r3z, tof2, mu_sun, vt2_fly, vt2_arr)
        
        dv_arr = math.sqrt((v3x-vt2_arr[0])**2 + (v3y-vt2_arr[1])**2 + (v3z-vt2_arr[2])**2)
        
        # Powered flyby ΔV
        v_inf_minus_x = vt1_fly[0] - vfx
        v_inf_minus_y = vt1_fly[1] - vfy
        v_inf_minus_z = vt1_fly[2] - vfz
        
        v_inf_plus_x = vfx - vt2_fly[0]  # Note sign flip for outgoing
        v_inf_plus_y = vfy - vt2_fly[1]
        v_inf_plus_z = vfz - vt2_fly[2]
        
        dv_flyby = powered_flyby_dv(v_inf_minus_x, v_inf_minus_y, v_inf_minus_z,
                                   v_inf_plus_x, v_inf_plus_y, v_inf_plus_z, mu_flyby)
        
        dv_out[dep_idx, fly_idx, arr_idx] = dv_dep + dv_flyby + dv_arr
    
    flyby_kernel[blockspergrid, threadsperblock](
        dep_mjd_gpu, fly_mjd_gpu, arr_mjd_gpu,
        r_dep_gpu, v_dep_gpu, r_fly_gpu, v_fly_gpu, r_arr_gpu, v_arr_gpu,
        mu_sun, mu_flyby, dv_total_gpu)
    
    cuda.synchronize()
    t_end = time.time()
    
    dv_total_cpu = dv_total_gpu.copy_to_host()
    print(f"GPU execution: {t_end-t_start:.2f}s")
    
    return dv_total_cpu, dep_range, fly_range, arr_range, r_planets, v_planets
