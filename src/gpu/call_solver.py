# Setting path to allow func calls
# from sibling folder /utils/
import sys
sys.path.append("..")

# External Python Libraries

import time
import numpy as np
from numba import cuda, types
import math 
import os

# Functions from sibling folder /utils/

from utils.uplanet import uplanet
from utils.utils import date2mjd2000, kep2car, jd2date, printLine
from utils.astroConstants import astroConstants
from gpu.gpu_lambert import lam_solve_cuda

def call_solver(departureStart, departureEnd, 
                arrivalStart, arrivalEnd, 
                nPoints, planetID1, planetID2):


    # GPU kernel (function) is defined within the call_solver function
    # due to the limitation of the GPU being unable to return values
    @cuda.jit
    def porkchop_kernel(dep_mjd, arr_mjd,
                        re_e, ve_e,
                        rm_m, vm_m,
                        mu_val,
                        dv_out):
        
        # dep_idx and arr_idx are 2D grid indices 
        dep_idx, arr_idx = cuda.grid(2)
        n_dep = dep_mjd.shape[0]
        n_arr = arr_mjd.shape[0]
        if dep_idx >= n_dep or arr_idx >= n_arr:
            return # necesary to avoid out-of-bounds access when npoints is not perfectly divisible by block size

        # Get departure and arrival mjd as single vars
        d_mjd = dep_mjd[dep_idx]
        a_mjd = arr_mjd[arr_idx]
        if a_mjd <= d_mjd:
            dv_out[dep_idx, arr_idx] = 99999 #really high number 

        else: 
            # states
            # this is necessary as the GPU does not support vector operations
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
            lam_solve_cuda(r1x, r1y, r1z,
                        r2x, r2y, r2z,
                        tof, mu_val,
                        vt, vt2,
                        MAX_ITER=200,
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

    # Heliocentric, therefore mu = muSun
    mu = astroConstants(4)

    # mjd2000 conversion
    departure_start_mjd = date2mjd2000(departureStart)
    departure_end_mjd = date2mjd2000(departureEnd)
    arrival_start_mjd = date2mjd2000(arrivalStart)
    arrival_end_mjd = date2mjd2000(arrivalEnd)

    # grid init
    dep_range = np.linspace(departure_start_mjd, departure_end_mjd, nPoints)
    arr_range = np.linspace(arrival_start_mjd, arrival_end_mjd, nPoints)

    # planet states array allocation 
    r_dep = np.empty((nPoints, 3))
    v_dep = np.empty((nPoints, 3))
    r_arriv = np.empty((nPoints, 3))
    v_arriv = np.empty((nPoints, 3))

    # console output
    wid = os.get_terminal_size().columns
    printLine()
    print(f"{nPoints} trajectories will be computed on the GPU ")

    print("Precomputing planet states")
    for i in range(nPoints):
        
        # Departure Planet
        a, e, inc, Om, om, theta = uplanet(dep_range[i], planetID1)
        h = np.sqrt(mu * a * (1 - e**2))
        r, v = kep2car(h, e, inc, Om, om, theta, mu)
        r_dep[i] = np.asarray(r).flatten()
        v_dep[i] = np.asarray(v).flatten()

        # Arrival Planet
        aa, ea, ia, Oma, oma, thetaa = uplanet(arr_range[i], planetID2)
        ha = np.sqrt(mu * aa * (1 - ea**2))
        r2, v2 = kep2car(ha, ea, ia, Oma, oma, thetaa, mu)
        r_arriv[i] = np.asarray(r2).flatten()
        v_arriv[i] = np.asarray(v2).flatten()
        
    # allocate result array 
    delta_v_solutions_gpu = np.zeros((nPoints, nPoints), dtype=np.float32)

    # copy inputs to GPU
    dep_mjd_gpu = cuda.to_device(dep_range.astype(np.float32))
    arr_mjd_gpu = cuda.to_device(arr_range.astype(np.float32))
    r_dep_gpu = cuda.to_device(r_dep.astype(np.float32))
    v_dep_gpu = cuda.to_device(v_dep.astype(np.float32))
    r_arriv_gpu = cuda.to_device(r_arriv.astype(np.float32))
    v_arriv_gpu = cuda.to_device(v_arriv.astype(np.float32))
    dv_out = cuda.to_device(delta_v_solutions_gpu)

    # assigning threads and blocks as per best practices
    threadsperblock = (16, 16)
    blockspergrid_x = (nPoints + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (nPoints + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    printLine()
    print(f"Launching GPU")
    
    # execute the kernel
    # start the timer
    t_gpu_start = time.time()
    porkchop_kernel[blockspergrid, threadsperblock](
        dep_mjd_gpu, arr_mjd_gpu,
        r_dep_gpu, v_dep_gpu,
        r_arriv_gpu, v_arriv_gpu,
        mu,
        dv_out)
    cuda.synchronize()
    t_gpu_end = time.time()

    delta_v_solutions_gpu = dv_out.copy_to_host()
    
    #print(f"GPU Lowest Delta-V Possible: {min_dv:.4f} km/s")
    #print(f"GPU kernel time: {t_gpu_end - t_gpu_start:.4f} s")

    print(f"GPU kernel execution time: {t_gpu_end - t_gpu_start:.2f} seconds")

    return delta_v_solutions_gpu, dep_range, arr_range, r_dep, v_dep, r_arriv, v_arriv