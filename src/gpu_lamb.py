import numpy as np
from numba import cuda

from utils.uplanet import uplanet
from utils.utils import date2mjd2000, kep2car, car2kep
from utils.astroConstants import astroConstants
from utils.lambert import lam_solve          # CPU reference
from gpu.gpu_lambert import lam_solve_dev    # GPU device Lambert

def main():
    # heliocentric
    mu = astroConstants(4)

    # Departure: Jun 7, 2003 at 22:27:34
    departure = np.array([2003, 6, 7, 22, 27, 34])
    departure_mjd = date2mjd2000(departure)
    print(f"Departure MJD2000: {departure_mjd}")

    # Earth at departure
    a, e, i, Om, om, theta = uplanet(departure_mjd, 3)
    h = np.sqrt(mu * a * (1 - e**2))
    print(f"Earth orbital elements at departure:\n  a={a}, e={e}, i={i}, Om={Om}, om={om}, theta={theta}")

    # Arrival: Dec 28, 2003 at 14:26:08
    arrival = np.array([2003, 12, 28, 14, 26, 8])
    arrival_mjd = date2mjd2000(arrival)
    print(f"Arrival MJD2000: {arrival_mjd}")

    # Mars at arrival
    aa, ea, ia, Oma, oma, thetaa = uplanet(arrival_mjd, 4)
    ha = np.sqrt(mu * aa * (1 - ea**2))
    print(f"Mars orbital elements at arrival:\n  a={aa}, e={ea}, i={ia}, Om={Oma}, om={oma}, theta={thetaa}")

    # Cartesian states
    re, ve = kep2car(h, e, i, Om, om, theta, mu)
    rm, vm = kep2car(ha, ea, ia, Oma, oma, thetaa, mu)
    re = np.asarray(re).flatten()
    ve = np.asarray(ve).flatten()
    rm = np.asarray(rm).flatten()
    vm = np.asarray(vm).flatten()

    # Time of flight in seconds
    TOF = (arrival_mjd - departure_mjd) * 24 * 60 * 60

    # --- CPU Lambert ---
    vt_cpu, vt2_cpu = lam_solve(re, rm, TOF, mu, 10_000)
    delv1_cpu = np.linalg.norm(vt_cpu - ve)
    delv2_cpu = np.linalg.norm(vm - vt2_cpu)
    total_delv_cpu = delv1_cpu + delv2_cpu
    print(f"CPU total delta-V: {total_delv_cpu:.6f} km/s")

    # --- GPU Lambert (single-thread kernel wrapper) ---
    @cuda.jit
    def lambert_kernel(R1, R2, mu_val, tof, V1_out, V2_out):
        i = cuda.grid(1)
        if i == 0:
            v1 = cuda.local.array(3, dtype=float)
            v2 = cuda.local.array(3, dtype=float)
            lam_solve_dev(R1[0], R1[1], R1[2],
                          R2[0], R2[1], R2[2],
                          tof, mu_val,
                          v1, v2,
                          MAX_ITER=1000,
                          traj_pro=True)
            for k in range(3):
                V1_out[k] = v1[k]
                V2_out[k] = v2[k]

    d_R1 = cuda.to_device(re.astype(np.float64))
    d_R2 = cuda.to_device(rm.astype(np.float64))
    V1_out = cuda.to_device(np.zeros(3, dtype=np.float64))
    V2_out = cuda.to_device(np.zeros(3, dtype=np.float64))

    lambert_kernel[1, 1](d_R1, d_R2, mu, TOF, V1_out, V2_out)
    vt_gpu = V1_out.copy_to_host()
    vt2_gpu = V2_out.copy_to_host()

    delv1_gpu = np.linalg.norm(vt_gpu - ve)
    delv2_gpu = np.linalg.norm(vm - vt2_gpu)
    total_delv_gpu = delv1_gpu + delv2_gpu

    print(f"GPU total delta-V: {total_delv_gpu:.6f} km/s")
    print(f"|CPU-GPU| delta-V difference: {abs(total_delv_cpu - total_delv_gpu):.6e} km/s")

    print("\nCPU velocities:")
    print("  vt :", vt_cpu)
    print("  vt2:", vt2_cpu)
    print("GPU velocities:")
    print("  vt :", vt_gpu)
    print("  vt2:", vt2_gpu)

if __name__ == "__main__":
    main()
