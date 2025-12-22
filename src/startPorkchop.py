import numpy as np
from gpu.call_solver import call_solver

depStart = np.array([2025, 1, 1, 0, 0, 0])
depEnd   = np.array([2026, 1, 1, 0, 0, 0])

arrStart = np.array([2036, 1, 1, 0, 0, 0])
arrEnd   = np.array([2055, 6, 1, 0, 0, 0])

pts = 2500

planet1 = 3  # Earth
planet2 = 4  # Mars

soln, depRange, arrRange, r1, v1, r2, v2 = call_solver(depStart, depEnd,
                                                       arrStart, arrEnd,
                                                       pts, planet1, planet2)


# min delta v (ignore zeros, NaNs and infinities)
valid_mask = np.isfinite(soln) & (soln > 0)
if not np.any(valid_mask):
    raise ValueError("No valid (non-zero, finite) delta-v values found in the solution matrix.")
min_dv = np.min(soln[valid_mask])

# date corresponding to min delta v
# use argmin on a masked version to avoid floating-point equality pitfalls and empty matches
flat_index = np.argmin(np.where(valid_mask, soln, np.inf))
min_idx = np.unravel_index(flat_index, soln.shape)
dep_mjd_min = depRange[min_idx[0]]
arr_mjd_min = arrRange[min_idx[1]]

print(f"Minimum delta-v: {min_dv:.4f} km/s")
print(f"Departure MJD: {dep_mjd_min:.4f}")
print(f"Arrival MJD: {arr_mjd_min:.4f}")


from utils.utils import mjd2000_to_date
print(f"Departure Date: {mjd2000_to_date(dep_mjd_min)}")
print(f"Arrival Date: {mjd2000_to_date(arr_mjd_min)}")

from utils.plotting import plot_porkchop
plot_porkchop(depRange, arrRange, soln,
                dep_mjd_min, arr_mjd_min,
                min_dv, "Earth_Mars_2030_2060.png")
