import os
import numpy as np

from gpu.call_solver import call_solver

depStart = np.array([2030, 1, 1, 0, 0, 0])
depEnd   = np.array([2061, 1, 1, 0, 0, 0])

arrStart = np.array([2030, 1, 1, 0, 0, 0])
arrEnd   = np.array([2061, 1, 1, 0, 0, 0])

pts = 2500

planet1 = 3  # Earth
planet2 = 4  # Mars

soln, depRange, arrRange, r1, v1, r2, v2 = call_solver(depStart, depEnd,
                                                       arrStart, arrEnd,
                                                       pts, planet1, planet2)

finite = soln[np.nonzero(soln)]
# remove NaN entries so the minimum ignores NaNs
finite = finite[~np.isnan(finite)]
if finite.size == 0:
    print("GPU Lowest Delta-V Possible: no valid solution")
else:
    minimum = np.min(finite)
    print(f"GPU Lowest Delta-V Possible: {minimum:.4f} km/s")

# Date of Minimum Delta-V
min_index = np.unravel_index(np.argmin(soln, where=~np.isnan(soln)), soln.shape)
dep_min = depRange[min_index[0]]
arr_min = arrRange[min_index[1]]
print(f"Departure Date of Minimum Delta-V: {dep_min[0]:04}-{dep_min[1]:02}-{dep_min[2]:02}")
print(f"Arrival Date of Minimum Delta-V: {arr_min[0]:04}-{arr_min[1]:02}-{arr_min[2]:02}")

# save porkchop figure to same directory
plot_fn = os.path.join(os.path.dirname(__file__), "Mars.png")

from utils.plotting import save_porkchop_plot
save_porkchop_plot(depRange, arrRange, soln, plot_fn, 5, 10, dpi = 600)
