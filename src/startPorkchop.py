import os
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

finite = soln[np.nonzero(soln)]
# remove NaN entries so the minimum ignores NaNs
finite = finite[~np.isnan(finite)]
if finite.size == 0:
    print("GPU Lowest Delta-V Possible: no valid solution")
else:
    minimum = np.min(finite)
    print(f"GPU Lowest Delta-V Possible: {minimum:.4f} km/s")

# save porkchop figure to same directory
plot_fn = os.path.join(os.path.dirname(__file__), "Mars.png")

from utils.plotting import save_porkchop_plot
save_porkchop_plot(depRange, arrRange, soln, filename=plot_fn, dpi = 600)
