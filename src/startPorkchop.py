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


from utils.plotting import save_porkchop_plot
save_porkchop_plot(depRange, arrRange, soln, filename="porkchop_plot.png", dpi = 600)
