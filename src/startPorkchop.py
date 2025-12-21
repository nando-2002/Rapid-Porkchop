import numpy as np
from gpu.call_solver import call_solver

depStart = np.array([2025, 1, 1, 0, 0, 0])
depEnd   = np.array([2026, 1, 1, 0, 0, 0])

arrStart = np.array([2036, 1, 1, 0, 0, 0])
arrEnd   = np.array([2055, 6, 1, 0, 0, 0])

pts = 1000

planet1 = 3  # Earth
planet2 = 4  # Mars

call_solver(depStart, depEnd, arrStart, arrEnd, pts, planet1, planet2)

