import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils.utils import jd2date


def get_optimal_dates(results):
    """Return formatted optimal departure, flyby and arrival date strings and min ΔV.

    Results is the same tuple returned by `call_flyby`: (dv_total, dep_range, fly_range, arr_range, ...)
    """
    dv_total, dep_range, fly_range, arr_range = results[:4]
    Z = np.array(dv_total, dtype=float).copy()
    valid_mask = ~np.isnan(Z)
    if not np.any(valid_mask):
        return None, None, None, None

    min_val = float(np.nanmin(Z))
    min_idx = np.unravel_index(np.nanargmin(Z), Z.shape)

    jd_offset = 2451545.0  # JD = MJD2000 + 2451545.0
    dep_jd = dep_range[min_idx[0]] + jd_offset
    fly_jd = fly_range[min_idx[1]] + jd_offset
    arr_jd = arr_range[min_idx[2]] + jd_offset

    def _fmt_jd(jd):
        Y, M, D, hrs, mins, secs = jd2date(jd)
        return f"{int(Y):04d}-{int(M):02d}-{int(D):02d} {int(hrs):02d}:{int(mins):02d}:{secs:05.2f}"

    return _fmt_jd(dep_jd), _fmt_jd(fly_jd), _fmt_jd(arr_jd), min_val

def save_flyby_plot(results, filename=None, min_dv_threshold=50, n_levels=10, dpi=600):
    """Create 3D isosurface-style plot of optimal flyby trajectories."""
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), "flyby_3d.png")
    
    dv_total, dep_range, fly_range, arr_range = results[:4]
    
    # Mask invalid high ΔV
    Z = np.array(dv_total, dtype=float).copy()
    Z[Z > min_dv_threshold] = np.nan
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert MJD2000 to years from 2030
    dep_years = 2030 + (dep_range - dep_range[0]) / 365.25
    fly_years = 2030 + (fly_range - fly_range[0]) / 365.25
    arr_years = 2030 + (arr_range - arr_range[0]) / 365.25
    
    X, Y, Z_grid = np.meshgrid(dep_years, fly_years, arr_years, indexing='ij')
    
    # Find minimum
    valid_mask = ~np.isnan(Z)
    if np.any(valid_mask):
        min_val = np.nanmin(Z)
        min_idx = np.unravel_index(np.argmin(Z, axis=None), Z.shape)
        
        # Print optimal dates (convert from MJD2000 -> JD -> calendar)
        dep_str, fly_str, arr_str, _ = get_optimal_dates(results)
        print(f"Optimal dates (min ΔV={min_val:.2f} km/s):\n  Departure: {dep_str} (MJD2000={dep_range[min_idx[0]]:.6f})\n  Flyby:     {fly_str} (MJD2000={fly_range[min_idx[1]]:.6f})\n  Arrival:   {arr_str} (MJD2000={arr_range[min_idx[2]]:.6f})")

        # Scatter low ΔV regions
        low_dv_mask = Z < min_dv_threshold
        ax.scatter(X[low_dv_mask], Y[low_dv_mask], Z_grid[low_dv_mask], 
                  c=Z[low_dv_mask], cmap='viridis', s=1, alpha=0.6)
        
        # Mark global minimum
        ax.scatter(X[min_idx], Y[min_idx], Z_grid[min_idx], 
                  c='red', s=100, label=f'Min ΔV: {min_val:.2f} km/s')
        
        ax.set_xlabel('Departure Year')
        ax.set_ylabel('Flyby Year')
        ax.set_zlabel('Arrival Year')
        ax.legend()
        
        title = (f'Optimal Flyby Trajectories (ΔV < {min_dv_threshold} km/s)\n'
                f'Minimum: {min_val:.2f} km/s at '
                f'{dep_years[min_idx[0]]:.1f}/{fly_years[min_idx[1]]:.1f}/{arr_years[min_idx[2]]:.1f}')
        ax.set_title(title)
    
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    return filename
