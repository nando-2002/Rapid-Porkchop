import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils.utils import jd2date

def save_flyby_plot(results, filename=None, min_dv_threshold=15.0, n_levels=10, dpi=600):
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
