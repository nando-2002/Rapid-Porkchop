import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
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

def save_flyby_density_plot(results, filename=None, sigma=1.5, vmin=5.0, vmax=20.0, 
                          n_contours=8, dpi=600, figsize=(12, 10)):
    """
    Pretty 3D density plot like Medium article - NO SKIMAGE REQUIRED.
    Uses pure Matplotlib + Scipy gaussian_filter only.
    """
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), "flyby_density.png")
    
    dv_total, dep_range, fly_range, arr_range = results[:4]
    
    # Convert MJD2000 to years from 2030
    dep_years = 2030 + (dep_range - dep_range[0]) / 365.25
    fly_years = 2030 + (fly_range - fly_range[0]) / 365.25
    arr_years = 2030 + (arr_range - arr_range[0]) / 365.25
    
    # Create coordinate grids
    X, Y, Z = np.meshgrid(dep_years, fly_years, arr_years, indexing='ij')
    
    # Volume data with masking
    volume = np.array(dv_total, dtype=float)
    volume = np.clip(volume, vmin, vmax)
    volume[volume > 30.0] = np.nan
    
    print(f"Volume range: {np.nanmin(volume):.1f} - {np.nanmax(volume):.1f} km/s")
    
    # === DENSITY FIELD (Scipy gaussian_filter only) ===
    low_dv_mask = volume < vmax
    density_volume = np.zeros_like(volume)
    density_volume[low_dv_mask] = 1.0 / (1.0 + volume[low_dv_mask])  # Inverse ΔV
    density_smoothed = gaussian_filter(density_volume, sigma=sigma)
    
    # === 3D DENSITY SCATTER (Medium article style) ===
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Flatten for scatter
    x_flat, y_flat, z_flat = X.flatten(), Y.flatten(), Z.flatten()
    density_flat = density_smoothed.flatten()
    
    # Filter valid points
    valid_mask = ~np.isnan(density_flat) & (density_flat > 0.01)
    x_valid = x_flat[valid_mask]
    y_valid = y_flat[valid_mask]
    z_valid = z_flat[valid_mask]
    density_valid = density_flat[valid_mask]
    
    # Density-based sizing & transparency (Medium technique)
    sizes = 25 * density_valid**2      # Quadratic for visibility
    alphas = 0.7 * density_valid**0.6  # Non-linear transparency
    
    # Color by actual ΔV
    colors = volume.flatten()[valid_mask]
    
    scatter = ax.scatter(x_valid, y_valid, z_valid,
                        c=colors, cmap='viridis_r',
                        s=sizes, alpha=alphas,
                        linewidth=0.2, edgecolors='black', antialiased=True)
    
    # === DENSITY ISO-CONTOURS (Pure NumPy slicing - no marching_cubes) ===
    # Sample planes through density field
    n_slices = 5
    for i in range(n_slices):
        slice_z = volume.shape[2] * i // (n_slices - 1)
        if slice_z < volume.shape[2]:
            slice_data = density_smoothed[:, :, slice_z]
            levels = np.linspace(0.1, 0.8, 4)
            
            # Contour lines on planes
            X_slice, Y_slice = np.meshgrid(dep_years, fly_years, indexing='ij')
            for level in levels:
                cs = ax.contour(X_slice, Y_slice, slice_data, levels=[level], 
                               colors='cyan', alpha=0.3, linewidths=1.5)
                ax.clabel(cs, inline=True, fontsize=8, fmt=f'{level:.1f}')
    
    # === GLOBAL MINIMUM STAR ===
    min_idx = np.unravel_index(np.nanargmin(volume), volume.shape)
    min_dv = volume[min_idx]
    ax.scatter(dep_years[min_idx[0]], fly_years[min_idx[1]], arr_years[min_idx[2]],
              c='red', s=250, marker='*', linewidth=4, 
              edgecolors='darkred', zorder=100,
              label=f'Global Optimum\nΔV = {min_dv:.2f} km/s')
    
    # === MEDIUM ARTICLE AESTHETICS ===
    ax.set_xlabel('Departure Year', fontsize=12, labelpad=12)
    ax.set_ylabel('Flyby Year', fontsize=12, labelpad=12)
    ax.set_zlabel('Arrival Year', fontsize=12, labelpad=12)
    
    # Perfect viewing angle
    ax.view_init(elev=25, azim=45)
    
    # Equal aspect cube
    max_range = np.array([dep_years[-1]-dep_years[0], 
                         fly_years[-1]-fly_years[0], 
                         arr_years[-1]-arr_years[0]]).max() / 2.0
    mid_x = (dep_years[-1] + dep_years[0]) * 0.5
    mid_y = (fly_years[-1] + fly_years[0]) * 0.5
    mid_z = (arr_years[-1] + arr_years[0]) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.08)
    cbar.set_label('ΔV [km/s]', fontsize=11)
    
    ax.legend(loc='upper left', fontsize=10)
    
    # Modern transparent style
    title = f'Flyby Mission ΔV Density Field (σ={sigma})\nGlobal Minimum: {min_dv:.2f} km/s'
    ax.set_title(title, fontsize=14, pad=20)
    
    # Transparent panes + grid
    fig.patch.set_alpha(0.0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # High-res save
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', transparent=True,
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ 3D Density plot saved: {filename}")
    print(f"   Low-ΔV clouds: σ={sigma}, {len(x_valid):,} points plotted")
    return filename

# Bonus: 2D projection (top view)
def save_density_projection(results, filename=None, sigma=1.5, dpi=300):
    """2D top-down projection of optimal flyby windows."""
    dv_total, dep_range, fly_range, _ = results[:4]
    dep_years = 2030 + (dep_range - dep_range[0]) / 365.25
    fly_years = 2030 + (fly_range - fly_range[0]) / 365.25
    
    # Project minimum ΔV along arrival axis
    min_projection = np.nanmin(dv_total, axis=2)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(min_projection.T, extent=[dep_years[0], dep_years[-1], 
                                        fly_years[0], fly_years[-1]],
               origin='lower', cmap='viridis_r', aspect='auto',
               vmin=5, vmax=15)
    plt.colorbar(label='Minimum ΔV [km/s]')
    plt.xlabel('Departure Year')
    plt.ylabel('Flyby Year')
    plt.title('Optimal Flyby Windows (Top Projection)')
    
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), "flyby_projection.png")
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    return filename