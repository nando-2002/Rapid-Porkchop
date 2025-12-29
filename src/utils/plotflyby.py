import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils.utils import jd2date

def save_flyby_plot(results, filename=None, min_dv_threshold=30, n_levels=10, dpi=600):
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


def save_flyby_isosurface(results, filename=None, level=None, min_dv_threshold=30, dpi=600, alpha=0.7):
    """Create a 3D isosurface-style plot of the ΔV volume.

    Tries to use skimage.measure.marching_cubes to extract an isosurface at `level`.
    If `level` is None a value near the global minimum is chosen. If scikit-image
    isn't available the function falls back to plotting three orthogonal contour
    slices through the minimum ΔV location.
    """
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), "flyby_isosurface.png")

    dv_total, dep_range, fly_range, arr_range = results[:4]

    # Prepare scalar volume and mask out high ΔV values
    V = np.array(dv_total, dtype=float).copy()
    V[V > min_dv_threshold] = np.nan

    # If there's no valid data, save an informative empty figure
    valid_mask = ~np.isnan(V)
    if not np.any(valid_mask):
        fig = plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, "No valid ΔV data below threshold", ha='center', va='center')
        plt.axis('off')
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close()
        return filename

    min_val = np.nanmin(V)
    max_val = np.nanmax(V)

    # Choose default isovalue if not provided (slightly above the minimum)
    if level is None:
        level = float(min_val + (max_val - min_val) * 0.1)
        if not np.isfinite(level) or level <= min_val:
            level = float(min_val + 0.1)

    # Convert MJD2000 to years from 2030 (same mapping as scatter plot)
    dep_years = 2030 + (dep_range - dep_range[0]) / 365.25
    fly_years = 2030 + (fly_range - fly_range[0]) / 365.25
    arr_years = 2030 + (arr_range - arr_range[0]) / 365.25

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Use matplotlib contour stacking to approximate an isosurface (no external deps)
    from matplotlib import cm
    from matplotlib.colors import Normalize

    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=min_val, vmax=max_val)
    facecolor = cmap((level - min_val) / max(1e-9, (max_val - min_val)))

    # We'll sample slices along each axis (limit to ~30 per axis for performance)
    max_slices = 30
    x_step = max(1, V.shape[0] // max_slices)
    y_step = max(1, V.shape[1] // max_slices)
    z_step = max(1, V.shape[2] // max_slices)

    # Prepare grids for contouring
    X_xy, Y_xy = np.meshgrid(dep_years, fly_years, indexing='ij')
    X_xz, Z_xz = np.meshgrid(dep_years, arr_years, indexing='ij')
    Y_yz, Z_yz = np.meshgrid(fly_years, arr_years, indexing='ij')

    # Temporary 2D axes for computing contours without drawing to the 3D axes
    fig2 = plt.figure(figsize=(4, 4))
    ax2 = fig2.add_subplot(111)

    found_any = False

    def _iter_contour_polygons(cs):
        """Yield Nx2 arrays of contour vertices from a ContourSet `cs`.
        Handles both modern Matplotlib (`cs.collections`) and older `cs.allsegs`.
        """
        # Preferred API (modern Matplotlib)
        if hasattr(cs, 'collections'):
            for coll in cs.collections:
                for path in coll.get_paths():
                    verts = path.vertices
                    if verts.shape[0] >= 3:
                        yield verts
        else:
            # Fallback API: cs.allsegs is a list (one element per level)
            allsegs = getattr(cs, 'allsegs', None)
            if allsegs and len(allsegs) > 0:
                for seg in allsegs[0]:
                    verts = np.asarray(seg)
                    if verts.shape[0] >= 3:
                        yield verts

    # XY slices (varying arrival index)
    for k in range(0, V.shape[2], z_step):
        z_val = arr_years[k]
        slice2d = V[:, :, k].copy()
        slice2d[np.isnan(slice2d)] = max_val + 1.0
        ax2.clear()
        cs = ax2.contour(X_xy, Y_xy, slice2d, levels=[level])
        for verts2d in _iter_contour_polygons(cs):
            poly3d = np.column_stack((verts2d[:, 0], verts2d[:, 1], np.full(len(verts2d), z_val)))
            poly = Poly3DCollection([poly3d], alpha=alpha)
            poly.set_facecolor(facecolor)
            poly.set_edgecolor('none')
            ax.add_collection3d(poly)
            found_any = True

    # XZ slices (varying flyby index)
    for j in range(0, V.shape[1], y_step):
        y_val = fly_years[j]
        slice2d = V[:, j, :].copy()
        slice2d[np.isnan(slice2d)] = max_val + 1.0
        ax2.clear()
        cs = ax2.contour(X_xz, Z_xz, slice2d, levels=[level])
        for verts2d in _iter_contour_polygons(cs):
            poly3d = np.column_stack((verts2d[:, 0], np.full(len(verts2d), y_val), verts2d[:, 1]))
            poly = Poly3DCollection([poly3d], alpha=alpha)
            poly.set_facecolor(facecolor)
            poly.set_edgecolor('none')
            ax.add_collection3d(poly)
            found_any = True

    # YZ slices (varying departure index)
    for i in range(0, V.shape[0], x_step):
        x_val = dep_years[i]
        slice2d = V[i, :, :].copy()
        slice2d[np.isnan(slice2d)] = max_val + 1.0
        ax2.clear()
        cs = ax2.contour(Y_yz, Z_yz, slice2d, levels=[level])
        for verts2d in _iter_contour_polygons(cs):
            poly3d = np.column_stack((np.full(len(verts2d), x_val), verts2d[:, 0], verts2d[:, 1]))
            poly = Poly3DCollection([poly3d], alpha=alpha)
            poly.set_facecolor(facecolor)
            poly.set_edgecolor('none')
            ax.add_collection3d(poly)
            found_any = True

    plt.close(fig2)

    # If nothing was found (no contour crosses), fall back to colored slices similar to before
    if not found_any:
        min_idx = np.unravel_index(np.nanargmin(V), V.shape)
        x_idx, y_idx, z_idx = min_idx
        facecolors = cmap(norm(V[:, :, z_idx]))
        ax.plot_surface(X_xy, Y_xy, np.full_like(X_xy, arr_years[z_idx]), facecolors=facecolors,
                        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=alpha)
        facecolors2 = cmap(norm(V[:, y_idx, :]))
        ax.plot_surface(X_xz, np.full_like(X_xz, fly_years[y_idx]), Z_xz, facecolors=facecolors2,
                        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=alpha)
        facecolors3 = cmap(norm(V[x_idx, :, :]))
        ax.plot_surface(np.full_like(Y_yz, dep_years[x_idx]), Y_yz, Z_yz, facecolors=facecolors3,
                        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=alpha)

    # Mark global minimum
    min_idx = np.unravel_index(np.nanargmin(V), V.shape)
    ax.scatter(dep_years[min_idx[0]], fly_years[min_idx[1]], arr_years[min_idx[2]],
               c='red', s=100, label=f'Min ΔV: {min_val:.2f} km/s')

    # Auto-scale axes
    ax.set_xlim(dep_years[0], dep_years[-1])
    ax.set_ylim(fly_years[0], fly_years[-1])
    ax.set_zlim(arr_years[0], arr_years[-1])
    ax.set_xlabel('Departure Year')
    ax.set_ylabel('Flyby Year')
    ax.set_zlabel('Arrival Year')
    ax.set_title(f'Flyby Isosurface at ΔV = {level:.2f} km/s (min {min_val:.2f} km/s)')

    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()
    return filename
