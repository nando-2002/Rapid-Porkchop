import os
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import jd2date

def save_porkchop_plot(dep_range, arr_range, dv_matrix, filename=None, dpi=550, n_ticks=6, cmap='CMRmap'):
    # dv_matrix expected shape (n_dep, n_arr)
    Z = np.array(dv_matrix, dtype=float).copy()
    Z[Z == 0] = np.nan  # mask invalid entries
    if filename is None:
        filename = os.path.join(os.path.dirname(__file__), "gpu_porkchop.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    # ensure grid orientation matches (x: dep, y: arr)
    X, Y = np.meshgrid(dep_range, arr_range)
    mesh = ax.pcolormesh(X, Y, Z.T, cmap=cmap, shading='auto', vmin = 5, vmax = 10)
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label('Delta-V (km/s)')

    def mjd2000_to_str(mjd):
        jd = mjd + 2400000.5 + 51544.5
        Y_, M_, D_, hrs, mins, secs = jd2date(jd)
        return f"{int(Y_)}-{int(M_):02d}-{int(D_):02d}"

    # set reasonable ticks
    xticks = np.linspace(dep_range[0], dep_range[-1], n_ticks)
    yticks = np.linspace(arr_range[0], arr_range[-1], n_ticks)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([mjd2000_to_str(x) for x in xticks], rotation=30)
    ax.set_yticklabels([mjd2000_to_str(y) for y in yticks])
    ax.set_xlabel('Departure date (YYYY-MM-DD)')
    ax.set_ylabel('Arrival date (YYYY-MM-DD)')

    # annotate minimum
    if np.any(~np.isnan(Z)):
        min_val = np.nanmin(Z)
        min_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
        min_dep_mjd = dep_range[min_idx[0]]
        min_arr_mjd = arr_range[min_idx[1]]
        ax.plot(min_dep_mjd, min_arr_mjd, 'ro', markersize=5)
        ax.annotate(f"Min {min_val:.2f} km/s", (min_dep_mjd, min_arr_mjd),
                    textcoords="offset points", xytext=(10, -10), color='white')

    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return filename