import os
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import jd2date

def plot_porkchop(dep_range, arr_range, delta_v_matrix, dep_min, arr_min, min_dv, name):
    """
    Plots the porkchop plot using matplotlib.

    Parameters:
    dep_range (np.ndarray): Array of departure MJDs.
    arr_range (np.ndarray): Array of arrival MJDs.
    delta_v_matrix (np.ndarray): 2D array of delta-v values.
    dep_min (float): Departure MJD corresponding to minimum delta-v.
    arr_min (float): Arrival MJD corresponding to minimum delta-v.
    min_dv (float): Minimum delta-v value.
    """
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(arr_range, dep_range)
    cp = plt.contourf(X, Y, delta_v_matrix, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Delta-V (km/s)')
    plt.plot(arr_min, dep_min, 'ro')  # Mark the minimum delta-v point
    plt.text(arr_min, dep_min, f'Min Δv: {min_dv:.2f} km/s', color='white', fontsize=10,
             verticalalignment='bottom', horizontalalignment='right')

    def mjd2000_to_yyyymmdd(mjd):
        # convert MJD2000 -> JD -> Y, M, D and format as YYYYMMDD
        jd = mjd + 2400000.5 + 51544.5
        Y_, M_, D_, hrs, mins, secs = jd2date(jd)
        return f"{int(Y_):04d}{int(M_):02d}{int(D_):02d}"

    # set reasonable ticks (use 6 ticks by default)
    n_ticks = 6
    xticks = np.linspace(arr_range[0], arr_range[-1], n_ticks)
    yticks = np.linspace(dep_range[0], dep_range[-1], n_ticks)
    plt.xticks(xticks, [mjd2000_to_yyyymmdd(x) for x in xticks], rotation=30)
    plt.yticks(yticks, [mjd2000_to_yyyymmdd(y) for y in yticks])

    plt.xlabel('Arrival date (YYYYMMDD)')
    plt.ylabel('Departure date (YYYYMMDD)')
    plt.title('Porkchop Plot')
    plt.grid()
    plt.savefig(name)