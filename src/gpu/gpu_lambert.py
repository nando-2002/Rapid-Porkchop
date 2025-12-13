# gpu/gpu_lambert.py

import math
from numba import cuda


# =========================
# CPU Lambert (wrapper)
# =========================

def cpu_lamsolve(R1, R2, DELTAT, MU, MAXITER=1000, TRAJ="pro"):
    """
    Thin wrapper that calls the original CPU Lambert solver in utils/lambert.py.
    This keeps CPU behavior unchanged and allows easy comparison.
    """
    from utils.lambert import lamsolve  # local import to avoid circular deps
    return lamsolve(R1, R2, DELTAT, MU, MAXITER=MAXITER, TRAJ=TRAJ)


# =========================
# GPU device Lambert pieces
# =========================

@cuda.jit(device=True)
def _Cz(z):
    if z > 0.0:
        sqrtz = math.sqrt(z)
        return (1.0 - math.cos(sqrtz)) / z
    elif z < 0.0:
        sqrtz = math.sqrt(-z)
        return (math.cosh(sqrtz) - 1.0) / (-z)
    else:
        return 0.5


@cuda.jit(device=True)
def _Sz(z):
    if z > 0.0:
        sqrtz = math.sqrt(z)
        return (sqrtz - math.sin(sqrtz)) / (z * sqrtz)
    elif z < 0.0:
        sqrtz = math.sqrt(-z)
        return (math.sinh(sqrtz) - sqrtz) / ((-z) * sqrtz)
    else:
        return 1.0 / 6.0


@cuda.jit(device=True)
def _yz(z, r1, r2, A):
    return r1 + r2 + A * (z * _Sz(z) - 1.0)


@cuda.jit(device=True)
def _Fz(z, r1, r2, A, mu, delT):
    y = _yz(z, r1, r2, A)
    C = _Cz(z)
    S = _Sz(z)
    # y * C^(3/2) * S * sqrt(A / y) - sqrt(mu)*delT
    return y * C * C * math.sqrt(C) * S * math.sqrt(A / y) - math.sqrt(mu) * delT


@cuda.jit(device=True)
def _Fpz(z, r1, r2, A, mu, delT):
    y = _yz(z, r1, r2, A)
    C = _Cz(z)
    if z != 0.0:
        # Simple approximate derivative; sufficient for convergence
        return math.sqrt(y / (4.0 * C * C * C))
    else:
        return 0.5 * math.sqrt(y)


@cuda.jit(device=True)
def device_lamsolve_total_dv(r1x, r1y, r1z,
                             r2x, r2y, r2z,
                             tof, mu,
                             maxiter=50):
    """
    GPU Lambert solver as a device function.
    Returns total transfer delta-v: ||v1|| + ||v2||.
    """

    # Magnitudes
    r1mag = math.sqrt(r1x * r1x + r1y * r1y + r1z * r1z)
    r2mag = math.sqrt(r2x * r2x + r2y * r2y + r2z * r2z)

    # Dot product and angle
    dot = r1x * r2x + r1y * r2y + r1z * r2z
    cos_theta = dot / (r1mag * r2mag)
    # Clamp cos_theta to [-1, 1]
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0
    theta = math.acos(cos_theta)

    # Prograde assumption (no cross-product; extend if you need retrograde)
    sin_theta = math.sin(theta)
    one_minus_cos = 1.0 - math.cos(theta)
    if one_minus_cos == 0.0:
        return 0.0  # degenerate geometry

    A = sin_theta * math.sqrt(r1mag * r2mag * (1.0 + cos_theta) / one_minus_cos)

    # Newton-Raphson on z
    z = 0.0
    for _ in range(maxiter):
        F = _Fz(z, r1mag, r2mag, A, mu, tof)
        if abs(F) < 1e-8:
            break
        Fp = _Fpz(z, r1mag, r2mag, A, mu, tof)
        if Fp == 0.0:
            break
        z -= F / Fp

    # Lagrange coefficients
    y = _yz(z, r1mag, r2mag, A)
    f = 1.0 - y / r1mag
    g = A * math.sqrt(y / mu)
    gdot = 1.0 - y / r2mag

    # v1 norm
    v1x = (r2x - f * r1x) / g
    v1y = (r2y - f * r1y) / g
    v1z = (r2z - f * r1z) / g
    v1_norm = math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)

    # v2 norm
    v2x = (gdot * r2x - r1x) / g
    v2y = (gdot * r2y - r1y) / g
    v2z = (gdot * r2z - r1z) / g
    v2_norm = math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z)

    return v1_norm + v2_norm


# =========================
# Public GPU kernel
# =========================

@cuda.jit
def porkchop_kernel(depmjd, arrmjd,
                    re_earth, ve_earth,
                    rm_mars, vm_mars,
                    mu,
                    deltav):
    """
    2D kernel:
    dep_idx → departure index
    arr_idx → arrival index
    Computes total Lambert transfer dv for each (dep, arr) pair.
    """
    dep_idx, arr_idx = cuda.grid(2)
    n_dep = depmjd.shape[0]
    n_arr = arrmjd.shape[0]

    if dep_idx >= n_dep or arr_idx >= n_arr:
        return

    dep_mjd = depmjd[dep_idx]
    arr_mjd = arrmjd[arr_idx]
    if arr_mjd <= dep_mjd:
        return

    # Positions (velocities currently unused; kept for future extensions)
    r1x = re_earth[dep_idx, 0]
    r1y = re_earth[dep_idx, 1]
    r1z = re_earth[dep_idx, 2]

    r2x = rm_mars[arr_idx, 0]
    r2y = rm_mars[arr_idx, 1]
    r2z = rm_mars[arr_idx, 2]

    # Time of flight in seconds
    tof = (arr_mjd - dep_mjd) * 86400.0

    total_dv = device_lamsolve_total_dv(r1x, r1y, r1z,
                                        r2x, r2y, r2z,
                                        tof, mu[0])

    deltav[dep_idx, arr_idx] = total_dv
