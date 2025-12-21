# gpu/gpu_lambert.py

import math
from numba import cuda

#-------------------------
# CUDA Stumpff Functions
#-------------------------

@cuda.jit(device=True)
def C_dev(z):
    if z > 0.0:
        root = math.sqrt(z)
        return (1.0 - math.cos(root)) / z
    elif z < 0.0:
        root = math.sqrt(-z)
        return (math.cosh(root) - 1.0) / (-z)
    else:
        return 0.5


@cuda.jit(device=True)
def S_dev(z):
    if z > 0.0:
        root = math.sqrt(z)
        return (root - math.sin(root)) / (z * root)
    elif z < 0.0:
        root = math.sqrt(-z)
        return (math.sinh(root) - root) / ((-z) * root)
    else:
        return 1.0 / 6.0


@cuda.jit(device=True)
def y_dev(z, r1, r2, A):
    # y(z, r1, r2, A) = r1 + r2 + A * ((z * S(z) - 1) / sqrt(C(z)))
    Cz = C_dev(z)
    if Cz <= 0.0:
        # Avoid sqrt of non-positive due to numerical issues
        return r1 + r2
    return r1 + r2 + A * ((z * S_dev(z) - 1.0) / math.sqrt(Cz))


@cuda.jit(device=True)
def F_dev(z, r1, r2, A, mu, delT):
    """
    F(z) from lambert.py, in scalar/device form.
    F(z) = ( (y/C)^1.5 ) * S + A*sqrt(y) - sqrt(mu)*delT
    """
    Cz = C_dev(z)
    yv = y_dev(z, r1, r2, A)
    if Cz == 0.0 or yv <= 0.0:
        # Error Handling; just return something large
        return 1e9
    term1 = (yv / Cz) ** 1.5 * S_dev(z)
    term2 = A * math.sqrt(yv)
    return term1 + term2 - math.sqrt(mu) * delT


@cuda.jit(device=True)
def Fp_dev(z, r1, r2, A, mu, delT):
    """
    F'(z) from lambert.py, scalar/device form.
    """
    if z == 0.0:
        y0 = y_dev(0.0, r1, r2, A)
        if y0 <= 0.0:
            return 1e9
        term1 = math.sqrt(2.0) / 40.0 * (y0 ** 1.5)
        term2 = A / 8.0 * (math.sqrt(y0) + A * math.sqrt(1.0 / (2.0 * y0)))
        return term1 + term2
    else:
        Cz = C_dev(z)
        Sz = S_dev(z)
        yz_val = y_dev(z, r1, r2, A)
        if Cz == 0.0 or yz_val <= 0.0:
            return 1e9 # Error handling 

        # (y/C)^(3/2)
        base = yz_val / Cz
        termA = base ** 1.5

        # 1/(2z) * (C - 3 S / (2C))
        termB = (1.0 / (2.0 * z)) * (Cz - 3.0 * Sz / (2.0 * Cz))

        # 3 S^2 / (4 C)
        termC = 3.0 * Sz * Sz / (4.0 * Cz)

        part1 = termA * (termB + termC)

        part2_inner1 = 3.0 * Sz / Cz * math.sqrt(yz_val)
        part2_inner2 = A * math.sqrt(Cz / yz_val)
        part2 = A / 8.0 * (part2_inner1 + part2_inner2)

        return part1 + part2


#-------------------------
# CUDA Lambert solver
#-------------------------

@cuda.jit(device=True)
def lam_solve_cuda(R1x, R1y, R1z,
                  R2x, R2y, R2z,
                  DELTA_T, MU,
                  V1, V2,
                  MAX_ITER= 250,
                  traj_pro=True):
    """
    Device version of lam_solve.
    R1, R2 split into components.
    V1, V2 are length-3 cuda.local arrays passed by caller.
    """

    # R_CROSS = np.cross(R1, R2) (only z-component needed)
    RCROSS_z = R1x * R2y - R1y * R2x

    # Magnitudes
    R1_MAG = math.sqrt(R1x * R1x + R1y * R1y + R1z * R1z)
    R2_MAG = math.sqrt(R2x * R2x + R2y * R2y + R2z * R2z)

    # THETA = arccos(dot / (|R1||R2|))
    dot = R1x * R2x + R1y * R2y + R1z * R2z
    cos_th = dot / (R1_MAG * R2_MAG)
    if cos_th > 1.0:
        cos_th = 1.0
    elif cos_th < -1.0:
        cos_th = -1.0
    THETA = math.acos(cos_th)

    # if (R_CROSS[2] >= 0) and (TRAJ == 'retro'): THETA = 2π - THETA
    # elif (R_CROSS[2] < 0) and (TRAJ == 'pro'):  THETA = 2π - THETA
    if RCROSS_z >= 0.0 and (not traj_pro):
        THETA = 2.0 * math.pi - THETA
    elif RCROSS_z < 0.0 and traj_pro:
        THETA = 2.0 * math.pi - THETA

    # Semi-major axis factor A
    denom = 1.0 - math.cos(THETA)
    if denom == 0.0:
        # Error handling; set some default or return
        V1[0] = 0.0; V1[1] = 0.0; V1[2] = 0.0
        V2[0] = 0.0; V2[1] = 0.0; V2[2] = 0.0
        return

    A = math.sin(THETA) * math.sqrt((R1_MAG * R2_MAG) / denom)

    # Newton initial guess for z
    z = 0.0
    while F_dev(z, R1_MAG, R2_MAG, A, MU, DELTA_T) < 0.0:
        z += 0.1
        if z > 1.0e6:
            # mimic ValueError and return 0 
            V1[0] = 0.0; V1[1] = 0.0; V1[2] = 0.0
            V2[0] = 0.0; V2[1] = 0.0; V2[2] = 0.0
            return

    # Newton Raphson iterations
    for _ in range(MAX_ITER):
        Fz_val = F_dev(z, R1_MAG, R2_MAG, A, MU, DELTA_T)
        Fpz_val = Fp_dev(z, R1_MAG, R2_MAG, A, MU, DELTA_T)
        if Fpz_val == 0.0:
            break
        z_new = z - Fz_val / Fpz_val
        z = z_new

    Y = y_dev(z, R1_MAG, R2_MAG, A)

    # Lagrange functions
    f = 1.0 - Y / R1_MAG
    g = A * math.sqrt(Y / MU)
    gdot = 1.0 - Y / R2_MAG

    inv_g = 1.0 / g

    # V1 = (1/g)*(R2 - f*R1)
    V1[0] = inv_g * (R2x - f * R1x)
    V1[1] = inv_g * (R2y - f * R1y)
    V1[2] = inv_g * (R2z - f * R1z)

    # V2 = (1/g)*(gdot*R2 - R1)
    V2[0] = inv_g * (gdot * R2x - R1x)
    V2[1] = inv_g * (gdot * R2y - R1y)
    V2[2] = inv_g * (gdot * R2z - R1z)

    # there is no return for CUDA functions