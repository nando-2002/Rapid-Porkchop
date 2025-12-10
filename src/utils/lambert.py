# Local Python library (not used anymore)
# from stumpff_func import F, Fp, y 

# PyPi Library Import
import numpy as np

def lam_solve(R1, R2, DELTA_T, MU, MAX_ITER = 1000, TRAJ = 'pro'):
    
    '''
    Solves Lambert's problem using universal variables to find initial and final
    velocity vectors (V1, V2) given the input of initial and final position vectors
    (R1, R2) and time elapsed between the 2 position (Delta T). 

    The algorithm is taken from "Orbital Mechanics for Engineering Students" by 
    Howard D. Curtis, Algorithm 5.2 and Example 5.2

    Input Parameters
    R1, R2 : ndarray
        Initial and final position vectors (km)
    DELTA_T : float
        Time of flight from R1 to R2 (s)
    MU : float
        Gravitational parameter (km^3/s^2)
    MAX_ITER : int, optional
        Maximum number of iterations for Newton's method
    TRAJ string : str, optional
        'pro' for prograde orbit, 'retro' for retrograde orbit (default is 'pro')
    
    Returns:
    V1, V2 : ndarray
        Initial and final velocity vectors (km/s) 
    
    Author:
        Prthik Karthikeyan 02/12/2025
    '''


    R_CROSS = np.cross(R1, R2)
    R1_MAG = np.linalg.norm(R1)
    R2_MAG = np.linalg.norm(R2)

    # Angle between the position vectors in rad
    # (R1 X R2) Z - Component influences the calculation of THETA  

    THETA = np.arccos( np.dot(R1, R2) / (R1_MAG*R2_MAG) )
    
    if (R_CROSS[2] >= 0) and (TRAJ == 'retro'):
        THETA = 2*np.pi - THETA
    elif (R_CROSS[2] < 0) and (TRAJ == 'pro'):
        THETA = 2*np.pi - THETA

    # Semi-major axis (km)
    A = np.sin( THETA ) * np.sqrt( (R1_MAG*R2_MAG) / (1 - np.cos(THETA)) ) 

    # Initial guess for z, before newton raphson solver
    z = 0

    while F(z, R1_MAG, R2_MAG, A, MU, DELTA_T) < 0 :
        z = z + 0.1
        if ( z > 1e6):
            raise ValueError("ERROR : Infinite initial guess for z")
    
    for i in range(MAX_ITER):
        z = z - F(z, R1_MAG, R2_MAG, A, MU, DELTA_T) / Fp(z, R1_MAG, R2_MAG, A, MU, DELTA_T)
 
    Y = y(z, R1_MAG, R2_MAG, A)

    # Lagrange functions
    f = 1 - Y / R1_MAG
    g = A * np.sqrt( Y / MU )
    gdot = 1 - Y / R2_MAG

    # Final velocity vectors
    V1 = (1 / g) * (R2 - f*R1)
    V2 = (1 / g) * (gdot*R2 - R1)

    return V1, V2

def C(z):
    if (z > 0):
        return ( ( 1 - np.cos(np.sqrt(z)) ) / (z) )
    elif (z < 0):
        return ( ( np.cosh(np.sqrt(-z)) - 1 ) / ( -z ) )
    else (z == 0):
        return 0.5
    
def S(z):
    if (z > 0):
        return ( ( np.sqrt(z) - np.sin(np.sqrt(z)) ) / (z)**(3/2) )
    elif (z < 0):
        return ( ( np.sinh(np.sqrt(-z)) - np.sqrt(-z) ) / ( -z )**(3/2) )
    else (z == 0):
        return 1/6

def y(z, r1, r2, A):
    return (
        r1 + r2 + A * (
            (z * S(z) - 1) / ( np.sqrt( C(z) ))
        )
    ) 

def F(z, r1, r2, A, mu, delT):
    return (
        ( (y(z, r1, r2, A) / C(z))**(3/2) )*S(z) + A*np.sqrt( y(z, r1, r2, A) ) - (np.sqrt(mu) * delT)
    ) 

def Fp(z, r1, r2, A, mu, delT):
    if z == 0:
        return np.sqrt(2) / 40 * y(0, r1, r2, A) ** 1.5 + A / 8 * (np.sqrt(y(0, r1, r2, A)) + A * np.sqrt(1 / 2 / y(0, r1, r2, A)))
    else:
        return (y(z, r1, r2, A) / C(z)) ** 1.5 * (1 / 2 / z * (C(z) - 3 * S(z) / 2 / C(z)) + 3 * S(z) ** 2 / 4 / C(z)) + A / 8 * (3 * S(z) / C(z) * np.sqrt(y(z, r1, r2, A)) + A * np.sqrt(C(z) / y(z, r1, r2, A)))

    