import numpy as np

def car2kep(r, v, mu):

    '''
    Returns Keplerian orbital elements when given Cartesian coordinates for
    position and velocity. 

    The algorithm is taken from "Orbital Mechanics for Engineering Students" by 
    Howard D. Curtis, Algorithm 4.2 

    Input Parameters
    r : ndarray 
        Position of an orbiting object
    v : ndarray 
        Velocity of an orbiting object   
    mu : float
        Gravitational parameter (km^3/s^2)

    Returns: 
        h_mag, semi_major_axis, ecc_mag, ran, inclin, arg_per, true_anom

    THIS COMMENT IS PENDING I AM TOO LAZY TO WRITE THIS NOW

    Author: 
        Prthik Karthikeyan 07/12/25
    '''
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    v_radial = np.dot(r, v) / r_mag
    
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    inclin = np.arccos(h[2] / h_mag)
    
    K = np.array([0, 0, 1])
    N = np.cross(K, h)
    N_mag = np.linalg.norm(N)

    ran = np.arccos( N[0] / N_mag)
    if (N[1] < 0):
        ran = 2*np.pi - ran
    
    #ecc = ( 1 / mu ) * ( ( v_mag**2 - (mu / r_mag)*r - r_mag*v_radial*v ) )
    brack1 = ( 1 / mu )
    brack2 = np.dot((v_mag**2 - (mu / r_mag)), r)
    brack3 = np.dot((r_mag*v_radial),v)
    ecc = brack1 * ( brack2 - brack3)


    ecc_mag = np.linalg.norm(ecc)

    ecc_mag_verif = np.sqrt( 1 + (h_mag**2 / mu**2) * (v_mag**2 - (2*mu / r_mag) )  )
    
    arg_per = np.arccos( np.dot(N, ecc) / (N_mag * ecc_mag))

    if (ecc[2] < 0):
        arg_per = 2*np.pi - arg_per

    true_anom = np.arccos( np.dot(ecc, r) / (ecc_mag * r_mag) )
    
    if (v_radial < 0):
        true_anom = 2*np.pi - true_anom

    perigee = (h_mag**2 / mu) * (1 / ( 1 + ecc_mag * np.cos(0) ) )
    apogee = (h_mag**2 / mu) * (1 / ( 1 + ecc_mag * np.cos(np.pi) ) )
    semi_major_axis = (perigee + apogee) * 0.5

    print(f"(a) R Magnitude : ", r_mag)
    print(f"(b) V Magnitude : ", v_mag)
    print(f"(c) V Radial : ", v_radial)
    print(f"(d) h : ", h)
    print(f"(e) h Magnitude : ", h_mag)
    print(f"(f) Inclination : ", np.rad2deg(inclin))
    print(f"(g) N : ", N)
    print(f"(h) N Magnitude : ", N_mag)
    print(f"(i) Right Ascension : ", np.rad2deg(ran))
    print(f"(j) Eccentricity : ", ecc)
    print(f"(k) Eccentricity Magnitude : ", ecc_mag, " or ", ecc_mag_verif)
    print(f"(l) Argument of Perigee : ", np.rad2deg(arg_per))
    print(f"(m) True Anomaly : ", np.rad2deg(true_anom))
    print(f" Also, Semi Major Axis : ", semi_major_axis)

    return h_mag, semi_major_axis, ecc_mag, ran, inclin, arg_per, true_anom




'''
r1 = np.array([-6045, -3490, 2500])
v1 = np.array([-3.457, 6.618, 2.533])

car2kep(r1, v1, 398600)

page 217 curtis
'''