import numpy as np

def eph_asteroid_2009hc98(time_mjd2000):
    """
    Ephemerides for asteroid 2009 HC98 from JPL data (ID 257323)
    
    Parameters:
    -----------
    time_mjd2000 : float
        Time of ephemeris in MJD2000 [days]
    
    Returns:
    --------
    kep : array [6]
        Keplerian parameters: [a, e, i, Om, om, f]
        a: semimajor axis [km]
        e: eccentricity
        i: inclination [rad]
        Om: RAAN [rad] 
        om: argument of pericenter [rad]
        f: true anomaly [rad]
    mass : float
        Mass [kg] (estimated from H magnitude)
    M : float
        Mean anomaly at time [rad]
    """
    
    # Hardcoded JPL data for 2009 HC98
    epoch0_mjd = 60200           # Epoch MJD
    a_AU = 3.0669                # Semi-major axis [AU]
    e = 0.2287                   # Eccentricity
    inc_deg = 4.9604             # Inclination [deg]
    Om_deg = 124.1222            # RAAN [deg]
    w_deg = 151.6065             # Argument of pericenter [deg]
    M0_deg = 212.3667            # Mean anomaly at epoch [deg]
    H = 16.6600                  # Absolute magnitude
    
    # Constants (no scipy)
    AU = 1.495978707e8           # km
    mu_sun_km = 1.3271244e11     # km^3/s^2 (solar gravitational parameter)
    pi = np.pi
    
    # Convert to consistent units
    a = a_AU * AU                # Semi-major axis [km]
    inc = np.deg2rad(inc_deg)    # Inclination [rad]
    node = np.deg2rad(Om_deg)    # RAAN [rad]
    w = np.deg2rad(w_deg)        # Argument of pericenter [rad]
    M0 = np.deg2rad(M0_deg)      # Mean anomaly at epoch [rad]
    
    pi2 = 2 * pi
    
    # Orbital parameters
    orbital_period = pi2 * np.sqrt(a**3 / mu_sun_km)  # seconds
    mean_motion = pi2 / orbital_period                # rad/s
    
    # Time conversions
    epoch0_mjd2000 = epoch0_mjd - 51544.5
    epoch_mjd2000 = time_mjd2000
    
    # Mean anomaly at requested time
    delta_t = (epoch_mjd2000 - epoch0_mjd2000) * 86400  # seconds
    M = M0 + mean_motion * delta_t                      # rad
    
    # Reduce to [0, 2pi]
    nrev = np.fix(M / pi2)
    M = M - nrev * pi2
    M = np.mod(M, pi2)
    
    # Solve Kepler's equation (Newton iteration, max 5 iterations)
    # Fixed initial guess (original had division by zero issue)
    Ek = M                          # Simple initial guess
    for i in range(5):
        F1 = Ek - e * np.sin(Ek) - M
        Ek = Ek - F1 / (1 - e * np.cos(Ek))
    
    # True anomaly from eccentric anomaly
    f = 2 * np.arctan2(np.sqrt(1 + e) * np.tan(Ek / 2), np.sqrt(1 - e))
    f = np.mod(f, pi2)
    
    # Keplerian elements
    kep = np.array([a, e, inc, node, w, f])
    
    # Mass estimation from H-magnitude (H-d relation)
    d = (-2.522e-2 * H**5 + 3.2961 * H**4 - 1.7249e2 * H**3 + 
         4.5231e3 * H**2 - 5.9509e4 * H + 3.1479e5)  # km
    density = 2100  # kg/m^3
    mass = (4 * pi / 3) * (0.5 * d * 1000)**3 * density  # kg
    
    return kep, mass, M

# Example usage
# if __name__ == "__main__":
#     time = 60200  # MJD2000 (epoch time)
#     kep, mass, M = eph_asteroid_2009hc98(time)
#     print("Keplerian elements:", kep)
#     print("Mass [kg]:", mass)
#     print("Mean anomaly [rad]:", M)
