import numpy as np

# constants replacing astroConstants(2) and astroConstants(4)
AU_KM = 149597870.7                 # km
MU_SUN_KM3_S2 = 1.32712440018e11    # km^3/s^2

_AST_DATA = None 

def _load_asteroid_data_csv(csv_path, *, delimiter=",", skiprows=0, usecols=None):
    """
    Expected columns (numeric, in this exact order):
      [epoch_mjd, sma_AU, ecc, inc_deg, RAAN_deg, argPer_deg, M0_deg, H_mag]
    """
    global _AST_DATA
    if _AST_DATA is None:
        _AST_DATA = np.loadtxt(
            csv_path, delimiter=delimiter, skiprows=skiprows, usecols=usecols, dtype=float
        )
        if _AST_DATA.ndim == 1:
            _AST_DATA = _AST_DATA[None, :]
    return _AST_DATA

def eph_asteroids(time_mjd2000, asteroid_id, csv_path,
                  *, id_is_one_based=True, delimiter=",", skiprows=0, usecols=None):
    data = _load_asteroid_data_csv(
        csv_path, delimiter=delimiter, skiprows=skiprows, usecols=usecols
    )

    idx = asteroid_id - 1 if id_is_one_based else asteroid_id
    row = data[idx, :]

    # CSV order per your note / MATLAB comment: epoch,sma,ecc,inc,RAAN,argPer,M,H [file:3]
    epoch0_mjd = row[0]
    sma_km = row[1] * AU_KM
    ecc = row[2]
    inc = np.deg2rad(row[3])
    node = np.deg2rad(row[4])   # RAAN
    w = np.deg2rad(row[5])      # argument of periapsis
    M0 = np.deg2rad(row[6])     # mean anomaly at epoch
    H = row[7]

    pi2 = 2.0 * np.pi
    orbital_period = pi2 * np.sqrt((sma_km**3) / MU_SUN_KM3_S2)  # s
    mean_motion = pi2 / orbital_period                           # rad/s

    # MATLAB: epoch0_mjd2000 = epoch0_mjd - 51544.5 [file:3]
    epoch0_mjd2000 = epoch0_mjd - 51544.5

    # Mean anomaly at requested time [file:3]
    M = M0 + mean_motion * (time_mjd2000 - epoch0_mjd2000) * 86400.0
    M = M - np.fix(M / pi2) * pi2  # reduce to first revolution [file:3]

    # Kepler solve: 5 Newton iterations (same as MATLAB) [file:3]
    Ek = M + ecc * np.sin(M) / (1.0 - np.sin(M + ecc) + np.sin(M))
    for _ in range(5):
        F1 = ecc * np.cos(Ek) - 1.0
        Ek = Ek + (Ek - ecc * np.sin(Ek) - M) / F1

    # True anomaly [file:3]
    f = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.tan(0.5 * Ek),
        np.sqrt(1.0 - ecc)
    )
    f = np.mod(f, 2.0 * np.pi)

    kep = np.array([sma_km, ecc, inc, node, w, f], dtype=float)
    M = np.mod(M, 2.0 * np.pi)

    # H -> diameter polynomial (copied exactly) [file:3]
    d = (-2.522e-2 * H**5
         + 3.2961 * H**4
         - 1.7249e2 * H**3
         + 4.5231e3 * H**2
         - 5.9509e4 * H
         + 3.1479e5)

    density = 2.0 * 1000.0  # MATLAB: 2 kg/dm^3 -> 2000 kg/m^3 [file:3]
    mass = 4.0 * np.pi / 3.0 * (0.5 * d)**3 * density

    return kep, mass, M

# kep, mass, M = eph_asteroids(
#     time_mjd2000=0,
#     asteroid_id=257323,
#     csv_path="AsteroidsElements_num.csv",
#     skiprows=0,
#     usecols=[2,3,4,5,6,7,8,9],  # adjust to your file
#     id_is_one_based=True
# )

# print(kep)